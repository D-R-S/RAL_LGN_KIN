// ============================================================================
//  lgn/urdf_loader.hpp
//  Parses a URDF file or XML string into a KinematicTree.
//  Dependency: TinyXML2
// ============================================================================
#pragma once
#include "kinematic_tree.hpp"
#include <tinyxml2.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace lgn {

class URDFLoader {
public:
    static KinematicTree from_file(const std::string& path) {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS)
            throw std::runtime_error("URDFLoader: cannot open " + path);
        return parse(doc);
    }

    static KinematicTree from_string(const std::string& xml) {
        tinyxml2::XMLDocument doc;
        if (doc.Parse(xml.c_str()) != tinyxml2::XML_SUCCESS)
            throw std::runtime_error("URDFLoader: XML parse failed");
        return parse(doc);
    }

private:
    struct Wire {
        std::string parent_link;
        std::string child_link;
        int         joint_idx;
    };

    static KinematicTree parse(tinyxml2::XMLDocument& doc) {
        auto* robot = doc.FirstChildElement("robot");
        if (!robot)
            throw std::runtime_error("URDFLoader: no <robot> element");

        KinematicTree tree;
        std::unordered_map<std::string, int> lmap;
        std::vector<Wire> wires;
        int dof_count = 0;

        // ── Pass 1: links ────────────────────────────────────────────────────
        for (auto* el = robot->FirstChildElement("link"); el;
             el = el->NextSiblingElement("link"))
        {
            const char* na = el->Attribute("name");
            if (!na) continue;
            Link lk;
            lk.name = na;

            auto* inertial = el->FirstChildElement("inertial");
            if (inertial) {
                auto* mass_el = inertial->FirstChildElement("mass");
                if (mass_el && mass_el->Attribute("value"))
                    lk.inertial.mass = std::stod(mass_el->Attribute("value"));

                auto* origin = inertial->FirstChildElement("origin");
                if (origin) lk.inertial.com = parse_xyz(origin);

                auto* iten = inertial->FirstChildElement("inertia");
                if (iten) lk.inertial.inertia = parse_inertia(iten);
            }
            lmap[lk.name] = tree.add_link(lk);
        }

        // ── Pass 2: joints ───────────────────────────────────────────────────
        for (auto* el = robot->FirstChildElement("joint"); el;
             el = el->NextSiblingElement("joint"))
        {
            const char* na = el->Attribute("name");
            const char* ty = el->Attribute("type");
            if (!na || !ty) continue;

            Joint jt;
            jt.name = na;

            std::string ts(ty);
            if      (ts == "revolute")   jt.type = JointType::Revolute;
            else if (ts == "continuous") jt.type = JointType::Continuous;
            else if (ts == "prismatic")  jt.type = JointType::Prismatic;
            else if (ts == "floating")   jt.type = JointType::Floating;
            else if (ts == "planar")     jt.type = JointType::Planar;
            else if (ts == "ball")       jt.type = JointType::Ball;   // A3a
            else                         jt.type = JointType::Fixed;

            jt.dof_count = joint_dof_count(jt.type);
            if (jt.dof_count > 0) {
                jt.dof_index  = dof_count;
                dof_count    += jt.dof_count;
            }

            // Fixed transform: parent link origin → joint origin
            auto* origin = el->FirstChildElement("origin");
            Vec3 xyz = Vec3::Zero(), rpy = Vec3::Zero();
            if (origin) {
                xyz = parse_xyz(origin);
                rpy = parse_rpy(origin);
            }
            jt.T_parent_joint = T_from_Rp(rpy_to_rot(rpy), xyz);

            // Joint axis (default z; ignored for ball/floating joints)
            auto* axis_el = el->FirstChildElement("axis");
            if (axis_el && axis_el->Attribute("xyz"))
                jt.axis = parse_vec3(axis_el->Attribute("xyz")).normalized();
            else
                jt.axis = Vec3::UnitZ();

            // Joint limits
            auto* lim = el->FirstChildElement("limit");
            if (lim) {
                auto da = [&](const char* a, double def) {
                    const char* v = lim->Attribute(a);
                    return v ? std::stod(v) : def;
                };
                jt.limits.lower      = da("lower",    -M_PI);
                jt.limits.upper      = da("upper",     M_PI);
                jt.limits.velocity   = da("velocity", 10.0);
                jt.limits.effort     = da("effort",   10.0);
                jt.limits.has_limits = true;
            } else {
                jt.limits.has_limits = (jt.type != JointType::Continuous
                                     && jt.type != JointType::Ball);
            }

            auto* par = el->FirstChildElement("parent");
            auto* chi = el->FirstChildElement("child");
            if (!par || !chi) continue;

            std::string par_name = par->Attribute("link") ? par->Attribute("link") : "";
            std::string chi_name = chi->Attribute("link") ? chi->Attribute("link") : "";
            if (!lmap.count(chi_name)) continue;

            jt.child_link = lmap[chi_name];
            int jidx = tree.add_joint(jt);
            wires.push_back({ par_name, chi_name, jidx });
        }

        // ── Pass 3: wire parent→child, find root ─────────────────────────────
        std::unordered_set<int> has_parent;

        for (const Wire& w : wires) {
            if (!lmap.count(w.parent_link)) continue;
            int par_idx = lmap[w.parent_link];
            int chi_idx = tree.joint(w.joint_idx).child_link;
            tree.link(par_idx).child_joints.push_back(w.joint_idx);
            tree.link(chi_idx).parent_joint = w.joint_idx;
            has_parent.insert(chi_idx);
        }

        int root = 0;
        for (auto& [name, idx] : lmap)
            if (!has_parent.count(idx)) { root = idx; break; }
        tree.set_root(root);
        tree.set_dof_count(dof_count);

        // Build dof_names
        std::vector<std::string> dnames(dof_count);
        for (const Wire& w : wires) {
            const Joint& jt = tree.joint(w.joint_idx);
            if (jt.dof_index < 0) continue;
            for (int d = 0; d < jt.dof_count; ++d) {
                int idx = jt.dof_index + d;
                if (idx < dof_count)
                    dnames[idx] = (jt.dof_count > 1)
                        ? jt.name + "_" + std::to_string(d)
                        : jt.name;
            }
        }
        tree.set_dof_names(dnames);
        tree.finalize();

        std::cout << "[lgn] Loaded: "
                  << tree.n_links()     << " links, "
                  << tree.n_dof()       << " DOFs, "
                  << tree.tips().size() << " tip(s)\n";
        return tree;
    }

    // ── XML helpers ──────────────────────────────────────────────────────────
    static Vec3 parse_xyz(tinyxml2::XMLElement* el) {
        const char* a = el->Attribute("xyz");
        return a ? parse_vec3(a) : Vec3::Zero();
    }
    static Vec3 parse_rpy(tinyxml2::XMLElement* el) {
        const char* a = el->Attribute("rpy");
        return a ? parse_vec3(a) : Vec3::Zero();
    }
    static Vec3 parse_vec3(const char* s) {
        std::istringstream ss(s);
        double x, y, z; ss >> x >> y >> z;
        return { x, y, z };
    }
    static Mat3 parse_inertia(tinyxml2::XMLElement* el) {
        auto g = [&](const char* a) {
            const char* v = el->Attribute(a);
            return v ? std::stod(v) : 0.0;
        };
        Mat3 I;
        I << g("ixx"), g("ixy"), g("ixz"),
             g("ixy"), g("iyy"), g("iyz"),
             g("ixz"), g("iyz"), g("izz");
        return I;
    }
};

} // namespace lgn
