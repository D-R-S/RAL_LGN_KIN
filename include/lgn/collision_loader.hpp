// ============================================================================
//  lgn/collision_loader.hpp
//  Parses URDF <collision> elements and populates a ContactWorld.
//
//  Kept separate from contacts.hpp (pure math) and urdf_loader.hpp
//  (kinematic structure) to avoid circular dependencies.
//
//  Include order when using all three:
//    #include <lgn/urdf_loader.hpp>     // kinematic tree from URDF
//    #include <lgn/contacts.hpp>        // ContactWorld, primitives
//    #include <lgn/collision_loader.hpp> // populate ContactWorld from URDF
//
//  Supported URDF collision shapes:
//    <sphere radius="r"/>
//    <box size="x y z"/>                → Box{half_extents}
//    <cylinder radius="r" length="l"/>  → Capsule{r, l/2}  (axis = local z)
//    <mesh .../>                        → silently skipped; add manually
//
//  Example:
//    auto tree = lgn::URDFLoader::from_file("robot.urdf");
//    lgn::ContactWorld world;
//    lgn::load_collision_geometry("robot.urdf", tree, world);
//    world.add_static_plane(lgn::Vec3::UnitY(), 0.0); // ground
// ============================================================================
#pragma once
#include "contacts.hpp"    // ContactWorld, Sphere, Capsule, Box
#include "core.hpp"        // T_from_Rp, rpy_to_rot
#include <tinyxml2.h>
#include <sstream>
#include <string>
#include <iostream>

namespace lgn {

inline void load_collision_geometry(
    const std::string& urdf_path,
    const KinematicTree& tree,
    ContactWorld& world)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(urdf_path.c_str()) != tinyxml2::XML_SUCCESS) {
        std::cerr << "[lgn] collision_loader: cannot open " << urdf_path << "\n";
        return;
    }
    auto* robot = doc.FirstChildElement("robot");
    if (!robot) return;

    int n_loaded = 0;
    int n_skipped = 0;  // mesh shapes

    for (auto* link_el = robot->FirstChildElement("link"); link_el;
         link_el = link_el->NextSiblingElement("link"))
    {
        const char* lname = link_el->Attribute("name");
        if (!lname) continue;

        // Skip links not in the tree (e.g. visual-only links)
        int li = -1;
        try { li = tree.link_index(lname); }
        catch (...) { continue; }
        (void)li;

        for (auto* col_el = link_el->FirstChildElement("collision"); col_el;
             col_el = col_el->NextSiblingElement("collision"))
        {
            // ── Parse <origin> ────────────────────────────────────────────────
            T_mat T_lc = T_identity();
            auto* orig = col_el->FirstChildElement("origin");
            if (orig) {
                auto parse = [](tinyxml2::XMLElement* e, const char* a) -> Vec3 {
                    const char* v = e->Attribute(a);
                    if (!v) return Vec3::Zero();
                    std::istringstream ss(v);
                    double x,y,z; ss >> x >> y >> z;
                    return {x, y, z};
                };
                Vec3 xyz = parse(orig, "xyz");
                Vec3 rpy = parse(orig, "rpy");
                T_lc = T_from_Rp(rpy_to_rot(rpy), xyz);
            }

            auto* geom = col_el->FirstChildElement("geometry");
            if (!geom) continue;

            // ── Sphere ────────────────────────────────────────────────────────
            if (auto* s = geom->FirstChildElement("sphere")) {
                double r = s->DoubleAttribute("radius", 0.01);
                world.add_collider(tree, lname, Sphere{r}, T_lc);
                ++n_loaded;

            // ── Box ───────────────────────────────────────────────────────────
            } else if (auto* b = geom->FirstChildElement("box")) {
                const char* sv = b->Attribute("size");
                Vec3 half{0.05, 0.05, 0.05};
                if (sv) {
                    std::istringstream ss(sv);
                    double x,y,z; ss >> x >> y >> z;
                    half = {x/2, y/2, z/2};
                }
                world.add_collider(tree, lname, Box{half}, T_lc);
                ++n_loaded;

            // ── Cylinder → Capsule (axis = local z per URDF spec) ─────────────
            } else if (auto* c = geom->FirstChildElement("cylinder")) {
                double r  = c->DoubleAttribute("radius", 0.01);
                double hl = c->DoubleAttribute("length", 0.05) / 2.0;
                world.add_collider(tree, lname, Capsule{r, hl}, T_lc);
                ++n_loaded;

            // ── Mesh — skip, log once ─────────────────────────────────────────
            } else if (geom->FirstChildElement("mesh")) {
                ++n_skipped;
            }
        }
    }

    std::cout << "[lgn] collision_loader: " << n_loaded << " primitives loaded";
    if (n_skipped > 0)
        std::cout << ", " << n_skipped << " mesh shapes skipped"
                  << " (add Sphere/Box/Capsule approximations manually)";
    std::cout << "\n";
}

} // namespace lgn
