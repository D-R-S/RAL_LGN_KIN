#!/usr/bin/env python3
# ============================================================================
#  tools/twin_sim_render.py
#
#  Flicker-free renderer for twin_sim_a's CSV stream.
#
#  Reads stdin, draws three persistent axes:
#    ┌─────────────────────────────────────┐
#    │  Dynamics (lgn | pinocchio)         │   top, full width
#    ├──────────────────┬──────────────────┤
#    │  Energy / link   │  Lyapunov        │   bottom (Lyap only if chaos)
#    └──────────────────┴──────────────────┘
#
#  USAGE
#      ./twin_sim_a --scenario 3 | python3 twin_sim_render.py
#      ./twin_sim_a --chaos      | python3 twin_sim_render.py
#      python3 twin_sim_render.py < run.csv      # replay
#
#  ── WHY THIS IS FLICKER-FREE ────────────────────────────────────────────
#
#  Single rendering path: matplotlib.animation.FuncAnimation with
#  blit=True. Every artist that changes is returned from the update
#  callback; FuncAnimation handles the blit-region bookkeeping itself.
#
#  Hand-rolled `restore_region` + `draw_idle` does NOT compose — the
#  draw_idle schedules a full-canvas redraw that races the blit and
#  produces the artist-only flicker you saw. Using FuncAnimation cleanly
#  avoids that.
#
#  All axes are LOCKED — no autoscale, ever:
#    • dynamics: fixed equal-aspect frame around both anchor points
#    • energy:   y-range frozen at startup from header (analytical bound)
#    • Lyapunov: fixed [0, duration] × [1e-20, 1e2] log
# ============================================================================
import sys
import re
import math
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import matplotlib
# Pick a windowed backend BEFORE importing pyplot.
try:
    matplotlib.use("TkAgg", force=True)
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


# ── Stream parsing ──────────────────────────────────────────────────────────

@dataclass
class Header:
    n: int = 7
    dt: float = 1e-3
    duration: float = 10.0
    damping: float = 0.0
    chaos: bool = False
    link_L: float = 0.5
    link_m: float = 0.1
    scenario: str = ""


@dataclass
class Frame:
    t: float = 0.0
    residual: float = 0.0
    delta_norm: float = float("nan")
    pose_lgn_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pose_lgn_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pose_pin_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pose_pin_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ke_pin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pe_pin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ke_lgn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pe_lgn: np.ndarray = field(default_factory=lambda: np.zeros(0))


def parse_header_line(hdr: Header, line: str) -> None:
    body = line[1:].strip()
    m = re.match(r'scenario\s*=\s*"(.*)"\s*$', body)
    if m:
        hdr.scenario = m.group(1)
        return
    for tok in body.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if   k == "n":        hdr.n = int(v)
        elif k == "dt":       hdr.dt = float(v)
        elif k == "duration": hdr.duration = float(v)
        elif k == "damping":  hdr.damping = float(v)
        elif k == "chaos":    hdr.chaos = (v in ("1", "true", "True"))
        elif k == "link_L":   hdr.link_L = float(v)
        elif k == "link_m":   hdr.link_m = float(v)


def reader_thread(fh, q: queue.Queue, hdr: Header,
                  hdr_done: threading.Event) -> None:
    """Read stdin, group lines into Frame objects, push onto queue."""
    cur: Optional[Frame] = None
    energies: List[tuple] = []

    def flush(c: Optional[Frame]) -> None:
        if c is None:
            return
        if energies:
            energies.sort(key=lambda r: r[0])
            c.ke_pin = np.array([r[1] for r in energies])
            c.pe_pin = np.array([r[2] for r in energies])
            c.ke_lgn = np.array([r[3] for r in energies])
            c.pe_lgn = np.array([r[4] for r in energies])
        try:
            q.put(c, timeout=2.0)
        except queue.Full:
            pass  # drop frame if GUI is way behind

    for raw in fh:
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith("#"):
            parse_header_line(hdr, line)
            if line.startswith("# COLUMNS: ENERGY"):
                hdr_done.set()
            continue
        parts = line.split(",")
        kind = parts[0]
        if kind == "FRAME":
            flush(cur)
            energies = []
            cur = Frame()
            try:
                cur.t          = float(parts[1])
                cur.residual   = float(parts[2])
                cur.delta_norm = (float("nan") if parts[3] == "nan"
                                              else float(parts[3]))
            except (IndexError, ValueError):
                pass
            hdr_done.set()
        elif kind == "POSE_LGN" and cur is not None:
            coords = np.array([float(x) for x in parts[1:]])
            cur.pose_lgn_x = coords[0::2]
            cur.pose_lgn_y = coords[1::2]
        elif kind == "POSE_PIN" and cur is not None:
            coords = np.array([float(x) for x in parts[1:]])
            cur.pose_pin_x = coords[0::2]
            cur.pose_pin_y = coords[1::2]
        elif kind == "ENERGY" and cur is not None:
            try:
                energies.append((
                    int  (parts[1]),
                    float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5]),
                ))
            except (IndexError, ValueError):
                pass
    flush(cur)
    try:
        q.put(None, timeout=1.0)  # EOF sentinel
    except queue.Full:
        pass


# ── Renderer ────────────────────────────────────────────────────────────────

def main() -> int:
    hdr = Header()
    hdr_done = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=256)

    t_reader = threading.Thread(
        target=reader_thread, args=(sys.stdin, q, hdr, hdr_done),
        daemon=True)
    t_reader.start()

    if not hdr_done.wait(timeout=5.0):
        print("renderer: no header within 5s; giving up.", file=sys.stderr)
        return 1

    n     = hdr.n
    L     = hdr.link_L
    m     = hdr.link_m
    chaos = hdr.chaos
    dur   = hdr.duration if hdr.duration > 0 else 10.0
    damp  = hdr.damping

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.color":       "#dddddd",
        "grid.linewidth":   0.6,
    })

    fig = plt.figure(figsize=(12.8, 9.0))
    try:
        fig.canvas.manager.set_window_title("twin_sim — fused view")
    except Exception:
        pass

    if chaos:
        gs = GridSpec(2, 2, figure=fig,
                      height_ratios=[1.1, 1.0],
                      hspace=0.34, wspace=0.22,
                      left=0.06, right=0.98, top=0.91, bottom=0.07)
        ax_dyn  = fig.add_subplot(gs[0, :])
        ax_en   = fig.add_subplot(gs[1, 0])
        ax_lyap = fig.add_subplot(gs[1, 1])
    else:
        gs = GridSpec(2, 1, figure=fig,
                      height_ratios=[1.1, 1.0],
                      hspace=0.34,
                      left=0.06, right=0.98, top=0.91, bottom=0.07)
        ax_dyn  = fig.add_subplot(gs[0, 0])
        ax_en   = fig.add_subplot(gs[1, 0])
        ax_lyap = None

    # ── Dynamics axis (LOCKED) ──────────────────────────────────────────
    reach = n * L
    pane_w = 2.5 * reach
    pane_h = 3.0 * reach
    anchor_x_lgn = pane_w * 0.80
    anchor_x_pin = pane_w * 0.80 + pane_w
    y_top = pane_h * (1.0 / 3.0)
    y_bot = -pane_h * (2.0 / 3.0)

    ax_dyn.set_xlim(0.0, 2.0 * pane_w)
    ax_dyn.set_ylim(y_bot, y_top + 0.15 * pane_h)
    ax_dyn.set_aspect("equal", adjustable="box")
    ax_dyn.set_title("dynamics")
    ax_dyn.set_xticks([])
    ax_dyn.set_yticks([])
    ax_dyn.autoscale(enable=False)  # belt + suspenders

    (line_lgn,) = ax_dyn.plot([], [], "-o", color="#E67E22",
                              lw=2.0, ms=7, animated=True)
    (line_pin,) = ax_dyn.plot([], [], "-o", color="#1F5FA8",
                              lw=2.0, ms=7, animated=True)
    ax_dyn.text(anchor_x_lgn, y_top + pane_h * 0.10, "lgn",
                ha="center", fontsize=11, color="#E67E22")
    ax_dyn.text(anchor_x_pin, y_top + pane_h * 0.10, "pinocchio",
                ha="center", fontsize=11, color="#1F5FA8")

    # ── Energy axis (LOCKED) ────────────────────────────────────────────
    # Y-range computed analytically from the scenario geometry, so we
    # never have to autoscale-and-rebuild the cached background.
    g = 9.81
    pe_shift = m * g * (n * L)
    # Conservative upper bound: a link falling from horizontal at the tip
    # can pick up KE ~ m*g*h where h = reach. PE bound from the shift.
    # Use 1.5× the larger of the two to leave headroom for numerical
    # transients in the first few steps.
    e_bound = 1.5 * max(m * g * reach, pe_shift)
    if e_bound <= 0.0:
        e_bound = 1.0
    ax_en.set_xlim(-0.5, n - 0.5)
    ax_en.set_ylim(-0.05 * e_bound, e_bound)
    ax_en.set_xlabel(f"link index  (0=base, {n-1}=tip)")
    ax_en.set_ylabel("energy  [J]")
    ax_en.set_title("KE / PE per link")
    ax_en.autoscale(enable=False)

    xs_link = np.arange(n)
    (l_ke_pin,) = ax_en.plot(xs_link, np.zeros(n), "-o",
                             color="#1F5FA8", lw=2.0, ms=6, label="KE pin",
                             animated=True)
    (l_ke_lgn,) = ax_en.plot(xs_link, np.zeros(n), "-s",
                             color="#E67E22", lw=2.0, ms=6, label="KE lgn",
                             animated=True)
    (l_pe_pin,) = ax_en.plot(xs_link, np.zeros(n), "-o",
                             color="#9DB8D8", lw=1.2, ms=4, label="PE pin",
                             animated=True)
    (l_pe_lgn,) = ax_en.plot(xs_link, np.zeros(n), "-s",
                             color="#F5C99B", lw=1.2, ms=5, label="PE lgn",
                             animated=True)
    ax_en.legend(loc="upper right", fontsize=9)

    # ── Lyapunov axis (LOCKED) ──────────────────────────────────────────
    lyap_t: List[float] = []
    lyap_d: List[float] = []
    l_lyap = None
    if ax_lyap is not None:
        ax_lyap.set_yscale("log")
        ax_lyap.set_xlim(0.0, dur)
        ax_lyap.set_ylim(1e-20, 1e2)
        ax_lyap.set_xlabel("t  [s]")
        ax_lyap.set_ylabel(r"$\|\delta q(t)\|_2$  (log)")
        ax_lyap.set_title(f"Lyapunov   b={damp:g}")
        ax_lyap.autoscale(enable=False)
        (l_lyap,) = ax_lyap.plot([], [], "-", color="#B2182B", lw=1.8,
                                 animated=True)
        ax_lyap.legend([l_lyap], [r"$\|\delta q\|_2$"], loc="upper left",
                       fontsize=9)

    # ── Title text as animated artist on the dynamics axis ─────────────
    # NOTE: we cannot use fig.suptitle here because matplotlib's blit code
    # walks each animated artist's `.axes` attribute, and a Figure-level
    # text has axes=None → AttributeError in _blit_draw. Anchoring the
    # title to ax_dyn (in axes-fraction coords above its top edge) gives
    # it a valid parent axes and lets blit redraw it cleanly each frame.
    title_text = ax_dyn.text(
        0.5, 1.06, " ",
        transform=ax_dyn.transAxes,
        ha="center", va="bottom",
        fontsize=12, animated=True,
        clip_on=False,
    )

    # Latest frame (renderer reads from this; writer thread updates it)
    state = {"latest": None, "done": False}

    def pump_queue():
        """Drain queue; keep most recent frame, accumulate Lyapunov."""
        latest = state["latest"]
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                state["done"] = True
                break
            # Lyapunov accumulates every frame, even dropped ones.
            if chaos and not math.isnan(item.delta_norm):
                lyap_t.append(item.t)
                lyap_d.append(max(item.delta_norm, 1e-20))
            latest = item
        state["latest"] = latest

    def init():
        """Called by FuncAnimation before the first frame. Return the
        full list of animated artists so blit knows what to track."""
        line_lgn.set_data([], [])
        line_pin.set_data([], [])
        l_ke_pin.set_data(xs_link, np.zeros(n))
        l_ke_lgn.set_data(xs_link, np.zeros(n))
        l_pe_pin.set_data(xs_link, np.zeros(n))
        l_pe_lgn.set_data(xs_link, np.zeros(n))
        if l_lyap is not None:
            l_lyap.set_data([], [])
        title_text.set_text("twin_sim — waiting for first frame...")
        artists = [line_lgn, line_pin,
                   l_ke_pin, l_ke_lgn, l_pe_pin, l_pe_lgn,
                   title_text]
        if l_lyap is not None:
            artists.append(l_lyap)
        return artists

    def update(_frame_num):
        pump_queue()
        f = state["latest"]
        artists = [line_lgn, line_pin,
                   l_ke_pin, l_ke_lgn, l_pe_pin, l_pe_lgn,
                   title_text]
        if l_lyap is not None:
            artists.append(l_lyap)
        if f is None:
            return artists
        # Dynamics
        if f.pose_lgn_x.size:
            line_lgn.set_data(f.pose_lgn_x + anchor_x_lgn,
                              f.pose_lgn_y + y_top)
        if f.pose_pin_x.size:
            line_pin.set_data(f.pose_pin_x + anchor_x_pin,
                              f.pose_pin_y + y_top)
        # Energy
        if f.ke_pin.size == n:
            l_ke_pin.set_data(xs_link, f.ke_pin)
            l_ke_lgn.set_data(xs_link, f.ke_lgn)
            l_pe_pin.set_data(xs_link, f.pe_pin + pe_shift)
            l_pe_lgn.set_data(xs_link, f.pe_lgn + pe_shift)
        # Lyapunov
        if l_lyap is not None and len(lyap_t) >= 2:
            l_lyap.set_data(lyap_t, lyap_d)
        # Title
        title_text.set_text(
            f"twin_sim   t = {f.t:.2f} s   "
            f"||Δq||∞ = {f.residual:.2e}   b = {damp:.4f}"
        )
        return artists

    # interval=33ms ≈ 30 fps. The reader thread keeps the queue full;
    # update() always grabs the freshest frame.
    anim = FuncAnimation(
        fig, update, init_func=init,
        interval=33, blit=True, cache_frame_data=False,
    )
    # Keep a reference alive (else GC kills the animation in some versions).
    fig._twin_anim = anim  # type: ignore[attr-defined]

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
