# twin_sim scenarios

Each scenario is a heading + a fenced code block of `key = value` lines.
Numbering in headings is informational; the actual index used by
`--scenario K` matches the order of appearance (first block = 1).

Recognized keys: `n`, `duration`, `dt`, `eps`, `damping`, `chaos`, `q[i]`, `dq[i]`.
Values may be numbers or simple expressions: `pi`, `pi/2`, `-pi/4`, `2*pi`.
Indices `i` are 0-based and refer to the `n-1` revolute joints.

`damping` is viscous joint damping: tau_i = -b * dq_i, applied identically
to both engines (and the chaos twin). Useful values:
  - 0.000 — no damping (default)
  - 0.001 — barely perceptible, energy decays over minutes
  - 0.005 — visible decay over ~30s
  - 0.05  — clearly damped, settles in ~10s
  - 0.2+  — overdamped, no swinging

---

## 1. Horizontal release

The whole chain starts horizontal-left from the anchor, released from rest.
Classic dramatic swing — large PE → large KE conversion. Mild chaos, but
clean energy bookkeeping. Good first scenario to verify the setup.

```
n = 7
duration = 10
q[0] = pi/2
```

## 2. Inverted unstable

Chain pointing straight up (q[0] = -pi/2 rotates +Y to +X... wait, this is
horizontal-right actually). True inverted is q[0] = pi (link 1 points up
since link 0 is fixed and q[0] rotates link 1 about its parent's Z by pi).
The chain balances upside-down, then any numerical noise sends it falling.
Engines will diverge fast — perfect for the chaos plot.

```
n = 7
duration = 12
q[0] = pi
chaos = true
eps = 1e-10
```

## 3. Zigzag fold

Alternating joint angles fold the chain into a tight zigzag. Lots of
stored geometric tension; releases asymmetrically and chaotically.

```
n = 9
duration = 15
q[0] = pi/2
q[1] = -pi/2
q[2] = pi/2
q[3] = -pi/2
q[4] = pi/2
q[5] = -pi/2
chaos = true
```

## 4. Whip release

Chain folded back on itself — first half down, second half curls back up.
When released the curl unwinds explosively. Tip reaches very high speed.

```
n = 8
duration = 12
q[0] = pi
q[4] = pi
chaos = true
```

## 5. Spinning base

Base joint has large initial angular velocity. Centrifugal effects throw
the chain outward; combined with gravity this gets very chaotic very fast.

```
n = 7
duration = 8
q[0] = pi/2
dq[0] = 6.0
chaos = true
eps = 1e-10
```

## 6. Counter-rotation

Adjacent joints rotating in opposite directions at high speed. The chain
fights itself — maximum entanglement, hardest test for the integrators.

```
n = 7
duration = 10
q[0] = pi/2
dq[0] = 4.0
dq[1] = -4.0
dq[2] = 4.0
dq[3] = -4.0
chaos = true
eps = 1e-12
```

## 7. Near-equilibrium small swing

Chain barely off vertical (hanging straight down with a tiny offset).
Should be nearly periodic and nearly integrable. ΔE_tot should stay
microscopic — sanity check that the integrators are well-behaved.

```
n = 7
duration = 20
q[0] = pi
q[1] = 0.05
```

## 8. Long-horizon drift study

Same horizontal release as scenario 1, but for a full minute, with light
damping. Watch |ΔE_tot| and the Lyapunov plot. With b = 0.005 you should
see the classic three-act structure on the Lyapunov plot:
exponential growth → plateau (saturated chaos) → exponential decay
as both chains shed energy and converge toward q = 0.

```
n = 7
duration = 60
dt = 5e-4
q[0] = pi/2
chaos = true
eps = 1e-10
damping = 0.005
```

## 9. Damped chaos showcase

Inverted release with moderate damping — designed to show ||δq(t)||
rising fast (unstable fall), hitting the chaotic plateau, then
collapsing as damping brings both chains to rest. The most pedagogically
satisfying Lyapunov plot in the file: you see chaos *and* its eventual
death.

```
n = 7
duration = 45
q[0] = pi
chaos = true
eps = 1e-10
damping = 0.01
```

## 10. Damped near-equilibrium

Small swing with damping — a textbook damped oscillator in 6 DOF.
Should look like ringdown, with ||δq|| decaying smoothly from the start
(no chaotic growth phase because the small-amplitude regime is nearly
integrable). Good contrast scenario against #9.

```
n = 7
duration = 20
q[0] = pi
q[1] = 0.05
chaos = true
eps = 1e-8
damping = 0.02
```
