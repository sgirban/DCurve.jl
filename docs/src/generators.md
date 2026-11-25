# Curve Generators

`DCurve.jl` provides high-performance, allocation-conscious generators for common geometric shapes, classical differential-geometry benchmarks, discrete dynamical systems, and reproducible stochastic paths.

Every generator shares a consistent, Julian keyword interface:

| Keyword | Default | Purpose |
| :--- | :--- | :--- |
| `dim` | `2` (or `3`) | Target ambient dimension $D$. Planar shapes automatically embed into $\mathbb{R}^D$. |
| `center` | `nothing` | Shift the curve by a $D$-element vector (automatically overrides `dim = length(center)`). |
| `T` | `Float64` | Scalar precision (`Float32`, `Float64`, `BigFloat`, etc.). |
| `closed` | shape-dependent | Override periodic closure (`true` vs. `false`). |
| `to_device` | `identity` | Hook applied to the point matrix before wrapping (e.g., `to_device = CuArray`). |

---

## 1. Circles, Ellipses & Higher-Dimensional Embeddings

Full circles default to `closed = true` and avoid duplicating the seam vertex at $\theta = 2\pi$, whereas partial arcs default to `closed = false`.

```@example generators
using DCurve

# Full circle (64 points, closed = true) and positional shorthand
c_full = circle_curve(64; radius = 2.5)
c_short = circle_curve(2.5, 64)

# Quarter-circle arc in the first quadrant (closed = false)
c_arc = circle_curve(25; radius = 3.0, arc = (0.0, π / 2))

# Rotated ellipse with semi-axes a = 4.0, b = 1.5
c_ell = ellipse_curve(80; a = 4.0, b = 1.5, rotation = π / 6)

# Embed a 2D circle into 5D space at a custom center
c_5d = circle_curve(32; radius = 1.0, center = [1.0, -2.0, 3.0, 4.0, 5.0])
(ndim(c_5d), npoints(c_5d), isclosed(c_5d))
```

---

## 2. Regular Polygons, Stars & $N$-D Hypercubes

### Subdivided Regular Polygons & Stars
Use `points_per_edge > 1` on [`polygon_curve`](@ref) when you need straight edges discretized with interior vertices (useful for testing discrete curvature concentrated at corners vs. flat segments).

```@example generators
using DCurve

# Coarse 6-vertex hexagon vs. 60-vertex subdivided hexagon
hex_coarse = polygon_curve(6; radius = 2.0)
hex_fine   = polygon_curve(6; radius = 2.0, points_per_edge = 10)

# 5-pointed star polygon (10 alternating outer/inner vertices)
star5 = star_curve(5; outer_radius = 2.0, inner_radius = 0.75)
(npoints(hex_fine), npoints(star5))
```

### Hamiltonian Walks on $D$-Dimensional Hypercubes
[`hypercube_curve`](@ref) traverses all $2^D$ vertices of a $D$-dimensional hypercube in Gray-code order so consecutive vertices always differ along a single edge of length `side`:

```@example generators
using DCurve

square   = hypercube_curve(2; side = 2.0)                  # 4 vertices in 2D
cube     = hypercube_curve(3; side = 1.0)                  # 8 vertices in 3D
tesseract = hypercube_curve(4; side = 2.0, T = Float32)    # 16 vertices in 4D (Float32)
(ndim(tesseract), npoints(tesseract), eltype(tesseract))
```

---

## 3. Analytic Curves & Knots

Classic parametric curves used to benchmark discrete curvature, torsion, writhe, and self-intersection checks:

```@example generators
using DCurve

# Planar cardioid r(θ) = a(1 + cos θ) with a cusp at θ = π
card = cardioid_curve(120; a = 1.5)

# Lissajous figure (3:4 frequency ratio)
liss = lissajous_curve(200; a = 2.0, b = 1.5, fx = 3, fy = 4, delta = π / 2)

# Archimedean and logarithmic spirals
arch = spiral_curve(150; a = 0.0, b = 0.15, turns = 4.0, kind = :archimedean)
log_sp = spiral_curve(150; a = 0.2, b = 0.12, turns = 3.0, kind = :logarithmic)

# 3D Trefoil knot and general (p, q)-torus knots
trefoil = trefoil_curve(240; scale = 1.0)
cinquefoil = torus_knot_curve(2, 5, 300; R_major = 2.5, r_minor = 0.8)
(ndim(cinquefoil), npoints(cinquefoil), isclosed(cinquefoil))
```

---

## 4. Iterative & Recurrence Curves (`iterated_curve`)

[`iterated_curve`](@ref) generates trajectories directly from discrete recurrence laws—covering autonomous maps, numerical ODE integrators, and multi-step inertial recurrences in any dimension $D$.

### 1-Step Dynamical Systems ($x_{k+1} = f(x_k)$)

```@example generators
using DCurve

# 2D Hénon chaotic attractor trajectory
henon_step(x) = (1.0 - 1.4 * x[1]^2 + x[2], 0.3 * x[1])
c_henon = iterated_curve(henon_step, [0.1, 0.1], 500)

# 3D Lorenz system integrated with explicit Euler steps
function lorenz_euler(u; σ = 10.0, ρ = 28.0, β = 8 / 3, dt = 0.01)
    dx = σ * (u[2] - u[1])
    dy = u[1] * (ρ - u[3]) - u[2]
    dz = u[1] * u[2] - β * u[3]
    return (u[1] + dt * dx, u[2] + dt * dy, u[3] + dt * dz)
end

c_lorenz = iterated_curve(lorenz_euler, [1.0, 1.0, 1.0], 2_000; T = Float32)
(ndim(c_lorenz), npoints(c_lorenz), eltype(c_lorenz))
```

### Multi-Step Recurrences ($x_{k+1} = f(x_k, x_{k-1}, \dots, x_{k-p+1})$)
Pass a vector of $p$ initial states `history = [x₁, ..., xₚ]`. At each step, `step_fn` receives the most recent $p$ points in reverse chronological order `(x_curr, x_prev, ...)`:

```@example generators
using DCurve

# Damped inertial particle under a central restoring force:
# x_{k+1} = x_k + γ (x_k - x_{k-1}) - k_spring * x_k
function inertial_orbit(x_curr, x_prev)
    γ = 0.98
    k_spring = 0.04
    return @. x_curr + γ * (x_curr - x_prev) - k_spring * x_curr
end

history = [[1.0, 0.0], [0.99, 0.18]]
c_orbit = iterated_curve(inertial_orbit, history, 250)
c_orbit[end]
```

---

## 5. Reproducible Random Curves (`random_curve`)

[`random_curve`](@ref) accepts any `Random.AbstractRNG` (such as `Xoshiro(seed)` or `MersenneTwister(seed)`) and supports four distributions across arbitrary dimensions:

- `:normal` — Gaussian steps (`cumulative = true` gives Brownian motion; `false` gives i.i.d. point clouds).
- `:uniform` — Hypercube steps in $[-\text{step\_scale}, \text{step\_scale}]^D$.
- `:sphere` — Isotropic random walk with constant Euclidean step size `step_scale` on $S^{D-1}$.
- `:smooth` — Band-limited random Fourier curve with `harmonics` modes (ideal for smooth synthetic test curves).

```@example generators
using DCurve, Random

rng = Xoshiro(42)

# 4D isotropic random walk with constant step length 0.1
walk_4d = random_curve(200; dim = 4, distribution = :sphere, step_scale = 0.1, rng = rng)

# Smooth closed 3D random loop (5 Fourier harmonics, Float32)
smooth_loop = random_curve(
    128;
    dim = 3,
    distribution = :smooth,
    harmonics = 5,
    step_scale = 1.5,
    closed = true,
    T = Float32,
    rng = Xoshiro(7),
)

(ndim(smooth_loop), npoints(smooth_loop), isclosed(smooth_loop), eltype(smooth_loop))
```

---

## 6. Direct GPU Generation (`to_device`)

All generators accept `to_device` so you can construct the curve directly on a GPU (or custom array backend) without manually re-wrapping `DiscreteCurve`:

```julia
using CUDA, DCurve

# Allocate on host, transfer once to GPU, and wrap as DiscreteCurve{Float32, CuMatrix{Float32}}
gpu_knot = torus_knot_curve(3, 5, 10_000; T = Float32, to_device = cu)
```

---

## API Reference

```@autodocs
Modules = [DCurve]
Pages   = ["generators.jl"]
Order   = [:function]
```