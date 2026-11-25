```@meta
CurrentModule = DCurve
```

# Curve Constructors

Every curve in `DCurve.jl` is constructed as a [`DiscreteCurve`](@ref). All high-level constructors accept keyword arguments for topology (`closed::Bool`), element precision (`T`), and device transfer (`to_device::Function`).

## 1. Direct Matrix & Zero-Copy Views

Passing a `D × M` matrix wraps the underlying memory directly without copying. Slicing an existing matrix with `@view` preserves the view type:

```@example constructors
using DCurve

pts = [0.0 1.0 2.0 3.0;
       0.0 1.5 0.5 0.0]

c = DiscreteCurve(pts; closed = false)
```

```@example constructors
c_sub = DiscreteCurve(@view pts[:, 2:4]; closed = true)
```

## 2. Coordinate Vectors (`1D` to `ND`)

Pass one coordinate vector per dimension (`x`, `x, y`, `x, y, z`, ...). Integer inputs are automatically promoted to `Float64` unless `T` is specified:

```@example constructors
t = range(0.0, 2π, length = 100)

# 1D curve
c1d = DiscreteCurve(sin.(t))
```

```@example constructors
# 2D closed loop
c2d = DiscreteCurve(cos.(t), sin.(t); closed = true)
```

```@example constructors
# 4D curve with Float32 precision
c4d = DiscreteCurve(cos.(t), sin.(t), 0.2 .* t, -0.1 .* t; T = Float32)
```

## 3. Collections of Points (Tuples or Vectors)

Build curves directly from waypoints stored as a `Vector` or `Tuple` of points:

```@example constructors
waypoints = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 2)]
c_poly = DiscreteCurve(waypoints; closed = true)
```

## 4. Parametric Mappings `f(t)`

Discretize any parametric curve ``f: \mathbb{R} \to \mathbb{R}^D`` over an explicit parameter grid `ts` or uniformly across an interval `(t_start, t_end)` with `n` samples:

```@example constructors
# Uniform sampling over [0, 2π] in 3D
c_helix = DiscreteCurve(t -> (cos(t), sin(t), 0.1 * t), (0.0, 2π), 60)
```

```@example constructors
# Non-uniform parameter grid clustered near endpoints
ts_clustered = [sin(π / 2 * u)^2 * 2π for u in range(0.0, 1.0, length = 50)]
c_clustered = DiscreteCurve(t -> (cos(t), sin(t)), ts_clustered; closed = true)
```


## Constructors & Metadata API

```@autodocs
Modules = [DCurve]
Pages = ["constructors.jl"]
Order = [:type, :function, :constant]
```