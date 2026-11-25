using LinearAlgebra

"""
    DiscreteCurve{T<:Real, A<:AbstractMatrix{T}}

Discrete representation of a curve in ``\\mathbb{R}^D`` with ``M`` sampled points,
stored as a ``D \\times M`` matrix (`ndim = D`, `npoints = M`) for type stability
and identical efficiency on CPU (`Array`) and GPU (`CuArray`, `ROCArray`, etc.).

# Fields
- `points::A`: Coordinate matrix of size `(D, M)` where column `j` holds point `j`.
- `closed::Bool`: Whether the curve is closed (`true`) or open (`false`).

# Examples
```jldoctest
julia> pts = [0.0 1.0 2.0; 0.0 1.0 0.0];

julia> c = DiscreteCurve(pts; closed = false)
DiscreteCurve{Float64} (2D, 3 points, open)
  storage : Matrix{Float64}
  start   : [0.0, 0.0]
  end     : [2.0, 0.0]
```
"""
struct DiscreteCurve{T<:Real, A<:AbstractMatrix{T}}
    points::A
    closed::Bool

    function DiscreteCurve{T, A}(points::A, closed::Bool = false) where {T<:Real, A<:AbstractMatrix{T}}
        size(points, 1) >= 1 || throw(ArgumentError("Spatial dimension D must be >= 1"))
        size(points, 2) >= 1 || throw(ArgumentError("Number of points M must be >= 1"))
        new{T, A}(points, closed)
    end
end

"""
    DiscreteCurve(points::AbstractMatrix{T}; closed::Bool = false) where {T<:Real}

Wrap a `D × M` coordinate matrix `points` directly without copying, preserving its
element type `T` and storage backend (CPU `Matrix`, `SubArray` view, or GPU `CuMatrix`).

# Examples
```jldoctest
julia> mat = Float32[0 1 2 3; 0 2 1 0];

julia> c = DiscreteCurve(@view mat[:, 1:3]; closed = true)
DiscreteCurve{Float32} (2D, 3 points, closed)
  storage : SubArray{Float32, 2, Matrix{Float32}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
  start   : [0.0f0, 0.0f0]
  end     : [2.0f0, 1.0f0]
```
"""
function DiscreteCurve(points::AbstractMatrix{T}; closed::Bool = false) where {T<:Real}
    DiscreteCurve{T, typeof(points)}(points, closed)
end

"""
    DiscreteCurve(coords::AbstractVector{<:Real}...; T = Float64, closed = false, to_device = identity)

Construct an `N`-dimensional curve from `D` coordinate vectors of equal length `M`:
- `DiscreteCurve(x)` → 1D curve (`1 × M`)
- `DiscreteCurve(x, y)` → 2D curve (`2 × M`)
- `DiscreteCurve(x, y, z, ...)` → `D`-dimensional curve (`D × M`)

Integer inputs are promoted to `Float64` by default unless `T` is explicitly specified.
Pass a transfer function via `to_device` (e.g. `x -> adapt(CuArray, x)`) to place the
curve directly on a GPU.

# Examples
```jldoctest
julia> t = range(0.0, 1.0, length = 5);

julia> c = DiscreteCurve(t, t .^ 2; closed = false)
DiscreteCurve{Float64} (2D, 5 points, open)
  storage : Matrix{Float64}
  start   : [0.0, 0.0]
  end     : [1.0, 1.0]
```
"""
function DiscreteCurve(
    coords::AbstractVector{<:Real}...;
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    D = length(coords)
    D >= 1 || throw(ArgumentError("At least one coordinate vector is required"))
    M = length(first(coords))
    M >= 1 || throw(ArgumentError("Coordinate vectors cannot be empty"))
    all(v -> length(v) == M, coords) || throw(DimensionMismatch("All coordinate vectors must have the same length"))

    T_out = R === Float64 && eltype(first(coords)) <: AbstractFloat ? promote_type(map(eltype, coords)...) : R
    mat = Matrix{T_out}(undef, D, M)
    @inbounds for i in 1:D, j in 1:M
        mat[i, j] = T_out(coords[i][j])
    end
    DiscreteCurve(to_device(mat); closed = closed)
end

"""
    DiscreteCurve(coords::Tuple{Vararg{AbstractVector{<:Real}}}; T = Float64, closed = false, to_device = identity)

Construct a `DiscreteCurve` in `D` dimensions from a tuple of `D` coordinate vectors
of equal length `M`, such as `DiscreteCurve((xs, ys))` or `DiscreteCurve((xs, ys, zs))`.
"""
function DiscreteCurve(
    coords::Tuple{Vararg{AbstractVector{<:Real}}};
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    DiscreteCurve(coords...; T = R, closed = closed, to_device = to_device)
end

"""
    DiscreteCurve(pts::AbstractVector{P}; T = Float64, closed = false, to_device = identity) where {P<:Union{Tuple, AbstractVector}}

Construct a `DiscreteCurve` from a collection of `M` points, where each point is a `Tuple`
or `AbstractVector` of length `D`.

# Examples
```jldoctest
julia> poly = DiscreteCurve([(0, 0, 0), (1, 0, 0), (1, 1, 2)]; closed = true)
DiscreteCurve{Float64} (3D, 3 points, closed)
  storage : Matrix{Float64}
  start   : [0.0, 0.0, 0.0]
  end     : [1.0, 1.0, 2.0]
```
"""
function DiscreteCurve(
    pts::AbstractVector{P};
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {P<:Union{Tuple, AbstractVector}, R<:Real}
    isempty(pts) && throw(ArgumentError("Points collection cannot be empty"))
    first_pt = first(pts)
    D = length(first_pt)
    D >= 1 || throw(ArgumentError("Point dimension D must be >= 1"))
    M = length(pts)
    all(p -> length(p) == D, pts) || throw(DimensionMismatch("All points must have the same dimension"))

    T_out = R === Float64 && eltype(first_pt) <: AbstractFloat ? eltype(first_pt) : R
    mat = Matrix{T_out}(undef, D, M)
    @inbounds for j in 1:M, i in 1:D
        mat[i, j] = T_out(pts[j][i])
    end
    DiscreteCurve(to_device(mat); closed = closed)
end

"""
    DiscreteCurve(pts::Tuple{Vararg{Tuple}}; T = Float64, closed = false, to_device = identity)

Construct a `DiscreteCurve` from a tuple of coordinate tuples, e.g.
`DiscreteCurve(((0.0, 0.0), (1.0, 0.5), (1.0, 1.0)))`.
"""
function DiscreteCurve(
    pts::Tuple{Vararg{Tuple}};
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    DiscreteCurve(collect(pts); T = R, closed = closed, to_device = to_device)
end

# Internal helpers for parametric curve sampling
function _infer_dim(sample::Real)
    return 1
end

function _infer_dim(sample::Union{Tuple, AbstractVector})
    D = length(sample)
    D >= 1 || throw(ArgumentError("Point dimension must be >= 1"))
    return D
end

function _write_sample!(pts::AbstractMatrix{R}, col::Int, val::Real) where {R<:Real}
    size(pts, 1) == 1 || throw(DimensionMismatch("Scalar return value requires 1D curve (D == 1)"))
    @inbounds pts[1, col] = R(val)
    return nothing
end

function _write_sample!(pts::AbstractMatrix{R}, col::Int, val::Tuple) where {R<:Real}
    D = size(pts, 1)
    length(val) == D || throw(DimensionMismatch("Sample length $(length(val)) does not match curve dimension $D"))
    @inbounds for i in 1:D
        pts[i, col] = R(val[i])
    end
    return nothing
end

function _write_sample!(pts::AbstractMatrix{R}, col::Int, val::AbstractVector) where {R<:Real}
    D = size(pts, 1)
    length(val) == D || throw(DimensionMismatch("Sample length $(length(val)) does not match curve dimension $D"))
    @inbounds for i in 1:D
        pts[i, col] = R(val[i])
    end
    return nothing
end

"""
    DiscreteCurve(f::Function, ts::AbstractVector{<:Real}; T = Float64, closed = false, to_device = identity)
    parametric_curve(f::Function, ts::AbstractVector{<:Real}; T = Float64, closed = false, to_device = identity)

Discretize a parametric mapping ``f: \\mathbb{R} \\to \\mathbb{R}^D`` along an explicit
parameter grid or range `ts` (uniform or non-uniform). `f(t)` may return a `Real` (1D),
a `Tuple`, or an `AbstractVector`.

# Examples
```jldoctest
julia> c = DiscreteCurve(t -> (cos(t), sin(t)), range(0.0, π, length = 11))
DiscreteCurve{Float64} (2D, 11 points, open)
  storage : Matrix{Float64}
  start   : [1.0, 0.0]
  end     : [-1.0, 1.2246e-16]
```
"""
function DiscreteCurve(
    f::Function,
    ts::AbstractVector{<:Real};
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    M = length(ts)
    M >= 1 || throw(ArgumentError("Parameter range/vector ts cannot be empty"))

    first_val = f(R(ts[1]))
    D = _infer_dim(first_val)
    mat = Matrix{R}(undef, D, M)
    _write_sample!(mat, 1, first_val)

    @inbounds for j in 2:M
        _write_sample!(mat, j, f(R(ts[j])))
    end
    DiscreteCurve(to_device(mat); closed = closed)
end

"""
    DiscreteCurve(f::Function, tspan::Tuple{<:Real, <:Real}, n::Int = 100; T = Float64, closed = false, to_device = identity)
    parametric_curve(f::Function, tspan::Tuple{<:Real, <:Real}, n::Int = 100; T = Float64, closed = false, to_device = identity)

Discretize a parametric curve ``f(t)`` uniformly over `tspan = (t_start, t_end)` using `n`
points in any spatial dimension `D` (`1..N`).

# Examples
```jldoctest
julia> c = DiscreteCurve(t -> (t, 2 * t, 3 * t), (0.0, 2.0), 5; T = Float32)
DiscreteCurve{Float32} (3D, 5 points, open)
  storage : Matrix{Float32}
  start   : [0.0f0, 0.0f0, 0.0f0]
  end     : [2.0f0, 4.0f0, 6.0f0]
```
"""
function DiscreteCurve(
    f::Function,
    tspan::Tuple{<:Real, <:Real},
    n::Int = 100;
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    n >= 1 || throw(ArgumentError("Number of samples n must be >= 1"))
    ts = range(R(tspan[1]), R(tspan[2]); length = n)
    return DiscreteCurve(f, ts; T = R, closed = closed, to_device = to_device)
end

"""
    ndim(c::DiscreteCurve) -> Int

Return the ambient spatial dimension `D` of curve `c`.

# Examples
```jldoctest
julia> c = DiscreteCurve([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]);

julia> ndim(c)
3
```
"""
ndim(c::DiscreteCurve) = size(c.points, 1)

"""
    npoints(c::DiscreteCurve) -> Int

Return the number of sample points `M` along curve `c`.

# Examples
```jldoctest
julia> c = DiscreteCurve([1.0, 2.0, 3.0, 4.0]);

julia> npoints(c)
4
```
"""
npoints(c::DiscreteCurve) = size(c.points, 2)
"""
    isclosed(c::DiscreteCurve) -> Bool

Return `true` if curve `c` is marked as closed, `false` otherwise.

# Examples
```jldoctest
julia> c = DiscreteCurve([0.0, 1.0], [0.0, 1.0]; closed = true);

julia> isclosed(c)
true
```
"""
isclosed(c::DiscreteCurve) = c.closed


