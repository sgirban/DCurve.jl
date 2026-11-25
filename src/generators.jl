function _resolve_center(center::Nothing, dim::Int, min_dim::Int, ::Type{R}) where {R<:Real}
    dim >= min_dim || throw(ArgumentError("Dimension dim must be >= $min_dim"))
    return zeros(R, dim)
end

function _resolve_center(center::AbstractVector{<:Real}, dim::Int, min_dim::Int, ::Type{R}) where {R<:Real}
    D = length(center)
    D >= min_dim || throw(ArgumentError("Center length must be >= $min_dim"))
    return R[R(x) for x in center]
end

"""
    circle_curve(n::Int; radius = 1.0, center = nothing, phase = 0.0, arc = (0.0, 2π), dim = 2, T = Float64, closed = nothing, to_device = identity)
    circle_curve(radius::Real, n::Int; kwargs...)

Sample `n` points along a circle (or circular arc) of `radius` in the first two coordinates
of ``\\mathbb{R}^D``, shifted by `center` and rotated by `phase`. Defaults to `closed = true`
for a full ``2\\pi`` circle and `closed = false` for partial arcs.
"""
function circle_curve(
    n::Int;
    radius = 1.0,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    phase = 0.0,
    arc::Tuple{<:Real, <:Real} = (0.0, 2π),
    dim::Int = 2,
    T::Type{R} = Float64,
    closed::Union{Nothing, Bool} = nothing,
    to_device::Function = identity,
) where {R<:Real}
    n >= 2 || throw(ArgumentError("Number of points n must be >= 2"))
    c = _resolve_center(center, dim, 2, R)
    r = R(radius)
    θ0 = R(arc[1]) + R(phase)
    θ1 = R(arc[2]) + R(phase)
    is_full = isapprox(abs(R(arc[2]) - R(arc[1])), R(2π); atol = sqrt(eps(R)))
    is_closed = isnothing(closed) ? is_full : closed

    ts = is_full ? range(θ0, θ1; length = n + 1)[1:n] : range(θ0, θ1; length = n)
    pts = repeat(c, 1, n)
    @inbounds for j in 1:n
        θ = ts[j]
        pts[1, j] += r * cos(θ)
        pts[2, j] += r * sin(θ)
    end
    DiscreteCurve(to_device(pts); closed = is_closed)
end

circle_curve(radius::Real, n::Int; kwargs...) = circle_curve(n; radius = radius, kwargs...)

"""
    ellipse_curve(n::Int; a = 1.0, b = 0.5, center = nothing, rotation = 0.0, dim = 2, T = Float64, closed = true, to_device = identity)

Sample `n` points on an ellipse with semi-axes `a` and `b`, rotated by `rotation` radians
and centered at `center` in ``\\mathbb{R}^D``.
"""
function ellipse_curve(
    n::Int;
    a = 1.0,
    b = 0.5,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rotation = 0.0,
    dim::Int = 2,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    n >= 3 || throw(ArgumentError("Number of points n must be >= 3"))
    c = _resolve_center(center, dim, 2, R)
    ra, rb = R(a), R(b)
    sα, cα = sincos(R(rotation))
    ts = range(zero(R), R(2π); length = n + 1)[1:n]

    pts = repeat(c, 1, n)
    @inbounds for j in 1:n
        x = ra * cos(ts[j])
        y = rb * sin(ts[j])
        pts[1, j] += cα * x - sα * y
        pts[2, j] += sα * x + cα * y
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    polygon_curve(sides::Int; radius = 1.0, center = nothing, rotation = 0.0, points_per_edge = 1, dim = 2, T = Float64, closed = true, to_device = identity)

Construct a regular `sides`-gon (`sides >= 3`) inscribed in a circle of `radius`.
Set `points_per_edge > 1` to subdivide each straight edge uniformly (`sides * points_per_edge` total points).
"""
function polygon_curve(
    sides::Int;
    radius = 1.0,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rotation = 0.0,
    points_per_edge::Int = 1,
    dim::Int = 2,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    sides >= 3 || throw(ArgumentError("Number of sides must be >= 3"))
    points_per_edge >= 1 || throw(ArgumentError("points_per_edge must be >= 1"))
    c = _resolve_center(center, dim, 2, R)
    r = R(radius)
    θ0 = R(rotation)

    verts = [
        (c[1] + r * cos(θ0 + R(2π) * (k - 1) / sides),
         c[2] + r * sin(θ0 + R(2π) * (k - 1) / sides))
        for k in 1:sides
    ]

    total = sides * points_per_edge
    pts = repeat(c, 1, total)
    col = 1
    @inbounds for k in 1:sides
        v1 = verts[k]
        v2 = verts[mod1(k + 1, sides)]
        for s in 0:(points_per_edge - 1)
            α = R(s) / R(points_per_edge)
            pts[1, col] = (one(R) - α) * v1[1] + α * v2[1]
            pts[2, col] = (one(R) - α) * v1[2] + α * v2[2]
            col += 1
        end
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    hypercube_curve(dim::Int; side = 1.0, center = nothing, T = Float64, closed = true, to_device = identity)

Construct a closed Hamiltonian cycle along the edges of a `dim`-dimensional hypercube
(`dim >= 2`, visiting all ``2^{\\text{dim}}`` vertices via Gray-code ordering).
"""
function hypercube_curve(
    dim::Int;
    side = 1.0,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    c = _resolve_center(center, dim, 2, R)
    D = length(c)
    D <= 20 || throw(ArgumentError("hypercube_curve supports dim <= 20 (2^dim vertices)"))
    half = R(side) / R(2)
    M = 1 << D
    pts = Matrix{R}(undef, D, M)
    @inbounds for j in 0:(M - 1)
        gray = j ⊻ (j >> 1)
        for d in 1:D
            bit = (gray >> (d - 1)) & 1
            pts[d, j + 1] = c[d] + (bit == 1 ? half : -half)
        end
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    star_curve(points::Int; outer_radius = 1.0, inner_radius = 0.4, center = nothing, rotation = π / 2, dim = 2, T = Float64, closed = true, to_device = identity)

Construct a symmetric `points`-pointed star polygon (`2 * points` vertices alternating
between `outer_radius` and `inner_radius`).
"""
function star_curve(
    points::Int;
    outer_radius = 1.0,
    inner_radius = 0.4,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rotation = π / 2,
    dim::Int = 2,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    points >= 2 || throw(ArgumentError("Star must have points >= 2"))
    c = _resolve_center(center, dim, 2, R)
    r_out, r_in = R(outer_radius), R(inner_radius)
    θ0 = R(rotation)
    M = 2 * points
    pts = repeat(c, 1, M)
    @inbounds for k in 0:(M - 1)
        r = iseven(k) ? r_out : r_in
        θ = θ0 + R(π) * k / points
        pts[1, k + 1] += r * cos(θ)
        pts[2, k + 1] += r * sin(θ)
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    cardioid_curve(n::Int; a = 1.0, center = nothing, rotation = 0.0, dim = 2, T = Float64, closed = true, to_device = identity)

Sample `n` points on a cardioid ``r(\\theta) = a (1 + \\cos\\theta)`` in ``\\mathbb{R}^D``.
"""
function cardioid_curve(
    n::Int;
    a = 1.0,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rotation = 0.0,
    dim::Int = 2,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    n >= 4 || throw(ArgumentError("Number of points n must be >= 4"))
    c = _resolve_center(center, dim, 2, R)
    scale = R(a)
    sα, cα = sincos(R(rotation))
    ts = range(zero(R), R(2π); length = n + 1)[1:n]
    pts = repeat(c, 1, n)
    @inbounds for j in 1:n
        θ = ts[j]
        r = scale * (one(R) + cos(θ))
        x = r * cos(θ)
        y = r * sin(θ)
        pts[1, j] += cα * x - sα * y
        pts[2, j] += sα * x + cα * y
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    torus_knot_curve(p::Int, q::Int, n::Int; R_major = 2.0, r_minor = 0.7, center = nothing, dim = 3, T = Float64, closed = true, to_device = identity)

Sample `n` points on a `(p, q)`-torus knot embedded in ``\\mathbb{R}^D`` (`dim >= 3`).
For `p = 2, q = 3`, this produces a trefoil knot.
"""
function torus_knot_curve(
    p::Int,
    q::Int,
    n::Int;
    R_major = 2.0,
    r_minor = 0.7,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dim::Int = 3,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    n >= 6 || throw(ArgumentError("Number of points n must be >= 6"))
    c = _resolve_center(center, dim, 3, R)
    Rm, rm = R(R_major), R(r_minor)
    ts = range(zero(R), R(2π); length = n + 1)[1:n]
    pts = repeat(c, 1, n)
    @inbounds for j in 1:n
        t = ts[j]
        ρ = Rm + rm * cos(q * t)
        pts[1, j] += ρ * cos(p * t)
        pts[2, j] += ρ * sin(p * t)
        pts[3, j] += rm * sin(q * t)
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    trefoil_curve(n::Int; scale = 1.0, center = nothing, dim = 3, T = Float64, closed = true, to_device = identity)

Sample `n` points on a 3D trefoil knot embedded in ``\\mathbb{R}^D`` (`dim >= 3`).
"""
function trefoil_curve(
    n::Int;
    scale = 1.0,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dim::Int = 3,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    n >= 6 || throw(ArgumentError("Number of points n must be >= 6"))
    c = _resolve_center(center, dim, 3, R)
    s = R(scale)
    ts = range(zero(R), R(2π); length = n + 1)[1:n]
    pts = repeat(c, 1, n)
    @inbounds for j in 1:n
        t = ts[j]
        pts[1, j] += s * (sin(t) + R(2) * sin(R(2) * t))
        pts[2, j] += s * (cos(t) - R(2) * cos(R(2) * t))
        pts[3, j] += s * (-sin(R(3) * t))
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    lissajous_curve(n::Int; a = 1.0, b = 1.0, fx = 3, fy = 2, delta = π / 2, center = nothing, dim = 2, T = Float64, closed = true, to_device = identity)

Sample `n` points on a Lissajous curve ``x(t) = a\\sin(f_x t + \\delta)``, ``y(t) = b\\sin(f_y t)``.
"""
function lissajous_curve(
    n::Int;
    a = 1.0,
    b = 1.0,
    fx::Real = 3,
    fy::Real = 2,
    delta = π / 2,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dim::Int = 2,
    T::Type{R} = Float64,
    closed::Bool = true,
    to_device::Function = identity,
) where {R<:Real}
    n >= 4 || throw(ArgumentError("Number of points n must be >= 4"))
    c = _resolve_center(center, dim, 2, R)
    ra, rb = R(a), R(b)
    w_x, w_y, δ = R(fx), R(fy), R(delta)
    ts = range(zero(R), R(2π); length = n + 1)[1:n]
    pts = repeat(c, 1, n)
    @inbounds for j in 1:n
        t = ts[j]
        pts[1, j] += ra * sin(w_x * t + δ)
        pts[2, j] += rb * sin(w_y * t)
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    spiral_curve(n::Int; a = 0.0, b = 0.2, turns = 3.0, kind = :archimedean, center = nothing, dim = 2, T = Float64, closed = false, to_device = identity)

Sample `n` points along a planar spiral (`kind = :archimedean` for ``r = a + b\\theta``, or
`kind = :logarithmic` for ``r = a e^{b\\theta}``) in ``\\mathbb{R}^D``.
"""
function spiral_curve(
    n::Int;
    a = 0.0,
    b = 0.2,
    turns = 3.0,
    kind::Symbol = :archimedean,
    center::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dim::Int = 2,
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    n >= 2 || throw(ArgumentError("Number of points n must be >= 2"))
    kind in (:archimedean, :logarithmic) ||
        throw(ArgumentError("kind must be :archimedean or :logarithmic, got :$kind"))
    c = _resolve_center(center, dim, 2, R)
    ra, rb = R(a), R(b)
    ts = range(zero(R), R(2π * turns); length = n)
    pts = repeat(c, 1, n)
    @inbounds for j in 1:n
        θ = ts[j]
        r = kind === :archimedean ? (ra + rb * θ) : (ra * exp(rb * θ))
        pts[1, j] += r * cos(θ)
        pts[2, j] += r * sin(θ)
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    iterated_curve(step_fn::Function, x0::AbstractVector{<:Real}, n::Int; T = Float64, closed = false, to_device = identity)
    iterated_curve(step_fn::Function, history::AbstractVector{<:AbstractVector{<:Real}}, n::Int; T = Float64, closed = false, to_device = identity)

Generate an `n`-point curve in ``\\mathbb{R}^D`` from a recurrence relation:
- **1-step recurrence:** starts from `x0` and computes `x_{k+1} = step_fn(x_k)`.
- **`p`-step recurrence:** starts from `history = [x_1, ..., x_p]` and computes
  `x_{k+1} = step_fn(x_k, x_{k-1}, ..., x_{k-p+1})`.
"""
function iterated_curve(
    step_fn::Function,
    x0::AbstractVector{<:Real},
    n::Int;
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    D = length(x0)
    D >= 1 || throw(ArgumentError("Initial state x0 cannot be empty"))
    n >= 1 || throw(ArgumentError("Number of points n must be >= 1"))

    pts = Matrix{R}(undef, D, n)
    curr = R[R(v) for v in x0]
    @inbounds pts[:, 1] .= curr

    for j in 2:n
        nxt = step_fn(curr)
        length(nxt) == D || throw(DimensionMismatch("step_fn returned length $(length(nxt)), expected $D"))
        @inbounds for i in 1:D
            curr[i] = R(nxt[i])
            pts[i, j] = curr[i]
        end
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

function iterated_curve(
    step_fn::Function,
    history::AbstractVector{<:AbstractVector{<:Real}},
    n::Int;
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    p = length(history)
    p >= 1 || throw(ArgumentError("Initial history cannot be empty"))
    D = length(first(history))
    D >= 1 || throw(ArgumentError("State dimension D must be >= 1"))
    all(v -> length(v) == D, history) || throw(DimensionMismatch("All history states must have dimension $D"))
    n >= p || throw(ArgumentError("Number of points n ($n) must be >= history length ($p)"))

    pts = Matrix{R}(undef, D, n)
    buf = [R[R(v) for v in history[k]] for k in 1:p]
    @inbounds for k in 1:p
        pts[:, k] .= buf[k]
    end

    for j in (p + 1):n
        nxt = step_fn(reverse(buf)...)
        length(nxt) == D || throw(DimensionMismatch("step_fn returned length $(length(nxt)), expected $D"))
        new_state = R[R(nxt[i]) for i in 1:D]
        @inbounds pts[:, j] .= new_state
        popfirst!(buf)
        push!(buf, new_state)
    end
    DiscreteCurve(to_device(pts); closed = closed)
end

"""
    random_curve(n::Int; dim = 2, distribution = :normal, step_scale = 1.0, cumulative = true, harmonics = 4, rng = Random.default_rng(), T = Float64, closed = false, to_device = identity)

Generate an `n`-point random curve in ``\\mathbb{R}^D`` using `rng`:
- `:normal` — Gaussian increments (`cumulative = true` gives Brownian motion; `false` gives i.i.d. points).
- `:uniform` — Uniform increments in ``[-\\text{step\\_scale}, \\text{step\\_scale}]^D``.
- `:sphere` — Isotropic random walk with constant Euclidean step length `step_scale` on ``S^{D-1}``.
- `:smooth` — Smooth random Fourier curve with `harmonics` frequency modes.
"""
function random_curve(
    n::Int;
    dim::Int = 2,
    distribution::Symbol = :normal,
    step_scale = 1.0,
    cumulative::Bool = true,
    harmonics::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
    T::Type{R} = Float64,
    closed::Bool = false,
    to_device::Function = identity,
) where {R<:Real}
    dim >= 1 || throw(ArgumentError("Dimension dim must be >= 1"))
    n >= 2 || throw(ArgumentError("Number of points n must be >= 2"))
    s = R(step_scale)

    if distribution === :smooth
        harmonics >= 1 || throw(ArgumentError("harmonics must be >= 1"))
        pts = zeros(R, dim, n)
        ts = closed ? range(zero(R), R(2π); length = n + 1)[1:n] : range(zero(R), R(2π); length = n)
        for k in 1:harmonics
            amp = s / R(k)
            a_cos = randn(rng, R, dim) .* amp
            b_sin = randn(rng, R, dim) .* amp
            @inbounds for j in 1:n
                sk, ck = sincos(R(k) * ts[j])
                for d in 1:dim
                    pts[d, j] += a_cos[d] * ck + b_sin[d] * sk
                end
            end
        end
        return DiscreteCurve(to_device(pts); closed = closed)
    end

    pts = Matrix{R}(undef, dim, n)
    if distribution === :normal
        randn!(rng, pts)
        pts .*= s
    elseif distribution === :uniform
        rand!(rng, pts)
        pts .= (pts .* R(2) .- one(R)) .* s
    elseif distribution === :sphere
        randn!(rng, pts)
        @inbounds for j in 1:n
            nrm = zero(R)
            for d in 1:dim
                nrm += pts[d, j]^2
            end
            inv_nrm = nrm > zero(R) ? s / sqrt(nrm) : zero(R)
            for d in 1:dim
                pts[d, j] *= inv_nrm
            end
        end
    else
        throw(ArgumentError("Unsupported distribution :$distribution (choose :normal, :uniform, :sphere, or :smooth)"))
    end

    if cumulative
        @inbounds for d in 1:dim
            pts[d, 1] = zero(R)
        end
        cumsum!(pts, pts; dims = 2)
    end

    DiscreteCurve(to_device(pts); closed = closed)
end