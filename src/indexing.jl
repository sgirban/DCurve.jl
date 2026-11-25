@inline function _resolve_index(c::DiscreteCurve, i::Integer)
    n = npoints(c)
    if isclosed(c)
        return mod1(Int(i), n)
    else
        @boundscheck (1 <= i <= n) || throw(BoundsError(c, i))
        return Int(i)
    end
end

Base.firstindex(::DiscreteCurve) = 1
Base.lastindex(c::DiscreteCurve) = npoints(c)
Base.eachindex(c::DiscreteCurve) = Base.OneTo(npoints(c))
Base.length(c::DiscreteCurve) = npoints(c)
Base.IteratorEltype(::Type{<:DiscreteCurve}) = Base.EltypeUnknown()

@inline function Base.iterate(c::DiscreteCurve, state::Int = 1)
    state > npoints(c) && return nothing
    return (@inbounds c[state]), state + 1
end

"""
    c[i]
    c[indices]

Extract the `i`-th point (or a collection of points) from a [`DiscreteCurve`](@ref) as a
zero-copy view into `c.points`.

- **Single integer (`c[i]`):** returns a `D`-element column view (`@view c.points[:, idx]`).
  Supports `end` arithmetic such as `c[end - 1]`.
- **Index collections (`c[1:4]`, `c[[1, 3, 5]]`, `c[(1, 3, 4, 5)]`, `c[:]`):** returns a
  `D × K` submatrix view of the selected points.
- **Boundary behavior:**
  - If `isclosed(c) == true`, indices wrap periodically via `mod1(i, npoints(c))`
    (`c[0] == c[end]` and `c[end + 1] == c[1]`).
  - If `isclosed(c) == false`, out-of-bounds indices throw a `BoundsError`.

# Examples
```jldoctest
julia> c_open = DiscreteCurve([0.0 1.0 2.0; 0.0 2.0 0.0]; closed = false);

julia> c_open[end - 1]
2-element view(::Matrix{Float64}, :, 2) with eltype Float64:
 1.0
 2.0

julia> c_closed = DiscreteCurve([0.0 1.0 2.0; 0.0 2.0 0.0]; closed = true);

julia> c_closed[0] == c_closed[end] && c_closed[end + 1] == c_closed[1]
true

julia> c_closed[(0, 1, 4)]
2×3 view(::Matrix{Float64}, :, [3, 1, 1]) with eltype Float64:
 2.0  0.0  0.0
 0.0  0.0  0.0
"""
Base.@propagate_inbounds function Base.getindex(c::DiscreteCurve, i::Integer)
idx = _resolve_index(c, i)
return @view c.points[:, idx]
end

Base.@propagate_inbounds function Base.getindex(c::DiscreteCurve, r::AbstractRange{<:Integer})
n = npoints(c)
if isclosed(c)
# Fast path: in-bounds ranges keep contiguous/strided range views with zero index allocation
if isempty(r) || (minimum(r) >= 1 && maximum(r) <= n)
return @view c.points[:, r]
else
idxs = [mod1(Int(i), n) for i in r]
return @view c.points[:, idxs]
end
else
@boundscheck checkbounds(Base.OneTo(n), r)
return @view c.points[:, r]
end
end

Base.@propagate_inbounds function Base.getindex(c::DiscreteCurve, indices::AbstractVector{<:Integer})
n = npoints(c)
if isclosed(c)
all_inbounds = true
@inbounds for i in indices
if !(1 <= i <= n)
all_inbounds = false
break
end
end
if all_inbounds
return @view c.points[:, indices]
else
idxs = [mod1(Int(i), n) for i in indices]
return @view c.points[:, idxs]
end
else
@boundscheck checkbounds(Base.OneTo(n), indices)
return @view c.points[:, indices]
end
end

Base.@propagate_inbounds function Base.getindex(c::DiscreteCurve, indices::Tuple{Vararg{Integer}})
# map over an NTuple resolves all indices on the stack / in registers at compile time
idxs = map(i -> _resolve_index(c, i), indices)
return @view c.points[:, collect(Int, idxs)]
end

Base.getindex(c::DiscreteCurve, ::Colon) = @view c.points[:, :]

Base.@propagate_inbounds function Base.getindex(c::DiscreteCurve, i::Integer, d::Integer)
    idx = _resolve_index(c, i)
    @boundscheck (1 <= d <= ndim(c)) || throw(BoundsError(c.points, (d, idx)))
    return @inbounds c.points[d, idx]
end

Base.@propagate_inbounds function Base.getindex(c::DiscreteCurve, ::Colon, d::Integer)
    @boundscheck (1 <= d <= ndim(c)) || throw(BoundsError(c.points, (d, :)))
    return @view c.points[d, :]
end