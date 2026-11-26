const _MAKIE_PLOT_HOOK = Ref{Any}(nothing)
const _MAKIE_PLOT_BANG_HOOK = Ref{Any}(nothing)
const _PLOTS_PLOT_HOOK = Ref{Any}(nothing)
const _PLOTS_PLOT_BANG_HOOK = Ref{Any}(nothing)

"""
    DCURVE_PALETTE

Colorblind-safe scientific publication palette (Okabe–Ito inspired) used by default
across `DCurve.jl` plotting backends.
"""
const DCURVE_PALETTE = (
    "#0072B2", # 1. Deep Ocean Blue (primary curve)
    "#D55E00", # 2. Vermilion (vertices / 2nd curve)
    "#009E73", # 3. Emerald Teal (start marker / 3rd curve)
    "#E69F00", # 4. Warm Amber (end marker / 4th curve)
    "#CC79A7", # 5. Muted Magenta
    "#56B4E9", # 6. Sky Cyan
    "#334155", # 7. Slate Charcoal
    "#882255", # 8. Deep Wine
)

# Host-side coordinate extraction: single bulk transfer if backed by GPU, zero-copy if CPU Matrix
@inline _host_points(c::DiscreteCurve) = c.points isa Matrix ? c.points : Array(c.points)

function _check_plottable_dim(c::DiscreteCurve)
    D = ndim(c)
    (1 <= D <= 3) || throw(ArgumentError(
        "Cannot directly plot a $(D)D curve; project to 1D, 2D, or 3D first (e.g., `DiscreteCurve(c.points[1:3, :]; closed = isclosed(c))`)."
    ))
    return D
end

"""
    curveplot(c::DiscreteCurve; kwargs...)
    curveplot!(args...; kwargs...)

Publication-grade plotting entry point for a [`DiscreteCurve`](@ref).

# Visual & Geometry Options
- `edge_size::Real = 2.5` (alias: `linewidth`): stroke thickness of curve edges.
- `edge_color = "#0072B2"` (alias: `color`): edge color or per-vertex color vector.
- `show_vertices::Bool = false`: overlay crisp white-rimmed markers at each vertex.
- `vertex_size::Real = 7.5`: diameter of vertex markers.
- `vertex_color = "#D55E00"`: fill color of vertex markers.
- `show_direction::Bool = false`: draw scale-invariant tangent orientation arrows along segments.
- `direction_count::Int = 8`: target number of orientation arrows distributed along the curve.
- `arrow_size::Real = 13.0`: arrowhead size when `show_direction = true`.
- `arrow_scale::Real = 1.0`: multiplier for orientation arrow shaft length.
- `arrow_color = nothing`: arrow color (defaults to `edge_color`).
- `show_endpoints::Bool = false`: highlight `c[1]` (emerald circle) and `c[end]` (amber diamond).
- `endpoint_size::Real = 10.5`: marker size for start/end indicators.
"""
function curveplot(c::DiscreteCurve; kwargs...)
    _check_plottable_dim(c)
    if _MAKIE_PLOT_HOOK[] !== nothing
        return _MAKIE_PLOT_HOOK[](c; kwargs...)
    elseif _PLOTS_PLOT_HOOK[] !== nothing
        return _PLOTS_PLOT_HOOK[](c; kwargs...)
    else
        throw(ErrorException(
            "No plotting backend loaded. Run `using CairoMakie` (or `GLMakie` / `Makie`) or `using Plots` first."
        ))
    end
end

function curveplot!(args...; kwargs...)
    if _MAKIE_PLOT_BANG_HOOK[] !== nothing
        return _MAKIE_PLOT_BANG_HOOK[](args...; kwargs...)
    elseif _PLOTS_PLOT_BANG_HOOK[] !== nothing
        return _PLOTS_PLOT_BANG_HOOK[](args...; kwargs...)
    else
        throw(ErrorException(
            "No plotting backend loaded. Run `using CairoMakie` (or `GLMakie` / `Makie`) or `using Plots` first."
        ))
    end
end