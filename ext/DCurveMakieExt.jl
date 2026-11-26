module DCurveMakieExt

using DCurve
using Makie

function __init__()
    DCurve._MAKIE_PLOT_HOOK[] = Makie.plot
    DCurve._MAKIE_PLOT_BANG_HOOK[] = Makie.plot!
end

Makie.plottype(::DiscreteCurve) = CurvePlot
Makie.preferred_axis_type(c::DiscreteCurve) = ndim(c) >= 3 ? Makie.LScene : Makie.Axis

@recipe(CurvePlot, curve) do scene
    Theme(
        color = DCurve.DCURVE_PALETTE[1],
        edge_color = nothing,
        linewidth = 2.5,
        edge_size = nothing,
        linestyle = :solid,
        alpha = 1.0,
        show_vertices = false,
        vertex_size = 7.5,
        vertex_color = DCurve.DCURVE_PALETTE[2],
        show_endpoints = false,
        endpoint_size = 10.5,
        show_direction = false,
        direction_count = 8,
        arrow_size = 13.0,
        arrow_scale = 1.0,
        arrow_color = nothing,
        label = nothing,
    )
end

function _prepare_makie_geometry(c::DiscreteCurve)
    D = DCurve._check_plottable_dim(c)
    pts = DCurve._host_points(c)
    M = size(pts, 2)
    closed = isclosed(c) && M >= 2

    if D == 1
        verts = Point2f[Point2f(Float32(j), pts[1, j]) for j in 1:M]
        line_pts = closed ? push!(copy(verts), Point2f(Float32(M + 1), pts[1, 1])) : copy(verts)
        return D, verts, line_pts
    elseif D == 2
        verts = Point2f[Point2f(pts[1, j], pts[2, j]) for j in 1:M]
        line_pts = closed ? push!(copy(verts), verts[1]) : copy(verts)
        return D, verts, line_pts
    else
        verts = Point3f[Point3f(pts[1, j], pts[2, j], pts[3, j]) for j in 1:M]
        line_pts = closed ? push!(copy(verts), verts[1]) : copy(verts)
        return D, verts, line_pts
    end
end

function _compute_makie_arrows_2d(lpts::AbstractVector, k_req::Int, scale::Real, active::Bool)
    n_seg = length(lpts) - 1
    if !active || n_seg < 1 || k_req < 1
        return (Point2f[Point2f(0)], Vec2f[Vec2f(0)])
    end

    p_min = reduce((a, b) -> min.(a, b), lpts)
    p_max = reduce((a, b) -> max.(a, b), lpts)
    diag = sqrt(sum(abs2, p_max - p_min))
    target_len = Float32((diag > 0 ? 0.06 * diag : 0.1) * scale)

    step = max(1, fld(n_seg, k_req))
    origins = Point2f[]
    dirs = Vec2f[]
    for i in 1:step:n_seg
        d = lpts[i + 1] - lpts[i]
        seg_len = sqrt(sum(abs2, d))
        if seg_len > 0
            push!(origins, 0.5f0 * (lpts[i] + lpts[i + 1]))
            push!(dirs, Vec2f((target_len / seg_len) * d))
        end
    end
    isempty(origins) && return (Point2f[Point2f(0)], Vec2f[Vec2f(0)])
    return (origins, dirs)
end

function _compute_makie_arrows_3d(lpts::AbstractVector, k_req::Int, scale::Real, active::Bool)
    n_seg = length(lpts) - 1
    if !active || n_seg < 1 || k_req < 1
        return (Point3f[Point3f(0)], Vec3f[Vec3f(0)])
    end

    p_min = reduce((a, b) -> min.(a, b), lpts)
    p_max = reduce((a, b) -> max.(a, b), lpts)
    diag = sqrt(sum(abs2, p_max - p_min))
    target_len = Float32((diag > 0 ? 0.06 * diag : 0.1) * scale)

    step = max(1, fld(n_seg, k_req))
    origins = Point3f[]
    dirs = Vec3f[]
    for i in 1:step:n_seg
        d = lpts[i + 1] - lpts[i]
        seg_len = sqrt(sum(abs2, d))
        if seg_len > 0
            push!(origins, 0.5f0 * (lpts[i] + lpts[i + 1]))
            push!(dirs, Vec3f((target_len / seg_len) * d))
        end
    end
    isempty(origins) && return (Point3f[Point3f(0)], Vec3f[Vec3f(0)])
    return (origins, dirs)
end

function Makie.plot!(p::CurvePlot{<:Tuple{DiscreteCurve}})
    D_init = DCurve._check_plottable_dim(to_value(p[:curve]))
    geom = lift(_prepare_makie_geometry, p[:curve])
    line_pts = lift(g -> g[3], geom)
    vert_pts = lift(g -> g[2], geom)

    eff_linewidth = lift((es, lw) -> es === nothing ? lw : es, p[:edge_size], p[:linewidth])
    eff_color = lift((ec, c) -> ec === nothing ? c : ec, p[:edge_color], p[:color])
    eff_arrow_color = lift((ac, c) -> ac === nothing ? c : ac, p[:arrow_color], eff_color)

    lines!(
        p,
        line_pts;
        color = eff_color,
        linewidth = eff_linewidth,
        linestyle = p[:linestyle],
        alpha = p[:alpha],
        linecap = :round,
        joinstyle = :round,
        label = p[:label],
    )

    scatter!(
        p,
        vert_pts;
        visible = p[:show_vertices],
        markersize = p[:vertex_size],
        color = p[:vertex_color],
        strokewidth = 1.1,
        strokecolor = :white,
    )

    start_pt = lift(vs -> isempty(vs) ? similar(vs, 0) : [first(vs)], vert_pts)
    end_pt = lift(vs -> length(vs) < 2 ? similar(vs, 0) : [last(vs)], vert_pts)

    scatter!(
        p,
        start_pt;
        visible = p[:show_endpoints],
        markersize = p[:endpoint_size],
        color = DCurve.DCURVE_PALETTE[3],
        marker = :circle,
        strokewidth = 1.2,
        strokecolor = :white,
    )

    scatter!(
        p,
        end_pt;
        visible = p[:show_endpoints],
        markersize = p[:endpoint_size],
        color = DCurve.DCURVE_PALETTE[4],
        marker = :diamond,
        strokewidth = 1.2,
        strokecolor = :white,
    )

    if D_init == 3
        arrows_3d = lift(
            (g, k, s, vis) -> _compute_makie_arrows_3d(g[3], k, s, vis),
            geom,
            p[:direction_count],
            p[:arrow_scale],
            p[:show_direction],
        )
        @static if isdefined(Makie, :arrows3d!)
            arrows3d!(
                p,
                lift(first, arrows_3d),
                lift(last, arrows_3d);
                visible = p[:show_direction],
                color = eff_arrow_color,
            )
        else
            arrows!(
                p,
                lift(first, arrows_3d),
                lift(last, arrows_3d);
                visible = p[:show_direction],
                color = eff_arrow_color,
                arrowsize = p[:arrow_size],
                linewidth = eff_linewidth,
            )
        end
    else
        arrows_2d = lift(
            (g, k, s, vis) -> _compute_makie_arrows_2d(g[3], k, s, vis),
            geom,
            p[:direction_count],
            p[:arrow_scale],
            p[:show_direction],
        )
        @static if isdefined(Makie, :arrows2d!)
            arrows2d!(
                p,
                lift(first, arrows_2d),
                lift(last, arrows_2d);
                visible = p[:show_direction],
                color = eff_arrow_color,
                shaftwidth = eff_linewidth,
                tipwidth = p[:arrow_size],
                tiplength = p[:arrow_size],
            )
        else
            arrows!(
                p,
                lift(first, arrows_2d),
                lift(last, arrows_2d);
                visible = p[:show_direction],
                color = eff_arrow_color,
                arrowsize = p[:arrow_size],
                linewidth = eff_linewidth,
            )
        end
    end

    return p
end

end