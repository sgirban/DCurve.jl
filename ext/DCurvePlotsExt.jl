module DCurvePlotsExt

using DCurve
using Plots

function __init__()
    DCurve._PLOTS_PLOT_HOOK[] = Plots.plot
    DCurve._PLOTS_PLOT_BANG_HOOK[] = Plots.plot!
end

function _extract_plots_arrays(c::DiscreteCurve)
    D = DCurve._check_plottable_dim(c)
    pts = DCurve._host_points(c)
    M = size(pts, 2)
    closed = isclosed(c) && M >= 2

    if D == 1
        xs_v = collect(1.0:Float64(M))
        ys_v = vec(pts[1, :])
        xs_l = closed ? push!(copy(xs_v), Float64(M + 1)) : copy(xs_v)
        ys_l = closed ? push!(copy(ys_v), ys_v[1]) : copy(ys_v)
        return D, (xs_v, ys_v), (xs_l, ys_l)
    elseif D == 2
        xs_v = vec(pts[1, :])
        ys_v = vec(pts[2, :])
        xs_l = closed ? push!(copy(xs_v), xs_v[1]) : copy(xs_v)
        ys_l = closed ? push!(copy(ys_v), ys_v[1]) : copy(ys_v)
        return D, (xs_v, ys_v), (xs_l, ys_l)
    else
        xs_v = vec(pts[1, :])
        ys_v = vec(pts[2, :])
        zs_v = vec(pts[3, :])
        xs_l = closed ? push!(copy(xs_v), xs_v[1]) : copy(xs_v)
        ys_l = closed ? push!(copy(ys_v), ys_v[1]) : copy(ys_v)
        zs_l = closed ? push!(copy(zs_v), zs_v[1]) : copy(zs_v)
        return D, (xs_v, ys_v, zs_v), (xs_l, ys_l, zs_l)
    end
end

function _render_plots!(
    plt,
    c::DiscreteCurve;
    show_vertices::Bool = false,
    show_endpoints::Bool = false,
    show_direction::Bool = false,
    direction_count::Int = 8,
    vertex_size::Real = 4.5,
    vertex_color = DCurve.DCURVE_PALETTE[2],
    endpoint_size::Real = 6.5,
    color = nothing,
    edge_color = nothing,
    linewidth::Real = 2.5,
    edge_size = nothing,
    arrow_scale::Real = 1.0,
    arrow_color = nothing,
    label = "",
    aspect_ratio = nothing,
    kwargs...,
)
    D, verts, line = _extract_plots_arrays(c)
    ar = aspect_ratio === nothing ? (D == 2 ? :equal : :auto) : aspect_ratio

    series_idx = length(plt.series_list) + 1
    auto_color = DCurve.DCURVE_PALETTE[mod1(series_idx, length(DCurve.DCURVE_PALETTE))]
    eff_color = edge_color !== nothing ? edge_color : (color !== nothing ? color : auto_color)
    eff_lw = edge_size !== nothing ? edge_size : linewidth
    eff_arrow_color = arrow_color !== nothing ? arrow_color : eff_color

    if D <= 2
        Plots.plot!(
            plt,
            line[1],
            line[2];
            color = eff_color,
            linewidth = eff_lw,
            label = label,
            aspect_ratio = ar,
            gridalpha = 0.2,
            framestyle = :box,
            kwargs...,
        )
    else
        Plots.plot!(
            plt,
            line[1],
            line[2],
            line[3];
            color = eff_color,
            linewidth = eff_lw,
            label = label,
            gridalpha = 0.2,
            kwargs...,
        )
    end

    if show_vertices
        if D <= 2
            Plots.scatter!(
                plt,
                verts[1],
                verts[2];
                markersize = vertex_size,
                color = vertex_color,
                markerstrokecolor = :white,
                markerstrokewidth = 0.8,
                label = "",
            )
        else
            Plots.scatter!(
                plt,
                verts[1],
                verts[2],
                verts[3];
                markersize = vertex_size,
                color = vertex_color,
                markerstrokecolor = :white,
                markerstrokewidth = 0.8,
                label = "",
            )
        end
    end

    if show_endpoints && !isempty(verts[1])
        if D <= 2
            Plots.scatter!(
                plt,
                [verts[1][1]],
                [verts[2][1]];
                markersize = endpoint_size,
                color = DCurve.DCURVE_PALETTE[3],
                markershape = :circle,
                markerstrokecolor = :white,
                label = "",
            )
            if length(verts[1]) >= 2
                Plots.scatter!(
                    plt,
                    [verts[1][end]],
                    [verts[2][end]];
                    markersize = endpoint_size,
                    color = DCurve.DCURVE_PALETTE[4],
                    markershape = :diamond,
                    markerstrokecolor = :white,
                    label = "",
                )
            end
        else
            Plots.scatter!(
                plt,
                [verts[1][1]],
                [verts[2][1]],
                [verts[3][1]];
                markersize = endpoint_size,
                color = DCurve.DCURVE_PALETTE[3],
                markershape = :circle,
                markerstrokecolor = :white,
                label = "",
            )
            if length(verts[1]) >= 2
                Plots.scatter!(
                    plt,
                    [verts[1][end]],
                    [verts[2][end]],
                    [verts[3][end]];
                    markersize = endpoint_size,
                    color = DCurve.DCURVE_PALETTE[4],
                    markershape = :diamond,
                    markerstrokecolor = :white,
                    label = "",
                )
            end
        end
    end

    if show_direction && D <= 2
        n_seg = length(line[1]) - 1
        if n_seg >= 1 && direction_count >= 1
            xmin, xmax = extrema(line[1])
            ymin, ymax = extrema(line[2])
            diag = hypot(xmax - xmin, ymax - ymin)
            target_len = (diag > 0 ? 0.06 * diag : 0.1) * arrow_scale

            step = max(1, fld(n_seg, direction_count))
            idxs = 1:step:n_seg
            ox = Float64[]
            oy = Float64[]
            dx = Float64[]
            dy = Float64[]
            for i in idxs
                vx = line[1][i + 1] - line[1][i]
                vy = line[2][i + 1] - line[2][i]
                seg_len = hypot(vx, vy)
                if seg_len > 0
                    push!(ox, 0.5 * (line[1][i] + line[1][i + 1]))
                    push!(oy, 0.5 * (line[2][i] + line[2][i + 1]))
                    push!(dx, (target_len / seg_len) * vx)
                    push!(dy, (target_len / seg_len) * vy)
                end
            end
            Plots.quiver!(plt, ox, oy; quiver = (dx, dy), color = eff_arrow_color, linewidth = eff_lw, label = "")
        end
    end

    return plt
end

function Plots.plot(c::DiscreteCurve; prefer_makie::Bool = true, kwargs...)
    if prefer_makie && DCurve._MAKIE_PLOT_HOOK[] !== nothing
        return DCurve._MAKIE_PLOT_HOOK[](c; kwargs...)
    end
    plt = Plots.plot()
    return _render_plots!(plt, c; kwargs...)
end

function Plots.plot!(plt::Plots.AbstractPlot, c::DiscreteCurve; kwargs...)
    return _render_plots!(plt, c; kwargs...)
end

function Plots.plot!(c::DiscreteCurve; prefer_makie::Bool = true, kwargs...)
    if prefer_makie && DCurve._MAKIE_PLOT_BANG_HOOK[] !== nothing
        return DCurve._MAKIE_PLOT_BANG_HOOK[](c; kwargs...)
    end
    return _render_plots!(Plots.current(), c; kwargs...)
end

end 