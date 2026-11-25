function _format_point(v::AbstractVector{<:Real})
    coords = join((sprint(show, round(x; sigdigits = 5)) for x in v), ", ")
    return string("[", coords, "]")
end

function Base.show(io::IO, c::DiscreteCurve{T}) where {T}
    D = ndim(c)
    M = npoints(c)
    state = isclosed(c) ? "closed" : "open"
    print(io, "DiscreteCurve(", D, "D, ", M, " pts, ", T, ", ", state, ")")
end

function Base.show(io::IO, ::MIME"text/plain", c::DiscreteCurve{T, A}) where {T, A}
    D = ndim(c)
    M = npoints(c)
    state = isclosed(c) ? "closed" : "open"

    println(io, "DiscreteCurve{", T, "} (", D, "D, ", M, " points, ", state, ")")
    println(io, "  storage : ", A)

    p_first = vec(Array(@view c.points[:, 1:1]))
    print(io, "  start   : ", _format_point(p_first))
    if M > 1
        p_last = vec(Array(@view c.points[:, M:M]))
        println(io)
        print(io, "  end     : ", _format_point(p_last))
    end
end