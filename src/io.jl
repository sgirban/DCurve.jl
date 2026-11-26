using Mmap

const _DCRV_MAGIC = (0x44, 0x43, 0x52, 0x56) # "DCRV"
const _DCRV_VERSION = 0x01
const _DCRV_HEADER_BYTES = 32
const _FORMAT_PRIORITY = (".dcrv", ".bin", ".obj", ".ply", ".pcd", ".csv", ".tsv", ".json")

@inline _dtype_code(::Type{Float16}) = 0x01
@inline _dtype_code(::Type{Float32}) = 0x02
@inline _dtype_code(::Type{Float64}) = 0x03
@inline _dtype_code(::Type{Int32}) = 0x04
@inline _dtype_code(::Type{Int64}) = 0x05
@inline _dtype_code(::Type{ComplexF32}) = 0x06
@inline _dtype_code(::Type{ComplexF64}) = 0x07
@inline _dtype_code(::Type{BigFloat}) = 0xff
@inline _dtype_code(::Type{T}) where {T} =
    throw(ArgumentError("Unsupported binary element type $T; convert to Float32, Float64, or BigFloat first."))

function _code_to_dtype(code::UInt8)
    code == 0x01 && return Float16
    code == 0x02 && return Float32
    code == 0x03 && return Float64
    code == 0x04 && return Int32
    code == 0x05 && return Int64
    code == 0x06 && return ComplexF32
    code == 0x07 && return ComplexF64
    code == 0xff && return BigFloat
    throw(ArgumentError("Unknown .dcrv dtype code: 0x$(string(code, base = 16))"))
end

function _write_dcrv_header(io::IO, flags::UInt8, code::UInt8, dim_or_count::Int64, npts::Int64)
    write(io, _DCRV_MAGIC[1], _DCRV_MAGIC[2], _DCRV_MAGIC[3], _DCRV_MAGIC[4])
    write(io, _DCRV_VERSION)
    write(io, flags)
    write(io, code)
    write(io, 0x00)
    write(io, htol(dim_or_count))
    write(io, htol(npts))
    write(io, htol(Int64(0)))
    return nothing
end

function _read_dcrv_header(io::IO)
    m1, m2, m3, m4 = read(io, UInt8), read(io, UInt8), read(io, UInt8), read(io, UInt8)
    (m1, m2, m3, m4) == _DCRV_MAGIC || throw(ArgumentError("Invalid .dcrv file: bad magic header."))
    ver = read(io, UInt8)
    ver == _DCRV_VERSION || throw(ArgumentError("Unsupported .dcrv version: $ver"))
    flags = read(io, UInt8)
    code = read(io, UInt8)
    _ = read(io, UInt8) # reserved
    dim_or_count = ltoh(read(io, Int64))
    npts = ltoh(read(io, Int64))
    _ = read(io, Int64) # reserved64
    return flags, code, Int(dim_or_count), Int(npts)
end

function _write_matrix_payload(io::IO, pts::AbstractMatrix{T}) where {T}
    host = pts isa Matrix{T} ? pts : Matrix{T}(pts)
    if T === BigFloat
        for x in host
            s = string(x)
            write(io, htol(Int32(ncodeunits(s))))
            write(io, codeunits(s))
        end
    else
        write(io, host)
    end
    return nothing
end

function _read_matrix_payload(io::IO, ::Type{T}, D::Int, M::Int) where {T}
    pts = Matrix{T}(undef, D, M)
    if T === BigFloat
        for idx in eachindex(pts)
            len = Int(ltoh(read(io, Int32)))
            buf = read(io, len)
            pts[idx] = parse(BigFloat, String(buf))
        end
    else
        read!(io, pts)
    end
    return pts
end

function _to_3d_host(c::DiscreteCurve)
    pts = c.points isa Matrix ? c.points : Array(c.points)
    D, M = size(pts)
    D <= 3 || throw(ArgumentError("3D mesh/cloud formats support up to 3D curves (got $(D)D)."))
    T = eltype(pts)
    out = zeros(T, 3, M)
    if D == 1
        for j in 1:M
            out[1, j] = T(j)
            out[2, j] = pts[1, j]
        end
    else
        out[1:D, :] .= pts
    end
    return out, D
end


"""
    save_dcrv(path::AbstractString, c::DiscreteCurve)

Write a [`DiscreteCurve`](@ref) to a 32-byte SIMD-aligned binary `.dcrv` file.
"""
function save_dcrv(path::AbstractString, c::DiscreteCurve{T}) where {T}
    D, M = size(c.points)
    flags = isclosed(c) ? 0x01 : 0x00
    code = _dtype_code(T)
    open(path, "w") do io
        _write_dcrv_header(io, flags, code, Int64(D), Int64(M))
        _write_matrix_payload(io, c.points)
    end
    return path
end

"""
    load_dcrv(path::AbstractString; mmap::Bool = false, eltype::Union{Nothing, Type} = nothing)

Read a [`DiscreteCurve`](@ref) from a binary `.dcrv` file. Set `mmap = true` for
zero-copy memory-mapped loading in microseconds.
"""
function load_dcrv(path::AbstractString; mmap::Bool = false, eltype::Union{Nothing, Type} = nothing)
    flags, code, D, M = open(_read_dcrv_header, path, "r")
    (flags & 0x02) == 0 || throw(ArgumentError("File '$path' is a multi-curve bundle; use `load_curves` instead."))

    closed = (flags & 0x01) != 0
    T_file = _code_to_dtype(code)

    pts = if mmap && T_file !== BigFloat && (eltype === nothing || eltype === T_file)
        open(path, "r") do io
            Mmap.mmap(io, Matrix{T_file}, (D, M), _DCRV_HEADER_BYTES)
        end
    else
        open(path, "r") do io
            seek(io, _DCRV_HEADER_BYTES)
            raw = _read_matrix_payload(io, T_file, D, M)
            eltype === nothing || eltype === T_file ? raw : Matrix{eltype}(raw)
        end
    end

    return DiscreteCurve(pts; closed = closed)
end

"""
    save_curves(path::AbstractString, curves::AbstractVector{<:DiscreteCurve})

Pack thousands of curves into a single contiguous `.dcrv` archive with an offset table.
"""
function save_curves(path::AbstractString, curves::AbstractVector{<:DiscreteCurve})
    out_path = isempty(splitext(path)[2]) ? path * ".dcrv" : path
    N = length(curves)
    open(out_path, "w") do io
        _write_dcrv_header(io, 0x02, 0x00, Int64(N), Int64(0))
        table_pos = position(io)
        offsets = zeros(Int64, N)
        write(io, offsets)

        for (i, c) in enumerate(curves)
            offsets[i] = Int64(position(io))
            T = eltype(c)
            D, M = size(c.points)
            write(io, isclosed(c) ? 0x01 : 0x00)
            write(io, _dtype_code(T))
            write(io, htol(Int64(D)))
            write(io, htol(Int64(M)))
            _write_matrix_payload(io, c.points)
        end

        seek(io, table_pos)
        for off in offsets
            write(io, htol(off))
        end
    end
    return out_path
end

"""
    load_curves(path::AbstractString; eltype::Union{Nothing, Type} = nothing) -> Vector{DiscreteCurve}

Load all curves from a multi-curve `.dcrv` archive created by [`save_curves`](@ref).
"""
function load_curves(path::AbstractString; eltype::Union{Nothing, Type} = nothing)
    resolved = isfile(path) ? path : (isfile(path * ".dcrv") ? path * ".dcrv" : path)
    return open(resolved, "r") do io
        flags, _, N, _ = _read_dcrv_header(io)
        (flags & 0x02) != 0 || throw(ArgumentError("File '$resolved' is a single curve; use `load_dcrv` instead."))

        offsets = [Int(ltoh(read(io, Int64))) for _ in 1:N]
        out = Vector{DiscreteCurve}(undef, N)

        for i in 1:N
            seek(io, offsets[i])
            cflags = read(io, UInt8)
            code = read(io, UInt8)
            D = Int(ltoh(read(io, Int64)))
            M = Int(ltoh(read(io, Int64)))
            T_file = _code_to_dtype(code)
            raw = _read_matrix_payload(io, T_file, D, M)
            pts = eltype === nothing || eltype === T_file ? raw : Matrix{eltype}(raw)
            out[i] = DiscreteCurve(pts; closed = (cflags & 0x01) != 0)
        end
        return out
    end
end


"""
    save_obj(path::AbstractString, c::DiscreteCurve)

Export a [`DiscreteCurve`](@ref) as a Wavefront `.obj` file with `v` vertex records
and an `l` polyline connectivity record (compatible with Blender, Rhino, MeshLab, Houdini).
"""
function save_obj(path::AbstractString, c::DiscreteCurve)
    pts3, orig_dim = _to_3d_host(c)
    M = size(pts3, 2)
    open(path, "w") do io
        println(io, "# DCurve.jl Wavefront OBJ")
        println(io, "# dcurve: dim=", orig_dim, ", closed=", isclosed(c))
        for j in 1:M
            println(io, "v ", pts3[1, j], ' ', pts3[2, j], ' ', pts3[3, j])
        end
        if M >= 2
            print(io, "l")
            for j in 1:M
                print(io, ' ', j)
            end
            isclosed(c) && print(io, " 1")
            println(io)
        end
    end
    return path
end

"""
    load_obj(path::AbstractString; eltype::Type{T} = Float64, closed::Union{Nothing, Bool} = nothing, dim::Union{Nothing, Int} = nothing)

Load a [`DiscreteCurve`](@ref) from a Wavefront `.obj` file, reconstructing vertex
traversal order and closed-loop status from `l` polyline elements if present.
"""
function load_obj(
    path::AbstractString;
    eltype::Type{T} = Float64,
    closed::Union{Nothing, Bool} = nothing,
    dim::Union{Nothing, Int} = nothing,
) where {T<:Number}
    verts = Vector{NTuple{3, T}}()
    line_indices = Int[]
    meta_dim = nothing
    meta_closed = nothing

    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        if startswith(line, '#')
            m_d = match(r"dim=(\d+)", line)
            m_d !== nothing && (meta_dim = parse(Int, m_d.captures[1]))
            m_c = match(r"closed=(true|false)", line)
            m_c !== nothing && (meta_closed = (m_c.captures[1] == "true"))
            continue
        end
        parts = split(line)
        tag = parts[1]
        if tag == "v" && length(parts) >= 4
            x = parse(T, parts[2])
            y = parse(T, parts[3])
            z = parse(T, parts[4])
            push!(verts, (x, y, z))
        elseif tag == "l" && length(parts) >= 3
            for k in 2:length(parts)
                # Handle potential slash notation (v/vt)
                idx_str = first(split(parts[k], '/'))
                idx = parse(Int, idx_str)
                push!(line_indices, idx < 0 ? length(verts) + idx + 1 : idx)
            end
        end
    end

    isempty(verts) && throw(ArgumentError("No vertices ('v') found in OBJ file '$path'."))

    detected_closed = false
    ordered_verts = if length(line_indices) >= 2
        if line_indices[1] == line_indices[end]
            detected_closed = true
            pop!(line_indices)
        end
        [verts[i] for i in line_indices]
    else
        verts
    end

    target_dim = dim !== nothing ? dim : (meta_dim !== nothing ? meta_dim : 3)
    M = length(ordered_verts)
    pts = Matrix{T}(undef, target_dim, M)
    for j in 1:M
        pts[1, j] = ordered_verts[j][1]
        target_dim >= 2 && (pts[2, j] = ordered_verts[j][2])
        target_dim >= 3 && (pts[3, j] = ordered_verts[j][3])
    end

    is_cl = closed !== nothing ? closed : (meta_closed !== nothing ? meta_closed : detected_closed)
    return DiscreteCurve(pts; closed = is_cl)
end


"""
    save_ply(path::AbstractString, c::DiscreteCurve; binary::Bool = false)

Export a [`DiscreteCurve`](@ref) to Stanford `.ply` format with `element vertex`
and `element edge` connectivity (compatible with CloudCompare, MeshLab, Open3D, Blender).
"""
function save_ply(path::AbstractString, c::DiscreteCurve; binary::Bool = false)
    pts3, orig_dim = _to_3d_host(c)
    M = size(pts3, 2)
    closed = isclosed(c) && M >= 2
    n_edges = M < 2 ? 0 : (closed ? M : M - 1)
    T = eltype(pts3)
    ply_scalar = T === Float32 ? "float" : "double"

    open(path, "w") do io
        println(io, "ply")
        println(io, binary ? "format binary_little_endian 1.0" : "format ascii 1.0")
        println(io, "comment dcurve: dim=", orig_dim, ", closed=", isclosed(c))
        println(io, "element vertex ", M)
        println(io, "property ", ply_scalar, " x")
        println(io, "property ", ply_scalar, " y")
        println(io, "property ", ply_scalar, " z")
        println(io, "element edge ", n_edges)
        println(io, "property int vertex1")
        println(io, "property int vertex2")
        println(io, "end_header")

        if binary
            for j in 1:M
                write(io, htol(pts3[1, j]), htol(pts3[2, j]), htol(pts3[3, j]))
            end
            for j in 1:(M - 1)
                write(io, htol(Int32(j - 1)), htol(Int32(j)))
            end
            if closed
                write(io, htol(Int32(M - 1)), htol(Int32(0)))
            end
        else
            for j in 1:M
                println(io, pts3[1, j], ' ', pts3[2, j], ' ', pts3[3, j])
            end
            for j in 1:(M - 1)
                println(io, j - 1, ' ', j)
            end
            if closed
                println(io, M - 1, " 0")
            end
        end
    end
    return path
end

function _ply_type_size(tname::AbstractString)
    tname in ("char", "uchar", "int8", "uint8") && return 1
    tname in ("short", "ushort", "int16", "uint16") && return 2
    tname in ("int", "uint", "int32", "uint32", "float", "float32") && return 4
    tname in ("double", "float64", "int64", "uint64") && return 8
    throw(ArgumentError("Unsupported PLY property type '$tname'."))
end

function _read_ply_binary_val(io::IO, tname::AbstractString)
    tname in ("float", "float32") && return Float64(ltoh(read(io, Float32)))
    tname in ("double", "float64") && return ltoh(read(io, Float64))
    tname in ("int", "int32") && return Float64(ltoh(read(io, Int32)))
    tname in ("uint", "uint32") && return Float64(ltoh(read(io, UInt32)))
    tname in ("short", "int16") && return Float64(ltoh(read(io, Int16)))
    tname in ("ushort", "uint16") && return Float64(ltoh(read(io, UInt16)))
    tname in ("char", "int8") && return Float64(read(io, Int8))
    tname in ("uchar", "uint8") && return Float64(read(io, UInt8))
    throw(ArgumentError("Unsupported PLY binary property '$tname'."))
end

"""
    load_ply(path::AbstractString; eltype::Type{T} = Float64, closed::Union{Nothing, Bool} = nothing, dim::Union{Nothing, Int} = nothing)

Load a [`DiscreteCurve`](@ref) from an ASCII or `binary_little_endian` `.ply` file,
reconstructing edge order and closedness if `element edge` is present.
"""
function load_ply(
    path::AbstractString;
    eltype::Type{T} = Float64,
    closed::Union{Nothing, Bool} = nothing,
    dim::Union{Nothing, Int} = nothing,
) where {T<:Number}
    open(path, "r") do io
        first_line = strip(readline(io))
        first_line == "ply" || throw(ArgumentError("File '$path' is not a valid PLY file."))

        fmt = "ascii"
        n_verts = 0
        n_edges = 0
        current_elem = ""
        vert_props = Tuple{String, String}[]
        edge_props = Tuple{String, String}[]
        meta_dim = nothing
        meta_closed = nothing

        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue
            line == "end_header" && break
            parts = split(line)
            if parts[1] == "format"
                fmt = parts[2]
            elseif parts[1] == "comment"
                m_d = match(r"dim=(\d+)", line)
                m_d !== nothing && (meta_dim = parse(Int, m_d.captures[1]))
                m_c = match(r"closed=(true|false)", line)
                m_c !== nothing && (meta_closed = (m_c.captures[1] == "true"))
            elseif parts[1] == "element"
                current_elem = parts[2]
                cnt = parse(Int, parts[3])
                if current_elem == "vertex"
                    n_verts = cnt
                elseif current_elem == "edge"
                    n_edges = cnt
                end
            elseif parts[1] == "property" && parts[2] != "list"
                if current_elem == "vertex"
                    push!(vert_props, (parts[2], parts[3]))
                elseif current_elem == "edge"
                    push!(edge_props, (parts[2], parts[3]))
                end
            end
        end

        fmt in ("ascii", "binary_little_endian") ||
            throw(ArgumentError("Unsupported PLY format '$fmt' (supported: ascii, binary_little_endian)."))

        raw_verts = Vector{NTuple{3, T}}(undef, n_verts)
        edges = Tuple{Int, Int}[]

        if fmt == "ascii"
            ix = findfirst(p -> p[2] == "x", vert_props)
            iy = findfirst(p -> p[2] == "y", vert_props)
            iz = findfirst(p -> p[2] == "z", vert_props)
            for j in 1:n_verts
                vals = split(strip(readline(io)))
                x = ix === nothing ? zero(T) : parse(T, vals[ix])
                y = iy === nothing ? zero(T) : parse(T, vals[iy])
                z = iz === nothing ? zero(T) : parse(T, vals[iz])
                raw_verts[j] = (x, y, z)
            end
            iv1 = findfirst(p -> p[2] == "vertex1", edge_props)
            iv2 = findfirst(p -> p[2] == "vertex2", edge_props)
            if iv1 !== nothing && iv2 !== nothing
                for _ in 1:n_edges
                    vals = split(strip(readline(io)))
                    push!(edges, (parse(Int, vals[iv1]) + 1, parse(Int, vals[iv2]) + 1))
                end
            end
        else
            for j in 1:n_verts
                x = zero(T); y = zero(T); z = zero(T)
                for (ptype, pname) in vert_props
                    v = _read_ply_binary_val(io, ptype)
                    if pname == "x"
                        x = T(v)
                    elseif pname == "y"
                        y = T(v)
                    elseif pname == "z"
                        z = T(v)
                    end
                end
                raw_verts[j] = (x, y, z)
            end
            for _ in 1:n_edges
                v1 = 0; v2 = 0
                for (ptype, pname) in edge_props
                    v = round(Int, _read_ply_binary_val(io, ptype))
                    pname == "vertex1" && (v1 = v + 1)
                    pname == "vertex2" && (v2 = v + 1)
                end
                push!(edges, (v1, v2))
            end
        end

        detected_closed = false
        ordered_verts = if !isempty(edges)
            chain = Int[edges[1][1], edges[1][2]]
            for k in 2:length(edges)
                u, v = edges[k]
                if u == chain[end]
                    push!(chain, v)
                elseif v == chain[end]
                    push!(chain, u)
                end
            end
            if length(chain) >= 3 && chain[1] == chain[end]
                detected_closed = true
                pop!(chain)
            end
            [raw_verts[i] for i in chain]
        else
            raw_verts
        end

        target_dim = dim !== nothing ? dim : (meta_dim !== nothing ? meta_dim : 3)
        M = length(ordered_verts)
        pts = Matrix{T}(undef, target_dim, M)
        for j in 1:M
            pts[1, j] = ordered_verts[j][1]
            target_dim >= 2 && (pts[2, j] = ordered_verts[j][2])
            target_dim >= 3 && (pts[3, j] = ordered_verts[j][3])
        end

        is_cl = closed !== nothing ? closed : (meta_closed !== nothing ? meta_closed : detected_closed)
        return DiscreteCurve(pts; closed = is_cl)
    end
end


"""
    save_pcd(path::AbstractString, c::DiscreteCurve; binary::Bool = false)

Export a [`DiscreteCurve`](@ref) to Point Cloud Data (`.pcd` v0.7) format.
"""
function save_pcd(path::AbstractString, c::DiscreteCurve; binary::Bool = false)
    pts3, orig_dim = _to_3d_host(c)
    M = size(pts3, 2)
    pts_f32 = Float32.(pts3)

    open(path, "w") do io
        println(io, "# .PCD v0.7 - Point Cloud Data file format")
        println(io, "# dcurve: dim=", orig_dim, ", closed=", isclosed(c))
        println(io, "VERSION 0.7")
        println(io, "FIELDS x y z")
        println(io, "SIZE 4 4 4")
        println(io, "TYPE F F F")
        println(io, "COUNT 1 1 1")
        println(io, "WIDTH ", M)
        println(io, "HEIGHT 1")
        println(io, "VIEWPOINT 0 0 0 1 0 0 0")
        println(io, "POINTS ", M)
        println(io, binary ? "DATA binary" : "DATA ascii")
        if binary
            write(io, pts_f32)
        else
            for j in 1:M
                println(io, pts3[1, j], ' ', pts3[2, j], ' ', pts3[3, j])
            end
        end
    end
    return path
end

"""
    load_pcd(path::AbstractString; eltype::Type{T} = Float64, closed::Union{Nothing, Bool} = nothing, dim::Union{Nothing, Int} = nothing)

Load a [`DiscreteCurve`](@ref) from an ASCII or binary `.pcd` v0.7 file.
"""
function load_pcd(
    path::AbstractString;
    eltype::Type{T} = Float64,
    closed::Union{Nothing, Bool} = nothing,
    dim::Union{Nothing, Int} = nothing,
) where {T<:Number}
    open(path, "r") do io
        n_pts = 0
        data_mode = "ascii"
        fields = String[]
        sizes = Int[]
        types = String[]
        meta_dim = nothing
        meta_closed = nothing

        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue
            if startswith(line, '#')
                m_d = match(r"dim=(\d+)", line)
                m_d !== nothing && (meta_dim = parse(Int, m_d.captures[1]))
                m_c = match(r"closed=(true|false)", line)
                m_c !== nothing && (meta_closed = (m_c.captures[1] == "true"))
                continue
            end
            parts = split(line)
            key = uppercase(parts[1])
            if key == "FIELDS"
                fields = String.(parts[2:end])
            elseif key == "SIZE"
                sizes = parse.(Int, parts[2:end])
            elseif key == "TYPE"
                types = String.(parts[2:end])
            elseif key == "POINTS"
                n_pts = parse(Int, parts[2])
            elseif key == "DATA"
                data_mode = lowercase(parts[2])
                break
            end
        end

        ix = findfirst(==("x"), fields)
        iy = findfirst(==("y"), fields)
        iz = findfirst(==("z"), fields)
        raw_verts = Vector{NTuple{3, T}}(undef, n_pts)

        if data_mode == "ascii"
            for j in 1:n_pts
                vals = split(strip(readline(io)))
                x = ix === nothing ? zero(T) : parse(T, vals[ix])
                y = iy === nothing ? zero(T) : parse(T, vals[iy])
                z = iz === nothing ? zero(T) : parse(T, vals[iz])
                raw_verts[j] = (x, y, z)
            end
        elseif data_mode == "binary"
            for j in 1:n_pts
                x = zero(T); y = zero(T); z = zero(T)
                for (f, sz, tp) in zip(fields, sizes, types)
                    val = if tp == "F" && sz == 4
                        Float64(ltoh(read(io, Float32)))
                    elseif tp == "F" && sz == 8
                        ltoh(read(io, Float64))
                    elseif tp == "I" && sz == 4
                        Float64(ltoh(read(io, Int32)))
                    elseif tp == "U" && sz == 4
                        Float64(ltoh(read(io, UInt32)))
                    else
                        skip(io, sz)
                        0.0
                    end
                    f == "x" && (x = T(val))
                    f == "y" && (y = T(val))
                    f == "z" && (z = T(val))
                end
                raw_verts[j] = (x, y, z)
            end
        else
            throw(ArgumentError("Unsupported PCD DATA mode '$data_mode' (supported: ascii, binary)."))
        end

        target_dim = dim !== nothing ? dim : (meta_dim !== nothing ? meta_dim : 3)
        pts = Matrix{T}(undef, target_dim, n_pts)
        for j in 1:n_pts
            pts[1, j] = raw_verts[j][1]
            target_dim >= 2 && (pts[2, j] = raw_verts[j][2])
            target_dim >= 3 && (pts[3, j] = raw_verts[j][3])
        end

        is_cl = closed !== nothing ? closed : (meta_closed !== nothing ? meta_closed : false)
        return DiscreteCurve(pts; closed = is_cl)
    end
end

"""
    save_csv(path::AbstractString, c::DiscreteCurve; delim::Char = ',', header::Bool = true, include_meta::Bool = true)

Save a [`DiscreteCurve`](@ref) as a CSV/TSV file with one vertex per row (`M × D`).
"""
function save_csv(
    path::AbstractString,
    c::DiscreteCurve;
    delim::Char = ',',
    header::Bool = true,
    include_meta::Bool = true,
)
    pts = c.points isa Matrix ? c.points : Array(c.points)
    D, M = size(pts)
    axis_names = ("x", "y", "z", "w")

    open(path, "w") do io
        if include_meta
            println(io, "# dcurve: closed=", isclosed(c), ", dim=", D, ", npoints=", M)
        end
        if header
            for d in 1:D
                d > 1 && write(io, delim)
                write(io, d <= 4 ? axis_names[d] : "x$d")
            end
            println(io)
        end
        for j in 1:M
            for d in 1:D
                d > 1 && write(io, delim)
                print(io, pts[d, j])
            end
            println(io)
        end
    end
    return path
end

"""
    load_csv(path::AbstractString; delim::Char = ',', eltype::Type{T} = Float64, closed::Union{Nothing, Bool} = nothing, transposed::Bool = true)

Load a [`DiscreteCurve`](@ref) from a CSV/TSV file.
"""
function load_csv(
    path::AbstractString;
    delim::Char = ',',
    eltype::Type{T} = Float64,
    closed::Union{Nothing, Bool} = nothing,
    transposed::Bool = true,
) where {T<:Number}
    detected_closed = false
    rows = Vector{Vector{T}}()

    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        if startswith(line, '#')
            occursin("closed=true", line) && (detected_closed = true)
            continue
        end

        fields = split(line, delim)
        parsed = Vector{T}(undef, length(fields))
        ok = true
        for (k, f) in enumerate(fields)
            v = tryparse(T, strip(f))
            if v === nothing
                ok = false
                break
            end
            parsed[k] = v
        end
        ok && push!(rows, parsed)
    end

    isempty(rows) && throw(ArgumentError("No numeric coordinate rows found in '$path'."))
    ncols = length(rows[1])
    all(r -> length(r) == ncols, rows) || throw(ArgumentError("Inconsistent column counts in '$path'."))

    nrows = length(rows)
    pts = if transposed
        mat = Matrix{T}(undef, ncols, nrows)
        for j in 1:nrows, d in 1:ncols
            mat[d, j] = rows[j][d]
        end
        mat
    else
        mat = Matrix{T}(undef, nrows, ncols)
        for d in 1:nrows, j in 1:ncols
            mat[d, j] = rows[d][j]
        end
        mat
    end

    is_cl = closed === nothing ? detected_closed : closed
    return DiscreteCurve(pts; closed = is_cl)
end

"""
    save_json(path::AbstractString, c::DiscreteCurve; pretty::Bool = false)

Serialize a [`DiscreteCurve`](@ref) to JSON.
"""
function save_json(path::AbstractString, c::DiscreteCurve; pretty::Bool = false)
    pts = c.points isa Matrix ? c.points : Array(c.points)
    D, M = size(pts)
    sep = pretty ? "\n  " : ""
    inner_sep = pretty ? "\n    " : ""

    open(path, "w") do io
        write(io, "{", sep)
        write(io, "\"dimension\": ", string(D), ",", sep)
        write(io, "\"closed\": ", isclosed(c) ? "true" : "false", ",", sep)
        write(io, "\"eltype\": \"", string(eltype(pts)), "\",", sep)
        write(io, "\"points\": [")
        for j in 1:M
            j > 1 && write(io, ",")
            write(io, inner_sep, "[")
            for d in 1:D
                d > 1 && write(io, ", ")
                print(io, pts[d, j])
            end
            write(io, "]")
        end
        write(io, pretty ? "\n  ]\n}" : "]}")
    end
    return path
end

"""
    load_json(path::AbstractString; eltype::Type{T} = Float64, closed::Union{Nothing, Bool} = nothing)

Load a [`DiscreteCurve`](@ref) from a JSON file.
"""
function load_json(
    path::AbstractString;
    eltype::Type{T} = Float64,
    closed::Union{Nothing, Bool} = nothing,
) where {T<:Number}
    text = read(path, String)

    is_cl = if closed !== nothing
        closed
    else
        m_cl = match(r"\"closed\"\s*:\s*(true|false)", text)
        m_cl !== nothing && m_cl.captures[1] == "true"
    end

    m_pts = match(r"\"points\"\s*:\s*\[(.*)\]"s, text)
    m_pts === nothing && throw(ArgumentError("JSON file '$path' does not contain a \"points\" array."))
    body = m_pts.captures[1]

    rows = Vector{Vector{T}}()
    for m_vec in eachmatch(r"\[([^\[\]]+)\]", body)
        nums = split(m_vec.captures[1], ',')
        row = Vector{T}(undef, length(nums))
        for (k, s) in enumerate(nums)
            v = tryparse(T, strip(s))
            v === nothing && throw(ArgumentError("Invalid number '$s' in JSON points array."))
            row[k] = v
        end
        push!(rows, row)
    end

    isempty(rows) && throw(ArgumentError("Empty \"points\" array in '$path'."))
    D = length(rows[1])
    M = length(rows)
    all(r -> length(r) == D, rows) || throw(ArgumentError("Inconsistent vertex dimensions in '$path'."))

    pts = Matrix{T}(undef, D, M)
    for j in 1:M, d in 1:D
        pts[d, j] = rows[j][d]
    end
    return DiscreteCurve(pts; closed = is_cl)
end

function _resolve_load_path(path::AbstractString)
    base, ext = splitext(path)
    if !isempty(ext)
        isfile(path) && return path, lowercase(ext)
        throw(SystemError("opening file $(repr(path))", 2))
    end
    for candidate_ext in _FORMAT_PRIORITY
        candidate = path * candidate_ext
        if isfile(candidate)
            return candidate, candidate_ext
        end
    end

    isfile(path) && return path, ".dcrv"
    throw(ArgumentError("No curve file found for '$path' (searched extensions: $(join(_FORMAT_PRIORITY, ", ")))."))
end

"""
    save_curve(path::AbstractString, c::DiscreteCurve; kwargs...)
    save_curve(c::DiscreteCurve, path::AbstractString; kwargs...)

Save a [`DiscreteCurve`](@ref), automatically appending `.dcrv` if `path` has no file extension.

Supported extensions:
- `.dcrv` (default) / `.bin`: SIMD-aligned binary format ([`save_dcrv`](@ref))
- `.obj`: Wavefront OBJ with `v` vertices and `l` polyline edges ([`save_obj`](@ref))
- `.ply`: Stanford PLY with `vertex` and `edge` elements ([`save_ply`](@ref))
- `.pcd`: Point Cloud Data v0.7 ([`save_pcd`](@ref))
- `.csv` / `.tsv`: delimited text ([`save_csv`](@ref))
- `.json`: JSON format ([`save_json`](@ref))
"""
function save_curve(path::AbstractString, c::DiscreteCurve; kwargs...)
    base, raw_ext = splitext(path)
    target_path = isempty(raw_ext) ? base * ".dcrv" : path
    ext = lowercase(splitext(target_path)[2])

    if ext in (".dcrv", ".bin")
        return save_dcrv(target_path, c)
    elseif ext == ".obj"
        return save_obj(target_path, c)
    elseif ext == ".ply"
        return save_ply(target_path, c; kwargs...)
    elseif ext == ".pcd"
        return save_pcd(target_path, c; kwargs...)
    elseif ext == ".csv"
        return save_csv(target_path, c; delim = ',', kwargs...)
    elseif ext == ".tsv"
        return save_csv(target_path, c; delim = '\t', kwargs...)
    elseif ext == ".json"
        return save_json(target_path, c; kwargs...)
    else
        throw(ArgumentError("Unsupported file extension '$ext'. Use .dcrv, .obj, .ply, .pcd, .csv, .tsv, or .json."))
    end
end

save_curve(c::DiscreteCurve, path::AbstractString; kwargs...) = save_curve(path, c; kwargs...)

"""
    load_curve(path::AbstractString; kwargs...)

Load a [`DiscreteCurve`](@ref). If `path` omits a file extension (e.g., `load_curve("curve")`),
searches matching files with `.dcrv` taking highest priority:
`.dcrv` > `.bin` > `.obj` > `.ply` > `.pcd` > `.csv` > `.tsv` > `.json`.
"""
function load_curve(path::AbstractString; kwargs...)
    resolved_path, ext = _resolve_load_path(path)

    if ext in (".dcrv", ".bin")
        return load_dcrv(resolved_path; kwargs...)
    elseif ext == ".obj"
        return load_obj(resolved_path; kwargs...)
    elseif ext == ".ply"
        return load_ply(resolved_path; kwargs...)
    elseif ext == ".pcd"
        return load_pcd(resolved_path; kwargs...)
    elseif ext == ".csv"
        return load_csv(resolved_path; delim = ',', kwargs...)
    elseif ext == ".tsv"
        return load_csv(resolved_path; delim = '\t', kwargs...)
    elseif ext == ".json"
        return load_json(resolved_path; kwargs...)
    else
        throw(ArgumentError("Unsupported file extension '$ext'."))
    end
end

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