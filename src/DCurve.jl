module DCurve

using LinearAlgebra
using Random

include("constructors.jl")
include("indexing.jl")
include("generators.jl")
include("plotting.jl")
include("io.jl")

Base.eltype(::DiscreteCurve{T}) where {T} = T

export DiscreteCurve,
       line_curve,
       helix_curve,
       circle_curve,
       ellipse_curve,
       polygon_curve,
       hypercube_curve,
       star_curve,
       cardioid_curve,
       trefoil_curve,
       torus_knot_curve,
       lissajous_curve,
       spiral_curve,
       iterated_curve,
       random_curve,
       ndim,
       npoints,
       isclosed,
       save_curve, load_curve,
       save_dcrv, load_dcrv,
       save_curves, load_curves,
       save_obj, load_obj,
       save_ply, load_ply,
       save_pcd, load_pcd,
       save_csv, load_csv,
       save_json, load_json
end
