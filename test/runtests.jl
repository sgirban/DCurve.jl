using Test
using Random
using LinearAlgebra
using DCurve

@testset "DCurve.jl Full Suite" begin

    @testset "1. Core DiscreteCurve Constructors & Metadata (1D–5D)" begin
        for T in (Float32, Float64), D in 1:5
            M = 12
            pts = rand(T, D, M)

            # Direct matrix wrap (zero-copy)
            c_open = DiscreteCurve(pts)
            @test c_open isa DiscreteCurve{T, Matrix{T}}
            @test ndim(c_open) == D
            @test npoints(c_open) == M
            @test eltype(c_open) === T
            @test !isclosed(c_open)
            @test c_open.points === pts

            # SubArray view wrap
            sub_pts = @view pts[:, 2:8]
            c_view = DiscreteCurve(sub_pts; closed = true)
            @test c_view isa DiscreteCurve{T, typeof(sub_pts)}
            @test ndim(c_view) == D
            @test npoints(c_view) == 7
            @test isclosed(c_view)

            # Coordinate vectors
            coords = [collect(range(T(0), T(d), length = M)) for d in 1:D]
            c_coords = DiscreteCurve(coords...; closed = (D % 2 == 0))
            @test ndim(c_coords) == D
            @test npoints(c_coords) == M
            @test eltype(c_coords) === T
            @test isclosed(c_coords) == (D % 2 == 0)

            # Tuple of coordinate vectors
            c_tup_coords = DiscreteCurve(Tuple(coords); closed = true)
            @test c_tup_coords.points ≈ c_coords.points
            @test isclosed(c_tup_coords)

            # Vector of point tuples
            tuple_pts = [ntuple(d -> T(j * d), D) for j in 1:M]
            c_pts = DiscreteCurve(tuple_pts; T = T, closed = true)
            @test ndim(c_pts) == D
            @test npoints(c_pts) == M
            @test eltype(c_pts) === T
        end

        # Parametric function constructors
        for T in (Float32, Float64)
            ts = range(T(0), T(2π), length = 25)
            c1 = DiscreteCurve(t -> sin(t), ts; T = T)
            @test ndim(c1) == 1
            @test npoints(c1) == 25
            @test eltype(c1) === T

            c4 = DiscreteCurve(t -> (cos(t), sin(t), t, t^2), (0.0, 1.0), 16; T = T, closed = true)
            @test ndim(c4) == 4
            @test npoints(c4) == 16
            @test eltype(c4) === T
            @test isclosed(c4)
        end
    end

    @testset "2. Circle & Ellipse Generators" begin
        for T in (Float32, Float64)
            # Full circle defaults to closed = true and no duplicate seam point
            c = circle_curve(32; radius = 2.5, T = T)
            @test ndim(c) == 2
            @test npoints(c) == 32
            @test eltype(c) === T
            @test isclosed(c)
            for j in 1:32
                @test norm(c.points[:, j]) ≈ T(2.5) atol = sqrt(eps(T))
            end

            # Positional radius shorthand
            c_pos = circle_curve(1.5, 20; T = T)
            @test npoints(c_pos) == 20
            @test norm(c_pos.points[:, 1]) ≈ T(1.5)

            # Partial arc defaults to closed = false
            arc_c = circle_curve(16; radius = 3.0, arc = (0.0, π / 2), T = T)
            @test !isclosed(arc_c)
            @test arc_c.points[:, 1] ≈ T[3.0, 0.0] atol = sqrt(eps(T))
            @test arc_c.points[:, end] ≈ T[0.0, 3.0] atol = sqrt(eps(T))

            # Higher-dimensional embedding via center
            c4 = circle_curve(24; radius = 1.0, center = [1.0, 2.0, 3.0, 4.0], T = T)
            @test ndim(c4) == 4
            @test all(c4.points[3, :] .== T(3.0))
            @test all(c4.points[4, :] .== T(4.0))

            # Ellipse semi-axes & rotation
            el = ellipse_curve(40; a = 3.0, b = 1.0, rotation = π / 2, T = T)
            @test ndim(el) == 2
            @test npoints(el) == 40
            @test isclosed(el)
            @test maximum(abs, el.points[2, :]) ≈ T(3.0) atol = sqrt(eps(T))
            @test maximum(abs, el.points[1, :]) ≈ T(1.0) atol = sqrt(eps(T))
        end
    end

    @testset "3. Regular Polygons, Hypercubes & Stars" begin
        for T in (Float32, Float64)
            # Regular hexagon with edge subdivision
            hex = polygon_curve(6; radius = 2.0, points_per_edge = 5, T = T)
            @test ndim(hex) == 2
            @test npoints(hex) == 30
            @test eltype(hex) === T
            @test isclosed(hex)
            for k in 0:5
                @test norm(hex.points[:, 1 + 5k]) ≈ T(2.0) atol = sqrt(eps(T))
            end

            # Hypercubes in 2D (square), 3D (cube), 4D (tesseract), 5D
            for D in 2:5
                hc = hypercube_curve(D; side = 2.0, center = ones(T, D), T = T)
                @test ndim(hc) == D
                @test npoints(hc) == 2^D
                @test eltype(hc) === T
                @test isclosed(hc)
                @test size(unique(hc.points; dims = 2), 2) == 2^D
                for j in 1:(2^D)
                    nxt = mod1(j + 1, 2^D)
                    @test norm(hc.points[:, nxt] - hc.points[:, j]) ≈ T(2.0) atol = sqrt(eps(T))
                end
            end

            # 5-pointed star
            st = star_curve(5; outer_radius = 2.0, inner_radius = 0.5, T = T)
            @test ndim(st) == 2
            @test npoints(st) == 10
            @test isclosed(st)
            @test norm(st.points[:, 1]) ≈ T(2.0) atol = sqrt(eps(T))
            @test norm(st.points[:, 2]) ≈ T(0.5) atol = sqrt(eps(T))
        end
    end

    @testset "4. Analytic Curves: Cardioid, Torus Knot, Trefoil, Lissajous & Spiral" begin
        for T in (Float32, Float64)
            # Cardioid: at θ = 0, r = 2a; at θ = π, r = 0 (cusp)
            card = cardioid_curve(100; a = 1.5, T = T)
            @test ndim(card) == 2
            @test npoints(card) == 100
            @test isclosed(card)
            @test card.points[:, 1] ≈ T[3.0, 0.0] atol = sqrt(eps(T))
            @test card.points[:, 51] ≈ T[0.0, 0.0] atol = sqrt(eps(T))

            # Torus knot (p=3, q=5) in 4D
            tk = torus_knot_curve(3, 5, 120; R_major = 2.0, r_minor = 0.5, dim = 4, T = T)
            @test ndim(tk) == 4
            @test npoints(tk) == 120
            @test eltype(tk) === T
            @test isclosed(tk)
            @test all(tk.points[4, :] .== zero(T))

            # Trefoil knot
            tr = trefoil_curve(90; scale = 1.0, T = T)
            @test ndim(tr) == 3
            @test npoints(tr) == 90
            @test isclosed(tr)

            # Lissajous figure
            liss = lissajous_curve(80; a = 2.0, b = 3.0, fx = 3, fy = 2, T = T)
            @test ndim(liss) == 2
            @test npoints(liss) == 80
            @test isclosed(liss)
            @test maximum(abs, liss.points[1, :]) <= T(2.0) + sqrt(eps(T))
            @test maximum(abs, liss.points[2, :]) <= T(3.0) + sqrt(eps(T))

            # Archimedean & logarithmic spirals
            sp_arch = spiral_curve(50; a = 0.5, b = 0.1, turns = 2.0, kind = :archimedean, T = T)
            @test ndim(sp_arch) == 2
            @test npoints(sp_arch) == 50
            @test !isclosed(sp_arch)
            @test norm(sp_arch.points[:, 1]) ≈ T(0.5) atol = sqrt(eps(T))
            @test norm(sp_arch.points[:, end]) ≈ T(0.5 + 0.1 * 4π) atol = sqrt(eps(T))

            sp_log = spiral_curve(40; a = 1.0, b = 0.1, turns = 1.0, kind = :logarithmic, T = T)
            @test norm(sp_log.points[:, 1]) ≈ T(1.0) atol = sqrt(eps(T))
            @test norm(sp_log.points[:, end]) ≈ T(exp(0.1 * 2π)) atol = sqrt(eps(T))
        end
    end

    @testset "5. Iterated / Recurrence Curves (1-step & p-step)" begin
        for T in (Float32, Float64)
            # 1-step linear contraction in 3D: x_{k+1} = 0.5 * x_k
            c1 = iterated_curve(x -> T(0.5) .* x, T[8.0, -4.0, 2.0], 5; T = T)
            @test ndim(c1) == 3
            @test npoints(c1) == 5
            @test eltype(c1) === T
            @test c1.points[:, 1] ≈ T[8.0, -4.0, 2.0]
            @test c1.points[:, 5] ≈ T[0.5, -0.25, 0.125]

            # 2-step Fibonacci-style recurrence in 2D: x_{k+1} = x_k + x_{k-1}
            history = [T[1.0, 0.0], T[1.0, 1.0]]
            c2 = iterated_curve((x_curr, x_prev) -> x_curr .+ x_prev, history, 6; T = T, closed = true)
            @test ndim(c2) == 2
            @test npoints(c2) == 6
            @test isclosed(c2)
            @test c2.points[1, :] ≈ T[1, 1, 2, 3, 5, 8]
            @test c2.points[2, :] ≈ T[0, 1, 1, 2, 3, 5]
        end
    end

    @testset "6. Random Curves & RNG Reproducibility (1D–5D)" begin
        for T in (Float32, Float64), D in 1:5
            # Reproducibility with explicit Xoshiro seed
            c_a = random_curve(50; dim = D, distribution = :normal, step_scale = 0.25, rng = Xoshiro(1234), T = T)
            c_b = random_curve(50; dim = D, distribution = :normal, step_scale = 0.25, rng = Xoshiro(1234), T = T)
            @test ndim(c_a) == D
            @test npoints(c_a) == 50
            @test eltype(c_a) === T
            @test c_a.points == c_b.points
            @test all(iszero, c_a.points[:, 1])

            # Uniform non-cumulative cloud bounded in [-step_scale, step_scale]^D
            c_uni = random_curve(100; dim = D, distribution = :uniform, step_scale = 2.0, cumulative = false, rng = Xoshiro(7), T = T)
            @test all(abs.(c_uni.points) .<= T(2.0))

            # Spherical random walk: every step has Euclidean norm == step_scale
            c_sph = random_curve(30; dim = D, distribution = :sphere, step_scale = 0.5, cumulative = true, rng = Xoshiro(99), T = T)
            for j in 1:29
                step = c_sph.points[:, j + 1] - c_sph.points[:, j]
                @test norm(step) ≈ T(0.5) atol = sqrt(eps(T))
            end

            # Smooth Fourier random curve
            c_sm = random_curve(60; dim = D, distribution = :smooth, harmonics = 4, closed = true, rng = Xoshiro(11), T = T)
            @test ndim(c_sm) == D
            @test npoints(c_sm) == 60
            @test isclosed(c_sm)
        end
    end

    @testset "7. Device Transfer Hook & Input Validation" begin
        to_view = mat -> view(mat, :, :)
        c_dev = torus_knot_curve(2, 3, 24; T = Float32, to_device = to_view)
        @test c_dev.points isa SubArray{Float32, 2}

        @test_throws ArgumentError circle_curve(1)
        @test_throws ArgumentError circle_curve(10; dim = 1)
        @test_throws ArgumentError polygon_curve(2)
        @test_throws ArgumentError polygon_curve(4; points_per_edge = 0)
        @test_throws ArgumentError hypercube_curve(1)
        @test_throws ArgumentError hypercube_curve(21)
        @test_throws ArgumentError star_curve(1)
        @test_throws ArgumentError cardioid_curve(3)
        @test_throws ArgumentError torus_knot_curve(2, 3, 5)
        @test_throws ArgumentError torus_knot_curve(2, 3, 10; dim = 2)
        @test_throws ArgumentError spiral_curve(10; kind = :unknown)
        @test_throws ArgumentError iterated_curve(x -> x, Float64[], 5)
        @test_throws DimensionMismatch iterated_curve(x -> [1.0], [1.0, 2.0], 5)
        @test_throws ArgumentError iterated_curve((x, y) -> x, [[1.0], [2.0]], 1)
        @test_throws ArgumentError random_curve(10; distribution = :cauchy)
        @test_throws ArgumentError random_curve(10; distribution = :smooth, harmonics = 0)
    end
    @testset "8. Smart Point Indexing (Open & Closed, 1D–5D)" begin
        for T in (Float32, Float64), D in 1:5
            M = 6
            pts = reshape(T.(1:(D * M)), D, M)

            # --- Open curve ---
            c_open = DiscreteCurve(pts; closed = false)
            @test length(c_open) == M
            @test firstindex(c_open) == 1
            @test lastindex(c_open) == M
            @test c_open[1] == pts[:, 1]
            @test c_open[end] == pts[:, M]
            @test c_open[end - 1] == pts[:, M - 1]
            @test c_open[2:4] == pts[:, 2:4]
            @test c_open[[1, 3, 6]] == pts[:, [1, 3, 6]]
            @test c_open[(1, 3, 4, 6)] == pts[:, [1, 3, 4, 6]]
            @test c_open[:] == pts
            @test collect(c_open) == [pts[:, j] for j in 1:M]

            # Out-of-bounds on open curve throws BoundsError
            @test_throws BoundsError c_open[0]
            @test_throws BoundsError c_open[-1]
            @test_throws BoundsError c_open[M + 1]
            @test_throws BoundsError c_open[0:3]
            @test_throws BoundsError c_open[2:(M + 1)]
            @test_throws BoundsError c_open[[1, M + 1]]
            @test_throws BoundsError c_open[(0, 1, 2)]

            # --- Closed curve (modular wrap-around) ---
            c_closed = DiscreteCurve(pts; closed = true)
            @test c_closed[0] == pts[:, M]
            @test c_closed[-1] == pts[:, M - 1]
            @test c_closed[M + 1] == pts[:, 1]
            @test c_closed[2M + 2] == pts[:, 2]
            @test c_closed[end - 1] == pts[:, M - 1]

            # Ranges, vectors, and tuples with wrap-around
            @test c_closed[2:5] == pts[:, 2:5]
            @test c_closed[0:(M + 1)] == pts[:, [M, 1, 2, 3, 4, 5, 6, 1]]
            @test c_closed[[0, 1, M + 1]] == pts[:, [M, 1, 1]]
            @test c_closed[(0, 1, 3, M + 1)] == pts[:, [M, 1, 3, 1]]
            @test c_closed[()] == pts[:, Int[]]
        end
    end
end