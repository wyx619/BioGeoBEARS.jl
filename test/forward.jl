using BioGeoBEARS, Test, CSV, DataFrames, Random

@testset "Forward simulation on a fixed tree" begin
    ss = StateSpace(["A", "B"]; max_range_size = 1)
    q = [0.0 0 0; 0 -0.4 0.4; 0 0.2 -0.2]
    w = CladogenesisWeights([2, 2, 3, 3], [2, 2, 3, 3], [2, 3, 2, 3], ones(4))
    ref = CSV.read(joinpath(@__DIR__, "baselines", "forward_r.tsv"), DataFrame; delim = '\t')
    rngref = CSV.read(joinpath(@__DIR__, "baselines", "forward_rng_r.tsv"), DataFrame; delim = '\t')
    for tree in ("balanced", "unbalanced"), seed in (1, 7, 12345)
        tr = read_tree(joinpath(@__DIR__, "data", "bsm_$(tree).newick"))
        rng = RRng(seed)
        actual = simulate_biogeog_history(tr, ss, q, w, 2; rng = rng)
        expected = ref[(ref.tree .== tree) .& (ref.seed .== seed), :]
        @test actual.node_states == expected.state
        @test r_runif(rng, 8) == rngref[(rngref.tree .== tree) .& (rngref.seed .== seed), :value]
        @test all(actual.node_states .> 1)
        @test all(actual.left_corner[n] > 1 && actual.right_corner[n] > 1 for n in ntips(tr)+1:nnodes(tr))
    end
    tr = read_tree("((a:1,b:1):1,c:2);")
    @test_throws ArgumentError simulate_biogeog_history(tr, ss, q, w, 1)
    @test_throws DimensionMismatch simulate_biogeog_history(tr, ss, q[1:2, 1:2], w, 2)
    @test simulate_biogeog_history(tr, ss, Model(:DEC), 2; rng = RRng(1)).node_states[root_node(tr)] == 2
end

@testset "Forward simulation using six BioGeoBEARS models" begin
    data = joinpath(@__DIR__, "data")
    ss = StateSpace(["A", "B", "C"]; max_range_size = 3)
    ref = CSV.read(joinpath(@__DIR__, "baselines", "forward_models_r.tsv"), DataFrame; delim = '\t')
    for (spec, name) in ((:DEC, "DEC"), (:DECJ, "DEC+J"), (:DIVALIKE, "DIVALIKE"),
                         (:DIVALIKEJ, "DIVALIKE+J"), (:BAYAREALIKE, "BAYAREALIKE"),
                         (:BAYAREALIKEJ, "BAYAREALIKE+J")),
        tree in ("balanced", "unbalanced"), seed in (1, 7, 12345)
        tr = read_tree(joinpath(data, "bsm_$(tree).newick"))
        model = Model(spec); model["d"] = 0.3; model["e"] = 0.1
        spec in (:DECJ, :DIVALIKEJ, :BAYAREALIKEJ) && (model["j"] = 0.1)
        original_y = model["y"]
        actual = simulate_biogeog_history(tr, ss, model, 2; rng = RRng(seed))
        rows = ref[(ref.model .== name) .& (ref.tree .== tree) .& (ref.seed .== seed), :]
        @testset "$name $tree $seed" begin
            @test actual.node_states == rows.state
            @test model["y"] == original_y
        end
    end
end
