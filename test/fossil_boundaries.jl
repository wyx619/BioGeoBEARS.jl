using BioGeoBEARS, Test

@testset "Stratified fossil boundaries and histories" begin
    data = joinpath(@__DIR__, "data")
    tips = read_lagrange_phylip(joinpath(data, "bsm_small.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    strat = Stratification(timesfn = joinpath(data, "bsm_small_times.txt"),
        areas_allowed_fn = joinpath(data, "bsm_small_areas.txt"))
    m = Model(:DECJ); m["d"] = 0.8; m["e"] = 0.3; m["j"] = 0.1

    for (tree, fossil_age) in (("fossil_below", 0.4999), ("fossil_above", 0.5001))
        @testset "$tree" begin
            tr = read_tree(joinpath(data, "bsm_$(tree).newick"))
            tc = tip_condlikes(tr, tips; max_range_size = 3)
            @test node_times_bp(tr)[1] ≈ fossil_age atol = 1e-12
            map, rng = stochastic_map_stratified(tr, ss, m, strat, tc; seedval = 20, return_rng = true)
            reference_rng = joinpath(@__DIR__, "baselines", "r_model_$(tree)_areas_DECJJ_20_rng.txt")
            @test r_runif(rng, 16) == parse.(Float64, readlines(reference_rng)[2:end])
            events = sort(map.branch_events[1]; by = e -> e.abs_event_time, rev = true)
            @test length(events) == 2
            @test all(e -> fossil_age <= e.abs_event_time <= node_times_bp(tr)[parent_node(tr, 1)], events)
            @test all(events[i].to_state == events[i + 1].from_state for i in 1:length(events)-1)
            parent = parent_node(tr, 1)
            parent_split = split(split(map.clado_event_txt[parent], "->")[2], ',')
            @test events[1].from_range == parent_split[1]
            @test events[end].to_state == map.node_states[1]
        end
    end

    for (age, rejected) in ((0.5, true), (0.500005, true), (0.499995, true),
                             (0.50002, false), (0.49998, false))
        tr = read_tree("(((a:$(1-age),b:1):1,(c:1,d:1):1):1,(e:2,f:2):1);")
        if rejected
            @test_throws ArgumentError BioGeoBEARS._check_stratification(tr, strat)
        else
            @test BioGeoBEARS._check_stratification(tr, strat)[1] ≈ age atol = 1e-12
        end
    end
end
