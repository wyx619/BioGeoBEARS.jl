using BioGeoBEARS, Test, CSV, DataFrames

@testset "Stratified fossil models match R" begin
    data = joinpath(@__DIR__, "data")
    base = joinpath(@__DIR__, "baselines")
    tips = read_lagrange_phylip(joinpath(data, "bsm_small.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    strat = Stratification(timesfn = joinpath(data, "bsm_small_times.txt"),
                           areas_allowed_fn = joinpath(data, "bsm_small_areas.txt"))
    for (spec, tag) in ((:DIVALIKE, "DIVALIKE"), (:DIVALIKEJ, "DIVALIKEJJ"),
                        (:BAYAREALIKE, "BAYAREALIKE"), (:BAYAREALIKEJ, "BAYAREALIKEJJ")),
        tree in ("fossil_below", "fossil_above")
        @testset "$spec $tree" begin
            tr = read_tree(joinpath(data, "bsm_$(tree).newick"))
            tc = tip_condlikes(tr, tips; max_range_size = 3)
            model = Model(spec); model["d"] = 0.8; model["e"] = 0.3
            spec in (:DIVALIKEJ, :BAYAREALIKEJ) && (model["j"] = 0.1)
            prefix = joinpath(base, "r_model_$(tree)_areas_$(tag)_20")
            lnL = parse(Float64, readlines(prefix * "_lnL.txt")[2])
            @test stratified_loglikelihood(tr, ss, tc, model, strat; r_float32 = true) ≈ lnL atol = 1e-6
            ap = stratified_ancestral_probs(tr, ss, tc, model, strat; r_float32 = true)
            marg = reduce(vcat, [permutedims(parse.(Float64, split(line))) for line in readlines(prefix * "_marg.txt")[2:end]])
            @test ap.branch_top ≈ marg atol = 1e-6
            map, rng = stochastic_map_stratified(tr, ss, model, strat, tc; seedval = 20, return_rng = true)
            @test r_runif(rng, 16) == parse.(Float64, readlines(prefix * "_rng.txt")[2:end])
            refs = CSV.read(prefix * "_map.tsv", DataFrame; delim = '\t', missingstring = ["", "NA"])
            for n in 1:nnodes(tr)
                rows = refs[refs.node .== n, :]
                ref = is_tip(tr, n) ? rows[(rows.stratum .== 1) .& (rows.piececlass .== "subbranch"), :] :
                                      rows[.!ismissing.(rows.clado_event_txt), :]
                @test nrow(ref) == 1
                @test map.node_states[n] == ref.sampled_states_AT_nodes[1]
                is_tip(tr, n) || @test map.clado_event_txt[n] == ref.clado_event_txt[1]
            end
            events = CSV.read(prefix * "_ana.tsv", DataFrame; delim = '\t', missingstring = ["", "NA"])
            expected = sort([(Int(r.node), Float64(r.abs_event_time), String(r.event_type), String(r.event_txt)) for r in eachrow(events) if !ismissing(r.abs_event_time)]; by = x -> (x[1], x[2]))
            actual = sort([(n, e.abs_event_time, e.event_type, e.event_txt) for n in 1:nnodes(tr) for e in map.branch_events[n]]; by = x -> (x[1], x[2]))
            @test [(x[1], x[3], x[4]) for x in actual] == [(x[1], x[3], x[4]) for x in expected]
            @test Float64[x[2] for x in actual] ≈ Float64[x[2] for x in expected] atol = 1e-6
        end
    end
end
