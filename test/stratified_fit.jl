using BioGeoBEARS, Test, CSV, DataFrames

@testset "Stratified fit and mapping match R" begin
    data = joinpath(@__DIR__, "data")
    base = joinpath(@__DIR__, "baselines")
    tr = read_tree(joinpath(data, "bsm_balanced.newick"))
    tips = read_lagrange_phylip(joinpath(data, "bsm_small.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    tc = tip_condlikes(tr, tips; max_range_size = 3)
    strat = Stratification(timesfn = joinpath(data, "bsm_small_times.txt"),
                           areas_allowed_fn = joinpath(data, "bsm_small_areas.txt"))
    ref = only(eachrow(CSV.read(joinpath(base, "stratified_fit_DEC.tsv"), DataFrame; delim = '\t')))
    fit = fit_model(tr, ss, tc, Model(:DEC), strat; maxeval = 1000, calc_ancprobs = true)
    @test fit.loglik ≈ ref.lnL atol = 1e-5
    @test fit.model["d"] ≈ ref.d atol = 1e-4
    @test fit.model["e"] ≈ ref.e atol = 1e-4
    # Compare marginal probabilities at the exact R-fitted parameters so the
    # optimizers' stopping tolerances do not dominate the comparison.
    model = Model(:DEC); model["d"] = ref.d; model["e"] = ref.e
    ap = stratified_ancestral_probs(tr, ss, tc, model, strat; r_float32 = true)
    marg = reduce(vcat, [permutedims(parse.(Float64, split(line))) for line in
                        readlines(joinpath(base, "stratified_fit_DEC_marg.txt"))[2:end]])
    @test ap.branch_top ≈ marg atol = 1e-6
    maps, rng = run_bsm(tr, ss, model, strat, tc; seedval = 0, n_maps = 1, return_rng = true)
    map = only(maps)
    @test r_runif(rng, 16) == parse.(Float64, readlines(joinpath(base, "stratified_fit_rng.txt"))[2:end])
    _, batch_rng = run_bsm(tr, ss, model, strat, tc; seedval = 0, n_maps = 2, return_rng = true)
    clado = [CladoEventRow(i, t, "founder (j)", "AB->AB,C", "C", "") for (i, t) in enumerate((0.5, 1.0, 2.0))]
    ana = [AnaEventRow(i, 1, 0.0, t, "d", "AB", "ABC", "C", "", "-") for (i, t) in enumerate((0.5, 1.0, 2.0))]
    simulate_source_areas!([BSMTables(clado, ana)], ss, batch_rng)
    @test vcat([r.dispersal_from for r in clado], [r.dispersal_from for r in ana]) ==
          readlines(joinpath(base, "stratified_batch_sources.txt"))[2:end]
    refs = CSV.read(joinpath(base, "stratified_fit_DEC_map.tsv"), DataFrame; delim = '\t', missingstring = ["", "NA"])
    for n in 1:nnodes(tr)
        rows = refs[refs.node .== n, :]
        r = is_tip(tr, n) ? rows[(rows.stratum .== 1) .& (rows.piececlass .== "subbranch"), :] :
                            rows[.!ismissing.(rows.clado_event_txt), :]
        @test nrow(r) == 1
        @test map.node_states[n] == r.sampled_states_AT_nodes[1]
        is_tip(tr, n) || @test map.clado_event_txt[n] == r.clado_event_txt[1]
    end
    events = CSV.read(joinpath(base, "stratified_fit_DEC_ana.tsv"), DataFrame; delim = '\t', missingstring = ["", "NA"])
    expected = sort([(Int(r.node), Float64(r.abs_event_time), String(r.event_txt)) for r in eachrow(events) if !ismissing(r.abs_event_time)]; by = x -> (x[1], x[2]))
    actual = sort([(n, e.abs_event_time, e.event_txt) for n in 1:nnodes(tr) for e in map.branch_events[n]]; by = x -> (x[1], x[2]))
    @test [(x[1], x[3]) for x in actual] == [(x[1], x[3]) for x in expected]
    @test Float64[x[2] for x in actual] ≈ Float64[x[2] for x in expected] atol = 1e-6
end

@testset "Fitted stratified batch mapping matches R" begin
    data = joinpath(@__DIR__, "data")
    base = joinpath(@__DIR__, "baselines")
    tr = read_tree(joinpath(data, "bsm_balanced.newick"))
    tips = read_lagrange_phylip(joinpath(data, "bsm_small.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    tc = tip_condlikes(tr, tips; max_range_size = 3)
    strat = Stratification(timesfn = joinpath(data, "bsm_small_times.txt"),
                           areas_allowed_fn = joinpath(data, "bsm_small_areas.txt"))
    for (spec, tag) in ((:DEC, "DEC"), (:DECJ, "DECJJ"), (:DIVALIKE, "DIVALIKE"),
                        (:DIVALIKEJ, "DIVALIKEJJ"), (:BAYAREALIKE, "BAYAREALIKE"),
                        (:BAYAREALIKEJ, "BAYAREALIKEJJ"))
        @testset "$spec" begin
            ref = only(eachrow(CSV.read(joinpath(base, "stratified_fit_$(tag).tsv"), DataFrame; delim = '\t')))
            model = Model(spec)
            model["d"] = ref.d; model["e"] = ref.e
            spec in (:DECJ, :DIVALIKEJ, :BAYAREALIKEJ) && (model["j"] = ref.j)
            maps, rng = run_bsm(tr, ss, model, strat, tc; seedval = 0, n_maps = 2, return_rng = true)
            prefix = joinpath(base, "stratified_batch_$(tag)")
            @test r_runif(deepcopy(rng), 16) == parse.(Float64, readlines(prefix * "_rng.txt")[2:end])
            for (i, map) in enumerate(maps)
                refs = CSV.read(prefix * "_map$(i).tsv", DataFrame; delim = '\t', missingstring = ["", "NA"])
                for n in 1:nnodes(tr)
                    rows = refs[refs.node .== n, :]
                    r = is_tip(tr, n) ? rows[(rows.stratum .== 1) .& (rows.piececlass .== "subbranch"), :] :
                                        rows[.!ismissing.(rows.clado_event_txt), :]
                    @test nrow(r) == 1
                    @test map.node_states[n] == r.sampled_states_AT_nodes[1]
                    is_tip(tr, n) || @test map.clado_event_txt[n] == r.clado_event_txt[1]
                end
                events = CSV.read(prefix * "_ana$(i).tsv", DataFrame; delim = '\t')
                expected = sort([(Int(r.node), Float64(r.abs_event_time), String(r.event_txt)) for r in eachrow(events)]; by = x -> (x[1], x[2]))
                actual = sort([(n, e.abs_event_time, e.event_txt) for n in 1:nnodes(tr) for e in map.branch_events[n]]; by = x -> (x[1], x[2]))
                @test [(x[1], x[3]) for x in actual] == [(x[1], x[3]) for x in expected]
                @test Float64[x[2] for x in actual] ≈ Float64[x[2] for x in expected] atol = 1e-6
            end
            tables = [bsm_tables(map, tr) for map in maps]
            simulate_source_areas!(tables, ss, rng)
            summary = Matrix(CSV.read(prefix * "_summary.tsv", DataFrame; delim = '\t'))
            @test count_ana_clado_events(tables, ss).summary_counts_BSMs ≈ summary atol = 1e-12
            if spec == :DEC
                window = Matrix(CSV.read(prefix * "_window.tsv", DataFrame; delim = '\t'))
                @test count_ana_clado_events(tables, ss; timeperiod = (1.0, 2.0)).summary_counts_BSMs ≈ window atol = 1e-12
            end
        end
    end
end

@testset "Additional stratified model fits match R" begin
    data = joinpath(@__DIR__, "data")
    base = joinpath(@__DIR__, "baselines")
    tr = read_tree(joinpath(data, "bsm_balanced.newick"))
    tips = read_lagrange_phylip(joinpath(data, "bsm_small.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    tc = tip_condlikes(tr, tips; max_range_size = 3)
    strat = Stratification(timesfn = joinpath(data, "bsm_small_times.txt"),
                           areas_allowed_fn = joinpath(data, "bsm_small_areas.txt"))
    for (spec, tag) in ((:DECJ, "DECJJ"), (:DIVALIKE, "DIVALIKE"), (:BAYAREALIKE, "BAYAREALIKE"),
                        (:DIVALIKEJ, "DIVALIKEJJ"), (:BAYAREALIKEJ, "BAYAREALIKEJJ"))
        @testset "$spec" begin
            ref = only(eachrow(CSV.read(joinpath(base, "stratified_fit_$(tag).tsv"), DataFrame; delim = '\t')))
            fit = fit_model(tr, ss, tc, Model(spec), strat; maxeval = 1000)
            @test fit.loglik ≈ ref.lnL atol = 1e-5
            @test fit.model["d"] ≈ ref.d atol = 5e-4
            @test fit.model["e"] ≈ ref.e atol = 5e-4
            if spec in (:DECJ, :DIVALIKEJ, :BAYAREALIKEJ)
                @test fit.model["j"] ≈ ref.j atol = (spec == :DIVALIKEJ ? 1e-3 : 5e-4)
            end
            model = Model(spec)
            model["d"] = ref.d; model["e"] = ref.e; model["j"] = ref.j
            @test stratified_loglikelihood(tr, ss, tc, model, strat) ≈ ref.lnL atol = 1e-6
            ap = stratified_ancestral_probs(tr, ss, tc, model, strat; r_float32 = true)
            prefix = "stratified_fit_$(tag)"
            marg = reduce(vcat, [permutedims(parse.(Float64, split(line))) for line in
                                readlines(joinpath(base, prefix * "_marg.txt"))[2:end]])
            @test ap.branch_top ≈ marg atol = 1e-6
            map = only(run_bsm(tr, ss, model, strat, tc; seedval = 0, n_maps = 1))
            refs = CSV.read(joinpath(base, prefix * "_map.tsv"), DataFrame; delim = '\t', missingstring = ["", "NA"])
            for n in 1:nnodes(tr)
                rows = refs[refs.node .== n, :]
                r = is_tip(tr, n) ? rows[(rows.stratum .== 1) .& (rows.piececlass .== "subbranch"), :] :
                                    rows[.!ismissing.(rows.clado_event_txt), :]
                @test nrow(r) == 1
                @test map.node_states[n] == r.sampled_states_AT_nodes[1]
                is_tip(tr, n) || @test map.clado_event_txt[n] == r.clado_event_txt[1]
            end
            events = CSV.read(joinpath(base, prefix * "_ana.tsv"), DataFrame; delim = '\t', missingstring = ["", "NA"])
            expected = sort([(Int(r.node), Float64(r.abs_event_time), String(r.event_txt)) for r in eachrow(events) if !ismissing(r.abs_event_time)]; by = x -> (x[1], x[2]))
            actual = sort([(n, e.abs_event_time, e.event_txt) for n in 1:nnodes(tr) for e in map.branch_events[n]]; by = x -> (x[1], x[2]))
            @test [(x[1], x[3]) for x in actual] == [(x[1], x[3]) for x in expected]
            @test Float64[x[2] for x in actual] ≈ Float64[x[2] for x in expected] atol = 1e-6
        end
    end
end
