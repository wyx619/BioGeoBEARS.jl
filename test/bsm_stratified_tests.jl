    @testset "M7 stratified 19-tip lnL matches R (Psychotria_M0strat)" begin
        # The package's own 19-tip stratified example: 5 strata with
        # boundaries 0.5/1.9/3.7/5.1/10.  R's calc_loglike_sp_stratified()
        # returns -38.382387333210 at d = 0.1, e = 0.05 (agreement is 2.6e-8,
        # float32 accumulation).  This case pins the chainsaw semantics: R
        # gives tip-branch pieces the full stratum thickness even when the
        # newick's tips sit slightly above the present, and its TTL selects
        # each node's row from the stratum where the node was internal or the
        # root of a subtree piece.
        d = joinpath(DATA, "psychotria_m0strat")
        tr = read_tree(joinpath(d, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(d, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        strat = Stratification(timesfn = joinpath(d, "timeperiods.txt"),
                               areas_allowed_fn = joinpath(d, "areas_allowed_all1s.txt"))
        m = Model(:DEC); m["d"] = 0.1; m["e"] = 0.05
        @test stratified_loglikelihood(tr, ss, tipcl, m, strat;
                                       r_float32 = true) ≈
              -38.382387333210 atol = 1e-6
    end
    @testset "M7 stratified 19-tip BSM matches R (Psychotria_M0strat)" begin
        # R's stochastic_mapping_on_stratified() at seed 12345, d = 0.1,
        # e = 0.05, maxtries = 40000, on the package's own 19-tip stratified
        # example.  The map consumes one reseeded stream per piece in R's
        # piecenum order (the newick's postorder), samples each subtree-piece
        # root by pushing the cut state FORWARD along the piece's root branch
        # (delta %*% exp(tQ), not exp(tQ) %*% delta), and reproduces R's
        # sampled node states and cladogenesis events on all 37 nodes.
        d = joinpath(DATA, "psychotria_m0strat")
        tr = read_tree(joinpath(d, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(d, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        strat = Stratification(timesfn = joinpath(d, "timeperiods.txt"),
                               areas_allowed_fn = joinpath(d, "areas_allowed_all1s.txt"))
        m = Model(:DEC); m["d"] = 0.1; m["e"] = 0.05
        mp = stochastic_map_stratified(tr, ss, m, strat, tipcl;
                                      seedval = 12345, maxtries = 40000)
        @test mp.node_states ==
              [4, 4, 3, 3, 4, 4, 3, 2, 2, 4, 5, 3, 2, 2, 2, 2, 2, 2, 3,
               6, 2, 2, 12, 9, 9, 9, 4, 3, 4, 2, 2, 6, 3, 15, 2, 6, 6]
        @test mp.clado_event_txt[20] == "KO->K,KO"
        @test mp.clado_event_txt[21] == "K->K,K"
        @test mp.clado_event_txt[22] == "K->K,K"
        @test mp.clado_event_txt[23] == "KOM->OM,K"
        @test mp.clado_event_txt[24] == "OM->OM,O"
        @test mp.clado_event_txt[25] == "OM->OM,M"
        @test mp.clado_event_txt[26] == "OM->M,O"
        @test mp.clado_event_txt[27] == "M->M,M"
        @test mp.clado_event_txt[28] == "O->O,O"
        @test mp.clado_event_txt[29] == "M->M,M"
        @test mp.clado_event_txt[30] == "K->K,K"
        @test mp.clado_event_txt[31] == "K->K,K"
        @test mp.clado_event_txt[32] == "KO->O,K"
        @test mp.clado_event_txt[33] == "O->O,O"
        @test mp.clado_event_txt[34] == "OMH->OM,H"
        @test mp.clado_event_txt[35] == "K->K,K"
        @test mp.clado_event_txt[36] == "KO->KO,O"
        @test mp.clado_event_txt[37] == "KO->K,KO"
        # R's table_w_anagenetic_events for the same map (7 events; event
        # times agree to ~1e-8, the float32-vs-double rate-extraction ulp --
        # 0.1 and 0.35 are not exactly representable in float32, unlike the
        # 3-taxon example's 0.25).
        ana = Tuple{Int,Float64,String}[]
        for n in 1:37, e in mp.branch_events[n]
            push!(ana, (n, e.abs_event_time, e.event_type))
        end
        sort!(ana, by = a -> (a[2], a[1]))
        @test length(ana) == 7
        @test [a[1] for a in ana] == [10, 18, 34, 34, 23, 32, 23]
        @test [a[3] for a in ana] == ["e", "e", "d", "d", "d", "d", "d"]
        @test [a[2] for a in ana] ≈
              [0.644559681043029, 1.02398679224187, 2.28493468679426,
               2.54318785724709, 2.67103227926017, 2.99438724185532,
               3.03492074785353] atol = 1e-7
        # R's ML_marginal_prob_each_state_at_branch_top_AT_node (max diff
        # over all 37 x 16 entries is 1.5e-8, the stratified float32
        # accumulation): spot-check three rows.
        ap = stratified_ancestral_probs(tr, ss, tipcl, m, strat;
                                        r_float32 = true)
        @test ap.branch_top[20, :] ≈
              [0.0, 0.4776101833702634, 0.0049435193444563241,
               0.00027950671709055686, 0.00011726123352729784,
               0.30937979489394363, 0.047955183734048494,
               0.038184372435945697, 0.0018527401631906421,
               0.0010647305160753002, 0.00015136848538740332,
               0.05254948321942688, 0.042775271651912097,
               0.0090368483428153179, 0.00070515277405604128,
               0.013394583117860914] atol = 1e-7
        @test ap.branch_top[23, :] ≈
              [0.0, 0.024089949356886759, 0.0064629792661978257,
               0.0010953275652785407, 9.8264352075966169e-06,
               0.44004550685259525, 0.092956815581022159,
               0.00093715781375754994, 0.0037878800994734134,
               0.00043542321309362596, 8.993950512338227e-05,
               0.40343113700997102, 0.010940797256708349,
               0.0034770570731653891, 0.00061895163203189304,
               0.011621251339487262] atol = 1e-7
        @test ap.branch_top[34, :] ≈
              [0.0, 0.027471605692889849, 0.016154361467321555,
               0.091767782250434346, 0.089836884400794417,
               0.0042474043786311635, 0.061877215557423107,
               0.061709891333883667, 0.03660345671689233,
               0.036399425838984562, 0.39172431813391118,
               0.0068823671626181753, 0.0068594765251192976,
               0.099614872163122967, 0.060056536664173696,
               0.0087944017137996967] atol = 1e-7
    end

    include("fossil_boundaries.jl")

    @testset "M7 stratified small trees match R" begin
        for (tree, areas, seed, spec, tag, d_rate, e_rate) in
            (("balanced", "areas", 1, :DEC, "DEC", 0.1, 0.05), ("balanced", "areas", 7, :DEC, "DEC", 0.1, 0.05),
             ("unbalanced", "allareas", 1, :DEC, "DEC", 0.1, 0.05), ("unbalanced", "allareas", 7, :DEC, "DEC", 0.1, 0.05),
             ("nearboundary", "allareas", 1, :DEC, "DEC", 0.1, 0.05),
             ("balanced", "areas", 1, :DECJ, "DECJJ", 0.1, 0.05),
             ("balanced", "areas", 7, :DIVALIKE, "DIVALIKE", 0.1, 0.05),
             ("balanced", "areas", 7, :BAYAREALIKE, "BAYAREALIKE", 0.1, 0.05),
             ("nearboundary", "allareas", 7, :DEC, "DEC", 0.3, 0.1),
             ("fossil", "areas", 1, :DEC, "DEC", 0.3, 0.1),
             ("fossil", "areas", 7, :DEC, "DEC", 0.3, 0.1),
             ("fossil_old", "areas", 7, :DEC, "DEC", 0.3, 0.1),
             ("fossil_below", "areas", 20, :DECJ, "DECJJ", 0.8, 0.3),
             ("fossil_above", "areas", 20, :DECJ, "DECJJ", 0.8, 0.3),
             ("fossil_old", "areas", 20, :DECJ, "DECJJ", 0.8, 0.3),
             ("two_fossils", "areas", 20, :DECJ, "DECJJ", 0.8, 0.3))
            @testset "$tree $areas $seed $tag" begin
                tr = read_tree(joinpath(DATA, "bsm_$(tree).newick"))
                tips = read_lagrange_phylip(joinpath(DATA, "bsm_small.data"))
                ss = StateSpace(tips.areanames; max_range_size = 3)
                tipcl = tip_condlikes(tr, tips; max_range_size = 3)
                strat = Stratification(timesfn = joinpath(DATA, "bsm_small_times.txt"),
                    areas_allowed_fn = joinpath(DATA, "bsm_small_$(areas).txt"))
                m = Model(spec); m["d"] = d_rate; m["e"] = e_rate
                spec == :DECJ && (m["j"] = 0.1)
                if tree == "fossil"
                    @test stratified_loglikelihood(tr, ss, tipcl, m, strat; r_float32 = true) ≈ -12.454654094198 atol = 1e-6
                elseif tree == "fossil_old" && spec == :DEC
                    @test stratified_loglikelihood(tr, ss, tipcl, m, strat; r_float32 = true) ≈ -12.083081378022 atol = 1e-6
                elseif tree == "fossil_below"
                    @test stratified_loglikelihood(tr, ss, tipcl, m, strat; r_float32 = true) ≈ -16.780878205653 atol = 1e-6
                elseif tree == "fossil_above"
                    @test stratified_loglikelihood(tr, ss, tipcl, m, strat; r_float32 = true) ≈ -16.780617690486 atol = 1e-6
                elseif tree == "two_fossils"
                    @test stratified_loglikelihood(tr, ss, tipcl, m, strat; r_float32 = true) ≈ -15.791973070328 atol = 1e-6
                end
                mp = stochastic_map_stratified(tr, ss, m, strat, tipcl; seedval = seed, maxtries = 40000)
                prefix = "r_model_$(tree)_$(areas)_$(tag)_$(seed)"
                map = CSV.read(joinpath(BASELINES, prefix * "_map.tsv"), DataFrame; delim = '\t', missingstring = ["", "NA"])
                for n in 1:nnodes(tr)
                    rows = map[map.node .== n, :]
                    if is_tip(tr, n)
                        ref = rows[(rows.stratum .== 1) .& (rows.piececlass .== "subbranch"), :]
                        @test nrow(ref) == 1
                        @test mp.node_states[n] == ref.sampled_states_AT_nodes[1]
                    else
                        ref = rows[.!ismissing.(rows.clado_event_txt), :]
                        @test nrow(ref) == 1
                        @test mp.node_states[n] == ref.sampled_states_AT_nodes[1]
                        @test mp.clado_event_txt[n] == ref.clado_event_txt[1]
                    end
                end
                events = CSV.read(joinpath(BASELINES, prefix * "_ana.tsv"), DataFrame; delim = '\t', missingstring = ["", "NA"])
                expected = [(Int(r.node), Float64(r.abs_event_time), String(r.event_type), String(r.event_txt)) for r in eachrow(events) if !ismissing(r.abs_event_time)]
                actual = [(n, e.abs_event_time, e.event_type, e.event_txt) for n in 1:nnodes(tr) for e in mp.branch_events[n]]
                if tree in ("fossil_below", "fossil_above")
                    fossil_events = filter(x -> x[1] == 1, actual)
                    @test length(fossil_events) == 2
                    @test all(x -> x[2] >= node_times_bp(tr)[1], fossil_events)
                end
                sort!(expected, by = x -> (x[1], x[2])); sort!(actual, by = x -> (x[1], x[2]))
                @test length(actual) == length(expected)
                if length(actual) == length(expected)
                    @test [(x[1], x[3], x[4]) for x in actual] == [(x[1], x[3], x[4]) for x in expected]
                    @test Float64[x[2] for x in actual] ≈ Float64[x[2] for x in expected] atol = 1e-6
                end
            end
        end
    end

    @testset "M7c stratified model maps match R" begin
        d = joinpath(DATA, "psychotria_m0strat")
        tr = read_tree(joinpath(d, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(d, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        strat = Stratification(timesfn = joinpath(d, "timeperiods.txt"),
                               areas_allowed_fn = joinpath(d, "areas_allowed_all1s.txt"))
        roots = Dict(2 => [25, 30], 3 => [23, 31, 35, 36], 4 => [21], 5 => [20])
        function rows(path)
            lines = readlines(path)
            header = split(lines[1], '\t')
            return [Dict(header .=> split(line, '\t'; keepempty = true)) for line in lines[2:end]]
        end
        for (spec, tag, ll, seed, d_rate, e_rate) in ((:DIVALIKE, "DIVALIKE", -36.589556511496, 12345, 0.1, 0.05),
                                 (:DIVALIKEJ, "DIVALIKEJJ", -34.658001656250, 12345, 0.1, 0.05),
                                 (:BAYAREALIKE, "BAYAREALIKE", -53.186618720633, 12345, 0.1, 0.05),
                                 (:BAYAREALIKEJ, "BAYAREALIKEJJ", -37.915353730891, 12345, 0.1, 0.05),
                                 (:BAYAREALIKE, "BAYAREALIKE_seed7", -70.581174578755, 7, 0.3, 0.1))
            @testset "$tag" begin
                m = Model(spec); m["d"] = d_rate; m["e"] = e_rate
                endswith(tag, "JJ") && (m["j"] = 0.1)
                @test stratified_loglikelihood(tr, ss, tipcl, m, strat; r_float32 = true) ≈ ll atol = 1e-6
                mp = stochastic_map_stratified(tr, ss, m, strat, tipcl; seedval = seed, maxtries = 40000)
                states = Dict{Int,Int}(); clado = Dict{Int,String}(); from_clado = Set{Int}()
                for r in rows(joinpath(d, "r_model_$(tag)_map.tsv"))
                    n = parse(Int, r["node"]); stratum = parse(Int, r["stratum"])
                    event = r["clado_event_txt"]
                    has_event = event != "NA" && !isempty(event)
                    if n <= 19
                        r["piececlass"] == "subbranch" && stratum == 1 && (states[n] = parse(Int, r["sampled_states_AT_nodes"]))
                    elseif r["piececlass"] == "subtree"
                        if has_event
                            states[n] = parse(Int, r["sampled_states_AT_nodes"]); push!(from_clado, n)
                        elseif !(n in from_clado) && (stratum == 5 || n in roots[stratum] || !haskey(states, n))
                            states[n] = parse(Int, r["sampled_states_AT_nodes"])
                        end
                    end
                    has_event && (clado[n] = event)
                end
                @test mp.node_states == [states[n] for n in 1:37]
                @test [mp.clado_event_txt[n] for n in 20:37] == [clado[n] for n in 20:37]
                expected = [(parse(Int, r["node"]), parse(Float64, r["abs_event_time"]), r["event_type"], r["event_txt"]) for r in rows(joinpath(d, "r_model_$(tag)_ana.tsv")) if !isempty(r["abs_event_time"])]
                actual = [(n, e.abs_event_time, e.event_type, e.event_txt) for n in 1:37 for e in mp.branch_events[n]]
                sort!(expected, by = x -> (x[2], x[1])); sort!(actual, by = x -> (x[2], x[1]))
                same_events = length(actual) == length(expected) &&
                    [(x[1], x[3], x[4]) for x in actual] == [(x[1], x[3], x[4]) for x in expected] &&
                    isapprox([x[2] for x in actual], [x[2] for x in expected]; atol = 1e-6)
                @test same_events
                marginal = reduce(vcat, [permutedims(parse.(Float64, split(line))) for line in readlines(joinpath(d, "r_model_$(tag)_marg.txt"))[2:end]])
                ap = stratified_ancestral_probs(tr, ss, tipcl, m, strat; r_float32 = true)
                @test ap.branch_top ≈ marginal atol = 1e-7
            end
        end
    end

