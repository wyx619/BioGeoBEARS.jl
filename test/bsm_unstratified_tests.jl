    @testset "M7 BSM matches R" begin
        # One stochastic map on the 3-taxon example, compared against
        # `stochastic_map_given_inputs()` from R.  Every sampled state, both
        # cladogenesis labels and each anagenetic event (including its time)
        # must match; that is only possible when Q, the cladogenesis table and
        # the RNG stream are all bit-identical to R's.
        tr = read_tree(joinpath(DATA, "bsm_3taxa.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "bsm_3taxa.data"))
        ss = StateSpace(tips.areanames; max_range_size = 3)
        tipcl = tip_condlikes(tr, tips; max_range_size = 3)

        params = Dict("de005_002" => (0.05, 0.02), "de03_01" => (0.3, 0.1))

        # R stores Q as C `float`, and its diagonal is the negated float32 of a
        # double row sum.  Reproduce that with `r_float32 = true`.
        for line in readlines(joinpath(BASELINES, "bsm_3taxa_qdiag.tsv"))
            parts = split(line, '\t')
            d, e = params[parts[1]]
            m = Model(:DEC); m["d"] = d; m["e"] = e
            qd = diag(Matrix(qmatrix(ss, m; r_float32 = true)))
            @test qd == [parse(Float64, x) for x in parts[2:end]]
        end

        one_map = function (d, e, seed)
            m = Model(:DEC); m["d"] = d; m["e"] = e
            q = qmatrix(ss, m; r_float32 = true)
            w = cladogenesis_weights(ss, m; r_float32 = true)
            dp = downpass(tr, ss, tipcl, q, w; r_float32 = true)
            ap = ancestral_probs(tr, ss, tipcl, q, w; r_float32 = true)
            stochastic_map(tr, ss, q, w, dp, ap; seedval = seed,
                           include_null_range = true, r_float32 = true)
        end

        str(x) = ismissing(x) ? "" : String(x)
        bb(x) = x == -1 ? 0 : x        # R writes NA for the root's branch bottom

        tt = CSV.read(joinpath(BASELINES, "bsm_3taxa_trtable.tsv"), DataFrame;
                      delim = '\t')
        ana = CSV.read(joinpath(BASELINES, "bsm_3taxa_ana.tsv"), DataFrame;
                       delim = '\t')

        for (tag, (d, e)) in params, seed in (12345, 1, 2)
            b = one_map(d, e, seed)

            for row in eachrow(tt[(tt.case .== tag) .& (tt.seed .== seed), :])
                n = Int(row.node)
                @test b.node_states[n] == row.node_state
                @test b.branch_bottom_states[n] == bb(row.brbot_state)
                @test b.left_corner[n] == bb(row.left_corner)
                @test b.right_corner[n] == bb(row.right_corner)
                @test b.clado_event_type[n] == str(row.clado_type)
                @test b.clado_event_txt[n] == str(row.clado_txt)
            end

            got = reduce(vcat, [b.branch_events[n] for n in 1:nnodes(tr)],
                         init = AnaEvent[])
            rows = ana[(ana.case .== tag) .& (ana.seed .== seed), :]
            @test length(got) == nrow(rows)
            for (g, row) in zip(got, eachrow(rows))
                @test g.from_range == row.from_range
                @test g.to_range == row.to_range
                @test g.event_time ≈ row.event_time rtol = 1e-14
                @test g.abs_event_time ≈ row.abs_event_time rtol = 1e-14
                @test g.event_type == row.event_type
                @test g.dispersal_to == row.dispersal_to
            end
        end

        # same seed, same map; different seed, different map
        @test one_map(0.05, 0.02, 12345).node_states ==
              one_map(0.05, 0.02, 12345).node_states
        @test one_map(0.05, 0.02, 12345).node_states !=
              one_map(0.05, 0.02, 7).node_states
    end

    @testset "M7 small tree maps match R" begin
        table = CSV.read(joinpath(BASELINES, "bsm_small_trtable.tsv"), DataFrame; delim = '\t')
        events = CSV.read(joinpath(BASELINES, "bsm_small_ana.tsv"), DataFrame; delim = '\t', types = Dict(:desc => String))
        cases = (("balanced_dec", "balanced", 0.1, 0.05, 0.0),
                 ("unbalanced_dec", "unbalanced", 0.3, 0.1, 0.0),
                 ("short_dec", "short", 0.1, 0.05, 0.0),
                 ("balanced_decj", "balanced", 0.1, 0.05, 0.1),
                 ("unbalanced_decj", "unbalanced", 0.3, 0.1, 0.1))
        for (tag, tree, d, e, j) in cases
            @testset "$tag" begin
                tr = read_tree(joinpath(DATA, "bsm_$(tree).newick"))
                tips = read_lagrange_phylip(joinpath(DATA, "bsm_small.data"))
                ss = StateSpace(tips.areanames; max_range_size = 3)
                tipcl = tip_condlikes(tr, tips; max_range_size = 3)
                m = Model(j == 0 ? :DEC : :DECJ); m["d"] = d; m["e"] = e; m["j"] = j
                q = qmatrix(ss, m; r_float32 = true)
                w = cladogenesis_weights(ss, m; r_float32 = true)
                dp = downpass(tr, ss, tipcl, q, w; r_float32 = true)
                ap = ancestral_probs(tr, ss, tipcl, q, w; r_float32 = true)
                descendants(n) = is_tip(tr, n) ? [n] : vcat(descendants(left_child(tr, n)), descendants(right_child(tr, n)))
                sig(n) = join(sort(descendants(n)), "-")
                nodes = Dict(sig(n) => n for n in 1:nnodes(tr))
                for seed in (1, 7, 12345)
                    mp = stochastic_map(tr, ss, q, w, dp, ap; seedval = seed, r_float32 = true, maxtries = 100000)
                    ref = table[(table.case .== tag) .& (table.seed .== seed), :]
                    for row in eachrow(ref)
                        n = nodes[string(row.desc)]
                        @test mp.node_states[n] == row.node_state
                        @test mp.branch_bottom_states[n] == max(0, row.brbot_state)
                        @test mp.left_corner[n] == max(0, row.left_corner)
                        @test mp.right_corner[n] == max(0, row.right_corner)
                        @test mp.clado_event_txt[n] == (ismissing(row.clado_txt) ? "" : row.clado_txt)
                    end
                    expected = events[(events.case .== tag) .& (events.seed .== seed), :]
                    got = [(sig(n), e.from_range, e.to_range, e.event_time, e.abs_event_time, e.event_type) for n in 1:nnodes(tr) for e in mp.branch_events[n]]
                    want = [(row.desc, row.from_range, row.to_range, row.event_time, row.abs_event_time, row.event_type) for row in eachrow(expected)]
                    sort!(got); sort!(want)
                    @test length(got) == length(want)
                    if length(got) == length(want)
                        @test [(x[1], x[2], x[3], x[6]) for x in got] == [(x[1], x[2], x[3], x[6]) for x in want]
                        @test Float64[x[4] for x in got] ≈ Float64[x[4] for x in want] atol = 1e-7
                        @test Float64[x[5] for x in got] ≈ Float64[x[5] for x in want] atol = 1e-7
                    end
                end
            end
        end
    end

