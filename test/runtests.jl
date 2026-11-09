using BioGeoBEARS
include("multiple_trees.jl")
include("forward.jl")
include("global_fit.jl")
include("global_real.jl")
include("event_timeperiod.jl")
using Test
using LinearAlgebra
using SparseArrays
using CSV, DataFrames
using StableRNGs
using Statistics

const TESTDIR = @__DIR__
const DATA = joinpath(TESTDIR, "data")
const BASELINES = joinpath(TESTDIR, "baselines")

"Read a `prt()` baseline written by `temp/make_tree_baseline.R`."
function load_prt_baseline(fn)
    rows = [split(line, ',') for line in readlines(fn)[2:end]]
    return (
        node = [parse(Int, r[1]) for r in rows],
        ancestor = [r[2] == "NA" ? 0 : parse(Int, r[2]) for r in rows],
        edge_length = [r[3] == "NA" ? 0.0 : parse(Float64, r[3]) for r in rows],
        node_ht = [parse(Float64, r[4]) for r in rows],
        time_bp = [parse(Float64, r[5]) for r in rows],
        node_type = [r[6] for r in rows],
    )
end

maxabs(a, b) = maximum(abs.(a .- b))

"Copy a tree with a different root stem (Newick cannot add one after parsing)."
with_root_edge(tr, len) = BGTree(tr.tip_labels, tr.node_labels, tr.parent, tr.left,
                                 tr.right, tr.edge_length, tr.postorder, tr.ntip,
                                 tr.nnode, Float64(len))

@testset "BioGeoBEARS.jl" begin
    include("workflow.jl")
    include("source_strata.jl")
    include("stratified_fit.jl")
    @testset "M0 package skeleton" begin
        @test BioGeoBEARS isa Module

        src = joinpath(pkgdir(BioGeoBEARS), "src")
        files = [
            "core/areas.jl",
            "core/tree.jl",
            "core/model.jl",
            "core/qmatrix.jl",
            "core/cladogenesis.jl",
            "core/fossils.jl",
            "likelihood/expm.jl",
            "likelihood/downpass.jl",
            "likelihood/uppass.jl",
            "likelihood/stratified.jl",
            "likelihood/detection.jl",
            "inference/optimize.jl",
            "inference/results.jl",
            "inference/model_selection.jl",
            "io/newick.jl",
            "io/phylip.jl",
            "io/config.jl",
            "simulate/bsm.jl",
            "simulate/events.jl",
            "plot/recipes.jl",
        ]
        for f in files
            @test isfile(joinpath(src, f))
        end
    end

    @testset "M0 BLAS backend is MKL" begin
        libs = BLAS.get_config().loaded_libs
        @test !isempty(libs)
        @test any(l -> occursin("mkl", lowercase(basename(l.libname))), libs)
    end

    trfn = joinpath(DATA, "Psychotria_5.2.newick")

    @testset "M1 read_tree" begin
        tr = read_tree(trfn)
        @test tr isa BGTree
        @test ntips(tr) == 19
        @test nnodes(tr) == 37
        @test root_node(tr) == 20
        @test is_tip(tr, 1) && is_tip(tr, 19) && !is_tip(tr, 20)
        @test is_root(tr, 20) && !is_root(tr, 21)
        @test parent_node(tr, 20) == 0
        @test left_child(tr, 20) == 21
        @test right_child(tr, 20) == 36
        @test tr.tip_labels[1] == "P_hawaiiensis_WaikamoiL1"
        @test check_tree(tr) === tr

        # a path and a Newick string describe the same tree
        tr2 = read_tree(read(trfn, String))
        @test tr2.parent == tr.parent
        @test tr2.left == tr.left
        @test tr2.right == tr.right
        @test tr2.edge_length == tr.edge_length

        # IO input
        tr3 = open(read_tree, trfn)
        @test tr3.parent == tr.parent

        # the optional stem after the closing parenthesis is kept
        @test root_edge(tr) == 0.0
        @test root_edge(read_tree("((A:1,B:1):1,C:2):0.5;")) == 0.5
        @test root_edge(read_tree("((A:1,B:1):1,C:2);")) == 0.0
        @test root_edge(read_tree("((A:1,B:1):1,C:2):1e-3;\n")) == 1e-3
    end

    @testset "M1 node numbering matches ape::read.tree()" begin
        tr = read_tree(trfn)
        ref = load_prt_baseline(joinpath(BASELINES, "psychotria_prt.csv"))

        @test nnodes(tr) == length(ref.node)
        @test collect(1:nnodes(tr)) == ref.node
        @test tr.parent == ref.ancestor
        @test tr.tip_labels == readlines(joinpath(BASELINES, "psychotria_tip_labels.txt"))
    end

    @testset "M1 prt_table matches R prt()" begin
        tr = read_tree(trfn)
        ref = load_prt_baseline(joinpath(BASELINES, "psychotria_prt.csv"))
        got = prt_table(tr)

        @test got.ancestor == ref.ancestor
        @test got.node_type == ref.node_type
        @test maxabs(got.edge_length, ref.edge_length) < 1e-9
        @test maxabs(got.node_ht, ref.node_ht) < 1e-9
        @test maxabs(got.time_bp, ref.time_bp) < 1e-9
        @test got.time_bp[1] < 1e-8                 # tips are at the present
        @test isapprox(got.time_bp[root_node(tr)], tree_height(tr))
    end

    @testset "M1 pruning order" begin
        tr = read_tree(trfn)
        order = pruning_order(tr)
        @test length(order) == nnodes(tr) - 1
        @test length(unique(order)) == length(order)
        @test !(root_node(tr) in order)

        position = zeros(Int, nnodes(tr))
        position[order] .= 1:length(order)
        for i in 1:nnodes(tr)
            (is_tip(tr, i) || is_root(tr, i)) && continue
            l = left_child(tr, i)
            r = right_child(tr, i)
            @test position[l] < position[i]
            @test position[r] < position[i]
            @test position[l] < position[r]         # left child first
        end
    end

    @testset "M1 read_tree rejects malformed input" begin
        @test_throws ArgumentError read_tree("((A:1,B:1,C:1):1,D:1);")   # polytomy
        @test_throws ArgumentError read_tree("((A,B:1):1,D:1);")         # no branch length
        @test_throws ArgumentError read_tree("not_a_file_and_not_newick")
    end

    # -----------------------------------------------------------------------
    # M1 state space
    # -----------------------------------------------------------------------

    "Rows of the R-generated states_list.tsv baseline, grouped by configuration."
    function load_states_baseline(fn)
        lines = readlines(fn)
        header = split(lines[1], '\t')
        col(k) = findfirst(==(k), header)
        groups = Dict{Tuple{Int,Int,Bool},Vector{NamedTuple}}()
        for line in lines[2:end]
            f = split(line, '\t')
            key = (parse(Int, f[col("nareas")]), parse(Int, f[col("maxareas")]),
                   f[col("include_null_range")] == "TRUE")
            entry = (
                nstates = parse(Int, f[col("nstates")]),
                numstates = parse(Int, f[col("numstates_from_numareas")]),
                indices = f[col("indices_0based")],
                split_ABC = f[col("split_ABC")],
                collapsed = f[col("collapsed")],
            )
            push!(get!(groups, key, NamedTuple[]), entry)
        end
        return groups
    end

    @testset "M1 state space matches cladoRcpp ordering" begin
        groups = load_states_baseline(joinpath(BASELINES, "states_list.tsv"))
        @test length(groups) == 18

        for ((n, m, null), entries) in groups
            areanames = [string(c) for c in 'A':'Z'][1:n]
            ss = StateSpace(areanames; max_range_size = m, include_null_range = null)

            @test length(ss) == entries[1].nstates
            @test numstates_from_numareas(n; max_range_size = m,
                                          include_null_range = null) == entries[1].numstates

            for (k, e) in enumerate(entries)
                expected = e.indices == "NA" ? Int[] :
                           parse.(Int, split(e.indices, ',')) .+ 1
                @test ss[k] == expected
                @test state_text(ss, k; split = true) == e.split_ABC
                @test state_text(ss, k; split = false) == e.collapsed
                @test state_index(ss, expected) == k
            end
        end
    end

    @testset "M1 StateSpace basics" begin
        ss = StateSpace(["A", "B", "C"])
        @test nareas(ss) == 3
        @test nstates(ss) == 8
        @test ss[1] == Int[]                       # null range is state 1
        @test state_text(ss, 1) == "_"
        @test states_list_text(ss; split = false) == ["_", "A", "B", "C", "AB", "AC", "BC", "ABC"]
        @test states_list_text(ss) == ["_", "A", "B", "C", "A,B", "A,C", "B,C", "A,B,C"]
        @test state_index(ss, [3, 1]) == 6         # order of the range does not matter
        @test state_index(ss, [1, 2, 3]) == 8
        @test state_index(ss, Int[]) == 1

        no_null = StateSpace(["A", "B", "C"]; include_null_range = false)
        @test nstates(no_null) == 7
        @test no_null[1] == [1]
        @test state_index(no_null, Int[]) == 0

        # max_range_size is clamped to the number of areas
        @test nstates(StateSpace(["A", "B"]; max_range_size = 9)) == 4

        @test_throws ArgumentError StateSpace(String[])
        @test_throws ArgumentError StateSpace(["A", "A"])
        @test_throws ArgumentError StateSpace(["A"]; max_range_size = 0)
    end

    # -----------------------------------------------------------------------
    # M1 tip ranges
    # -----------------------------------------------------------------------

    geogfn = joinpath(DATA, "Psychotria_geog.data")

    @testset "M1 read_lagrange_phylip" begin
        tips = read_lagrange_phylip(geogfn)
        @test tips isa TipRanges
        @test ntips(tips) == 19
        @test nareas(tips) == 4
        @test tips.areanames == ["K", "O", "M", "H"]
        @test tips.tipnames[1] == "P_mariniana_Kokee2"
        @test tip_range(tips, 1) == [1]
        @test !has_ambiguity(tips, 1)

        # rows match the data.frame R produced
        lines = readlines(joinpath(BASELINES, "psychotria_tipranges.csv"))
        header = split(lines[1], ',')
        @test replace.(header[2:end], "\"" => "") == tips.areanames
        for (i, line) in enumerate(lines[2:end])
            f = replace.(split(line, ','), "\"" => "")
            @test f[1] == tips.tipnames[i]
            @test String(view(tips.data, i, :)) == join(f[2:end])
        end

        # round trip
        out = joinpath(mktempdir(), "roundtrip.data")
        write_lagrange_phylip(out, tips)
        again = read_lagrange_phylip(out)
        @test again.tipnames == tips.tipnames
        @test again.areanames == tips.areanames
        @test again.data == tips.data
    end

    @testset "M1 read_lagrange_phylip rejects malformed input" begin
        dir = mktempdir()
        writefile(name, text) = (p = joinpath(dir, name); write(p, text); p)

        @test_throws ArgumentError read_lagrange_phylip(joinpath(dir, "nope.data"))

        # row shorter than the header says
        p = writefile("short.data", "2\t3 (A B C)\nt1\t110\nt2\t11\n")
        @test_throws ArgumentError read_lagrange_phylip(p)

        # duplicate area names
        p = writefile("dupareas.data", "1\t2 (A A)\nt1\t10\n")
        @test_throws ArgumentError read_lagrange_phylip(p)

        # a tip that is entirely unknown
        p = writefile("allq.data", "1\t2 (A B)\nt1\t??\n")
        @test_throws ArgumentError read_lagrange_phylip(p)
        @test ntips(read_lagrange_phylip(p; block_all_question_marks = false)) == 1

        # illegal character
        p = writefile("badchar.data", "1\t2 (A B)\nt1\t1x\n")
        @test_throws ArgumentError read_lagrange_phylip(p)

        # fewer tips than declared
        p = writefile("short.data2", "3\t2 (A B)\nt1\t10\n")
        @test_throws ArgumentError read_lagrange_phylip(p)
    end

    @testset "M1 tip_condlikes matches R" begin
        tr = read_tree(trfn)
        tips = read_lagrange_phylip(geogfn)

        got = tip_condlikes(tr, tips)
        @test size(got) == (19, 16)

        ref = [parse.(Float64, split(line, ','))
               for line in readlines(joinpath(BASELINES, "psychotria_tipcondlikes.txt"))]
        @test got == permutedims(reduce(hcat, ref))

        # one state per tip, and each row sums to one
        @test all(sum(got, dims = 2) .== 1)
        @test all(x -> x == 0 || x == 1, got)

        # row order follows the tree, not the geography file
        first_tip = findfirst(==("P_mariniana_MauiNui"), tips.tipnames)
        @test tr.tip_labels[1] != tips.tipnames[1]
        @test got[findfirst(==("P_mariniana_MauiNui"), tr.tip_labels),
                  state_index(StateSpace(tips.areanames), [3])] == 1
        @test tip_range(tips, first_tip) == [3]
    end

    @testset "M1 tip_condlikes edge cases" begin
        tr = read_tree("((A:1,B:1):1,C:1);")
        tips = TipRanges(["A", "B", "C"], ["x", "y"], ['1' '0'; '0' '1'; '?' '0'])
        @test ntips(tips) == 3 && nareas(tips) == 2
        ss = StateSpace(["x", "y"])

        # '?' treated as absent when ambiguities are off: "? 0" has no 1s at all,
        # so it becomes the null range
        lik = tip_condlikes(tr, tips)
        @test lik[3, 1] == 1
        @test sum(lik[3, :]) == 1

        # with ambiguities, a '?' admits every state with the required areas and
        # without the forbidden ones — but never the null range
        lik_amb = tip_condlikes(tr, tips; use_ambiguities = true)
        @test lik_amb[3, state_index(ss, [1])] == 1
        @test lik_amb[3, state_index(ss, [1, 2])] == 0    # y is a hard absence
        @test lik_amb[3, state_index(ss, [2])] == 0
        @test lik_amb[3, 1] == 0
        @test lik_amb[1, state_index(ss, [1])] == 1        # "1 0" is unambiguous
        @test sum(lik_amb[1, :]) == 1

        # a range outside the state space is an error, not a silent zero row
        big = TipRanges(["A", "B", "C"], ["x", "y"], ['1' '1'; '1' '1'; '1' '1'])
        @test_throws ArgumentError tip_condlikes(tr, big; max_range_size = 1)

        # a tip in the tree but missing from the geography file
        missing_tip = TipRanges(["A", "B"], ["x", "y"], ['1' '0'; '0' '1'])
        @test_throws ArgumentError tip_condlikes(tr, missing_tip)
    end

    # -----------------------------------------------------------------------
    # M1 configuration files
    # -----------------------------------------------------------------------

    @testset "M1 read_times" begin
        times = read_times(joinpath(DATA, "times_v1.txt"))
        @test times == [1.9, 3.7, 5.1]

        dir = mktempdir()
        unsorted = joinpath(dir, "unsorted.txt")
        write(unsorted, "1.9\n0.5\n")
        @test_throws ArgumentError read_times(unsorted)

        withzero = joinpath(dir, "zero.txt")
        write(withzero, "0\n1.9\n")
        @test_throws ArgumentError read_times(withzero)

        @test_throws ArgumentError read_times(joinpath(dir, "missing.txt"))
    end

    @testset "M1 read_matrix_with_names" begin
        rownames, colnames, mat = read_matrix_with_names(joinpath(DATA, "Psychotria_controls_v1.txt"))
        @test colnames == ["K", "O", "M", "H"]
        @test length(rownames) == 19
        @test rownames[1] == "P_mariniana_Kokee2"
        @test size(mat) == (19, 4)
        @test mat[1, :] == [20.0, 2.0, 10.0, 1.0]
    end

    @testset "M1 read_areas_allowed and prune_states" begin
        blocks = read_areas_allowed(joinpath(DATA, "areas_allowed_v1.txt"))
        @test length(blocks) == 3
        @test all(size(b) == (4, 4) for b in blocks)
        @test all(blocks[1] .== 1)

        ss = StateSpace(["K", "O", "M", "H"])
        @test nstates(prune_states(ss, blocks[1])) == nstates(ss)

        # second block forbids mixing K or O with anything
        pruned = prune_states(ss, blocks[2])
        @test pruned[1] == Int[]                    # null range survives
        @test state_index(pruned, [1, 2]) == 0
        @test state_index(pruned, [3, 4]) != 0
        @test nstates(pruned) == 1 + 4 + 1          # null, four singletons, M+H

        # third block allows nothing to combine
        @test nstates(prune_states(ss, blocks[3])) == 5

        @test_throws ArgumentError prune_states(ss, ones(3, 3))
    end

    # -----------------------------------------------------------------------
    # M2 model object, Q matrix and cladogenesis
    # -----------------------------------------------------------------------

    "Read a numeric matrix written by `write.table(sep=\"\\t\")`."
    function read_tsv_matrix(fn)
        rows = [[parse(Float64, x) for x in split(line, '\t')] for line in readlines(fn)]
        return permutedims(reduce(hcat, rows))
    end

    "Same, but R's `NA` becomes `NaN` (e.g. the root row of an uppass table)."
    function read_tsv_matrix_na(fn)
        rows = [[x == "NA" ? NaN : parse(Float64, x) for x in split(line, '\t')]
                for line in readlines(fn)]
        return permutedims(reduce(hcat, rows))
    end

    "Read one of the COO tables dumped by the R baseline script."
    function read_coo(fn)
        rows = [[parse(Float64, x) for x in split(line, '\t')] for line in readlines(fn)]
        return (ancestor = [Int(r[1]) for r in rows], left = [Int(r[2]) for r in rows],
                right = [Int(r[3]) for r in rows], weight = [r[4] for r in rows])
    end

    """
    Sum the weights of duplicate triples, so the table can be compared as a set.

    R indexes the non-null states from 0; adding 2 maps that onto this package's
    1-based indexing over the whole state space (the null range is state 1).
    """
    function coo_dict(anc, left, right, weight; r_indexing = false)
        out = Dict{Tuple{Int,Int,Int},Float64}()
        shift = r_indexing ? 2 : 0
        for i in eachindex(weight)
            key = (anc[i] + shift, left[i] + shift, right[i] + shift)
            out[key] = get(out, key, 0.0) + weight[i]
        end
        return out
    end

    "Maximum relative difference between two COO dictionaries."
    function coo_maxdiff(a, b)
        keys(a) == keys(b) || return Inf
        isempty(a) && return 0.0
        return maximum(abs(get(a, k, 0.0) - get(b, k, 0.0)) /
                       max(1.0, abs(get(a, k, 0.0))) for k in keys(a))
    end

    @testset "M2 Model defaults" begin
        m = Model()
        @test length(m.params) == 23
        @test m["d"] == 0.01 && m["e"] == 0.01
        @test m["a"] == 0.0 && m["b"] == 1.0
        @test m["j"] == 0.0
        @test m["y"] == 1.0 && m["s"] == 1.0 && m["v"] == 1.0
        @test m["mx01"] == 0.0001 && m["mx01s"] == 0.0001
        @test m["mx01r"] == 0.5

        d = param(m, "d")
        @test d.lower ≈ 1e-12
        @test d.upper ≈ 5 - 1e-12
        @test d.kind == "free"
        @test param(m, "a").kind == "fixed"
        @test param(m, "ysv").kind == "3-j"
        @test param(m, "mx01s").kind == "mx01"

        @test free_names(Model(:DEC)) == ["d", "e"]
        @test free_names(Model(:DECJ)) == ["d", "e", "j"]
        @test init_params(Model(:DEC)) == [0.01, 0.01]
        @test lower_bounds(Model(:DEC)) == [1e-12, 1e-12]
        @test upper_bounds(Model(:DEC)) == [5 - 1e-12, 5 - 1e-12]
        @test_throws ArgumentError Model(:M0)

        # the four derived models, each of which changes both an event weight
        # and one mx01 parameter
        div = Model(:DIVALIKE)
        @test div["s"] == 0.0 && div["y"] == 1.0 && div["v"] == 1.0
        @test div["mx01v"] == 0.5 && div["mx01s"] == 0.0001
        @test div["j"] == 0.0
        @test param(div, "ysv").kind == "2-j"

        divj = Model(:DIVALIKEJ)
        @test free_names(divj) == ["d", "e", "j"]
        @test divj["s"] == 0.0 && divj["mx01v"] == 0.5
        @test divj["j"] == 0.0001
        @test param(divj, "j").upper ≈ 1.99999

        bay = Model(:BAYAREALIKE)
        @test bay["s"] == 0.0 && bay["v"] == 0.0 && bay["y"] == 1.0
        @test bay["mx01y"] == 0.9999 && bay["mx01v"] == 0.0001
        @test param(bay, "ysv").kind == "1-j"

        bayj = Model(:BAYAREALIKEJ)
        @test free_names(bayj) == ["d", "e", "j"]
        @test bayj["s"] == 0.0 && bayj["v"] == 0.0 && bayj["mx01y"] == 0.9999
        @test param(bayj, "j").upper ≈ 0.99999

        # every model shares the same 23-row table, so a model name can never
        # silently change the meaning of a parameter lookup
        for spec in (:DEC, :DECJ, :DIVALIKE, :DIVALIKEJ, :BAYAREALIKE, :BAYAREALIKEJ)
            @test length(Model(spec).params) == 23
        end

        # bounds are enforced on assignment
        m["d"] = 0.5
        @test m["d"] == 0.5
        @test_throws ArgumentError (m["d"] = 10.0)
        @test_throws ArgumentError (m["d"] = -1.0)

        # the optimiser's vector round-trips through the model
        set_free_values!(m, [0.3, 0.7])
        @test free_values(m) == [0.3, 0.7]
        @test_throws ArgumentError set_free_values!(m, [1.0])
    end

    @testset "M2 maxent matrices match R" begin
        for c in (0.0001, 0.001, 0.1, 0.5, 0.9, 0.9999)
            tag = c == 0.0001 ? "1e-04" : string(c)
            for (kind, fn) in (("sub", relative_probabilities_of_subsets),
                               ("vic", relative_probabilities_of_vicariants))
                ref = read_tsv_matrix(joinpath(BASELINES, "maxent_$(kind)_$(tag).tsv"))
                @test fn(5, c) == ref
            end
        end

        # the DEC default collapses to an exact step matrix
        @test relative_probabilities_of_subsets(4, 0.0001) ==
              [1.0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0]
        @test relative_probabilities_of_vicariants(4, 0.0001) ==
              [0.0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0]

        @test_throws ArgumentError relative_probabilities_of_subsets(4, 0.0)
        @test_throws ArgumentError relative_probabilities_of_vicariants(0, 0.5)

        # a single state can only have one mean
        @test maxent_probs(0.0, [7]) == [1.0]
        # the constraint is the requested mean: 2 is the mean of [1,2,3], so the
        # maximum-entropy answer is uniform; 3 is the largest state, so the only
        # distribution with that mean is a point mass there
        @test maxent_probs(2.0, [1, 2, 3]) ≈ [1 / 3, 1 / 3, 1 / 3]
        @test maxent_probs(3.0, [1, 2, 3]) == [0.0, 0.0, 1.0]
        @test maxent_probs(-1.0, [1, 2, 3]) == [1.0, 0.0, 0.0]
    end

    @testset "M2 Q matrix matches R" begin
        # cladoRcpp rounds every source rate and each accumulation to float.
        ss4 = StateSpace(["A", "B", "C", "D"]; max_range_size = 4)
        m4 = Model(:DEC); m4["d"] = 0.3; m4["e"] = 0.1
        q4 = qmatrix(ss4, m4; r_float32 = true)
        @test q4[12, 16] == Float64(Float32(0.3) + Float32(0.3) + Float32(0.3))
        @test q4[12, 16] == 0.9000000357627869
        areas = ["K", "O", "M", "H"]
        ss = StateSpace(areas)

        cases = [
            ("dec", ss, Model(:DEC), 0.01, 0.01),
            ("dec_de", ss, Model(:DEC), 0.05, 0.02),
            ("dec_mrs2", StateSpace(areas; max_range_size = 2), Model(:DEC), 0.01, 0.01),
        ]
        for (tag, space, m, d, e) in cases
            m["d"] = d
            m["e"] = e
            ref = read_tsv_matrix(joinpath(BASELINES, "qmat_$(tag).tsv"))
            got = Matrix(qmatrix(space, m))
            @test size(got) == size(ref)
            # R's cladoRcpp accumulates in C `float`, so its Q is float32-rounded;
            # 1e-8 relative is the tightest tolerance that is meaningful.
            @test maximum(abs.(got .- ref)) < 1e-8
            @test maximum(abs.(row_sums(got))) < 1e-12
        end

        # the diagonal is exactly minus the off-diagonal row sum, even for rows
        # that have no off-diagonal entries at all
        q = qmatrix(ss, Model(:DEC))
        for i in 1:size(q, 1)
            off = sum(q[i, :]) - q[i, i]
            @test q[i, i] ≈ -off atol = 1e-15
        end

        # the cladogenesis parameters never enter Q, so every model with the same
        # d and e produces the same matrix — the models differ only in their
        # cladogenesis weights
        for spec in (:DECJ, :DIVALIKE, :DIVALIKEJ, :BAYAREALIKE, :BAYAREALIKEJ)
            @test Matrix(qmatrix(ss, Model(spec))) == Matrix(qmatrix(ss, Model(:DEC)))
        end

        zero_model = Model(:DEC)
        zero_model["d"] = 0.0
        zero_model["e"] = 0.0
        qz = Matrix(qmatrix(ss, zero_model))
        @test all(iszero, qz)                  # no NaN, no -0.0 surprises
        @test all(iszero, row_sums(qz))

        # restricting the state space keeps the row sums at zero
        pruned = prune_states(ss, read_areas_allowed(joinpath(DATA, "areas_allowed_v1.txt"))[3])
        @test maximum(abs.(row_sums(Matrix(qmatrix(pruned, Model(:DEC)))))) < 1e-12

        # dispersal multipliers scale the expansion terms
        mult = ones(4, 4); mult[1, 2] = 3.0
        qm = Matrix(qmatrix(ss, Model(:DEC); dispersal_multipliers = mult))
        @test qm[2, 6] ≈ 3 * 0.01              # K -> KO costs d * mult[K, O]
        @test qm[2, 7] ≈ 0.01                  # K -> KM unchanged
    end

    @testset "M2 cladogenesis weights match R" begin
        areas = ["K", "O", "M", "H"]
        ss = StateSpace(areas)

        dec = Model(:DEC); dec["d"] = 0.01; dec["e"] = 0.01
        decj = Model(:DECJ); decj["d"] = 0.01; decj["e"] = 0.01; decj["j"] = 0.1
        decj2 = Model(:DECJ); decj2["d"] = 0.01; decj2["e"] = 0.01; decj2["j"] = 0.5
        ss2 = StateSpace(areas; max_range_size = 2)

        div = Model(:DIVALIKE); div["d"] = 0.01; div["e"] = 0.01
        divj = Model(:DIVALIKEJ); divj["d"] = 0.01; divj["e"] = 0.01; divj["j"] = 0.1
        bay = Model(:BAYAREALIKE); bay["d"] = 0.01; bay["e"] = 0.01
        bayj = Model(:BAYAREALIKEJ); bayj["d"] = 0.01; bayj["e"] = 0.01; bayj["j"] = 0.1

        cases = [("dec", ss, dec), ("decj", ss, decj),
                 ("decj_j0.5", ss, decj2), ("dec_mrs2", ss2, dec),
                 ("divalike", ss, div), ("divalikej", ss, divj),
                 ("bayarealike", ss, bay), ("bayarealikej", ss, bayj)]
        for (tag, space, m) in cases
            ref = read_coo(joinpath(BASELINES, "coo_$(tag).tsv"))
            got = cladogenesis_weights(space, m)

            a = coo_dict(ref.ancestor, ref.left, ref.right, ref.weight; r_indexing = true)
            b = coo_dict(got.ancestor, got.left, got.right, got.weight)
            @test keys(a) == keys(b)                       # identical event set
            @test coo_maxdiff(a, b) < 1e-8                 # same weights

            # The *order* matters as well: the BSM split sampler draws an index
            # into the probability vector built from this table, so a different
            # enumeration would consume the RNG stream differently (M7).
            @test got.ancestor .- 2 == ref.ancestor
            @test got.left .- 2 == ref.left
            @test got.right .- 2 == ref.right
            @test got.weight ≈ ref.weight rtol = 1e-6

            # Rsp_rowsums, over the non-null states
            refrs = [parse(Float64, l) for l in readlines(joinpath(BASELINES, "rowsums_$(tag).tsv"))]
            # R's weights are C floats, so compare relatively
            @test row_sums(got, length(space))[2:end] ≈ refrs rtol = 1e-6
        end

        # DEC has no founder events; DEC+J adds exactly one pair per (ancestor, new area)
        n_dec = length(cladogenesis_weights(ss, dec))
        n_decj = length(cladogenesis_weights(ss, decj))
        @test n_dec == 104
        @test n_decj - n_dec == 56

        # The event sets follow the model definitions: DEC has 56 subset + 44
        # vicariance + 4 sympatry events.  DIVALIKE drops the subsets and gains
        # the six equal 2|2 splits of the four-area range (56 removed, 6 added).
        # BAYAREALIKE keeps sympatry only, i.e. one event per non-null state.
        @test length(cladogenesis_weights(ss, div)) == 54
        @test length(cladogenesis_weights(ss, bay)) == 15
        @test length(cladogenesis_weights(ss, divj)) - 54 == 56
        @test length(cladogenesis_weights(ss, bayj)) - 15 == 56
    end

    @testset "M2 cladogenesis invariants" begin
        ss = StateSpace(["A", "B", "C"])
        m = Model(:DECJ); m["j"] = 0.2
        w = cladogenesis_weights(ss, m)

        @test all(w.ancestor .>= 2)                        # the null range never speciates
        @test all(w.left .>= 1) && all(w.right .>= 1)
        @test all(w.weight .> 0)
        @test length(w.ancestor) == length(w.left) == length(w.right) == length(w.weight)

        # every daughter of a vicariance or subset event is a subset of its ancestor
        for i in eachindex(w.weight)
            a = ss[w.ancestor[i]]
            l = ss[w.left[i]]
            r = ss[w.right[i]]
            @test issubset(l, a) || issubset(r, a) || isempty(intersect(l, a))
        end

        # row sums are positive for every non-null state, so no node divides by zero
        rs = row_sums(w, length(ss))
        @test all(rs[2:end] .> 0)
        @test rs[1] == 0

        @test_throws ArgumentError cladogenesis_weights(ss, m; dispersal_multipliers = ones(2, 2))
    end

    @testset "M3 linked parameters" begin
        # DEC+J with j = 0.1: ysv = 2.9 and y = s = v = 2.9/3, NOT 1.  R applies
        # this inside the likelihood but not inside get_Qmat_COOmat().
        m = Model(:DECJ); m["j"] = 0.1
        @test m["y"] == 1.0                       # untouched until linking
        ml = linked(m)
        @test ml["ysv"] ≈ 2.9
        @test ml["ys"] ≈ 2.9 * 2 / 3
        @test ml["y"] ≈ 2.9 / 3
        @test ml["s"] ≈ 2.9 / 3
        @test ml["v"] ≈ 2.9 / 3
        @test m["ysv"] ≈ 3 - 1e-5                 # the original is not modified

        # DEC has j = 0, so the links reproduce the default 1:1:1 split
        dec = linked(Model(:DEC))
        @test dec["ysv"] ≈ 3.0
        @test dec["y"] == dec["s"] == dec["v"] == 1.0

        # DIVALIKE / BAYAREALIKE keep their fixed zeros
        @test linked(Model(:DIVALIKE))["s"] == 0.0
        @test linked(Model(:BAYAREALIKE))["v"] == 0.0
        @test linked(Model(:BAYAREALIKE))["mx01y"] == 0.9999

        # mx01* follow mx01 unless overridden
        mm = Model(); mm["mx01"] = 0.3
        @test linked(mm)["mx01j"] ≈ 0.3

        # fixed parameters may sit outside their optimiser bounds (j = 0 is
        # below the 1e-5 lower bound, and that is legal for a fixed parameter)
        @test Model(:DEC)["j"] == 0.0
        @test_throws ArgumentError (Model(:DECJ)["j"] = -1.0)
    end

    @testset "M3 matrix exponential" begin
        ss = StateSpace(["K", "O", "M", "H"])
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        q = Matrix(qmatrix(ss, m))

        @test transition_matrix(q, 0.0) == Matrix{Float64}(I, 16, 16)
        p = transition_matrix(q, 2.372081244)
        @test maximum(abs.(sum(p, dims = 2) .- 1)) < 1e-12   # row-stochastic
        @test minimum(p) > -1e-15                             # no negative probs

        # matches EXPOKIT's dgpadm on the same matrix.  R's Q is float32-rounded,
        # so feed the same matrix to both rather than comparing our Q to R's
        # exponential.
        qref = read_tsv_matrix(joinpath(BASELINES, "qmat_dec.tsv"))
        ref = read_tsv_matrix(joinpath(BASELINES, "expm_dec_t2.372.tsv"))
        @test maximum(abs.(transition_matrix(qref, 2.372081244) .- ref)) < 1e-14

        # the null-range row is zero, so the null range is absorbing: exp(tQ)
        # keeps it at itself and cannot reach any other state
        @test all(iszero, q[1, :])
        @test p[1, 1] ≈ 1.0
        @test maximum(abs.(p[1, 2:end])) < 1e-15

        # exp(tA)·0 = 0 (plan §6.5), for both the dense and the Krylov path
        z = zeros(16)
        @test all(iszero, apply_transition(q, 1.0, z))
        @test all(iszero, apply_transition(sparse(q), 1.0, z))

        # dense and sparse paths agree
        v = zeros(16); v[2] = 1.0
        @test apply_transition(q, 0.7, v) ≈ apply_transition(sparse(q), 0.7, v) rtol = 1e-9
    end

    @testset "M3 log-likelihood matches R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))

        # R's node combination accumulates in float32 (plan §6.10), so lnL can
        # only agree to ~1e-7; 1e-6 absolute is the meaningful tolerance.
        for row in eachrow(CSV.read(joinpath(BASELINES, "lnL_psychotria.tsv"), DataFrame))
            spec = Symbol(replace(String(row.model), "+" => ""))
            ss = StateSpace(tips.areanames; max_range_size = Int(row.max_range_size))
            m = Model(spec)
            m["d"] = row.d; m["e"] = row.e; m["j"] = row.j
            got = loglikelihood(tr, ss, tips, m)
            @test got ≈ row.lnL atol = 1e-6
        end

        for row in eachrow(CSV.read(joinpath(BASELINES, "lnL_psychotria_params.tsv"), DataFrame))
            ss = StateSpace(tips.areanames; max_range_size = Int(row.max_range_size))
            m = Model(:DEC); m["d"] = row.d; m["e"] = row.e
            @test loglikelihood(tr, ss, tips, m) ≈ row.lnL atol = 1e-6
        end
    end

    @testset "M3 downpass internals match R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)

        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        r = downpass(tr, ss, tipcl, qmatrix(ss, m), cladogenesis_weights(ss, m))

        # per-node totals (R's computed_likelihoods_at_each_node)
        reftot = [parse(Float64, l) for l in readlines(joinpath(BASELINES, "node_totals_dec.tsv"))]
        for n in 1:nnodes(tr)
            is_tip(tr, n) && continue
            @test r.node_total[n] ≈ reftot[n] rtol = 1e-6
        end
        @test r.loglik ≈ sum(log(reftot[n]) for n in 1:nnodes(tr) if !is_tip(tr, n)) atol = 1e-6

        # per-node conditional likelihoods (R's condlikes_of_each_state)
        refcl = read_tsv_matrix(joinpath(BASELINES, "condlikes_dec.tsv"))
        @test size(refcl) == size(r.node_condlikes)
        for n in 1:nnodes(tr)
            is_tip(tr, n) && continue
            scale = maximum(abs.(view(refcl, n, :)))
            @test maximum(abs.(view(r.node_condlikes, n, :) .- view(refcl, n, :))) < 1e-6 * scale
        end

        # the normalised vectors sum to 1 at every internal node
        for n in 1:nnodes(tr)
            is_tip(tr, n) && continue
            @test sum(r.node_relprobs[n, :]) ≈ 1.0
            @test r.node_relprobs[n, 1] == 0.0        # null range never speciates
        end

        # the model is not mutated by the likelihood call
        before = free_values(m)
        loglikelihood(tr, ss, tips, m)
        @test free_values(m) == before

        # a zero-length branch is treated as a hook: no cladogenesis event, and
        # the two daughter likelihoods are multiplied elementwise
        @test length(cladogenesis_weights(ss, m)) == 104
    end

    @testset "M3 ancestral probabilities match R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)

        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        a = ancestral_probs(tr, ss, tips, m)

        @test a.downpass.loglik ≈ loglikelihood(tr, ss, tips, m)

        # R leaves the root's branch-bottom uppass as NA, so that row is skipped
        for (fn, got) in (("mltop_dec.tsv", a.branch_top),
                          ("mlbottom_dec.tsv", a.branch_bottom),
                          ("uppass_top_dec.tsv", a.uppass_top))
            ref = read_tsv_matrix_na(joinpath(BASELINES, fn))
            @test size(ref) == size(got)
            for n in 1:nnodes(tr)
                is_tip(tr, n) && continue
                n == root_node(tr) && fn == "mlbottom_dec.tsv" && continue
                @test maximum(abs.(view(got, n, :) .- view(ref, n, :))) < 1e-6
            end
        end

        # every row is a probability vector
        for n in 1:nnodes(tr)
            @test sum(a.branch_top[n, :]) ≈ 1.0
            if !is_tip(tr, n) && n != root_node(tr)
                @test sum(a.branch_bottom[n, :]) ≈ 1.0
            end
        end

        # at the root the marginal probability IS the downpass probability: R
        # deliberately avoids multiplying the downpass by itself
        @test a.branch_top[root_node(tr), :] ≈ a.downpass.node_relprobs[root_node(tr), :]
        @test a.uppass_top[root_node(tr), :] ≈ fill(1 / 16, 16)   # uniform start

        # the null range is impossible at internal nodes
        for n in 1:nnodes(tr)
            is_tip(tr, n) && continue
            @test a.branch_top[n, 1] == 0.0
        end

        # all six models produce valid probability vectors
        for spec in (:DEC, :DECJ, :DIVALIKE, :DIVALIKEJ, :BAYAREALIKE, :BAYAREALIKEJ)
            mm = Model(spec); mm["j"] = 0.1
            aa = ancestral_probs(tr, ss, tips, mm)
            @test all(isfinite, aa.branch_top)
            @test all(abs.(sum(aa.branch_top, dims = 2) .- 1) .< 1e-9)
        end
    end

    @testset "M3 root edge and fixed nodes match R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        ns = nstates(ss)

        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        q = qmatrix(ss, m)
        w = cladogenesis_weights(ss, m)
        ref = CSV.read(joinpath(BASELINES, "rootedge_fixnode_dec.tsv"), DataFrame)

        @test loglikelihood(tr, ss, m, tipcl) ≈ ref.base[1] atol = 1e-6

        # --- root stem -----------------------------------------------------
        @test root_edge(tr) == 0.0
        tre = with_root_edge(tr, 0.1)
        @test root_edge(tre) == 0.1

        dp = downpass(tre, ss, tipcl, q, w; rootedge = true)
        @test dp.loglik ≈ ref.rootedge[1] atol = 1e-6
        @test length(dp.rootedge_bottom) == ns
        @test sum(dp.rootedge_bottom) ≈ 1.0
        # the stem adds one exp(tQ) and one log() term, nothing else
        @test dp.loglik - loglikelihood(tre, ss, m, tipcl) ≈
              log(sum(vec(apply_transition(q, 0.1, dp.node_relprobs[root_node(tre), :]))))
        # rootedge = false ignores a stem that is present
        @test loglikelihood(tre, ss, m, tipcl) ≈ ref.base[1] atol = 1e-6

        # --- one fixed node ------------------------------------------------
        fx1 = zeros(ns); fx1[3] = 1.0
        dpf = downpass(tr, ss, tipcl, q, w; fixnode = 20, fixlikes = fx1)
        @test dpf.loglik ≈ ref.fixnode20[1] atol = 1e-6
        @test dpf.node_relprobs[20, :] ≈ fx1          # the node is pinned
        refdp = read_tsv_matrix(joinpath(BASELINES, "fixnode_dec_downpass.tsv"))
        @test maximum(abs.(dpf.node_relprobs .- refdp)) < 1e-6

        ap = ancestral_probs(tr, ss, tipcl, q, w; fixnode = 20, fixlikes = fx1)
        @test ap.branch_top[20, :] ≈ fx1
        refm = read_tsv_matrix(joinpath(BASELINES, "fixnode_dec_marginal.tsv"))
        @test maximum(abs.(ap.branch_top .- refm)) < 1e-6

        # --- several fixed nodes -------------------------------------------
        fx2 = zeros(2, ns); fx2[1, 3] = 1.0; fx2[2, 5] = 1.0
        @test downpass(tr, ss, tipcl, q, w; fixnode = [20, 21], fixlikes = fx2).loglik ≈
              ref.fixnode2021[1] atol = 1e-6

        # malformed input is rejected
        @test_throws ArgumentError downpass(tr, ss, tipcl, q, w; fixnode = 20)
        @test_throws ArgumentError downpass(tr, ss, tipcl, q, w; fixnode = 999, fixlikes = fx1)
        @test_throws ArgumentError downpass(tr, ss, tipcl, q, w; fixnode = [20, 21], fixlikes = fx1)
    end

    @testset "M3 Mk model matches phytools" begin
        # phytools' rerootingMethod() fits the same Mk model but includes the
        # root frequencies; BioGeoBEARS conditions on the data instead, so the
        # two log-likelihoods differ by log(1/4).  Baselines come from
        # temp/make_mk_baseline.R.
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 1, include_null_range = false)

        rate = parse(Float64, readlines(joinpath(BASELINES, "mk_rate.tsv"))[1])
        m = Model()
        set_param!(m, "d"; kind = "fixed", init = 0.0, est = 0.0)
        set_param!(m, "e"; kind = "fixed", init = 0.0, est = 0.0)
        set_param!(m, "s"; kind = "fixed", init = 0.0, est = 0.0)
        set_param!(m, "v"; kind = "fixed", init = 0.0, est = 0.0)
        set_param!(m, "ysv"; kind = "1-j")
        set_param!(m, "ys"; kind = "ysv*1/1")
        set_param!(m, "y"; kind = "1-j")
        set_param!(m, "mx01y"; kind = "fixed", init = 0.9999, est = 0.9999)
        set_param!(m, "a"; kind = "free", init = rate, est = rate)

        got = loglikelihood(tr, ss, tips, m)
        bgb = parse(Float64, readlines(joinpath(BASELINES, "mk_bgb_lnL.tsv"))[1])
        phy = parse(Float64, readlines(joinpath(BASELINES, "mk_lnL.tsv"))[1])
        @test got ≈ bgb atol = 1e-6
        @test got ≈ phy - log(0.25) atol = 1e-6

        # marginal ancestral states: phytools stores one row per internal node
        # in ape's order, which is our rows 20:37
        ap = ancestral_probs(tr, ss, tips, m)
        ref = read_tsv_matrix(joinpath(BASELINES, "mk_marginal_anc.tsv"))
        @test size(ref) == (nnodes(tr) - ntips(tr), nstates(ss))
        @test maximum(abs.(ap.branch_top[(ntips(tr) + 1):nnodes(tr), :] .- ref)) < 1e-6
        @test ap.branch_top[root_node(tr), :] ≈ ref[1, :] atol = 1e-6
    end

    @testset "M3 likelihood invariants" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))

        # a null range is excluded from the state space without breaking the
        # index bookkeeping in cladogenesis_weights
        ss = StateSpace(tips.areanames; max_range_size = 4, include_null_range = false)
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        w = cladogenesis_weights(ss, m)
        @test minimum(w.ancestor) == 1
        @test maximum(w.ancestor) == nstates(ss)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4, include_null_range = false)
        @test isfinite(loglikelihood(tr, ss, m, tipcl))

        # doubling d with e held at zero must not decrease the likelihood
        ss4 = StateSpace(tips.areanames; max_range_size = 4)
        lo = Model(:DEC); lo["d"] = 0.001; lo["e"] = 0.0
        hi = Model(:DEC); hi["d"] = 0.1;   hi["e"] = 0.0
        @test loglikelihood(tr, ss4, tips, lo) <= loglikelihood(tr, ss4, tips, hi)

        # every model is finite at the default parameters
        for spec in (:DEC, :DECJ, :DIVALIKE, :DIVALIKEJ, :BAYAREALIKE, :BAYAREALIKEJ)
            m = Model(spec); m["j"] = 0.1
            @test isfinite(loglikelihood(tr, ss4, tips, m))
        end

        # wrong-sized tip matrices are rejected
        @test_throws ArgumentError downpass(tr, ss4, zeros(2, 16), qmatrix(ss4, m),
                                            cladogenesis_weights(ss4, m))
    end

    @testset "M3 small trees stay finite" begin
        newicks = [
            "((t1:1e-7,t2:0.5):0.3,t3:2);",                       # one hook branch
            "((t1:100,t2:0.01):5,t3:0.001);",                     # extreme lengths
            "(((t1:1,t2:1):1,t3:1):1,t4:1);",                     # 4 tips
            "((t1:1e-6,t2:1e-6):1e-6,(t3:1e-6,t4:1e-6):1e-6);",   # all at the limit
        ]
        for nwk in newicks
            t = read_tree(nwk)
            ss = StateSpace(["A", "B", "C"]; max_range_size = 2)
            m = Model(:DECJ); m["d"] = 0.1; m["e"] = 0.05; m["j"] = 0.2

            tipcl = zeros(ntips(t), nstates(ss))
            for i in 1:ntips(t)
                tipcl[i, mod1(i, nstates(ss) - 1) + 1] = 1.0
            end

            @test isfinite(loglikelihood(t, ss, m, tipcl))
            a = ancestral_probs(t, ss, m, tipcl)
            @test all(isfinite, a.branch_top)
            @test all(abs.(sum(a.branch_top, dims = 2) .- 1) .< 1e-9)
        end
    end

    @testset "M4 maximum likelihood matches R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)

        specs = Dict("DEC" => :DEC, "DEC+J" => :DECJ,
                     "DIVALIKE" => :DIVALIKE, "DIVALIKE+J" => :DIVALIKEJ,
                     "BAYAREALIKE" => :BAYAREALIKE, "BAYAREALIKE+J" => :BAYAREALIKEJ)

        ref = CSV.read(joinpath(BASELINES, "ml_psychotria.tsv"), DataFrame)
        fits = BGBFit[]
        for row in eachrow(ref)
            f = fit_model(tr, ss, tips, Model(specs[row.model]); maxeval = 2000)
            push!(fits, f)
            @test f.loglik ≈ row.lnL atol = 1e-4
            @test f.model["d"] ≈ row.d atol = 1.5e-3
            @test f.model["e"] ≈ row.e atol = 1.5e-3
            @test f.model["j"] ≈ row.j atol = 1.5e-3
            @test f.numparams == row.numparams
            @test f.converged
        end

        # the strict local DEC baseline from the plan
        dec = fits[findfirst(f -> model_name(f) == "DEC", fits)]
        @test paramvalue(dec, "d") ≈ 0.035055 atol = 1e-3
        @test paramvalue(dec, "e") ≈ 0.028324 atol = 1e-3
        @test dec.loglik ≈ -34.543127 atol = 1e-4

        # --- model selection -----------------------------------------------
        df = summary_table(fits)
        @test nrow(df) == 6
        @test df.model[1] == "DEC+J"                 # sorted by AIC, best first
        @test df.AIC[1] == minimum(df.AIC)
        @test all(df.dAIC .>= 0)
        @test sum(df.AIC_wt) ≈ 1.0
        @test sum(df.AICc_wt) ≈ 1.0
        @test df.AIC ≈ 2 .* df.numparams .- 2 .* df.lnL
        @test df.AICc ≈ df.AIC .+ 2 .* df.numparams .* (df.numparams .+ 1) ./
                        (ntips(tr) .- df.numparams .- 1)

        @test akaike_weights([10.0, 10.0, 10.0]) ≈ fill(1 / 3, 3)
        @test akaike_weights([0.0, 2.0]) ≈ exp.(-0.5 .* [0.0, 2.0]) ./ sum(exp.(-0.5 .* [0.0, 2.0]))
        @test aicc(-10.0, 2, 20) ≈ 24 + 2 * 2 * 3 / 17
        @test_throws ArgumentError aicc(-10.0, 20, 19)
    end

    @testset "M4 multi-start and fit object" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)

        # ten random starts must not do worse than the R optimum
        f = fit_model(tr, ss, tips, Model(:DEC); nstarts = 10,
                      rng = StableRNG(20261104), maxeval = 1000)
        @test f.loglik >= -34.543127 - 1e-4
        @test f.model["d"] ≈ 0.035055 atol = 1.5e-3

        fj = fit_model(tr, ss, tips, Model(:DECJ); nstarts = 10,
                       rng = StableRNG(7), maxeval = 1000)
        @test fj.loglik >= -20.947589 - 1e-4
        @test fj.model["j"] ≈ 0.114287 atol = 1.5e-3

        # the fitted model has its linked parameters applied
        @test fj.model["ysv"] ≈ 3 - fj.model["j"]
        @test fj.model["y"] ≈ (3 - fj.model["j"]) / 3

        # ancestral probabilities are opt-in
        @test_throws ArgumentError ancprobs(f)
        fa = fit_model(tr, ss, tips, Model(:DEC); maxeval = 500, calc_ancprobs = true)
        @test size(ancprobs(fa).branch_top) == (nnodes(tr), nstates(ss))
        @test ancprobs(fa).downpass.loglik ≈ fa.loglik

        # summary helpers
        @test fit_summary(f).model == "DEC"
        @test fit_summary(f).AIC ≈ aic(f)
        @test nparams(f) == 2
        @test occursin("DEC", sprint(show, f))

        # a model with nothing to estimate is rejected
        @test_throws ArgumentError fit_model(tr, ss, tips, Model(:DEC);
                                             algorithms = ()) 
        fixed = Model(:DEC)
        set_param!(fixed, "d"; kind = "fixed")
        set_param!(fixed, "e"; kind = "fixed")
        @test_throws ArgumentError fit_model(tr, ss, tips, fixed)
    end

    @testset "M5 time stratification matches R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)

        exdir = joinpath(DATA, "m3strat")
        timesfn = joinpath(exdir, "timeperiods.txt")
        dmfn = joinpath(exdir, "dispersal_multipliers.txt")
        aafn = joinpath(exdir, "areas_allowed.txt")

        strat = Stratification(; timesfn = timesfn, dispersal_multipliers_fn = dmfn)
        strat_aa = Stratification(; timesfn = timesfn, dispersal_multipliers_fn = dmfn,
                                  areas_allowed_fn = aafn)

        @test strat.times == [0.5, 1.9, 3.7, 5.1, 10.0]
        @test length(strat.dispersal_multipliers) == 5
        @test isempty(strat.areas_allowed)
        @test length(strat_aa.areas_allowed) == 5

        # R's per-stratum state spaces: all 16 states, then 8, 4, 2, 2.
        allowed_counts = [length(BioGeoBEARS._allowed_states(ss, strat_aa, i)) for i in 1:5]
        @test allowed_counts == [16, 8, 4, 2, 2]

        cases = Dict("mult_d001_e001" => (0.01, 0.01, strat),
                     "mult_d003_e002" => (0.03, 0.02, strat),
                     "aa_d001_e001"   => (0.01, 0.01, strat_aa))

        ref = CSV.read(joinpath(BASELINES, "strat_lnL.tsv"), DataFrame)
        @test nrow(ref) == 6

        for row in eachrow(ref)
            d, e, s = cases[row.case]
            m = Model(:DEC); m["d"] = d; m["e"] = e
            got = stratified_loglikelihood(tr, ss, tipcl, m, s;
                                           dense = row.path == "dense")
            # R accumulates each split likelihood in float32 (§6.10 of the
            # plan), so a Float64 implementation sits ~1e-7 away from R's
            # printed value.  Dense and sparse paths agree with each other far
            # more tightly; that is checked below.
            @test got ≈ row.lnL atol = 5e-7
        end

        # our dense and sparse paths must agree, and R's two paths are identical
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        ll_dense = stratified_loglikelihood(tr, ss, tipcl, m, strat; dense = true)
        ll_sparse = stratified_loglikelihood(tr, ss, tipcl, m, strat; dense = false)
        @test ll_dense ≈ ll_sparse atol = 1e-10
        rvals = [row.lnL for row in eachrow(ref) if row.case == "mult_d001_e001"]
        @test rvals[1] ≈ rvals[2] atol = 1e-12

        # per-node conditional likelihoods, in original node order, match R's
        # condlikes_table restricted to the 37 original nodes
        cl = stratified_condlikes(tr, ss, tipcl, m, strat)
        refcl = read_tsv_matrix(joinpath(BASELINES, "strat_condlikes.tsv"))
        @test size(cl) == size(refcl) == (nnodes(tr), nstates(ss))
        for n in 1:nnodes(tr)
            @test maximum(abs.(view(cl, n, :) .- view(refcl, n, :))) < 1e-6
        end
        @test sum(log.(vec(sum(cl, dims = 2)))) ≈ -41.7801738907879 atol = 5e-7

        # no time breakpoint: one stratum must reproduce the plain downpass
        deg = Stratification([tree_height(tr) + 1.0])
        @test stratified_loglikelihood(tr, ss, tipcl, m, deg) ≈
              loglikelihood(tr, ss, m, tipcl) atol = 1e-12

        # the model is not mutated
        before = free_values(m)
        stratified_loglikelihood(tr, ss, tipcl, m, strat)
        @test free_values(m) == before

        # a boundary that lands on a node is rejected
        node_t = node_times_bp(tr)
        @test_throws ArgumentError stratified_loglikelihood(
            tr, ss, tipcl, m, Stratification([node_t[node_t .> 0.6][1], 10.0]))
        # the last boundary must be older than the root
        @test_throws ArgumentError stratified_loglikelihood(
            tr, ss, tipcl, m, Stratification([1.0]))
        @test_throws ArgumentError Stratification([0.5, 0.4])
        @test_throws ArgumentError Stratification([0.0, 0.5])

        # a stratified ML fit runs through the same optimiser, and can return
        # the marginal ancestral probabilities at the optimum
        f = fit_model(tr, ss, tipcl, Model(:DEC), strat; maxeval = 300,
                      calc_ancprobs = true)
        @test f.loglik ≈ stratified_loglikelihood(tr, ss, tipcl, f.model, strat) atol = 1e-9
        @test f.numparams == 2
        @test size(ancprobs(f).branch_top) == (nnodes(tr), nstates(ss))
        @test ancprobs(f).downpass.loglik ≈ f.loglik
    end

    @testset "M5 stratified ancestral probabilities match R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)

        exdir = joinpath(DATA, "m3strat")
        strat = Stratification(; timesfn = joinpath(exdir, "timeperiods.txt"),
                               dispersal_multipliers_fn = joinpath(exdir, "dispersal_multipliers.txt"))
        strat_aa = Stratification(; timesfn = joinpath(exdir, "timeperiods.txt"),
                                  dispersal_multipliers_fn = joinpath(exdir, "dispersal_multipliers.txt"),
                                  areas_allowed_fn = joinpath(exdir, "areas_allowed.txt"))
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01

        root = root_node(tr)
        for (tag, s) in (("mult", strat), ("aa", strat_aa))
            a = stratified_ancestral_probs(tr, ss, tipcl, m, s)

            @test size(a.branch_top) == (nnodes(tr), nstates(ss))
            @test a.downpass.loglik ≈ stratified_loglikelihood(tr, ss, tipcl, m, s) atol = 1e-10
            @test a.downpass.node_condlikes ≈ stratified_condlikes(tr, ss, tipcl, m, s)

            reftop = read_tsv_matrix(joinpath(BASELINES, "strat_mltop_$tag.tsv"))
            refbot = read_tsv_matrix_na(joinpath(BASELINES, "strat_mlbottom_$tag.tsv"))
            @test size(reftop) == size(a.branch_top)
            @test maximum(abs.(a.branch_top .- reftop)) < 1e-6
            # the root's branch bottom is NA in R; everywhere else compare values
            @test isequal(isnan.(a.branch_bottom), isnan.(refbot))
            ok = .!isnan.(refbot)
            @test maximum(abs.(a.branch_bottom[ok] .- refbot[ok])) < 1e-6

            # every non-root row is a probability vector
            for n in 1:nnodes(tr)
                @test sum(view(a.branch_top, n, :)) ≈ 1.0 atol = 1e-10
                n == root && continue
                @test sum(view(a.branch_bottom, n, :)) ≈ 1.0 atol = 1e-10
            end

            # the root's branch top is the downpass probability, and its
            # branch bottom is undefined (R stores NA there)
            @test view(a.branch_top, root, :) ≈ view(a.downpass.node_relprobs, root, :)
            @test all(isnan, view(a.branch_bottom, root, :))
            @test view(a.uppass_top, root, :) ≈ fill(1 / nstates(ss), nstates(ss))

            # the null range is impossible at internal nodes
            for n in 1:nnodes(tr)
                is_tip(tr, n) && continue
                @test a.branch_top[n, 1] == 0.0
            end

            # dense and sparse paths agree
            a2 = stratified_ancestral_probs(tr, ss, tipcl, m, s; dense = true)
            @test a.branch_top ≈ a2.branch_top atol = 1e-10
            @test a.uppass_top ≈ a2.uppass_top atol = 1e-10
        end

        # one stratum with no breakpoint must reproduce the non-stratified
        # uppass (the stratified code is a strict generalisation)
        deg = Stratification([tree_height(tr) + 1.0])
        a1 = stratified_ancestral_probs(tr, ss, tipcl, m, deg)
        b1 = ancestral_probs(tr, ss, m, tipcl)
        @test a1.branch_top ≈ b1.branch_top atol = 1e-10
        @test a1.uppass_top ≈ b1.uppass_top atol = 1e-10
        for n in 1:nnodes(tr)
            n == root && continue
            @test view(a1.branch_bottom, n, :) ≈ view(b1.branch_bottom, n, :) atol = 1e-10
        end

        # the model is not mutated
        before = free_values(m)
        stratified_ancestral_probs(tr, ss, tipcl, m, strat)
        @test free_values(m) == before
    end

    @testset "M5 stratified fixnode and areas_adjacency match R" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01

        exdir = joinpath(DATA, "m3strat")
        timesfn = joinpath(exdir, "timeperiods.txt")
        dmfn = joinpath(exdir, "dispersal_multipliers.txt")
        strat = Stratification(; timesfn = timesfn, dispersal_multipliers_fn = dmfn)

        onehot(k) = (v = zeros(nstates(ss)); v[k] = 1.0; v)

        # --- fixnode / fixlikes ------------------------------------------------
        # node 23 is a subtree root in stratum 3 and a piece tip in stratum 4;
        # node 31 is the root of another stratum-3 piece.
        cases = [
            ("fix1", 23, onehot(2), -47.001392687388282),
            ("fix2", [23, 31], vcat(onehot(2)', onehot(4)'), -71.673320009328989),
        ]
        for (tag, fn, fl, reflnl) in cases
            a = stratified_ancestral_probs(tr, ss, tipcl, m, strat;
                                           fixnode = fn, fixlikes = fl)
            ll = stratified_loglikelihood(tr, ss, tipcl, m, strat;
                                          fixnode = fn, fixlikes = fl)
            @test ll ≈ reflnl atol = 5e-7
            @test a.downpass.loglik ≈ ll
            @test a.downpass.node_condlikes ≈
                  stratified_condlikes(tr, ss, tipcl, m, strat; fixnode = fn, fixlikes = fl)

            reftop = read_tsv_matrix(joinpath(BASELINES, "strat_mltop_$tag.tsv"))
            refbot = read_tsv_matrix_na(joinpath(BASELINES, "strat_mlbottom_$tag.tsv"))
            @test maximum(abs.(a.branch_top .- reftop)) < 1e-6
            @test isequal(isnan.(a.branch_bottom), isnan.(refbot))
            ok = .!isnan.(refbot)
            @test maximum(abs.(a.branch_bottom[ok] .- refbot[ok])) < 1e-6

            # the fixed state dominates the fixed node's marginal probability
            n = fn isa Integer ? fn : fn[1]
            @test argmax(view(a.branch_top, n, :)) == 2
            @test a.branch_top[n, 2] > 0.99
        end

        # a fixed node changes the likelihood: the same tree without the fix
        # gives a different value
        ll0 = stratified_loglikelihood(tr, ss, tipcl, m, strat)
        @test ll0 ≈ -41.7801738907879 atol = 5e-7
        llfix = stratified_loglikelihood(tr, ss, tipcl, m, strat;
                                         fixnode = 23, fixlikes = onehot(2))
        @test !isapprox(llfix, ll0; atol = 1e-3)

        # fixnode is validated the same way as in `downpass`
        @test_throws ArgumentError stratified_loglikelihood(
            tr, ss, tipcl, m, strat; fixnode = 23)
        @test_throws ArgumentError stratified_loglikelihood(
            tr, ss, tipcl, m, strat; fixnode = 999, fixlikes = onehot(2))

        # two fixnodes inside the same piece: each keeps its own likelihood.
        # (R's master-table ordering can swap the two rows here; this package
        # aligns by node number, so the two fixes land on their own nodes.)
        fl24 = vcat(onehot(2)', onehot(3)')
        a24 = stratified_ancestral_probs(tr, ss, tipcl, m, strat;
                                         fixnode = [23, 24], fixlikes = fl24)
        @test argmax(view(a24.branch_top, 23, :)) == 2
        @test argmax(view(a24.branch_top, 24, :)) == 3
        @test all(isfinite, a24.branch_top)

        # --- areas_adjacency ---------------------------------------------------
        adjfn = joinpath(exdir, "areas_adjacency.txt")
        blocks = read_areas_adjacency(adjfn)
        @test length(blocks) == 5
        @test all(b -> b == blocks[1], blocks)

        strat_adj = Stratification(; timesfn = timesfn, dispersal_multipliers_fn = dmfn,
                                   areas_adjacency_fn = adjfn)
        @test length(strat_adj.areas_adjacency) == 5
        # a chain K-O-M-H: null + 4 singletons + 3 adjacent pairs
        @test [length(BioGeoBEARS._allowed_states(ss, strat_adj, i)) for i in 1:5] ==
              fill(8, 5)

        aadj = stratified_ancestral_probs(tr, ss, tipcl, m, strat_adj)
        lladj = stratified_loglikelihood(tr, ss, tipcl, m, strat_adj)
        @test lladj ≈ -57.81962257035417 atol = 5e-7
        reftop = read_tsv_matrix(joinpath(BASELINES, "strat_mltop_adj.tsv"))
        refbot = read_tsv_matrix_na(joinpath(BASELINES, "strat_mlbottom_adj.tsv"))
        @test maximum(abs.(aadj.branch_top .- reftop)) < 1e-6
        ok = .!isnan.(refbot)
        @test maximum(abs.(aadj.branch_bottom[ok] .- refbot[ok])) < 1e-6

        # adjacency is stricter than areas_allowed: states outside the chain are
        # dropped, so the constrained fit differs from the unconstrained one
        @test !isapprox(lladj, ll0; atol = 1e-3)

        # the two constraints intersect
        aa = read_areas_allowed(joinpath(exdir, "areas_allowed.txt"))
        dm = BioGeoBEARS._read_matrix_blocks(dmfn, "dispersal_multipliers")
        strat_aa = Stratification(read_times(timesfn); dispersal_multipliers = dm,
                                  areas_allowed = aa)
        both = Stratification(read_times(timesfn); dispersal_multipliers = dm,
                              areas_allowed = aa, areas_adjacency = blocks)
        for i in 1:5
            @test BioGeoBEARS._allowed_states(ss, both, i) ==
                  intersect(BioGeoBEARS._allowed_states(ss, strat_adj, i),
                            BioGeoBEARS._allowed_states(ss, strat_aa, i))
        end

        # --- fitting forwards the fixnode --------------------------------------
        f = fit_model(tr, ss, tipcl, Model(:DEC), strat; maxeval = 200,
                      calc_ancprobs = true, fixnode = 23, fixlikes = onehot(2))
        @test f.loglik ≈ stratified_loglikelihood(tr, ss, tipcl, f.model, strat;
                                                  fixnode = 23, fixlikes = onehot(2)) atol = 1e-9
        @test argmax(view(ancprobs(f).branch_top, 23, :)) == 2
    end

    # -----------------------------------------------------------------------
    # M6 imperfect detection and fossils
    # -----------------------------------------------------------------------

    @testset "M6 detection model" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))
        det = read_detections(joinpath(DATA, "Psychotria_detections_v1.txt"))
        ctl = read_controls(joinpath(DATA, "Psychotria_controls_v1.txt"))

        @test det isa DetectionCounts
        @test ntips(det) == 19
        @test nareas(det) == 4
        @test det.areanames == ["K", "O", "M", "H"]
        @test ctl.otunames == det.otunames
        @test size(det.counts) == (19, 4)

        # every value in the grid matches R, including the -Inf cases
        grid = CSV.read(joinpath(BASELINES, "det_calc_obs_like.csv"), DataFrame)
        @test nrow(grid) == 480
        maxdiff = 0.0
        for row in eachrow(grid)
            ours = calc_obs_like(row.truly_present, row.obs_target, row.obs_all;
                                 mean_frequency = row.mf, dp = row.dp, fdp = row.fdp)
            @test isequal(ours, row.lnL) || abs(ours - row.lnL) < 1e-12
            if isfinite(ours) && isfinite(row.lnL)
                maxdiff = max(maxdiff, abs(ours - row.lnL))
            end
        end
        @test maxdiff < 1e-12

        ss = StateSpace(det.areanames; max_range_size = 4, include_null_range = true)
        @test nstates(ss) == 16

        for (tag, mf, dp, fdp) in [("mf0.1_dp1_fdp0", 0.1, 1.0, 0.0),
                                   ("mf0.2_dp0.5_fdp0.02", 0.2, 0.5, 0.02)]
            ours = tiplikes_wDetectionModel(tr, ss, det, ctl;
                                            mean_frequency = mf, dp = dp, fdp = fdp)
            ref = read_tsv_matrix(joinpath(BASELINES, "det_tiplikes_$tag.tsv"))
            @test size(ours) == (19, 16)
            @test maxabs(ours, ref) < 1e-12

            oursraw = tiplikes_wDetectionModel(tr, ss, det, ctl;
                                               mean_frequency = mf, dp = dp, fdp = fdp,
                                               return_lnls = false)
            refraw = read_tsv_matrix(joinpath(BASELINES, "det_raw_$tag.tsv"))
            @test maxabs(oursraw, refraw) < 1e-14

            # the null range is given zero probability
            @test all(iszero, view(ours, :, 1))
        end

        # rows are matched to tips by name, not by position
        perm = reverse(1:ntips(det))
        det_rev = DetectionCounts(det.otunames[perm], det.areanames, det.counts[perm, :])
        ctl_rev = DetectionCounts(ctl.otunames[perm], ctl.areanames, ctl.counts[perm, :])
        base = tiplikes_wDetectionModel(tr, ss, det, ctl)
        @test maxabs(base, tiplikes_wDetectionModel(tr, ss, det_rev, ctl_rev)) == 0.0

        # null_range_gets_0_like=false leaves the null range with real likelihoods
        raw_null = tiplikes_wDetectionModel(tr, ss, det, ctl;
                                            mean_frequency = 0.2, dp = 0.5, fdp = 0.02,
                                            return_lnls = false,
                                            null_range_gets_0_like = false)
        row1 = findfirst(==(tr.tip_labels[1]), det.otunames)
        expected_null = Pdata_given_rangerow(falses(4), view(det.counts, row1, :),
                                             view(ctl.counts, row1, :);
                                             mean_frequency = 0.2, dp = 0.5, fdp = 0.02,
                                             return_lnls = false)
        @test expected_null > 0
        @test raw_null[1, 1] ≈ expected_null

        # dp=1, fdp=0, mf=0.1: the best state is exactly the observed range
        areas1 = findall(>(0), det.counts[row1, :])
        @test argmax(view(base, 1, :)) == state_index(ss, areas1)
        @test base[1, state_index(ss, areas1)] == 1.0

        # the matrix feeds the likelihood unchanged
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        plain = tip_condlikes(tr, tips; max_range_size = 4)
        m = Model(:DEC)
        @test isfinite(loglikelihood(tr, ss, m, base))
        @test !isapprox(loglikelihood(tr, ss, m, base),
                        loglikelihood(tr, ss, m, plain); atol = 1e-6)

        # validation
        bad = DetectionCounts(det.otunames[1:2], det.areanames, det.counts[1:2, :])
        @test_throws ArgumentError tiplikes_wDetectionModel(tr, ss, bad, ctl)
        @test_throws ArgumentError read_detections(joinpath(DATA, "nope.txt"))
    end

    @testset "M6 fossils" begin
        tr = read_tree(joinpath(DATA, "Psychotria_5.2.newick"))

        # --- add_hook matches R bit for bit ------------------------------------
        for (tipindex, depthtime, tag, name) in
                [(1, 0.5, "hook_tip1_0.5", "fossil_test"),
                 (3, 2.0, "hook_tip3_2.0", "fossil_deep")]
            ours = add_hook(tr, tr.tip_labels[tipindex], depthtime; newtipname = name)
            ref = read_tree(joinpath(BASELINES, tag * ".newick"))
            @test ours.tip_labels == ref.tip_labels
            @test ours.parent == ref.parent
            @test ours.left == ref.left
            @test ours.right == ref.right
            @test maxabs(ours.edge_length, ref.edge_length) < 1e-12
            @test ntips(ours) == ntips(tr) + 1
            @test nnodes(ours) == nnodes(tr) + 2
            check_tree(ours)

            # the fossil sits at the requested depth (minus its own 1e-7 hook)
            p = prt_table(ours)
            fn = findfirst(==(name), ours.tip_labels)
            @test p.time_bp[fn] ≈ depthtime atol = 1e-6
        end

        # a hook that overshoots the root leaves the tree alone
        @test add_hook(tr, tr.tip_labels[1], 100.0).parent == tr.parent
        @test_throws ArgumentError add_hook(tr, "not_a_tip", 1.0)
        @test_throws ArgumentError add_hook(tr, tr.tip_labels[1], -1.0)

        # --- candidate branches match R ---------------------------------------
        branches = CSV.read(joinpath(BASELINES, "det_possible_branches.csv"), DataFrame)
        for row in eachrow(branches)
            pair = parse.(Int, split(row.pair, "-"))
            ours = sort(possible_fossil_branches(tr, tr.tip_labels[pair];
                                                 stem_or_crown = row.stem_or_crown))
            @test ours == sort(parse.(Int, split(row.nodes)))
        end
        @test_throws ArgumentError possible_fossil_branches(
            tr, [tr.tip_labels[1]]; stem_or_crown = "crown")
        @test_throws ArgumentError possible_fossil_branches(
            tr, [tr.tip_labels[1], tr.tip_labels[2]]; stem_or_crown = "sideways")

        # --- random placement is reproducible and respects the bracket --------
        rec = FossilRecord(fossil_taxon = "test1", stem_or_crown = "both",
                           min_age = 1.0, max_age = 2.0, range_coded = "1000",
                           is_mrca_of = tr.tip_labels[1], and_mrca_of = tr.tip_labels[3],
                           geog_constraint = "presence-only",
                           constraint_type = "side_branch", specifier_type = "specifiers")

        t1, added1 = add_fossils_randomly(tr, [rec]; rng = StableRNG(1))
        t1b, _ = add_fossils_randomly(tr, [rec]; rng = StableRNG(1))
        @test added1 == ["fossil_test1"]
        @test t1.tip_labels == t1b.tip_labels
        @test maxabs(t1.edge_length, t1b.edge_length) == 0.0
        @test ntips(t1) == ntips(tr) + 1
        check_tree(t1)

        p1 = prt_table(t1)
        fn1 = findfirst(==("fossil_test1"), t1.tip_labels)
        @test 1.0 <= p1.time_bp[fn1] <= 2.0
        @test is_internal(t1, p1.ancestor[fn1])
        @test p1.edge_length[fn1] > 0

        # the attachment point lies inside the constraining clade
        @test findfirst(==(tr.tip_labels[1]), t1.tip_labels) !== nothing

        # a different seed moves the fossil
        t2, _ = add_fossils_randomly(tr, [rec]; rng = StableRNG(2))
        @test prt_table(t2).time_bp[findfirst(==("fossil_test1"), t2.tip_labels)] !=
              p1.time_bp[fn1]

        # direct ancestors are coded as 1e-7 hooks
        reca = FossilRecord(fossil_taxon = "anc1", min_age = 1.0, max_age = 2.0,
                            range_coded = "1000", is_mrca_of = tr.tip_labels[2],
                            and_mrca_of = tr.tip_labels[5],
                            constraint_type = "direct_ancestor",
                            specifier_type = "specifiers")
        t3, added3 = add_fossils_randomly(tr, [reca]; rng = StableRNG(7))
        p3 = prt_table(t3)
        fn3 = findfirst(==("fossil_anc1"), t3.tip_labels)
        @test added3 == ["fossil_anc1"]
        @test p3.edge_length[fn3] == 1e-7
        @test 1.0 <= p3.time_bp[fn3] <= 2.0

        # species and rank specifiers
        recs = FossilRecord(fossil_taxon = "sp1", min_age = 0.5, max_age = 1.0,
                            range_coded = "0100", taxon_specifiers = tr.tip_labels[4],
                            specifier_type = "species")
        t4, added4 = add_fossils_randomly(tr, [recs]; rng = StableRNG(3))
        @test added4 == ["fossil_sp1"]

        recg = FossilRecord(fossil_taxon = "g1", min_age = 0.5, max_age = 1.0,
                            range_coded = "0010", taxon_specifiers = "P_mariniana",
                            specifier_type = "genus")
        t5, added5 = add_fossils_randomly(tr, [recg]; rng = StableRNG(4))
        @test added5 == ["fossil_g1"]
        check_tree(t5)

        # fixnode rows are not tips and are skipped
        recf = FossilRecord(fossil_taxon = "fix", min_age = 0.5, max_age = 1.0,
                            range_coded = "0010", constraint_type = "fixnode")
        tf, addedf = add_fossils_randomly(tr, [recf]; rng = StableRNG(1))
        @test isempty(addedf)
        @test tf.parent == tr.parent

        # an impossible age bracket leaves the tree unchanged
        recx = FossilRecord(fossil_taxon = "old", min_age = 50.0, max_age = 60.0,
                            range_coded = "1000", is_mrca_of = tr.tip_labels[1],
                            and_mrca_of = tr.tip_labels[3])
        tx, addedx = add_fossils_randomly(tr, [recx]; rng = StableRNG(1))
        @test isempty(addedx)
        @test tx.parent == tr.parent

        # --- fossil ranges -----------------------------------------------------
        tips = read_lagrange_phylip(joinpath(DATA, "Psychotria_geog.data"))
        newtips = add_fossil_ranges(tips, ["fossil_test1"], [rec])
        @test ntips(newtips) == ntips(tips) + 1
        @test newtips.tipnames[end] == "fossil_test1"
        @test join(newtips.data[end, :]) == "1???"      # presence-only
        @test join(tips.data[1, :]) == "1000"           # input untouched

        recabs = FossilRecord(fossil_taxon = "abs", min_age = 1.0, max_age = 2.0,
                              range_coded = "1000", geog_constraint = "absence-only")
        newtips2 = add_fossil_ranges(tips, ["fossil_abs"], [recabs])
        @test join(newtips2.data[end, :]) == "?000"

        recboth = FossilRecord(fossil_taxon = "both", min_age = 1.0, max_age = 2.0,
                               range_coded = "1000", geog_constraint = "both")
        newtips3 = add_fossil_ranges(tips, ["fossil_both"], [recboth])
        @test join(newtips3.data[end, :]) == "1000"

        @test_throws ArgumentError add_fossil_ranges(tips, ["fossil_test1"], [recabs])
        @test_throws ArgumentError add_fossil_ranges(tips, ["nope"], [rec])

        # --- read_fossils round trip -------------------------------------------
        dir = mktempdir()
        path = joinpath(dir, "fossils.tsv")
        write(path, "fossil_taxon\tstem_or_crown\tmin_age\tmax_age\trange_coded\tis_mrca_of\tand_mrca_of\tgeog_constraint\tconstraint_type\tspecifier_type\n" *
                     "test1\tboth\t1.0\t2.0\t1000\t$(tr.tip_labels[1])\t$(tr.tip_labels[3])\tpresence-only\tside_branch\tspecifiers\n")
        recs2 = read_fossils(path)
        @test length(recs2) == 1
        @test recs2[1].fossil_taxon == "test1"
        @test recs2[1].min_age == 1.0 && recs2[1].max_age == 2.0
        @test recs2[1].geog_constraint == "presence-only"
        @test recs2[1].specifier_type == "specifiers"
        @test_throws ArgumentError read_fossils(joinpath(dir, "nope.tsv"))
        @test_throws ArgumentError FossilRecord(fossil_taxon = "x", min_age = 5.0, max_age = 1.0)

        # tables can also come from a DataFrame or named tuples
        fdf = DataFrame(fossil_taxon = ["a"], min_age = [1.0], max_age = [2.0],
                        range_coded = ["1000"])
        @test fossil_records(fdf)[1].fossil_taxon == "a"
        @test fossil_records([(fossil_taxon = "c", min_age = 1, max_age = 2)])[1].stem_or_crown ==
              "both"

        # --- the resulting tree feeds the likelihood ---------------------------
        ss = StateSpace(tips.areanames; max_range_size = 4, include_null_range = true)
        tipcl = tip_condlikes(t1, newtips; max_range_size = 4, include_null_range = true)
        @test size(tipcl) == (ntips(t1), 16)
        @test isfinite(loglikelihood(t1, ss, Model(:DEC), tipcl))
    end

    @testset "M7 R RNG stream" begin
        # Baselines come from R's own stream: set.seed(12345) followed by the
        # calls listed below.  These must match bit for bit — the whole point
        # of RRng is that BSM event counts reproduce R's.
        cases = Dict{String, Vector{Float64}}()
        for row in eachrow(CSV.read(joinpath(BASELINES, "rng_cases.csv"), DataFrame))
            push!(get!(cases, row.case, Float64[]), row.value)
        end

        r = RRng(12345)
        @test r_runif(r, 200) == cases["unif"]

        r = RRng(12345)
        @test [r_rexp(r, 2.0) for _ in 1:100] == cases["exp2"]

        r = RRng(12345)
        @test [r_rexp(r, 0.5) for _ in 1:100] == cases["exp05"]

        sample_cases = [
            ("s_w4", [0.1, 0.2, 0.3, 0.4], 10, true),
            ("s_eq4", fill(0.25, 4), 5, true),
            ("s_eq3", fill(1 / 3, 3), 5, true),
            ("s_eq16", fill(1 / 16, 16), 5, true),
            ("s_z4", [0.0, 0.5, 0.5, 0.0], 5, true),
            ("s_nor", fill(1.0, 10), 10, false),
            ("s_norw", collect(1.0:10.0), 4, false),
        ]
        for (case, prob, k, replace) in sample_cases
            r = RRng(12345)
            @test Float64.(r_sample_prob(r, prob, k; replace = replace)) ==
                  cases[case]
        end

        # the scalar form agrees with the vector form
        r = RRng(12345)
        @test r_sample_prob(r, [0.1, 0.2, 0.3, 0.4]) == 2
        @test r_sample_prob(RRng(12345), [0.1, 0.2, 0.3, 0.4], 1)[1] == 2

        # seeding is reproducible and seeds differ
        @test r_runif(RRng(1), 5) == r_runif(RRng(1), 5)
        @test r_runif(RRng(1), 5) != r_runif(RRng(2), 5)

        # uniforms stay inside (0, 1) even after many draws
        r = RRng(99)
        u = r_runif(r, 5000)
        @test all(x -> 0.0 < x < 1.0, u)

        @test_throws ArgumentError r_sample_prob(RRng(1), [-0.1, 0.5])
        @test_throws ArgumentError r_sample_prob(RRng(1), [0.0, 0.0])
    end

    @testset "M7 expm is bit-exact with R" begin
        # `exp(tQ)` for the 3-taxon DEC d=0.3, e=0.1 case, written by R's
        # `expokit_dgpadm_Qmat()`.  The port must agree to the last bit: BSM
        # sampling flips on a one-ulp difference (plan §5.4).
        tr = read_tree(joinpath(DATA, "bsm_3taxa.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "bsm_3taxa.data"))
        ss = StateSpace(tips.areanames; max_range_size = 3)
        m = Model(:DEC); m["d"] = 0.3; m["e"] = 0.1
        q = qmatrix(ss, m; r_float32 = true)

        for t in (1, 2)
            ref = permutedims(reduce(hcat, [
                [parse(Float64, x) for x in split(l, '\t')]
                for l in readlines(joinpath(BASELINES, "expm_dgpadm_t$t.tsv"))]))
            @test transition_matrix(q, Float64(t)) == ref
        end

        # R's sum() accumulates in a long double; Julia's pairwise sum does not
        # agree, and the difference is enough to move `branch_bottom`.
        v = [0.0, 0.15677703558496056, 0.15677703558496056, 0.019979337182182439,
             0.26835039643347425, 0.053358594196685066, 0.053358594196685052,
             0.082765424968277074]
        @test r_sum(v) == 0.79136641814722497
        @test sum(v) != r_sum(v)
        @test r_sum(v, false) == sum(v)
    end

    @testset "M7 stratified tree pieces match R master_table" begin
        # R's `section_the_tree()` master table for the 3-taxon "islands coming
        # up" example: two strata, three subbranches in the youngest, one
        # subtree below.  `subedge_length` is the length inside the stratum.
        d = joinpath(DATA, "strat_m3")
        tr = read_tree(joinpath(d, "tree.newick"))
        strat = Stratification(timesfn = joinpath(d, "timeperiods.txt"),
                               areas_allowed_fn = joinpath(d, "areas_allowed_noC.txt"))
        tbl = tree_pieces(tr, strat)

        lines = readlines(joinpath(BASELINES, "strat_master_table_m3areas.tsv"))[2:end]
        rows = [split(l, '\t') for l in lines]
        got = [(r.node, i, r.piecenum, r.piececlass, r.subnode_type, r.subedge_length,
                r.subtime_bp, r.time_top, r.time_bot)
               for (i, ps) in enumerate(tbl) for r in ps]

        @test length(got) == length(rows)
        for (g, row) in zip(got, rows)
            @test g[1] == parse(Int, row[1])          # node
            @test g[2] == parse(Int, row[2])          # stratum
            @test g[3] == parse(Int, row[3])          # piecenum
            @test g[4] == row[4]                      # piececlass
            @test g[5] == row[5]                      # subnode_type
            if row[6] == "NA"
                @test isnan(g[6])
            else
                @test g[6] ≈ parse(Float64, row[6]) atol = 1e-12
            end
            # R accumulates SUBtime_bp and the stratum's top by subtraction,
            # so those two drift at the 1e-14 level; the structure is exact.
            @test g[7] ≈ parse(Float64, row[7]) atol = 1e-12
            @test g[8] ≈ parse(Float64, row[8]) atol = 1e-12
            @test g[9] ≈ parse(Float64, row[9]) atol = 1e-12
        end

        # one subtree piece in the oldest stratum, rooted at the tree's root
        @test all(r.piececlass == "subbranch" for r in tbl[1])
        @test length(tbl[2]) == 5
        @test count(r -> r.subnode_type == "root", tbl[2]) == 1
        @test [r.node for r in tbl[2] if r.subnode_type == "root"] == [root_node(tr)]

        # `stratified_bsm_inputs` restricts each stratum's rate matrix to the
        # states its areas_allowed block permits: C is unavailable in the old
        # stratum, so {null, A, B, AB} survive out of eight.
        tips = read_lagrange_phylip(joinpath(d, "geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 3)
        tipcl = tip_condlikes(tr, tips; max_range_size = 3)
        m = Model(:DEC); m["d"] = 0.1; m["e"] = 0.05
        ap = stratified_ancestral_probs(tr, ss, tipcl, m, strat)
        bi = stratified_bsm_inputs(tr, ss, m, strat, ap)

        @test length(bi.q) == 2
        @test size(bi.q[1]) == (8, 8)
        @test size(bi.q[2]) == (4, 4)
        @test bi.keep[1] == collect(1:8)
        @test length(bi.keep[2]) == 4
        for i in 1:2
            @test all(abs.(vec(sum(bi.q[i]; dims = 2))) .< 1e-12)
            @test length(bi.sub_P[i]) == length(bi.pieces[i])
        end
        # the youngest stratum is three subbranches of 0.1, each with its
        # own transition matrix; the oldest is a single subtree
        @test all(p -> p.kind == :subbranch, bi.pieces[1])
        @test all(abs.(bi.pieces[1][k].length - 0.1) < 1e-12 for k in 1:3)
        @test size(bi.sub_P[1][1]) == (8, 8)
        @test all(abs.(vec(sum(bi.sub_P[1][1]; dims = 2)) .- 1) .< 1e-12)
        @test bi.pieces[2][1].kind == :subtree
        @test size(bi.sub_P[2][1]) == (0, 0)
        @test sum(bi.root_probs) ≈ 1

        # branch-top conditioning: the row of P for the bottom state, weighted
        # by the downpass likelihoods, with the null range zeroed and the whole
        # vector normalised.
        top_likes = fill(1.0, size(bi.q[1], 1))
        p = stratified_branch_top_probs(bi, 1, 1, bi.keep[1][5], top_likes)
        @test sum(p) ≈ 1
        @test p[1] == 0                        # null range cannot survive
        row = copy(bi.sub_P[1][1][5, :])
        row[1] = 0.0
        @test p ≈ row ./ sum(row) atol = 1e-12
        @test_throws ArgumentError stratified_branch_top_probs(bi, 2, 1, bi.keep[2][1], top_likes)
        @test_throws ArgumentError stratified_branch_top_probs(bi, 2, 1, 4, top_likes)
        @test stratified_branch_top_probs(bi, 1, 1, bi.keep[1][5], top_likes) ==
              stratified_branch_top_probs(bi, 1, 1, bi.keep[1][5], top_likes)

        # R's seed chain: accumulated, incremented by counter*90127, wrapped
        @test next_piece_seed(12345, 2) == 12345 + 2 * 90127
        @test next_piece_seed(2147483000, 10) ==
              mod(2147483000 + 10 * 90127, 2147483647)
        @test next_piece_seed(next_piece_seed(12345, 2), 3) ==
              mod(12345 + 2 * 90127 + 3 * 90127, 2147483647)

        # A whole stratified map: every branch's events must replay from its
        # bottom state to its top state, and the tips must end on their data.
        r = stochastic_map_stratified(tr, ss, m, strat, tipcl; seedval = 12345)
        @test r.node_states[1] == 4 && r.node_states[2] == 4   # human, chimp in C
        @test r.node_states[3] == 5                            # gorilla in AB
        for n in 1:nnodes(tr)
            cur = r.branch_bottom_states[n]
            r.branch_bottom_states[n] == 0 && continue
            for e in r.branch_events[n]
                @test e.from_state == cur
                cur = e.to_state
            end
            @test cur == r.node_states[n]
        end
        # same seed, same map; a different seed gives a different map
        r2 = stochastic_map_stratified(tr, ss, m, strat, tipcl; seedval = 12345)
        @test r2.node_states == r.node_states
        @test length(r2.branch_events) == length(r.branch_events)
        r3 = stochastic_map_stratified(tr, ss, m, strat, tipcl; seedval = 7)
        @test r3.node_states != r.node_states ||
              [length(e) for e in r3.branch_events] != [length(e) for e in r.branch_events]
        # the youngest stratum's events fall inside the 0.1-long tip slices;
        # node 3 also carries the events of its older subtree segment
        for n in 1:2
            for e in r.branch_events[n]
                @test 0 <= e.abs_event_time <= 0.1 + 1e-12
            end
        end
        # Against R's runBSM() for the same seed.  R's
        # `table_w_anagenetic_events` round-trips its numbers through text
        # (~16 digits), so the ages are compared with a tolerance; the
        # underlying simulation is bit-identical -- an instrumented run of
        # R's stochastic_map_branch() prints the same waiting time Julia does.
        @test r.node_states == [4, 4, 5, 3, 3]
        @test sum(length, r.branch_events) == 5
        @test [e.event_type for e in r.branch_events[1]] == ["d", "e"]
        @test [e.event_type for e in r.branch_events[2]] == ["d", "e"]
        @test [e.event_type for e in r.branch_events[3]] == ["d"]
        @test [e.trynum for e in r.branch_events[1]] == [6385, 6385]
        # node 2 is unreachable by forward simulation within maxtries, so R
        # falls back to a manual history and labels the events maxtries + 2
        @test [e.trynum for e in r.branch_events[2]] == [40002, 40002]
        @test [e.abs_event_time for e in r.branch_events[1]] ≈
              [0.087206047400832196, 0.064667761041751995] atol = 1e-12
        @test [e.abs_event_time for e in r.branch_events[2]] ≈
              [0.032171196723356797, 0.027499274094589101] atol = 1e-12
        @test r.branch_events[3][1].abs_event_time ≈ 1.9356794227637999 atol = 1e-12
    end

    @testset "M7 ape pruningwise order" begin
        for line in readlines(joinpath(BASELINES, "bsm_pruning_order.tsv"))[2:end]
            newick, expected = split(line, '\t')
            @test BioGeoBEARS._ape_pruning_nodes(read_tree(newick)) == parse.(Int, split(expected, ','))
        end
    end

    @testset "M7 zero-rate branch histories" begin
        ss = StateSpace(["A", "B"]; max_range_size = 2)
        q = zeros(nstates(ss), nstates(ss))
        rng = BioGeoBEARS.RRng(7)
        before = rng.mti
        @test isempty(BioGeoBEARS.stochastic_map_branch(rng, 1.0, 2, 2, q, ss))
        @test rng.mti == before
        @test_throws ErrorException BioGeoBEARS.stochastic_map_branch(
            BioGeoBEARS.RRng(7), 1.0, 2, 4, q, ss; maxtries = 1, manual_history = false)
        events = BioGeoBEARS.stochastic_map_branch(
            BioGeoBEARS.RRng(7), 1.0, 2, 4, q, ss; maxtries = 1)
        @test length(events) == 1
        @test events[1].event_txt == "A->AB"
    end

    include("bsm_unstratified_tests.jl")

    @testset "M7 BSM event counts match R" begin
        # 50 stochastic maps per parameter set, then R's source-area simulation
        # and event counting.  This exercises the whole M7 chain: RNG stream,
        # map sampling, source-area draws and the counting tables.
        tr = read_tree(joinpath(DATA, "bsm_3taxa.newick"))
        tips = read_lagrange_phylip(joinpath(DATA, "bsm_3taxa.data"))
        ss = StateSpace(tips.areanames; max_range_size = 3)
        tipcl = tip_condlikes(tr, tips; max_range_size = 3)
        n = nareas(ss)

        read_mat = function (fn)
            lines = readlines(fn)
            hdr = split(lines[1], '\t')
            rows = [split(l, '\t') for l in lines[2:end]]
            mat = [parse(Float64, r[j]) for r in rows, j in 2:length(hdr)]
            (row = [r[1] for r in rows], col = hdr[2:end], mat = mat)
        end

        counts_for = function (d, e)
            m = Model(:DEC); m["d"] = d; m["e"] = e
            q = qmatrix(ss, m; r_float32 = true)
            w = cladogenesis_weights(ss, m; r_float32 = true)
            dp = downpass(tr, ss, tipcl, q, w; r_float32 = true)
            ap = ancestral_probs(tr, ss, tipcl, q, w; r_float32 = true)
            maps, rng = run_bsm(tr, ss, q, w, dp, ap; seedval = 12345, n_maps = 50,
                                include_null_range = true, r_float32 = true)
            tables = [bsm_tables(b, tr) for b in maps]
            simulate_source_areas!(tables, ss, rng)
            return tables, count_ana_clado_events(tables, ss)
        end

        for (tag, d, e) in (("de005_002", 0.05, 0.02), ("de03_01", 0.3, 0.1))
            tables, c = counts_for(d, e)

            ref = read_mat(joinpath(BASELINES, "bsm_counts_summary_$tag.tsv"))
            @test ref.row == ["means", "stdevs", "sums"]
            @test c.summary_counts_BSMs ≈ ref.mat atol = 1e-12

            for (nm, got) in (("all", c.all_dispersals_counts_fromto_means),
                              ("d", c.d_counts_fromto_means),
                              ("a", c.a_counts_fromto_means),
                              ("founder", c.founder_counts_fromto_means))
                r = read_mat(joinpath(BASELINES, "bsm_counts_$(nm)_$tag.tsv"))
                @test got ≈ r.mat atol = 1e-12
            end

            er = read_mat(joinpath(BASELINES, "bsm_counts_e_$tag.tsv"))
            etot = sum(c.e_counts_rectangle; dims = 1)
            @test vec(etot) ≈ vec(er.mat) atol = 1e-12

            for (nm, got, names) in (("vic", c.unique_vic_counts, c.vic_events),
                                     ("sub", c.unique_sub_counts, c.sub_events),
                                     ("sym", c.unique_sym_counts, c.sym_events))
                rl = readlines(joinpath(BASELINES, "bsm_counts_$(nm)_$tag.tsv"))[2:end]
                if isempty(rl)
                    @test isempty(names)
                else
                    @test names == [String(split(l, '\t')[1]) for l in rl]
                    @test vec(mean(got; dims = 1)) ≈
                          [parse(Float64, split(l, '\t')[2]) for l in rl] atol = 1e-12
                end
            end
        end

        # Map m of run_bsm uses seedval + m.
        m = Model(:DEC); m["d"] = 0.05; m["e"] = 0.02
        q = qmatrix(ss, m; r_float32 = true)
        w = cladogenesis_weights(ss, m; r_float32 = true)
        dp = downpass(tr, ss, tipcl, q, w; r_float32 = true)
        ap = ancestral_probs(tr, ss, tipcl, q, w; r_float32 = true)
        maps, _ = run_bsm(tr, ss, q, w, dp, ap; seedval = 100, n_maps = 3,
                          include_null_range = true, r_float32 = true)
        for k in 1:3
            @test maps[k].node_states ==
                  stochastic_map(tr, ss, q, w, dp, ap; seedval = 100 + k,
                                 include_null_range = true, r_float32 = true).node_states
        end

        # calc_bsm_mean_node_states: one-hot per map, mean = sums / nmaps
        st = calc_bsm_mean_node_states(maps, tr, ss)
        @test size(st.state_counts) == (nnodes(tr), nstates(ss), 3)
        @test all(sum(st.state_counts[n, :, i]) == 1
                  for n in 1:nnodes(tr), i in 1:3)
        @test st.mean_bsm_probs == st.sum_bsm_counts ./ 3
        @test all(0 .<= st.mean_bsm_probs .<= 1)

        # source areas are only drawn for events that have a destination
        tables, c2 = counts_for(0.05, 0.02)
        for t in tables
            for r in t.ana
                if r.dispersal_to in ("", "-")
                    @test r.dispersal_from == ""
                else
                    @test r.dispersal_from in ss.areanames
                end
            end
        end

        # print_counts_lists_to_file writes every 2-D table and skips vectors
        dir = mktempdir()
        files = print_counts_lists_to_file(c2, "m0"; dir = dir)
        @test !isempty(files)
        @test all(isfile, files)
        @test any(endswith(f, "summary_counts_BSMs.txt") for f in files)
        @test !any(endswith(f, "founder_totals_list.txt") for f in files)

        # uniquify_clado_events puts the smaller daughter on the left, and
        # sorts equal-size daughters (mirroring R's swap rules)
        t1 = BSMTables([CladoEventRow(1, 1.0, "vicariance (v)", "AB->B,A", "", "")], AnaEventRow[])
        @test uniquify_clado_events(t1)[1].event_txt == "AB->A,B"
        t2 = BSMTables([CladoEventRow(1, 1.0, "subset (s)", "ABC->AB,C", "", "")], AnaEventRow[])
        @test uniquify_clado_events(t2)[1].event_txt == "ABC->C,AB"
    end

    @testset "M4b Cyrtandra (59 tips, 128 states) matches R" begin
        # The package's own Cyrtandra example: 7 areas, max_range_size 7, so
        # 128 states and 116 matrix exponentials per likelihood.  R's
        # calc_loglike_for_optim takes about 1030 ms for this call and returns
        # -64.945711038449; Julia takes about 110 ms.
        d = joinpath(DATA, "cyrtandra")
        tr = read_tree(joinpath(d, "Cyrtandra.newick"))
        tips = read_lagrange_phylip(joinpath(d, "Cyrtandra_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 7)
        @test ntips(tr) == 59
        @test nstates(ss) == 128
        tipcl = tip_condlikes(tr, tips; max_range_size = 7)
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        @test loglikelihood(tr, ss, m, tipcl) ≈ -64.945711038449 atol = 1e-5
    end
    @testset "M2 Psychotria dispersal multipliers match R" begin
        # The package's own Psychotria_M2 example: dispersal is only allowed
        # eastward (K->O->M->H), so every rate that crosses an area backwards
        # is zeroed.  R's calc_loglike_for_optim with the same multiplier file
        # returns -38.506060390949 at d = e = 0.01.
        d = joinpath(DATA, "psychotria_m2")
        tr = read_tree(joinpath(d, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(d, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        lines = filter(l -> !isempty(strip(l)) && strip(l) != "END",
                       readlines(joinpath(d, "dispersal_multipliers.txt")))
        dm = permutedims(reduce(hcat, [parse.(Float64, split(l)) for l in lines[2:end]]))
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        q = qmatrix(ss, m; dispersal_multipliers = dm)
        w = cladogenesis_weights(ss, m; dispersal_multipliers = dm)
        @test q[3, 5] == 0.0          # K->M is forbidden
        @test downpass(tr, ss, tipcl, q, w).loglik ≈
              -38.506060390949 atol = 1e-5
    end

    @testset "M4b distance-weighted dispersal matches R" begin
        # The package's own Psychotria_M4_dists example: a distances matrix
        # (K-O 80, O-M 60, M-H 40, ...) raised to the model's `x` exponent.
        # R returns -36.871816095556 at x = 0 (no effect) and -49.958099552586
        # at x = -1 (inverse-distance weighting), both at d = e = 0.01.
        d = joinpath(DATA, "psychotria_m4")
        tr = read_tree(joinpath(d, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(d, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        _, dists = read_area_matrix(joinpath(d, "Hawaii_KOMH_distances.txt"))
        m = Model(:DEC); m["d"] = 0.01; m["e"] = 0.01
        for (x, ref) in ((0.0, -36.871816095556), (-1.0, -49.958099552586))
            m["x"] = x
            q = qmatrix(ss, m; distances = dists)
            w = cladogenesis_weights(ss, m; distances = dists)
            @test downpass(tr, ss, tipcl, q, w).loglik ≈ ref atol = 1e-5
        end
        # x = 0 must be identical to the plain DEC likelihood
        m["x"] = 0.0
        q0 = qmatrix(ss, m)
        @test qmatrix(ss, m; distances = dists) == q0
    end
    include("bsm_stratified_tests.jl")

    @testset "M7b stratified 19-tip DEC+J BSM matches R (Psychotria_M0strat)" begin
        # The same 19-tip stratified example with j = 0.1.  Verified end to
        # end against R's stochastic_mapping_on_stratified() at seed 12345:
        # all 37 node states, all 18 cladogenesis events and all 6
        # anagenetic events (node, type, direction and absolute time) agree.
        # DEC+J is the case that exposed R's prt() time_bp rounding: a 1e-8
        # piece-edge difference breaks exact founder-scenario ties in the
        # split sampling, so the piece geometry has to be bit-identical.
        d = joinpath(DATA, "psychotria_m0strat")
        tr = read_tree(joinpath(d, "Psychotria_5.2.newick"))
        tips = read_lagrange_phylip(joinpath(d, "Psychotria_geog.data"))
        ss = StateSpace(tips.areanames; max_range_size = 4)
        tipcl = tip_condlikes(tr, tips; max_range_size = 4)
        strat = Stratification(timesfn = joinpath(d, "timeperiods.txt"),
                               areas_allowed_fn = joinpath(d, "areas_allowed_all1s.txt"))
        m = Model(:DECJ); m["d"] = 0.1; m["e"] = 0.05; m["j"] = 0.1
        @test stratified_loglikelihood(tr, ss, tipcl, m, strat;
                                       r_float32 = true) ≈ -35.681047557044 atol = 1e-6
        mp = stochastic_map_stratified(tr, ss, m, strat, tipcl;
                                       seedval = 12345, maxtries = 40000)
        @test mp.node_states ==
              [4, 4, 3, 3, 4, 4, 3, 2, 2, 4, 5, 3, 2, 2, 2, 2, 2, 2, 3,
               7, 2, 2, 2, 4, 4, 4, 4, 3, 4, 2, 2, 2, 5, 5, 2, 12, 2]
        clado = Dict(
            20 => "KM->K,KM", 21 => "K->K,K", 22 => "K->K,K", 23 => "K->M,K",
            24 => "M->M,M", 25 => "M->M,M", 26 => "M->M,O", 27 => "M->M,M",
            28 => "O->O,O", 29 => "M->M,M", 30 => "K->K,K", 31 => "K->K,K",
            32 => "K->H,K", 33 => "H->H,O", 34 => "H->M,H", 35 => "K->K,K",
            36 => "KOM->K,OM", 37 => "K->K,K")
        for n in 20:37
            @test mp.clado_event_txt[n] == clado[n]
        end
        ana = Tuple{Int,Float64,String,String}[]
        for n in 1:37, e in mp.branch_events[n]
            push!(ana, (n, e.abs_event_time, e.event_type, e.event_txt))
        end
        sort!(ana, by = a -> (a[2], a[1]))
        @test length(ana) == 6
        @test [a[1] for a in ana] == [7, 13, 7, 19, 13, 36]
        @test [a[3] for a in ana] == ["e", "e", "d", "e", "d", "d"]
        @test [a[4] for a in ana] ==
              ["OM->O", "KO->K", "M->OM", "OM->O", "K->KO", "KM->KOM"]
        @test [a[2] for a in ana] ≈
              [0.0999879794893742, 0.543169363960624, 1.0072436070505,
               2.66967313469554, 2.88995751091862, 3.15028184447438] atol = 1e-7

        # Marginal ancestral probabilities.  R's uppass COO scales founder
        # weights by a leaked loop variable `m` (stratified_v1.R:3753 ->
        # :2946); DEC is immune (j = 0), DEC+J is not.  Verified to 2e-9
        # against a likelihood-only R run (no map to perturb the tables).
        ap = stratified_ancestral_probs(tr, ss, tipcl, m, strat; r_float32 = true)
        @test ap.branch_top[20, :] ≈ [
            0, 0.43279897801271977, 0.053203836229988186,
            0.0054837294588218274, 0.0041150667734979832,
            0.28567727186633496, 0.053934836922778916,
            0.04217550069984901, 0.0070891572448204326,
            0.0050665841865816762, 0.00081571951915761188,
            0.049451267018975807, 0.039834024749855278,
            0.0087402794577290509, 0.0015127550113287274,
            0.010100992847560768] atol = 1e-7
        @test ap.branch_top[23, :] ≈ [
            0, 0.39043914179014716, 0.12256140790682127,
            0.062058674000337821, 0.002264096231600871,
            0.17949054603149689, 0.096168046444113489,
            0.0036609973956057752, 0.022787700863960335,
            0.0014924836727114089, 0.00083850921878815912,
            0.10079030763130518, 0.0068311185834728835,
            0.0039255304308839518, 0.0011312622095032889,
            0.0055601775892515932] atol = 1e-7
        @test ap.branch_top[32, :] ≈ [
            0, 0.42011110750537989, 0.17041566433987085,
            0.089007686034797404, 0.084072278191504812,
            0.081702403987011785, 0.042875548468684195,
            0.04079785529856516, 0.010198404239383091,
            0.0097769543941155154, 0.0035292398937365418,
            0.016601033509279349, 0.015983087567648671,
            0.0056457327376437843, 0.0022288102317330723,
            0.0070541936006458155] atol = 1e-7
        @test ap.branch_top[34, :] ≈ [
            0, 0.0080926973641613463, 0.0081285091218583887,
            0.39905752986850151, 0.39389575571806829,
            0.00011430573025261812, 0.0035145125552929284,
            0.00347667144676029, 0.0036349936306331724,
            0.0035992390740344658, 0.16661446964346019,
            0.00013607571091766514, 0.00013461734105982856,
            0.0046343731363867801, 0.0047704258375873658,
            0.00019582382102510139] atol = 1e-7
    end
end
