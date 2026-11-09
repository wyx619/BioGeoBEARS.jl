using BioGeoBEARS, Test, CSV, DataFrames

@testset "Time-varying source areas match R" begin
    ss = StateSpace(["A", "B", "C"])
    young = ones(3, 3); young[1, 3] = 0
    old = ones(3, 3); old[2, 3] = 0
    times = [1.0, 3.0]
    clado = [CladoEventRow(i, t, "founder (j)", "AB->AB,C", "C", "") for (i, t) in enumerate((0.5, 1.0, 2.0))]
    ana = [AnaEventRow(i, 1, 0.0, t, "d", "AB", "ABC", "C", "", "-") for (i, t) in enumerate((0.5, 1.0, 2.0))]
    tables = [BSMTables(clado, ana)]
    simulate_source_areas!(tables, ss, RRng(42); dmat = [young, old], times = times)
    ref = CSV.read(joinpath(@__DIR__, "baselines", "source_strata.tsv"), DataFrame; delim = '\t')
    @test vcat([r.dispersal_from for r in clado], [r.dispersal_from for r in ana]) == ref.source
    @test_throws ArgumentError simulate_source_areas!(tables, ss, RRng(42); dmat = [young, old])
    @test_throws ArgumentError simulate_source_areas!(tables, ss, RRng(42); dmat = [young, old], times = [1.0, 1.0])
    @test_throws ArgumentError simulate_source_areas!(tables, ss, RRng(42); dmat = [young, old], times = [1.0, 2.0])

    young[1, 3] = 0.2; young[2, 3] = 0.8
    old[1, 3] = 0.8; old[2, 3] = 0.2
    clado = [CladoEventRow(i, t, "founder (j)", "AB->AB,C", "C", "") for (i, t) in enumerate(repeat([0.5, 1.0, 2.0], 4))]
    ana = [AnaEventRow(i, 1, 0.0, t, "d", "AB", "ABC", "C", "", "-") for (i, t) in enumerate(repeat([0.5, 1.0, 2.0], 4))]
    simulate_source_areas!([BSMTables(clado, ana)], ss, RRng(42); dmat = [young, old], times = times)
    @test vcat([r.dispersal_from for r in clado], [r.dispersal_from for r in ana]) ==
          readlines(joinpath(@__DIR__, "baselines", "source_strata_weighted.txt"))[2:end]
end
