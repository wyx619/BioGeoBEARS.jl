using BioGeoBEARS, Test

@testset "Event counts by time period" begin
    ss = StateSpace(["A", "B", "C"])
    times = (0.5, 1.0, 1.5, 2.0)
    clado = [CladoEventRow(i, t, "sympatry (y)", "A->A,A", "", "") for (i, t) in enumerate(times)]
    ana = [AnaEventRow(i, 1, 0.0, t, "d", "A", "AB", "B", "A", "-") for (i, t) in enumerate(times)]
    tables = [BSMTables(clado, ana), BSMTables(copy(clado), copy(ana))]
    result = count_ana_clado_events(tables, ss; timeperiod = (1.0, 2.0))
    @test result.sympatry_totals_list == [2, 2]
    @test result.d_totals_list == [2, 2]
    @test result.summary_counts_BSMs[3, 12] == 8
    @test count_ana_clado_events(tables, ss; timeperiod = [2.0, 1.0]).summary_counts_BSMs == result.summary_counts_BSMs
    @test count_ana_clado_events(tables, ss; timeperiod = (1.0, 1.0)).summary_counts_BSMs[3, 12] == 0
    @test length(tables[1].clado) == length(tables[1].ana) == 4
    @test_throws ArgumentError count_ana_clado_events(tables, ss; timeperiod = (1.0,))
    @test_throws ArgumentError count_ana_clado_events(tables, ss; timeperiod = (1.0, Inf))
end
