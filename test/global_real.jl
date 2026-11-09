using BioGeoBEARS, Test, CSV, DataFrames

@testset "Global search on Psychotria" begin
    data = joinpath(@__DIR__, "data")
    tr = read_tree(joinpath(data, "Psychotria_5.2.newick"))
    tips = read_lagrange_phylip(joinpath(data, "Psychotria_geog.data"))
    ss = StateSpace(tips.areanames; max_range_size = 4)
    refs = CSV.read(joinpath(@__DIR__, "baselines", "ml_psychotria.tsv"), DataFrame)
    fit_model(tr, ss, tips, Model(:DEC); algorithms = (:LN_BOBYQA,), maxeval = 10, global_maxeval = 10)
    for (spec, label) in ((:DEC, "DEC"), (:DECJ, "DEC+J"))
        target = only(refs[refs.model .== label, :lnL])
        localfit = nothing
        local_time = @elapsed localfit = fit_model(tr, ss, tips, Model(spec);
            algorithms = (:LN_BOBYQA,), maxeval = 400)
        for budget in (100, 400)
            fitted = nothing
            elapsed = @elapsed fitted = fit_model(tr, ss, tips, Model(spec);
                algorithms = (:LN_BOBYQA,), maxeval = 400, global_maxeval = budget)
            @test fitted.loglik >= localfit.loglik - 1e-8
            @test fitted.loglik ≈ target atol = 1e-3
            println("$label global=$budget local_lnL=$(localfit.loglik) lnL=$(fitted.loglik) local_s=$(round(local_time, digits=3)) total_s=$(round(elapsed, digits=3))")
        end
    end
end
