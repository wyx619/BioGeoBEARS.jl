# Model selection criteria.

"""
    aic(loglik, numparams) -> Float64
    aic(f::BGBFit) -> Float64

Akaike Information Criterion, `2k - 2 lnL`.  Lower is better.
"""
aic(loglik::Real, numparams::Integer) = 2 * numparams - 2 * loglik

aic(f::BGBFit) = aic(f.loglik, f.numparams)

"""
    aicc(loglik, numparams, samplesize) -> Float64
    aicc(f::BGBFit) -> Float64

Small-sample AIC: `AIC + 2k(k+1)/(n - k - 1)`.  `samplesize` is the number of
tips, as in R's `getAICc()`.  Requires `k < n - 1`, otherwise the correction
has no meaning and an `ArgumentError` is thrown.
"""
function aicc(loglik::Real, numparams::Integer, samplesize::Integer)
    numparams >= samplesize - 1 &&
        throw(ArgumentError("AICc needs more samples than parameters " *
                            "(k = $numparams, n = $samplesize)"))
    return aic(loglik, numparams) +
           2 * numparams * (numparams + 1) / (samplesize - numparams - 1)
end

aicc(f::BGBFit) = aicc(f.loglik, f.numparams, f.ntips)

"""
    akaike_weights(values) -> Vector{Float64}

Akaike weights `exp(-Δ/2) / Σ exp(-Δ/2)` for a vector of AIC (or AICc) values.
They sum to one; the best model has the largest weight.  This is R's
`getAIC_weight_for_model1()` applied to every row.
"""
function akaike_weights(values::AbstractVector{<:Real})
    isempty(values) && return Float64[]
    best = minimum(values)
    rel = exp.(-0.5 .* (values .- best))
    return rel ./ sum(rel)
end

"""
    summary_table(fits; sort_by = :AIC) -> DataFrame

The model-comparison table R builds after a run: one row per fit, with the
log-likelihood, the free parameters, `d`/`e`/`j`, `AIC`, `AICc`, their
differences from the best model, and the Akaike weights.
"""
function summary_table(fits::AbstractVector{BGBFit}; sort_by::Symbol = :AIC)
    isempty(fits) && throw(ArgumentError("summary_table needs at least one fit"))
    rows = [fit_summary(f) for f in fits]

    df = DataFrame(model = [r.model for r in rows],
                   lnL = [r.lnL for r in rows],
                   numparams = [r.numparams for r in rows],
                   d = [r.d for r in rows],
                   e = [r.e for r in rows],
                   j = [r.j for r in rows],
                   AIC = [r.AIC for r in rows],
                   AICc = [r.AICc for r in rows])

    for col in (:AIC, :AICc)
        vals = df[!, col]
        df[!, Symbol("d", col)] = vals .- minimum(vals)
        df[!, Symbol(col, "_wt")] = akaike_weights(vals)
    end

    sort!(df, sort_by)
    return df
end
