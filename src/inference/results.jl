# Fitted model results.

"""
    BGBFit

The result of a maximum-likelihood fit, matching what R's `bears_optim_run()`
returns for the fields that matter downstream.

* `model` — the fitted [`Model`](@ref), with `est` at the optimum and the
  linked parameters (`ysv`, `y`, `s`, `v`, `mx01*`) already updated
* `loglik` — the maximised log-likelihood
* `numparams` — number of free parameters, for AIC
* `ntips` — number of tips, the sample size for AICc
* `converged` — whether the optimiser reached one of its stopping rules
* `algorithm` — the NLopt algorithm that produced the optimum
* `nfevals` / `iterations` — effort spent by that algorithm
* `message` — the optimiser's return code, as a string
* `ancprobs` — marginal ancestral state probabilities, or `nothing` when the
  fit was run with `calc_ancprobs = false`
"""
struct BGBFit
    model::Model
    loglik::Float64
    numparams::Int
    ntips::Int
    converged::Bool
    algorithm::Symbol
    nfevals::Int
    iterations::Int
    message::String
    ancprobs::Union{Nothing,AncestralProbs}
end

function BGBFit(model::Model, loglik::Real, ntips::Integer;
                converged::Bool = true, algorithm::Symbol = :unknown,
                nfevals::Integer = 0, iterations::Integer = 0,
                message::AbstractString = "",
                ancprobs::Union{Nothing,AncestralProbs} = nothing)
    return BGBFit(model, float(loglik), count(p -> p.kind == "free", model.params),
                  Int(ntips), converged, algorithm, Int(nfevals), Int(iterations),
                  String(message), ancprobs)
end

function Base.show(io::IO, f::BGBFit)
    print(io, "BGBFit(", model_name(f.model), "): loglik = ", round(f.loglik, digits = 6),
          ", ", f.numparams, " params, AIC = ", round(aic(f), digits = 6))
end

"""
    loglikelihood(f::BGBFit) -> Float64

The maximised log-likelihood of a fit.
"""
loglikelihood(f::BGBFit) = f.loglik

"Number of free parameters in a fit."
nparams(f::BGBFit) = f.numparams

"""
    model_name(f::BGBFit) -> String

Short model label, e.g. `"DEC+J"`.
"""
model_name(f::BGBFit) = model_name(f.model)

"""
    paramvalue(f::BGBFit, name) -> Float64

The fitted estimate of parameter `name`, e.g. `paramvalue(f, "d")`.
"""
paramvalue(f::BGBFit, name::AbstractString) = f.model[name]

"""
    params_table(f::BGBFit)

The full parameter table of the fitted model, as [`params_table`](@ref) returns
it for a [`Model`](@ref).
"""
params_table(f::BGBFit) = params_table(f.model)

"""
    ancprobs(f::BGBFit) -> AncestralProbs

Marginal ancestral state probabilities.  Throws if the fit was run without
`calc_ancprobs = true`.
"""
function ancprobs(f::BGBFit)
    f.ancprobs === nothing &&
        throw(ArgumentError("this fit has no ancestral probabilities; refit with calc_ancprobs=true"))
    return f.ancprobs
end

"""
    fit_summary(f::BGBFit) -> NamedTuple

The scalar fields R's summary table carries for one model: label, log-likelihood,
free parameters, `d`, `e`, `j`, and the information criteria.
"""
function fit_summary(f::BGBFit)
    # AICc is undefined when the model has as many parameters as tips; report
    # NaN rather than letting a reporting function throw on a small tree.
    aicc_value = f.numparams >= f.ntips - 1 ? NaN : aicc(f)
    return (model = model_name(f),
            lnL = f.loglik,
            numparams = f.numparams,
            d = paramvalue(f, "d"),
            e = paramvalue(f, "e"),
            j = paramvalue(f, "j"),
            AIC = aic(f),
            AICc = aicc_value)
end
