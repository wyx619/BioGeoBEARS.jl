# Parameter optimization.

const DEFAULT_ALGORITHMS = (:LN_BOBYQA, :LN_NELDERMEAD, :LN_SBPLX, :LN_COBYLA)

# Return values that mean the optimiser stopped for a legitimate reason.  R's
# optimx reports convcode 0 for all of these.
const _CONVERGED_CODES = (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED,
                          :STOPVAL_REACHED, :MAXEVAL_REACHED)

"""
    fit_model(tr, ss, tipcl, model; kwargs...) -> BGBFit
    fit_model(tr, ss, tips, model; kwargs...) -> BGBFit

Maximise the log-likelihood over the free parameters of `model`.

The starting point is `init_params(model)`; with `nstarts > 1` the remaining
starts are drawn log-uniformly inside the parameter bounds.  For every start
each algorithm in `algorithms` is run and the best result overall is kept.

Keyword arguments:

* `algorithms` — NLopt algorithms to race, default
  `(:LN_BOBYQA, :LN_NELDERMEAD, :LN_SBPLX, :LN_COBYLA)`
* `maxeval` — evaluation budget per algorithm and start (default 1000)
* `global_maxeval` — if positive, run deterministic DIRECT-L first and refine
  its best point with each local algorithm (default 0, disabled)
* `ftol_rel`, `xtol_rel` — relative stopping tolerances (default `1e-10`, `0`)
* `nstarts` — number of starting points (default 1)
* `rng` — RNG for the extra starting points
* `calc_ancprobs` — also compute marginal ancestral probabilities at the optimum
* `penalty` — log-likelihood substituted when the calculation returns a
  non-finite value, so the optimiser can back away from that region (R's
  `on_NaN_error`, default `-1e50`)

The returned [`BGBFit`](@ref) carries a copy of the model with the linked
parameters already updated, so `fit.model["ysv"]` is consistent with `fit.model["j"]`.
"""
function fit_model(tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                   model::Model;
                   algorithms = DEFAULT_ALGORITHMS,
                   maxeval::Integer = 1000,
                   global_maxeval::Integer = 0,
                   ftol_rel::Real = 1e-10,
                   xtol_rel::Real = 0.0,
                   nstarts::Integer = 1,
                   rng::AbstractRNG = Random.default_rng(),
                   calc_ancprobs::Bool = false,
                   penalty::Real = -1e50)
    objective = (x, work) -> _loglik_at(x, tr, ss, tipcl, work, penalty)
    ancprobs_fn = calc_ancprobs ?
        (fitted -> ancestral_probs(tr, ss, tipcl, qmatrix(ss, fitted),
                                   cladogenesis_weights(ss, fitted))) : nothing
    return _fit_model(objective, ancprobs_fn, model, ntips(tr);
                      algorithms = algorithms, maxeval = maxeval,
                      ftol_rel = ftol_rel, xtol_rel = xtol_rel,
                      nstarts = nstarts, rng = rng, global_maxeval = global_maxeval)
end

"""
    fit_model(tr, ss, tipcl, model, strat::Stratification; kwargs...) -> BGBFit
    fit_model(tr, ss, tips, model, strat::Stratification; kwargs...) -> BGBFit

Maximise the time-stratified log-likelihood.  Accepts the same optimiser
keywords as the non-stratified method; with `calc_ancprobs = true` the marginal
ancestral probabilities at the optimum are computed by
[`stratified_ancestral_probs`](@ref).  `fixnode` / `fixlikes` pin the state
likelihoods at one or more internal nodes throughout the optimisation.
"""
function fit_model(tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                   model::Model, strat::Stratification;
                   algorithms = DEFAULT_ALGORITHMS,
                   maxeval::Integer = 1000,
                   global_maxeval::Integer = 0,
                   ftol_rel::Real = 1e-10,
                   xtol_rel::Real = 0.0,
                   nstarts::Integer = 1,
                   rng::AbstractRNG = Random.default_rng(),
                   calc_ancprobs::Bool = false,
                   penalty::Real = -1e50,
                   fixnode = nothing, fixlikes = nothing)
    objective = (x, work) -> begin
        ll = try
            set_free_values!(work, x)
            update_linked!(work)
            stratified_loglikelihood(tr, ss, tipcl, work, strat;
                                     fixnode = fixnode, fixlikes = fixlikes)
        catch
            NaN
        end
        isfinite(ll) ? ll : float(penalty)
    end
    ancprobs_fn = calc_ancprobs ?
        (fitted -> stratified_ancestral_probs(tr, ss, tipcl, fitted, strat;
                                              fixnode = fixnode,
                                              fixlikes = fixlikes)) :
        nothing
    return _fit_model(objective, ancprobs_fn, model, ntips(tr);
                      algorithms = algorithms, maxeval = maxeval,
                      ftol_rel = ftol_rel, xtol_rel = xtol_rel,
                      nstarts = nstarts, rng = rng, global_maxeval = global_maxeval)
end

function _fit_model(objective, ancprobs_fn, model::Model, ntip::Int;
                    algorithms, maxeval, ftol_rel, xtol_rel, nstarts, rng, global_maxeval)
    nfree = count(p -> p.kind == "free", model.params)
    nfree > 0 || throw(ArgumentError("model $(model_name(model)) has no free parameters"))
    isempty(algorithms) && throw(ArgumentError("algorithms must not be empty"))
    nstarts >= 1 || throw(ArgumentError("nstarts must be at least 1"))
    maxeval > 0 || throw(ArgumentError("maxeval must be positive"))
    global_maxeval >= 0 || throw(ArgumentError("global_maxeval must be nonnegative"))

    lo = lower_bounds(model)
    hi = upper_bounds(model)
    work = deepcopy(model)      # the objective mutates it; keep the caller's model intact

    best_ll = -Inf
    best_x = Float64[]
    best_alg = :none
    best_ret = :FAILURE
    best_nfev = 0

    starts = _start_values(model, nstarts, rng)
    if global_maxeval > 0
        all(isfinite, lo) && all(isfinite, hi) && all(lo .< hi) ||
            throw(ArgumentError("global search requires finite, distinct parameter bounds"))
        opt = NLopt.Opt(:GN_DIRECT_L, nfree)
        NLopt.lower_bounds!(opt, lo)
        NLopt.upper_bounds!(opt, hi)
        NLopt.maxeval!(opt, global_maxeval)
        NLopt.max_objective!(opt, (x, grad) -> objective(x, work))
        fmax, xmax, ret = NLopt.optimize(opt, starts[1])
        if isfinite(fmax)
            best_ll, best_x, best_alg, best_ret = fmax, copy(xmax), :GN_DIRECT_L, ret
            best_nfev = NLopt.numevals(opt)
            push!(starts, copy(xmax))
        end
    end

    for start in starts
        for alg in algorithms
            opt = NLopt.Opt(alg, nfree)
            NLopt.lower_bounds!(opt, lo)
            NLopt.upper_bounds!(opt, hi)
            NLopt.maxeval!(opt, maxeval)
            NLopt.ftol_rel!(opt, float(ftol_rel))
            xtol_rel > 0 && NLopt.xtol_rel!(opt, float(xtol_rel))
            NLopt.max_objective!(opt, (x, grad) -> objective(x, work))

            fmax, xmax, ret = NLopt.optimize(opt, start)
            nfev = NLopt.numevals(opt)

            if isfinite(fmax) && fmax > best_ll
                best_ll = fmax
                best_x = copy(xmax)
                best_alg = alg
                best_ret = ret
                best_nfev = nfev
            end
        end
    end

    best_x == Float64[] &&
        throw(ErrorException("every optimisation run returned a non-finite likelihood"))

    fitted = update_linked!(set_free_values!(deepcopy(model), best_x))
    anc = ancprobs_fn === nothing ? nothing : ancprobs_fn(fitted)

    return BGBFit(fitted, best_ll, ntip;
                  converged = best_ret in _CONVERGED_CODES,
                  algorithm = best_alg,
                  nfevals = best_nfev,
                  message = String(best_ret),
                  ancprobs = anc)
end

function fit_model(tr::BGTree, ss::StateSpace, tips::TipRanges, model::Model,
                   strat::Stratification; use_ambiguities::Bool = false, kwargs...)
    tipcl = tip_condlikes(tr, tips;
                          max_range_size = ss.max_range_size,
                          include_null_range = ss.include_null_range,
                          use_ambiguities = use_ambiguities)
    return fit_model(tr, ss, tipcl, model, strat; kwargs...)
end

function fit_model(tr::BGTree, ss::StateSpace, tips::TipRanges, model::Model;
                   use_ambiguities::Bool = false, kwargs...)
    tipcl = tip_condlikes(tr, tips;
                          max_range_size = ss.max_range_size,
                          include_null_range = ss.include_null_range,
                          use_ambiguities = use_ambiguities)
    return fit_model(tr, ss, tipcl, model; kwargs...)
end

"""
    fit_model(tr, ss, tips; model = Model(:DEC), kwargs...) -> BGBFit

Convenience form with the model defaulting to DEC.
"""
fit_model(tr::BGTree, ss::StateSpace, tips::TipRanges;
          model::Model = Model(:DEC), kwargs...) =
    fit_model(tr, ss, tips, model; kwargs...)

# The log-likelihood at one parameter vector.  `work` is reused across calls so
# the optimiser does not allocate a new model every evaluation.
function _loglik_at(x, tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                    work::Model, penalty::Real)
    ll = try
        set_free_values!(work, x)
        update_linked!(work)
        downpass(tr, ss, tipcl, qmatrix(ss, work), cladogenesis_weights(ss, work)).loglik
    catch
        NaN
    end
    return isfinite(ll) ? ll : float(penalty)
end

# First start is the model's own initial values; the rest are log-uniform draws
# inside the bounds, which spans the several orders of magnitude that d and e
# usually cover.
function _start_values(model::Model, nstarts::Integer, rng::AbstractRNG)
    starts = Vector{Vector{Float64}}()
    push!(starts, init_params(model))
    nstarts == 1 && return starts

    lo = lower_bounds(model)
    hi = upper_bounds(model)
    for _ in 2:nstarts
        x = Vector{Float64}(undef, length(lo))
        for i in eachindex(lo)
            a = max(lo[i], 1e-12)
            b = max(hi[i], a * (1 + 1e-9))
            x[i] = exp(log(a) + rand(rng) * (log(b) - log(a)))
        end
        push!(starts, x)
    end
    return starts
end
