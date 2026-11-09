# Cladogenesis event weights.

"""
    maxent_probs(constraint, states; tol=1e-12) -> Vector{Float64}

Maximum-entropy distribution on `states` (a vector of numeric values) whose mean
equals `constraint`, with a uniform prior.

This is `FD::maxent()`: find `λ` such that `p_i ∝ exp(λ · states[i])` has the
requested mean, then normalise.  The mean is monotone in `λ`, so a bisection on
`λ` converges; `λ = 0` gives the uniform distribution.

A single state can only have one mean, so the constraint is ignored there.
"""
function maxent_probs(constraint::Real, states::AbstractVector{<:Real}; tol = 1e-12)
    n = length(states)
    n == 0 && throw(ArgumentError("maxent_probs needs at least one state"))
    n == 1 && return [1.0]

    # The mean of a distribution on `states` can never leave [min, max], so an
    # out-of-range constraint is clamped to the corresponding end point.  This is
    # not a special case bolted on for convenience: the callers routinely ask for
    # a mean below 1 (e.g. `quantile(0:(k+1), 0.0001)`), and R's FD::maxent
    # returns the same degenerate answer.
    if constraint <= minimum(states)
        out = zeros(n)
        out[argmin(states)] = 1.0
        return out
    elseif constraint >= maximum(states)
        out = zeros(n)
        out[argmax(states)] = 1.0
        return out
    end

    mean_state = sum(states) / n
    abs(constraint - mean_state) < tol && return fill(1.0 / n, n)

    shift = maximum(states)
    prob(λ) = begin
        w = exp.(λ .* (states .- shift))
        w ./ sum(w)
    end
    f(λ) = sum(prob(λ) .* states) - constraint

    lo, hi = 0.0, 1.0
    if f(0.0) < 0            # need more weight on the larger states
        while f(hi) < 0
            lo = hi
            hi *= 2
            hi > 1e6 && break
        end
    else                     # need more weight on the smaller states
        hi = 0.0
        lo = -1.0
        while f(lo) > 0
            hi = lo
            lo *= 2
            lo < -1e6 && break
        end
    end

    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
        hi - lo < 1e-14 && break
    end

    return prob((lo + hi) / 2)
end

"""
    relative_probabilities_of_subsets(max_numareas, maxent_constraint_01; na_val=0.0)

Matrix of relative probabilities of the smaller daughter's range size, given the
ancestor's range size, for sympatric-subset speciation (and, with a different
parameter, for founder events).

Row `a` is the ancestor's range size, column `d` the smaller daughter's.  Entries
below the diagonal (a daughter larger than its ancestor) are `na_val`.  The
result is rounded to three decimals, exactly as R does — the rounding is
load-bearing, it turns the near-step maxent distributions into exact 0/1 steps
for the default `maxent_constraint_01 = 0.0001`.
"""
function relative_probabilities_of_subsets(max_numareas::Integer,
                                           maxent_constraint_01::Real;
                                           na_val::Real = 0.0)
    max_numareas >= 1 ||
        throw(ArgumentError("max_numareas must be at least 1, got $max_numareas"))
    0.00001 <= maxent_constraint_01 <= 0.99999 ||
        throw(ArgumentError("maxent_constraint_01 must be in [0.00001, 0.99999], " *
                            "got $maxent_constraint_01"))

    M = fill(float(na_val), Int(max_numareas), Int(max_numareas))
    for a in 1:max_numareas
        tmpstates = collect(1:a)
        constraint = quantile(collect(0.0:(a + 1)), float(maxent_constraint_01))
        M[a, 1:a] .= maxent_probs(constraint, tmpstates)
    end
    return round.(M, digits = 3)
end

"""
    relative_probabilities_of_vicariants(max_numareas, maxent_constraint_01v; na_val=0.0)

Same table for vicariance, where the smaller daughter can be at most half the
ancestor (strictly smaller than the median range size).  Row 1 is all `na_val`:
a one-area ancestor cannot split.
"""
function relative_probabilities_of_vicariants(max_numareas::Integer,
                                              maxent_constraint_01v::Real;
                                              na_val::Real = 0.0)
    max_numareas >= 1 ||
        throw(ArgumentError("max_numareas must be at least 1, got $max_numareas"))
    0.00001 <= maxent_constraint_01v <= 0.99999 ||
        throw(ArgumentError("maxent_constraint_01v must be in [0.00001, 0.99999], " *
                            "got $maxent_constraint_01v"))

    M = fill(float(na_val), Int(max_numareas), Int(max_numareas))
    M[1, :] .= float(na_val)

    if max_numareas < 2
        M[1, :] .= 1.0
        return M
    end

    for a in 2:max_numareas
        tmpstates = collect(1:a)
        possible = [x for x in tmpstates if x < median(tmpstates)]
        constraint = quantile(collect(0.0:(length(possible) + 1)), float(maxent_constraint_01v))
        M[a, 1:length(possible)] .= maxent_probs(constraint, possible)
    end
    return round.(M, digits = 3)
end

_maxsize(row) = (i = findlast(!iszero, row); i === nothing ? 0 : i)

"""
    CladogenesisWeights

The cladogenesis events of one model, as parallel arrays: the ancestral state,
the two daughter states, and the event weight.

All indices are **1-based positions in the full `StateSpace`**, so they can be
used to index `qmatrix()` output directly.  R stores the same table 0-based and
with the null range removed, which shifts every index by two; the tests do that
translation explicitly rather than hiding it here.
"""
struct CladogenesisWeights
    ancestor::Vector{Int}
    left::Vector{Int}
    right::Vector{Int}
    weight::Vector{Float64}
end

Base.length(w::CladogenesisWeights) = length(w.weight)

function Base.show(io::IO, w::CladogenesisWeights)
    print(io, "CladogenesisWeights: ", length(w), " events")
end

"""
    row_sums(w::CladogenesisWeights, nstates; r_float32=false) -> Vector{Float64}

Sum of the weights of every event whose ancestor is each state — the
`Rsp_rowsums` that `calc_loglike_sp` normalises by.

With `r_float32` each weight is rounded to single precision before the sum,
reproducing R's `rcpp_calc_rowsums_for_COOweights_columnar()`, whose weights
are C `float`s accumulated into a `Rcpp::NumericVector` (double).  This only
matters for fixed-seed simulation that must match R bit for bit.
"""
function row_sums(w::CladogenesisWeights, nstates::Integer; r_float32::Bool = false)
    if r_float32
        # R's `cpp_calc_rowsums_for_COOweights_columnar()` accumulates into a
        # `Rcpp::NumericVector` (double) even though each weight is a C `float`.
        # So round the weights, then sum in double precision.
        out = zeros(Float64, nstates)
        for (k, a) in enumerate(w.ancestor)
            out[a] += Float64(Float32(w.weight[k]))
        end
        return out
    end
    out = zeros(Float64, nstates)
    for (k, a) in enumerate(w.ancestor)
        out[a] += w.weight[k]
    end
    return out
end

const _CLADO_MIN_PRECISION = 1e-30

"""
    cladogenesis_weights(ss::StateSpace, model::Model; dispersal_multipliers=nothing)

Enumerate every cladogenesis event allowed by `ss` and weight it by the model's
`y`, `s`, `v`, `j` and the four `mx01*` maxent parameters.

`dispersal_multipliers` (the same matrix [`qmatrix`](@ref) takes) scales founder
events only: the `j` weight of a jump from range `A` to a new area `b` is the
mean of `mult[a, b]` over `a ∈ A`.  R does exactly this, and it is why `j` can be
larger than 1 without producing impossible probabilities.

`founder_multiplier` scales every founder weight by a constant.  It exists for
[`stratified_ancestral_probs`](@ref): R leaks a loop variable `m` into its
uppass COO build, which multiplies DEC+J founder weights by 1/2/4/... in each
stratum.  The default 1 leaves the weights unchanged.
"""
function cladogenesis_weights(ss::StateSpace, model::Model;
                              dispersal_multipliers = nothing,
                              distances = nothing,
                              r_float32::Bool = false,
                              founder_multiplier::Real = 1)
    n = nareas(ss)
    mult = dispersal_multipliers === nothing ? ones(n, n) :
           Matrix{Float64}(dispersal_multipliers)
    size(mult) == (n, n) ||
        throw(ArgumentError("dispersal_multipliers is $(size(mult)), expected ($n, $n)"))
    mult = _apply_distance_exponent(mult, distances, model["x"], n)

    # The null range never undergoes cladogenesis, so it is left out of the
    # enumeration.  Its state index is 1 when it is present, so indices shift by
    # one only in that case.
    states = [state for state in ss.states if !isempty(state)]
    position = Dict{Vector{Int},Int}(state => k for (k, state) in enumerate(states))
    offset = ss.include_null_range ? 1 : 0
    full_index(k::Int) = k + offset

    maxareas = isempty(states) ? 0 : maximum(length, states)
    maxareas == 0 && return CladogenesisWeights(Int[], Int[], Int[], Float64[])

    mx01s = relative_probabilities_of_subsets(maxareas, model["mx01s"])
    mx01v = relative_probabilities_of_vicariants(maxareas, model["mx01v"])
    mx01j = relative_probabilities_of_subsets(maxareas, model["mx01j"])
    mx01y = relative_probabilities_of_subsets(maxareas, model["mx01y"])

    # Largest daughter size that any of the four event types gives non-zero
    # weight, per ancestral size.  This caps every loop below.
    allowed = max.(mx01s, mx01v, mx01j, mx01y) .> 0
    max_minsize = [_maxsize(allowed[a, :]) for a in 1:maxareas]

    y = model["y"]
    s = model["s"]
    v = model["v"]
    j = model["j"]

    by_size = [Int[] for _ in 1:maxareas]
    for (k, state) in enumerate(states)
        push!(by_size[length(state)], k)
    end

    anc = Int[]
    left = Int[]
    right = Int[]
    weight = Float64[]

    emit!(a, l, r, w) = begin
        abs(w) > _CLADO_MIN_PRECISION || return
        push!(anc, full_index(a))
        push!(left, full_index(l))
        push!(right, full_index(r))
        push!(weight, r_float32 ? Float64(Float32(w)) : w)
    end

    for (k, ancestor) in enumerate(states)
        ancsize = length(ancestor)
        cap = max_minsize[ancsize]

        # sympatry (range copying)
        if abs(y) > _CLADO_MIN_PRECISION && ancsize <= cap
            emit!(k, k, k, y * mx01y[ancsize, ancsize])
        end
    end

    # R emits one block per event type, each covering every ancestor in state
    # order, and inside the block a "primary" orientation followed by the
    # mirror image.  The exact order matters: `r_sample_prob` indexes into the
    # probability vector built from this table, so a different order would
    # consume the RNG stream differently.

    # subset: one daughter keeps the whole range, the other gets a proper subset
    if abs(s) > _CLADO_MIN_PRECISION
        for (k, ancestor) in enumerate(states)
            ancsize = length(ancestor)
            ancsize >= 2 || continue
            for desc_size in 1:max_minsize[ancsize]
                w = s * mx01s[ancsize, desc_size]
                abs(w) > _CLADO_MIN_PRECISION || continue
                for r in by_size[desc_size]
                    daughter = states[r]
                    all(x -> x in ancestor, daughter) || continue
                    length(daughter) == ancsize && continue      # that is sympatry
                    emit!(k, k, r, w)
                end
            end
        end
        # ... and the mirror image: the right daughter keeps the whole range.
        for (k, ancestor) in enumerate(states)
            ancsize = length(ancestor)
            ancsize >= 2 || continue
            for desc_size in 1:max_minsize[ancsize]
                w = s * mx01s[ancsize, desc_size]
                abs(w) > _CLADO_MIN_PRECISION || continue
                for r in by_size[desc_size]
                    daughter = states[r]
                    all(x -> x in ancestor, daughter) || continue
                    length(daughter) == ancsize && continue
                    emit!(k, r, k, w)
                end
            end
        end
    end

    # vicariance: the range splits in two.  R emits the equal-size splits
    # first (both orientations fall out of the loop over daughters), then the
    # unequal-size splits in the primary orientation, then their mirror.
    if abs(v) > _CLADO_MIN_PRECISION
        for (k, ancestor) in enumerate(states)
            ancsize = length(ancestor)
            ancsize >= 2 || continue
            for desc_size in 1:max_minsize[ancsize]
                w = v * mx01v[ancsize, desc_size]
                abs(w) > _CLADO_MIN_PRECISION || continue
                for r in by_size[desc_size]
                    daughter = states[r]
                    all(x -> x in ancestor, daughter) || continue
                    rest = [x for x in ancestor if !(x in daughter)]
                    isempty(rest) && continue
                    l = get(position, rest, 0)
                    l == 0 && continue
                    length(rest) == desc_size || continue
                    emit!(k, l, r, w)
                end
            end
        end
        for (k, ancestor) in enumerate(states)
            ancsize = length(ancestor)
            ancsize >= 2 || continue
            for desc_size in 1:max_minsize[ancsize]
                w = v * mx01v[ancsize, desc_size]
                abs(w) > _CLADO_MIN_PRECISION || continue
                for r in by_size[desc_size]
                    daughter = states[r]
                    all(x -> x in ancestor, daughter) || continue
                    rest = [x for x in ancestor if !(x in daughter)]
                    isempty(rest) && continue
                    l = get(position, rest, 0)
                    l == 0 && continue
                    length(rest) == desc_size && continue    # equal size: done above
                    emit!(k, l, r, w)
                end
            end
        end
        # mirror of the unequal-size splits
        for (k, ancestor) in enumerate(states)
            ancsize = length(ancestor)
            ancsize >= 2 || continue
            for desc_size in 1:max_minsize[ancsize]
                w = v * mx01v[ancsize, desc_size]
                abs(w) > _CLADO_MIN_PRECISION || continue
                for r in by_size[desc_size]
                    daughter = states[r]
                    all(x -> x in ancestor, daughter) || continue
                    rest = [x for x in ancestor if !(x in daughter)]
                    isempty(rest) && continue
                    l = get(position, rest, 0)
                    l == 0 && continue
                    length(rest) == desc_size && continue
                    emit!(k, r, l, w)
                end
            end
        end
    end

    # founder event: one daughter is a brand-new area outside the ancestor
    if abs(j) > _CLADO_MIN_PRECISION
        for (k, ancestor) in enumerate(states)
            ancsize = length(ancestor)
            for desc_size in 1:max_minsize[ancsize]
                base = j * founder_multiplier * mx01j[ancsize, desc_size]
                abs(base) > _CLADO_MIN_PRECISION || continue
                for r in by_size[desc_size]
                    daughter = states[r]
                    any(x -> x in ancestor, daughter) && continue
                    jprob = 0.0
                    for a in ancestor, b in daughter
                        jprob += mult[a, b]
                    end
                    jprob /= length(ancestor) * length(daughter)
                    emit!(k, k, r, base * jprob)
                end
            end
        end
        for (k, ancestor) in enumerate(states)
            ancsize = length(ancestor)
            for desc_size in 1:max_minsize[ancsize]
                base = j * founder_multiplier * mx01j[ancsize, desc_size]
                abs(base) > _CLADO_MIN_PRECISION || continue
                for r in by_size[desc_size]
                    daughter = states[r]
                    any(x -> x in ancestor, daughter) && continue
                    jprob = 0.0
                    for a in ancestor, b in daughter
                        jprob += mult[a, b]
                    end
                    jprob /= length(ancestor) * length(daughter)
                    emit!(k, r, k, base * jprob)
                end
            end
        end
    end

    return CladogenesisWeights(anc, left, right, weight)
end
