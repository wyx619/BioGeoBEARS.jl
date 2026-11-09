# Ancestral-state probability calculations.

"""
    AncestralProbs

Marginal ancestral state probabilities, matching R's `calc_loglike_sp()`
`return_what="all"` fields.

* `branch_top` — `nnodes × nstates`, probabilities at the node (R's
  `ML_marginal_prob_each_state_at_branch_top_AT_node`)
* `branch_bottom` — the same at the bottom of each node's branch
* `uppass_top` / `uppass_bottom` — the raw uppass probabilities
* `downpass` — the [`DownpassResult`](@ref) they were combined with

Rows sum to one.  The null-range column is zero for internal nodes.
"""
struct AncestralProbs
    branch_top::Matrix{Float64}
    branch_bottom::Matrix{Float64}
    uppass_top::Matrix{Float64}
    uppass_bottom::Matrix{Float64}
    downpass::DownpassResult
end

function Base.show(io::IO, a::AncestralProbs)
    print(io, "AncestralProbs: ", size(a.branch_top, 1), " nodes × ",
          size(a.branch_top, 2), " states, loglik = ", a.downpass.loglik)
end

"""
    ancestral_probs(tr, ss, tipcl, q, w; kwargs...) -> AncestralProbs

Run the downpass and the root-to-tip uppass and return the marginal ancestral
state probabilities.

`rootedge` and `fixnode`/`fixlikes` are forwarded to [`downpass`](@ref); fixed
nodes also have their uppass probability multiplied by the same fixed
likelihood vector, as R does (R's `use_fixnodes_on_uppass` block).  A root stem
changes the likelihood but not the uppass start, which stays uniform — that is
what R computes, since its `starting_probs` is only used for `length()`.
"""
function ancestral_probs(tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                         q::AbstractMatrix, w::CladogenesisWeights;
                         min_branchlength::Real = 1e-6,
                         rootedge::Bool = false,
                         fixnode = nothing, fixlikes = nothing,
                         r_float32::Bool = false)
    dp = downpass(tr, ss, tipcl, q, w; min_branchlength = min_branchlength,
                  rootedge = rootedge, fixnode = fixnode, fixlikes = fixlikes,
                  r_float32 = r_float32)
    nn = nnodes(tr)
    ns = nstates(ss)
    rs = row_sums(w, ns; r_float32 = r_float32)
    fixes = _normalise_fixes(nn, ns, fixnode, fixlikes)

    uppass_top = zeros(Float64, nn, ns)
    uppass_bottom = zeros(Float64, nn, ns)

    root = root_node(tr)
    uppass_top[root, :] .= 1.0 / ns        # R: 1/length(starting_probs)

    # Root first, then every other internal node parent-before-child.
    for anc in (root, reverse(pruning_order(tr))...)
        is_tip(tr, anc) && continue

        l = left_child(tr, anc)
        r = right_child(tr, anc)
        hook = branch_length(tr, l) < min_branchlength ||
               branch_length(tr, r) < min_branchlength
        ww = hook ? _identity_weights(ss) : w
        rrs = hook ? row_sums(ww, ns) : rs

        if anc != root
            mother = parent_node(tr, anc)
            sister = left_child(tr, mother) == anc ? right_child(tr, mother) :
                     left_child(tr, mother)
            sister_is_left = left_child(tr, mother) == sister

            # 2. uppass probabilities at the bottom of anc's branch
            L, R = _uppass_split(uppass_top[mother, :], ww, rrs, ss.include_null_range,
                                 sister_is_left ? dp.branch_bottom[sister, :] : nothing,
                                 sister_is_left ? nothing : dp.branch_bottom[sister, :],
                                 r_float32)
            uppass_bottom[anc, :] .= sister_is_left ? R : L
            # 3. forward exponentiation up to the focal node
            uppass_top[anc, :] .= apply_transition_transpose(
                q, branch_length(tr, anc), uppass_bottom[anc, :]; exact = r_float32)
        end

        # 4. propagate to the tips.  The tip's own uppass probability is the
        # marginal on ITS side of the split, with the sibling's downpass
        # likelihoods on the other side.
        if is_tip(tr, l)
            L, _ = _uppass_split(uppass_top[anc, :], ww, rrs, ss.include_null_range,
                                 nothing, dp.branch_bottom[r, :], r_float32)
            uppass_bottom[l, :] .= L
            uppass_top[l, :] .= apply_transition_transpose(
                q, branch_length(tr, l), uppass_bottom[l, :]; exact = r_float32)
        end
        if is_tip(tr, r)
            _, R = _uppass_split(uppass_top[anc, :], ww, rrs, ss.include_null_range,
                                 dp.branch_bottom[l, :], nothing, r_float32)
            uppass_bottom[r, :] .= R
            uppass_top[r, :] .= apply_transition_transpose(
                q, branch_length(tr, r), uppass_bottom[r, :]; exact = r_float32)
        end

        # R applies the fixed likelihoods to the uppass at the END of the node's
        # iteration, after the tip propagation above.
        if fixes !== nothing
            k = findfirst(==(anc), fixes[1])
            k === nothing || (uppass_top[anc, :] .*= view(fixes[2], k, :))
        end
    end

    # 5. combine
    top = dp.node_relprobs .* uppass_top
    top[root, :] .= dp.node_relprobs[root, :]
    _normalise_rows!(top, r_float32)

    bottom = dp.branch_bottom .* uppass_bottom
    _normalise_rows!(bottom, r_float32)

    return AncestralProbs(top, bottom, uppass_top, uppass_bottom, dp)
end

function _normalise_rows!(M::Matrix{Float64}, exact::Bool = false)
    for i in 1:size(M, 1)
        s = r_sum(view(M, i, :), exact)
        s > 0 && (M[i, :] ./= s)
    end
    return M
end

# Scenario probabilities for one split: for every (ancestor, left, right) event,
# weight it by the ancestral probability, the two daughter probabilities and the
# event's share of that ancestor's weight, then marginalise onto each daughter.
function _uppass_split(probs_anc, w::CladogenesisWeights, rs, include_null::Bool,
                       left_likes, right_likes, exact::Bool = false)
    ns = length(probs_anc)
    L = zeros(Float64, ns)
    R = zeros(Float64, ns)

    p = Vector{Float64}(undef, length(w.weight))
    for k in eachindex(w.weight)
        a = w.ancestor[k]
        if rs[a] <= 0
            p[k] = 0.0
            continue
        end
        # R's `apply(..., prod)` multiplies c(ancprobs, Lprobs, Rprobs, cond)
        # left to right, so keep that order: the last bits depend on it.
        lk = left_likes === nothing ? 1.0 : left_likes[w.left[k]]
        rk = right_likes === nothing ? 1.0 : right_likes[w.right[k]]
        p[k] = probs_anc[a] * lk * rk * (w.weight[k] / rs[a])
    end

    total = r_sum(p, exact)
    total > 0 || return (L, R)

    if exact
        # R normalises the whole scenario vector, then for each daughter state
        # sums the matching scenarios with its long-double `sum()`.
        probs = p ./ total
        left_idx = [Int[] for _ in 1:ns]
        right_idx = [Int[] for _ in 1:ns]
        for k in eachindex(w.left)
            push!(left_idx[w.left[k]], k)
            push!(right_idx[w.right[k]], k)
        end
        for n in 1:ns
            isempty(left_idx[n]) || (L[n] = r_sum(probs[left_idx[n]]))
            isempty(right_idx[n]) || (R[n] = r_sum(probs[right_idx[n]]))
        end
        return (L, R)
    end

    for k in eachindex(w.weight)
        q = p[k] / total
        L[w.left[k]] += q
        R[w.right[k]] += q
    end
    return (L, R)
end

# The "no cladogenesis" table R substitutes for hook branches: every non-null
# state splits into itself, with weight 1.
function _identity_weights(ss::StateSpace)
    anc = Int[]
    left = Int[]
    right = Int[]
    weight = Float64[]
    offset = ss.include_null_range ? 1 : 0
    i = offset
    for state in ss.states
        isempty(state) && continue
        i += 1
        push!(anc, i); push!(left, i); push!(right, i); push!(weight, 1.0)
    end
    return CladogenesisWeights(anc, left, right, weight)
end

"""
    ancestral_probs(tr, ss, model, tipcl; kwargs...) -> AncestralProbs

Build `Q` and the cladogenesis weights from `model` (with its linked parameters
applied) and return the marginal ancestral state probabilities.
"""
function ancestral_probs(tr::BGTree, ss::StateSpace, model::Model,
                         tipcl::AbstractMatrix; kwargs...)
    m = linked(model)
    return ancestral_probs(tr, ss, tipcl, qmatrix(ss, m), cladogenesis_weights(ss, m);
                           kwargs...)
end

"""
    ancestral_probs(tr, ss, tips::TipRanges, model; kwargs...) -> AncestralProbs

Convenience form that builds `Q` and the cladogenesis weights from `model`
(with its linked parameters applied, as [`loglikelihood`](@ref) does).
"""
function ancestral_probs(tr::BGTree, ss::StateSpace, tips::TipRanges, model::Model;
                         use_ambiguities::Bool = false, kwargs...)
    m = linked(model)
    tipcl = tip_condlikes(tr, tips;
                          max_range_size = ss.max_range_size,
                          include_null_range = ss.include_null_range,
                          use_ambiguities = use_ambiguities)
    return ancestral_probs(tr, ss, tipcl, qmatrix(ss, m), cladogenesis_weights(ss, m);
                           kwargs...)
end
