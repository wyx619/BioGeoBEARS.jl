# Likelihood downpass.

"""
    DownpassResult

The conditional likelihoods computed by [`downpass`](@ref), laid out like R's
`calc_loglike_sp(return_what="all")` output.

* `loglik` — the log-likelihood of the whole tree
* `node_condlikes` — `nnodes × nstates`, conditional likelihoods (not normalised)
* `node_relprobs` — the same, normalised per node
* `node_total` — the scalar that was divided out at each node
* `branch_bottom` — `nnodes × nstates`, the normalised conditional likelihoods
  at the bottom of each node's branch (R's
  `relative_probs_of_each_state_at_branch_bottom_below_node_DOWNPASS`); the
  uppass needs it for the sister branch
* `rootedge_bottom` — the conditional likelihoods at the bottom of the root
  stem, or an empty vector when the stem was not included

`loglik` includes the root-stem segment when it was used.
"""
struct DownpassResult
    loglik::Float64
    node_condlikes::Matrix{Float64}
    node_relprobs::Matrix{Float64}
    node_total::Vector{Float64}
    branch_bottom::Matrix{Float64}
    rootedge_bottom::Vector{Float64}
end

function Base.show(io::IO, r::DownpassResult)
    print(io, "DownpassResult: loglik = ", r.loglik)
end

"""
    downpass(tr, ss, tipcl, q, w; kwargs...) -> DownpassResult

Run the tip-to-root pass and return a [`DownpassResult`](@ref).

`tipcl` is the `ntips × nstates` tip conditional likelihood matrix from
[`tip_condlikes`](@ref); `q` is the rate matrix and `w` the cladogenesis weights,
both built for the same `ss`.  Splitting them out lets a caller reuse one `Q`
and one weight table across a whole optimisation.

Keyword arguments:

* `min_branchlength` — branches shorter than this are hook branches
* `probs_of_states_at_root` — optional prior on the root state, multiplied into
  the root's split likelihood
* `rootedge` — if `true` and the tree has a stem (`tr.root_edge > 0`), one more
  `exp(t·Q)` is applied below the root and its scalar is added to `loglik`
* `fixnode` / `fixlikes` — fix the state likelihoods at one or more nodes.  The
  fixed likelihood vector is multiplied into the node's split likelihood before
  normalisation, as R's `calc_loglike_sp()` does; this is how ML ancestral
  states are pinned down.
"""
function downpass(tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                  q::AbstractMatrix, w::CladogenesisWeights;
                  min_branchlength::Real = 1e-6,
                  probs_of_states_at_root = nothing,
                  rootedge::Bool = false,
                  fixnode = nothing, fixlikes = nothing,
                  r_float32::Bool = false)
    nn = nnodes(tr)
    ns = nstates(ss)
    size(tipcl) == (ntips(tr), ns) ||
        throw(ArgumentError("tip_condlikes is $(size(tipcl)), expected ($(ntips(tr)), $ns)"))

    if probs_of_states_at_root !== nothing
        length(probs_of_states_at_root) == ns ||
            throw(ArgumentError("probs_of_states_at_root has $(length(probs_of_states_at_root)) " *
                                "entries, expected $ns"))
    end

    fixes = _normalise_fixes(nn, ns, fixnode, fixlikes)

    condlikes = zeros(Float64, nn, ns)
    relprobs = zeros(Float64, nn, ns)
    totals = zeros(Float64, nn)
    branch_bottom = zeros(Float64, nn, ns)

    for t in 1:ntips(tr)
        condlikes[t, :] .= view(tipcl, t, :)
        relprobs[t, :] .= view(tipcl, t, :)
        # R sets the per-tip factor to rowSums/rowSums = 1: tips are observations,
        # not likelihoods to be accumulated.
        totals[t] = 1.0
    end

    rs = row_sums(w, ns; r_float32 = r_float32)

    # `pruning_order` excludes the root (it has no parent), so it is appended:
    # the root is where the last cladogenesis event happens and where the
    # log-likelihood is accumulated.
    for anc in (pruning_order(tr)..., root_node(tr))
        is_tip(tr, anc) && continue
        l = left_child(tr, anc)
        r = right_child(tr, anc)
        Ll = apply_transition(q, branch_length(tr, l), view(relprobs, l, :);
                              exact = r_float32)
        Lr = apply_transition(q, branch_length(tr, r), view(relprobs, r, :);
                              exact = r_float32)
        branch_bottom[l, :] .= Ll ./ r_sum(Ll, r_float32)
        branch_bottom[r, :] .= Lr ./ r_sum(Lr, r_float32)

        node = if branch_length(tr, l) < min_branchlength ||
                  branch_length(tr, r) < min_branchlength
            Ll .* Lr                      # hook branch: no cladogenesis event
        else
            _combine_split(Ll, Lr, w, rs, ss.include_null_range;
                           r_float32 = r_float32)
        end

        if fixes !== nothing
            k = findfirst(==(anc), fixes[1])
            k === nothing || (node = node .* view(fixes[2], k, :))
        end

        if probs_of_states_at_root !== nothing && is_root(tr, anc)
            node = node .* probs_of_states_at_root
        end

        tot = r_sum(node, r_float32)
        totals[anc] = tot
        condlikes[anc, :] .= node
        relprobs[anc, :] .= node ./ tot
    end

    loglik = 0.0
    for n in 1:nn
        is_tip(tr, n) && continue
        loglik += log(totals[n])
    end

    # Optional stem below the lowest fork.  R appends this segment to the node
    # tables; here it is kept apart so the per-node matrices stay one row per
    # node (R's append makes its own uppass recycling go wrong).
    rootedge_bottom = Float64[]
    if rootedge && root_edge(tr) > 0
        top = view(relprobs, root_node(tr), :)
        rootedge_bottom = vec(apply_transition(q, root_edge(tr), top;
                                               exact = r_float32))
        tot = r_sum(rootedge_bottom, r_float32)
        loglik += log(tot)
        rootedge_bottom ./= tot
    end

    return DownpassResult(loglik, condlikes, relprobs, totals, branch_bottom,
                          rootedge_bottom)
end

# Normalise `fixnode` / `fixlikes` into a node vector and a `k × nstates`
# matrix of likelihoods.  Returns `nothing` when no node is fixed.
function _normalise_fixes(nn::Int, ns::Int, fixnode, fixlikes)
    fixnode === nothing && return nothing
    fixlikes === nothing &&
        throw(ArgumentError("fixnode was given without fixlikes"))
    nodes = fixnode isa Integer ? Int[fixnode] : Int[fixnode...]
    isempty(nodes) && return nothing
    for n in nodes
        1 <= n <= nn ||
            throw(ArgumentError("fixnode contains $n, but the tree has $nn nodes"))
    end
    mat = if fixlikes isa AbstractVector
        length(nodes) == 1 ||
            throw(ArgumentError("fixlikes must be a matrix with one row per fixnode " *
                                "($(length(nodes)) rows), got a vector"))
        length(fixlikes) == ns ||
            throw(ArgumentError("fixlikes has $(length(fixlikes)) entries, expected $ns"))
        reshape(Float64.(fixlikes), 1, ns)
    else
        size(fixlikes, 1) == length(nodes) && size(fixlikes, 2) == ns ||
            throw(ArgumentError("fixlikes is $(size(fixlikes)), expected " *
                                "($(length(nodes)), $ns)"))
        Matrix{Float64}(fixlikes)
    end
    return (nodes, mat)
end

# One speciation event: sum the weight of every (left, right) pair that could
# produce each ancestral range, then normalise by that ancestor's total weight.
# R's `rcpp_calc_splitlikes_using_COOweights_columnar` does exactly this, in the
# null-range-free index space.
function _combine_split(Ll, Lr, w::CladogenesisWeights, rs, include_null::Bool;
                        r_float32::Bool = false)
    node = zeros(Float64, length(Ll))
    if r_float32
        # R's `cpp_calc_splitlikes_using_COOweights_columnar()` rounds each
        # `left * right * weight` term to a C `float` before accumulating in a
        # double, then rounds the `sum / rowsum` ratio to `float` as well.
        for k in eachindex(w.weight)
            t = Float64(Float32(Ll[w.left[k]] * Lr[w.right[k]] * w.weight[k]))
            node[w.ancestor[k]] += t
        end
        for a in eachindex(node)
            if include_null && a == 1
                node[1] = 0.0             # the null range never speciates
                continue
            end
            rs[a] > 0 && (node[a] = Float64(Float32(node[a] / rs[a])))
        end
        return node
    end
    for k in eachindex(w.weight)
        node[w.ancestor[k]] += Ll[w.left[k]] * Lr[w.right[k]] * w.weight[k]
    end
    for a in eachindex(node)
        if include_null && a == 1
            node[1] = 0.0             # the null range never speciates
            continue
        end
        rs[a] > 0 && (node[a] /= rs[a])
    end
    return node
end

"""
    loglikelihood(tr, ss, model, tipcl; kwargs...) -> Float64

Build `Q` and the cladogenesis weights from `model` and return the
log-likelihood of the tree given the tip conditional likelihoods.

The derived parameters (`ysv`, `ys`, `y`, `s`, `v`, `mx01*`) are recomputed from
the free ones first, as R's `calc_loglike_for_optim()` does — see
[`update_linked!`](@ref).  A copy is used, so `model` is not modified.
"""
function loglikelihood(tr::BGTree, ss::StateSpace, model::Model,
                       tipcl::AbstractMatrix; kwargs...)
    m = linked(model)
    q = qmatrix(ss, m)
    w = cladogenesis_weights(ss, m)
    return downpass(tr, ss, tipcl, q, w; kwargs...).loglik
end

"""
    loglikelihood(tr, ss, tips::TipRanges, model; kwargs...) -> Float64

Convenience form that derives the tip conditional likelihoods from a geography
table, matching the tree's tip order.
"""
function loglikelihood(tr::BGTree, ss::StateSpace, tips::TipRanges, model::Model;
                       use_ambiguities::Bool = false, kwargs...)
    tipcl = tip_condlikes(tr, tips;
                          max_range_size = ss.max_range_size,
                          include_null_range = ss.include_null_range,
                          use_ambiguities = use_ambiguities)
    return loglikelihood(tr, ss, model, tipcl; kwargs...)
end
