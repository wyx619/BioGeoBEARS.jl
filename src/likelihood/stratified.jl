# Time-stratified likelihood calculations.

"""
    Stratification

Settings for a time-stratified analysis: where the time boundaries are, and the
per-stratum dispersal multipliers / allowed-area constraints.

Build one with [`Stratification`](@ref):

```julia
strat = Stratification([0.5, 1.9, 3.7, 5.1, 10.0];
                       dispersal_multipliers = dm, areas_allowed = aa)
```

`times` are the time boundaries in millions of years before present, youngest
first.  The final value must be older than the root; it only bounds the oldest
stratum and is never used as a cut.

`dispersal_multipliers`, `areas_allowed` and `areas_adjacency` are either a
single square matrix (applied to every stratum) or one matrix per stratum.  An
empty vector means "no constraint".  `areas_allowed` keeps a state when every
area of the range is allowed from the range's first area (R's
`check_if_state_is_allowed()`); `areas_adjacency` keeps a state when all pairs
of its areas are flagged as adjacent (R's `check_if_state_is_allowed_by_adjacency()`).
When both are given the state must satisfy both, as in R.
"""
struct Stratification
    times::Vector{Float64}
    dispersal_multipliers::Vector{Matrix{Float64}}
    areas_allowed::Vector{Matrix{Float64}}
    areas_adjacency::Vector{Matrix{Float64}}
end

"""
    Stratification(times; dispersal_multipliers=nothing, areas_allowed=nothing,
                   areas_adjacency=nothing)

Build a [`Stratification`](@ref) from a vector of time boundaries.
"""
function Stratification(times::AbstractVector{<:Real};
                        dispersal_multipliers = nothing,
                        areas_allowed = nothing,
                        areas_adjacency = nothing)
    t = Float64.(times)
    isempty(t) && throw(ArgumentError("times must not be empty"))
    issorted(t) || throw(ArgumentError("times must be sorted youngest to oldest, got $(t)"))
    0.0 in t && throw(ArgumentError("time 0 must not be listed; the present is implicit"))
    n = length(t)
    dm = _per_stratum_matrices(dispersal_multipliers, n, "dispersal_multipliers")
    aa = _per_stratum_matrices(areas_allowed, n, "areas_allowed")
    adj = _per_stratum_matrices(areas_adjacency, n, "areas_adjacency")
    return Stratification(t, dm, aa, adj)
end

"""
    Stratification(; timesfn, dispersal_multipliers_fn=nothing,
                   areas_allowed_fn=nothing, areas_adjacency_fn=nothing)

Read the settings from files, in the same formats as R's `timesfn`,
`dispersal_multipliers_fn`, `areas_allowed_fn` and `areas_adjacency_fn`.
"""
function Stratification(; timesfn::AbstractString,
                        dispersal_multipliers_fn::Union{Nothing,AbstractString} = nothing,
                        areas_allowed_fn::Union{Nothing,AbstractString} = nothing,
                        areas_adjacency_fn::Union{Nothing,AbstractString} = nothing)
    times = read_times(timesfn)
    dm = dispersal_multipliers_fn === nothing ? nothing :
         _read_matrix_blocks(dispersal_multipliers_fn, "dispersal_multipliers")
    aa = areas_allowed_fn === nothing ? nothing : read_areas_allowed(areas_allowed_fn)
    adj = areas_adjacency_fn === nothing ? nothing : read_areas_adjacency(areas_adjacency_fn)
    return Stratification(times; dispersal_multipliers = dm, areas_allowed = aa,
                          areas_adjacency = adj)
end

function _per_stratum_matrices(x, n::Int, what::AbstractString)
    x === nothing && return Matrix{Float64}[]
    if x isa AbstractMatrix
        m = Matrix{Float64}(x)
        return [copy(m) for _ in 1:n]
    end
    mats = [Matrix{Float64}(m) for m in x]
    length(mats) == n ||
        throw(ArgumentError("$what has $(length(mats)) matrices, expected $n"))
    return mats
end

function Base.show(io::IO, s::Stratification)
    print(io, "Stratification: ", length(s.times), " periods, boundaries ",
          s.times, isempty(s.dispersal_multipliers) ? "" : ", dispersal multipliers",
          isempty(s.areas_allowed) ? "" : ", areas allowed",
          isempty(s.areas_adjacency) ? "" : ", areas adjacency")
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

"""
    stratified_loglikelihood(tr, ss, tipcl, model, strat; kwargs...) -> Float64
    stratified_loglikelihood(tr, ss, tips, model, strat; kwargs...) -> Float64

Log-likelihood of `tr` under a time-stratified model.

`strat` is a [`Stratification`](@ref) giving the time boundaries and the
per-stratum dispersal multipliers / allowed-area constraints.  `tipcl` is the
`ntips × nstates` tip conditional likelihood matrix, as returned by
[`tip_condlikes`](@ref).

Keyword arguments:

* `dense` — form `exp(t·Q)` densely instead of using the sparse Krylov path
  (default `false`; both paths agree to round-off, and R's `force_sparse`
  dense/sparse results are identical)
* `min_branchlength` — branches shorter than this are hook branches (default
  `1e-6`, as in R)
* `fixnode` / `fixlikes` — fix the state likelihoods at one or more internal
  nodes, as in [`downpass`](@ref); the vector is in the full state space

Every stratum rebuilds its own `Q` and cladogenesis weights from the subset of
states allowed in that period, exactly as R's `calc_loglike_sp_stratified()`
does.
"""
function stratified_loglikelihood(tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                                  model::Model, strat::Stratification;
                                  dense::Bool = false,
                                  min_branchlength::Real = 1e-6,
                                  fixnode = nothing, fixlikes = nothing,
                                  r_float32::Bool = false)
    all_cond = stratified_condlikes(tr, ss, tipcl, model, strat;
                                    dense = dense,
                                    min_branchlength = min_branchlength,
                                    fixnode = fixnode, fixlikes = fixlikes,
                                    r_float32 = r_float32)
    ll = 0.0
    for n in 1:nnodes(tr)
        s = r_sum(view(all_cond, n, :), r_float32)
        s > 0 || return -Inf
        ll += log(s)
    end
    return ll
end

"""
    stratified_condlikes(tr, ss, tipcl, model, strat; kwargs...) -> Matrix

The `nnodes × nstates` table of unnormalised conditional likelihoods at every
original node, one row per node, as accumulated across the time strata.  The
stratified log-likelihood is `sum(log.(rowsums))` of this table.

This is the stratified analogue of R's `condlikes_table` restricted to the rows
that correspond to original nodes.
"""
function stratified_condlikes(tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                              model::Model, strat::Stratification;
                              dense::Bool = false,
                              min_branchlength::Real = 1e-6,
                              fixnode = nothing, fixlikes = nothing,
                              r_float32::Bool = false)
    all_cond, _, _, _ = _stratified_downpass(tr, ss, tipcl, model, strat;
                                          dense = dense,
                                          min_branchlength = min_branchlength,
                                          fixnode = fixnode, fixlikes = fixlikes,
                                          r_float32 = r_float32)
    return all_cond
end

# Downpass shared by `stratified_condlikes` and `stratified_ancestral_probs`.
# Returns the unnormalised node likelihoods, the normalised branch-bottom
# messages (R's `relative_probs_..._branch_bottom_..._DOWNPASS`, with NaN at the
# root) and the piece sectioning used by both passes.
function _stratified_downpass(tr::BGTree, ss::StateSpace, tipcl::AbstractMatrix,
                              model::Model, strat::Stratification;
                              dense::Bool = false,
                              min_branchlength::Real = 1e-6,
                              fixnode = nothing, fixlikes = nothing,
                              r_float32::Bool = false)
    size(tipcl) == (ntips(tr), nstates(ss)) ||
        throw(ArgumentError("tip_condlikes is $(size(tipcl)), expected " *
                            "($(ntips(tr)), $(nstates(ss)))"))

    node_times = _check_stratification(tr, strat)
    m = linked(model)
    b = vcat(0.0, strat.times)
    nper = length(strat.times)
    nn = nnodes(tr)
    ns = nstates(ss)
    fixes = _normalise_fixes(nn, ns, fixnode, fixlikes)

    # Conditional likelihoods of every original node, in the full state space.
    # Original tips are the observations and are never recomputed.
    all_cond = zeros(Float64, nn, ns)
    all_cond[1:ntips(tr), :] .= tipcl

    # R's branch-bottom table: one row per node, the normalised message its
    # branch sends to its parent.  A node is either internal to a piece or a
    # piece tip, never both, so every non-root row is written exactly once.
    # R initialises the table to zero and inserts an NA row for the root, since
    # the root has no branch bottom.
    all_bottom = zeros(Float64, nn, ns)

    # `carried[v]` is the unnormalised conditional likelihood vector at the
    # bottom of node v's branch in the stratum that just ended.  For stratum 1
    # the entries are the tip observations.
    carried = Vector{Vector{Float64}}(undef, nn)
    for t in 1:ntips(tr)
        carried[t] = Vector{Float64}(tipcl[t, :])
    end

    pieces = _section_pieces(tr, node_times, b, nper)

    # `carried` as it stood when each stratum began: the message arriving at
    # the bottom of every node's branch from everything younger.  The
    # stratified stochastic map conditions each piece on exactly this.
    snaps = Vector{Matrix{Float64}}(undef, nper)

    for i in 1:nper
        snap = fill(NaN, nn, ns)
        for v in 1:nn
            isassigned(carried, v) && (snap[v, :] .= carried[v])
        end
        snaps[i] = snap

        keep = _allowed_states(ss, strat, i)
        sub_ss = _subset_statespace(ss, keep)
        mult = isempty(strat.dispersal_multipliers) ? nothing :
               strat.dispersal_multipliers[i]

        q = qmatrix(sub_ss, m; dispersal_multipliers = mult, r_float32 = r_float32)
        dense && (q = Matrix(q))
        w = cladogenesis_weights(sub_ss, m; dispersal_multipliers = mult, r_float32 = r_float32)

        for p in pieces[i]
            v = p.node

            if p.kind == :subbranch
                # A single branch segment crossing the whole period.
                fossil_age = is_tip(tr, v) && node_times[v] >= 0.001 ?
                             round(node_times[v]; digits = 7) : 0.0
                effective_length = fossil_age > b[i] ?
                    max(0.0, p.length - (fossil_age - b[i])) : p.length
                raw = apply_transition(q, effective_length, carried[v][keep]; exact = r_float32)
                carried[v] = _expand(raw, keep, ns)
                continue
            end

            sub = p.sub
            tipmat = Matrix{Float64}(undef, ntips(sub), length(keep))
            for (k, u) in enumerate(p.tip_orig)
                tipmat[k, :] .= carried[u][keep]
            end

            # Fixes that fall inside this piece, in the piece's own numbering
            # and reduced to the stratum's state space.
            fx = _piece_fixes(p, keep, fixes)

            res = downpass(sub, sub_ss, tipmat, q, w;
                           min_branchlength = min_branchlength,
                           rootedge = p.length > 0, r_float32 = r_float32,
                           fixnode = fx === nothing ? nothing : fx[1],
                           fixlikes = fx === nothing ? nothing : fx[2])

            for (k, u) in enumerate(p.internal_orig)
                sub_idx = ntips(sub) + k
                view(all_cond, u, keep) .= view(res.node_condlikes, sub_idx, :)
                view(all_bottom, u, keep) .= view(res.branch_bottom, sub_idx, :)
            end
            for (k, u) in enumerate(p.tip_orig)
                view(all_bottom, u, keep) .= view(res.branch_bottom, k, :)
            end

            if i < nper
                # R's carry is `conditional_likelihoods_at_bottom_of_root_branch`
                # (calc_loglike_sp_v1.R:2187): the transition matrix applied to
                # the piece root's NORMALISED downpass relprobs.  The scale
                # constants cancel against R's, which is why the TTL sums match
                # row for row.
                raw = apply_transition(q, p.length,
                                       view(res.node_relprobs, root_node(sub), :);
                                       exact = r_float32)
                carried[v] = _expand(raw, keep, ns)
            end
        end
    end

    all_bottom[root_node(tr), :] .= NaN

    return all_cond, all_bottom, pieces, snaps
end

function stratified_loglikelihood(tr::BGTree, ss::StateSpace, tips::TipRanges,
                                  model::Model, strat::Stratification;
                                  use_ambiguities::Bool = false, kwargs...)
    tipcl = tip_condlikes(tr, tips;
                          max_range_size = ss.max_range_size,
                          include_null_range = ss.include_null_range,
                          use_ambiguities = use_ambiguities)
    return stratified_loglikelihood(tr, ss, tipcl, model, strat; kwargs...)
end

"""
    stratified_ancestral_probs(tr, ss, tipcl, model, strat; kwargs...) -> AncestralProbs
    stratified_ancestral_probs(tr, ss, tips, model, strat; kwargs...) -> AncestralProbs

Marginal ancestral state probabilities under a time-stratified model.

Returns the same [`AncestralProbs`](@ref) structure as the non-stratified
[`ancestral_probs`](@ref).  `branch_top`/`branch_bottom` are the
downpass × uppass products, row-normalised; the root's branch-top row is the
downpass probability and its branch-bottom row is `NaN`, as in R (R stores `NA`
there because the root has no branch bottom).

`strat` is a [`Stratification`](@ref) giving the time boundaries and the
per-stratum constraints.  `dense` and `min_branchlength` are as in
[`stratified_loglikelihood`](@ref); `fixnode` / `fixlikes` fix the state
likelihoods at one or more internal nodes, exactly as in R's stratified uppass
(the fixed vector multiplies the node's branch-top probability after its
children have been processed, so it does not propagate further down the tree).
"""
function stratified_ancestral_probs(tr::BGTree, ss::StateSpace,
                                    tipcl::AbstractMatrix, model::Model,
                                    strat::Stratification;
                                    dense::Bool = false,
                                    min_branchlength::Real = 1e-6,
                                    fixnode = nothing, fixlikes = nothing,
                                    r_float32::Bool = false)
    size(tipcl) == (ntips(tr), nstates(ss)) ||
        throw(ArgumentError("tip_condlikes is $(size(tipcl)), expected " *
                            "($(ntips(tr)), $(nstates(ss)))"))

    node_times = _check_stratification(tr, strat)
    m = linked(model)
    b = vcat(0.0, strat.times)
    nper = length(strat.times)
    nn = nnodes(tr)
    ns = nstates(ss)
    fixes = _normalise_fixes(nn, ns, fixnode, fixlikes)

    all_cond, all_bottom, pieces, _ = _stratified_downpass(
        tr, ss, tipcl, model, strat; dense = dense,
        min_branchlength = min_branchlength,
        fixnode = fixnode, fixlikes = fixlikes, r_float32 = r_float32)
    lookup = _piece_lookup(pieces)

    up_top = zeros(Float64, nn, ns)
    up_bot = zeros(Float64, nn, ns)
    root = root_node(tr)
    up_top[root, :] .= 1.0 / ns        # R: 1/length(starting_probs)

    # Per-stratum, per-piece tip probabilities carried to the next stratum up.
    # A subtree piece stores a tips × states matrix; a subbranch stores a vector.
    piece_top = [Vector{Any}(undef, length(pieces[i])) for i in 1:nper]
    piece_bot = [Vector{Any}(undef, length(pieces[i])) for i in 1:nper]

    # R leaks the loop variable `for (m in 1:length(phy2$edge.length))`
    # (stratified_v1.R:3753) into the `m` argument of the UPASS COO build
    # (:2946): when `m` is non-NULL cladoRcpp multiplies every founder weight
    # by `m`.  Since subbranch pieces never run that loop, the leaked value is
    # the edge count of the LAST SUBTREE PIECE of the previously processed
    # (older) stratum; the oldest stratum keeps m = 1.  DEC is unaffected
    # (j = 0), but DEC+J's founder weights are scaled 1/2/4/... in the uppass.
    last_subtree_edges = ones(Int, nper)
    for s in 1:nper, p in pieces[s]
        p.kind == :subtree && (last_subtree_edges[s] = nnodes(p.sub) - 1)
    end
    up_founder_m = [i == nper ? 1 : last_subtree_edges[i + 1] for i in 1:nper]

    for i in nper:-1:1
        keep = _allowed_states(ss, strat, i)
        sub_ss = _subset_statespace(ss, keep)
        mult = isempty(strat.dispersal_multipliers) ? nothing :
               strat.dispersal_multipliers[i]

        q = qmatrix(sub_ss, m; dispersal_multipliers = mult, r_float32 = r_float32)
        dense && (q = Matrix(q))
        w = cladogenesis_weights(sub_ss, m; dispersal_multipliers = mult,
                                 r_float32 = r_float32,
                                 founder_multiplier = up_founder_m[i])

        for (jj, p) in enumerate(pieces[i])
            v = p.node

            if p.kind == :subbranch
                pjj, k = lookup[i + 1][v]
                prev_top = piece_top[i + 1][pjj]
                prev_bot = piece_bot[i + 1][pjj]
                anc_bottom = k == 0 ? prev_top : view(prev_top, k, :)
                anc_bot_branch = k == 0 ? prev_bot : view(prev_bot, k, :)

                # R's "phantom limb": if the node itself is younger than the
                # stratum's top, its branch section has not started yet, so the
                # probability is passed through without exponentiating.  This
                # applies to every subbranch except those in the youngest
                # stratum (where the node sits at the top).
                if node_times[v] < b[i]
                    piece_top[i][jj] = Vector{Float64}(anc_bottom)
                else
                    raw = vec(transition_matrix(q, p.length)' * anc_bottom[keep])
                    ss.include_null_range && (raw[1] = 0.0)
                    s = sum(raw)
                    s > 0 && (raw ./= s)
                    piece_top[i][jj] = _expand(raw, keep, ns)
                end
                piece_bot[i][jj] = Vector{Float64}(anc_bot_branch)
                continue
            end

            sub = p.sub
            ntip_sub = ntips(sub)

            # Uppass probability at the piece root's branch top.
            if i == nper
                root_top = Vector{Float64}(view(up_top, v, keep))
            else
                pjj, k = lookup[i + 1][v]
                prev_top = piece_top[i + 1][pjj]
                prev_bot = piece_bot[i + 1][pjj]
                anc_bottom = k == 0 ? prev_top : view(prev_top, k, :)
                anc_bot_branch = k == 0 ? prev_bot : view(prev_bot, k, :)

                if p.length > 0
                    raw = vec(transition_matrix(q, p.length)' * anc_bottom[keep])
                    ss.include_null_range && (raw[1] = 0.0)
                    s = sum(raw)
                    s > 0 && (raw ./= s)
                    root_top = raw
                else
                    root_top = Vector{Float64}(view(up_top, v, keep))
                end
                view(up_bot, v, keep) .= view(anc_bot_branch, keep)
            end

            # Downpass branch bottoms for the piece's nodes, reduced space.
            down_bottom = Matrix{Float64}(undef, nnodes(sub), length(keep))
            for (k, u) in enumerate(p.tip_orig)
                down_bottom[k, :] .= view(all_bottom, u, keep)
            end
            for (k, u) in enumerate(p.internal_orig)
                down_bottom[ntip_sub + k, :] .= view(all_bottom, u, keep)
            end

            ut, ub = _uppass_piece(sub, sub_ss, q, w, down_bottom, root_top,
                                   min_branchlength, _piece_fixes(p, keep, fixes))

            for (k, u) in enumerate(p.tip_orig)
                view(up_top, u, keep) .= view(ut, k, :)
                view(up_bot, u, keep) .= view(ub, k, :)
            end
            for (k, u) in enumerate(p.internal_orig)
                view(up_top, u, keep) .= view(ut, ntip_sub + k, :)
                # The piece root's bottom is not computed inside the piece; it
                # was set above from the stratum below (and is zero at the
                # global root), exactly as in R.
                if k > 1
                    view(up_bot, u, keep) .= view(ub, ntip_sub + k, :)
                end
            end
            view(up_top, v, keep) .= view(ut, root_node(sub), :)

            # Carry the tip probabilities to the stratum below in the full state
            # space, as R's `relprobs_at_tips_for_next_stratum_up` does.
            ptm = zeros(Float64, ntip_sub, ns)
            pbm = zeros(Float64, ntip_sub, ns)
            for k in 1:ntip_sub
                ptm[k, keep] .= view(ut, k, :)
                pbm[k, keep] .= view(ub, k, :)
            end
            piece_top[i][jj] = ptm
            piece_bot[i][jj] = pbm
        end
    end

    # Original tips: their uppass probability at the top of their branch comes
    # from the piece in the youngest stratum that holds them (R's final loop
    # over `piececlass == "orig_tip"` rows).
    for tn in 1:ntips(tr)
        youngest = findfirst(i -> haskey(lookup[i], tn), eachindex(lookup))
        youngest === nothing && throw(ArgumentError("tip $tn has no stratified tree piece"))
        pjj, k = lookup[youngest][tn]
        pt = piece_top[youngest][pjj]
        up_top[tn, :] .= k == 0 ? pt : view(pt, k, :)
    end

    # Final assembly: downpass × uppass, row-normalised.  The root is special
    # in both tables, exactly as in R.
    top_down = _normalise_rows_copy(all_cond)
    top = top_down .* up_top
    top[root, :] .= view(top_down, root, :)
    _normalise_rows_fallback!(top, top_down)

    bottom = all_bottom .* up_bot
    bottom[root, :] .= view(all_bottom, root, :)
    _normalise_rows_fallback!(bottom, all_bottom)

    # The stratified analogue of a `DownpassResult`: the node likelihoods, the
    # normalised branch bottoms and the per-node scalars.  There is no separate
    # root stem, so `rootedge_bottom` is empty.
    node_total = vec(sum(all_cond, dims = 2))
    ll = sum(log, node_total)
    dp = DownpassResult(ll, all_cond, top_down, node_total, all_bottom, Float64[])
    return AncestralProbs(top, bottom, up_top, up_bot, dp)
end

function stratified_ancestral_probs(tr::BGTree, ss::StateSpace, tips::TipRanges,
                                    model::Model, strat::Stratification;
                                    use_ambiguities::Bool = false, kwargs...)
    tipcl = tip_condlikes(tr, tips;
                          max_range_size = ss.max_range_size,
                          include_null_range = ss.include_null_range,
                          use_ambiguities = use_ambiguities)
    return stratified_ancestral_probs(tr, ss, tipcl, model, strat; kwargs...)
end

# The uppass loop for one piece.  Identical in structure to `ancestral_probs`,
# except that the root's branch-top probability is supplied instead of being
# uniform, and the sister downpass probabilities are the piece's own.
function _uppass_piece(sub::BGTree, ss::StateSpace, q::AbstractMatrix,
                       w::CladogenesisWeights, down_bottom::Matrix{Float64},
                       root_top::Vector{Float64}, min_branchlength::Real,
                       fixes = nothing)
    nn = nnodes(sub)
    ns = nstates(ss)
    rs = row_sums(w, ns)

    ut = zeros(Float64, nn, ns)
    ub = zeros(Float64, nn, ns)

    root = root_node(sub)
    ut[root, :] .= root_top

    for anc in (root, reverse(pruning_order(sub))...)
        is_tip(sub, anc) && continue

        l = left_child(sub, anc)
        r = right_child(sub, anc)
        hook = branch_length(sub, l) < min_branchlength ||
               branch_length(sub, r) < min_branchlength
        ww = hook ? _identity_weights(ss) : w
        rrs = hook ? row_sums(ww, ns) : rs

        if anc != root
            mother = parent_node(sub, anc)
            sister = left_child(sub, mother) == anc ? right_child(sub, mother) :
                     left_child(sub, mother)
            sister_is_left = left_child(sub, mother) == sister

            L, R = _uppass_split(ut[mother, :], ww, rrs, ss.include_null_range,
                                 sister_is_left ? view(down_bottom, sister, :) : nothing,
                                 sister_is_left ? nothing : view(down_bottom, sister, :))
            ub[anc, :] .= sister_is_left ? R : L
            ut[anc, :] .= vec(transition_matrix(q, branch_length(sub, anc))' *
                              view(ub, anc, :))
        end

        if is_tip(sub, l)
            L, _ = _uppass_split(ut[anc, :], ww, rrs, ss.include_null_range,
                                 nothing, view(down_bottom, r, :))
            ub[l, :] .= L
            ut[l, :] .= vec(transition_matrix(q, branch_length(sub, l))' *
                            view(ub, l, :))
        end
        if is_tip(sub, r)
            _, R = _uppass_split(ut[anc, :], ww, rrs, ss.include_null_range,
                                 view(down_bottom, l, :), nothing)
            ub[r, :] .= R
            ut[r, :] .= vec(transition_matrix(q, branch_length(sub, r))' *
                            view(ub, r, :))
        end

        # R's fixnode-on-uppass block: applied at the end of the node's turn,
        # after its children have been computed, and without renormalising.
        if fixes !== nothing
            k = findfirst(==(anc), fixes[1])
            k === nothing || (view(ut, anc, :) .*= view(fixes[2], k, :))
        end
    end

    return ut, ub
end

# Row-normalise a copy of `M` (used for the downpass branch tops).
function _normalise_rows_copy(M::AbstractMatrix)
    out = Matrix{Float64}(undef, size(M))
    for i in 1:size(M, 1)
        s = sum(view(M, i, :))
        out[i, :] .= s > 0 ? view(M, i, :) ./ s : view(M, i, :)
    end
    return out
end

# Row-normalise in place; rows whose sum is not a positive finite number are
# replaced by the fallback row (R's NaN handling in the final assembly).
function _normalise_rows_fallback!(M::Matrix{Float64}, fallback::AbstractMatrix)
    for i in 1:size(M, 1)
        s = sum(view(M, i, :))
        if isfinite(s) && s > 0
            view(M, i, :) ./= s
        else
            view(M, i, :) .= view(fallback, i, :)
        end
    end
    return M
end

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

"""
    tree_pieces(tr, strat) -> Vector{Vector{NamedTuple}}

The tree sliced into strata, in the shape of R's `master_table`.

`tree_pieces(tr, strat)[i]` holds the pieces of stratum `i` (youngest first).
Every piece is a `NamedTuple` with

* `node` — the node at the top of the piece, in the original numbering
* `piecenum` — its index within the stratum
* `piececlass` — `"subbranch"` for a single branch, `"subtree"` otherwise
* `subnode_type` — `"tip"`, `"internal"` or `"root"` within the piece
* `subedge_length` — the branch length inside this stratum; `NaN` for a
  subtree root, which has no branch below it
* `subtime_bp` — the node's time before present, measured from the bottom of
  its stratum (0 for anything at or below that boundary, which includes every
  piece tip)
* `time_top` / `time_bot` — the stratum's boundaries

R keeps all of this in one flat `master_table` with the stratum number in a
column; grouping by stratum is more useful here, since every stochastic map
walks one stratum at a time.
"""
function tree_pieces(tr::BGTree, strat::Stratification)
    node_times = _check_stratification(tr, strat)
    b = vcat(0.0, strat.times)
    nper = length(strat.times)
    secs = _section_pieces(tr, node_times, b, nper)

    out = Vector{Vector{NamedTuple}}(undef, nper)
    for i in 1:nper
        rows = NamedTuple[]
        for (j, p) in enumerate(secs[i])
            if p.kind == :subbranch
                push!(rows, (node = p.node, piecenum = j, piececlass = "subbranch",
                             subnode_type = "tip", subedge_length = p.length,
                             subtime_bp = max(node_times[p.node] - b[i], 0.0),
                             time_top = b[i], time_bot = b[i + 1]))
                continue
            end
            sub = p.sub
            for k in 1:ntips(sub)
                orig = p.tip_orig[k]
                push!(rows, (node = orig, piecenum = j, piececlass = "subtree",
                             subnode_type = "tip",
                             subedge_length = branch_length(sub, k),
                             subtime_bp = max(node_times[orig] - b[i], 0.0),
                             time_top = b[i], time_bot = b[i + 1]))
            end
            for k in 1:(nnodes(sub) - ntips(sub))
                orig = p.internal_orig[k]
                idx = ntips(sub) + k
                push!(rows, (node = orig, piecenum = j, piececlass = "subtree",
                             subnode_type = orig == p.node ? "root" : "internal",
                             subedge_length = orig == p.node ? NaN :
                                              branch_length(sub, idx),
                             subtime_bp = max(node_times[orig] - b[i], 0.0),
                             time_top = b[i], time_bot = b[i + 1]))
            end
        end
        out[i] = rows
    end
    return out
end

"""
    StratifiedBSMInputs

Everything a stratified stochastic map needs, precomputed once and reusable
across maps.  Built by [`stratified_bsm_inputs`](@ref).

Fields:

* `pieces` — per stratum, the [`tree_pieces`](@ref) decomposition (internal
  `_Piece` records: the node at the top, `:subbranch`/`:subtree`, the
  stratum-local length, and the sub-tree for subtree pieces)
* `q` / `w` / `rs` — the rate matrix, cladogenesis weights and their row sums,
  restricted to the states allowed in that stratum
* `keep` — the allowed state indices in the full state space
* `sub_P` — per stratum and piece, `exp(q·length)` for a subbranch (empty for
  subtrees, which are exponentiated internally by the recursive map)
* `root_probs` — the marginal ancestral probabilities at the root
* `bounds` — the stratum boundaries, `[0; times]`
"""
struct StratifiedBSMInputs{P}
    pieces::Vector{Vector{P}}
    q::Vector{Matrix{Float64}}
    w::Vector{CladogenesisWeights}
    wl::Vector{CladogenesisWeights}
    rs::Vector{Vector{Float64}}
    keep::Vector{Vector{Int}}
    sub_P::Vector{Vector{Matrix{Float64}}}
    root_probs::Vector{Float64}
    bounds::Vector{Float64}
end

function Base.show(io::IO, s::StratifiedBSMInputs)
    n = sum(length, s.pieces)
    print(io, "StratifiedBSMInputs: ", length(s.pieces), " strata, ", n, " pieces, ",
          length(s.root_probs), " states")
end

"""
    stratified_bsm_inputs(tr, ss, model, strat, ap; r_float32 = false)

Precompute the stratified stochastic-mapping inputs: one rate matrix and
cladogenesis table per stratum (restricted to that stratum's allowed states),
the transition matrix of every subbranch, and the root state probabilities
from `ap` (a [`stratified_ancestral_probs`](@ref) result).

R builds the same thing in `get_inputs_for_stochastic_mapping_stratified()`,
one list entry per time period, and then reuses it for every map.  Grouping by
stratum here keeps the per-piece transition matrices next to the pieces that
use them, so a map never has to recompute an exponential.
"""
function stratified_bsm_inputs(tr::BGTree, ss::StateSpace, model::Model,
                               strat::Stratification, ap::AncestralProbs;
                               r_float32::Bool = false)
    node_times = _check_stratification(tr, strat)
    b = vcat(0.0, strat.times)
    nper = length(strat.times)
    # R's get_inputs_for_stochastic_mapping() builds the BSM's split COO with
    # get_Qmat_COOmat_from_BioGeoBEARS_run_object(), which does NOT apply the
    # linked-parameter pass: the split sampling uses y = s = v = 1 even for
    # DEC+J.  Every conditioning row the map reads, though — the piece-root
    # sample, the split's branch-bottom likes and the daughter branch-top
    # sample (stochastic_map_given_inputs.R:481/761/1169) — comes from the
    # LIKELIHOOD's per-piece tables, which were computed with the linked
    # y = s = v = 2.9/3.  So `w` (unlinked) drives the split sampling while
    # `wl` (linked) drives the per-piece downpass/uppass conditioning.  The q
    # matrices are anagenetic and unaffected.
    m = model
    ml = linked(model)

    pieces = _section_pieces(tr, node_times, b, nper)
    qs = Vector{Matrix{Float64}}(undef, nper)
    ws = Vector{CladogenesisWeights}(undef, nper)
    wls = Vector{CladogenesisWeights}(undef, nper)
    rss = Vector{Vector{Float64}}(undef, nper)
    keeps = Vector{Vector{Int}}(undef, nper)
    subP = Vector{Vector{Matrix{Float64}}}(undef, nper)

    for i in 1:nper
        keep = _allowed_states(ss, strat, i)
        mult = isempty(strat.dispersal_multipliers) ? nothing :
               strat.dispersal_multipliers[i]
        sub_ss = _subset_statespace(ss, keep)
        qs[i] = Matrix(qmatrix(sub_ss, m; dispersal_multipliers = mult,
                               r_float32 = r_float32))
        ws[i] = cladogenesis_weights(sub_ss, m; dispersal_multipliers = mult,
                                     r_float32 = r_float32)
        wls[i] = cladogenesis_weights(sub_ss, ml; dispersal_multipliers = mult,
                                      r_float32 = r_float32)
        rss[i] = row_sums(ws[i], length(keep); r_float32 = r_float32)
        keeps[i] = keep
        ps = Matrix{Float64}[]
        for p in pieces[i]
            if p.kind == :subbranch
                push!(ps, transition_matrix(qs[i], p.length; exact = r_float32))
            else
                push!(ps, zeros(0, 0))
            end
        end
        subP[i] = ps
    end

    return StratifiedBSMInputs(pieces, qs, ws, wls, rss, keeps, subP,
                               collect(ap.branch_top[root_node(tr), :]), b)
end

"""
    stratified_branch_top_probs(inputs, stratum, piece, bottom_state, top_likes)

The state distribution at the top of one subbranch, given the state at its
bottom and the downpass likelihoods above it.

R computes `condprobs_branch_bot %*% P`, zeroes the null range (impossible in
a survivor) and normalises.  Because the bottom is a single state, that is just
the corresponding row of the piece's transition matrix, weighted by
`top_likes`.  Both vectors live in the stratum's restricted state space
([`StratifiedBSMInputs`](@ref) `keep`).
"""
function stratified_branch_top_probs(bi::StratifiedBSMInputs, stratum::Int,
                                     piece::Int, bottom_state::Int, top_likes;
                                     r_float32::Bool = true)
    keep = bi.keep[stratum]
    pos = findfirst(==(bottom_state), keep)
    pos === nothing &&
        throw(ArgumentError("state $bottom_state is not allowed in stratum $stratum"))
    P = bi.sub_P[stratum][piece]
    isempty(P) &&
        throw(ArgumentError("piece $piece of stratum $stratum is a subtree, " *
                            "which has no single-branch transition matrix"))
    p = P[pos, :] .* top_likes
    p[1] = 0.0                       # the null range cannot survive
    s = r_sum(p, r_float32)
    s > 0 || throw(ArgumentError("all branch-top probabilities are zero"))
    return p ./ s
end

"""
    next_piece_seed(seedval, counter) -> Int

R's stratified BSM reseeds before every piece and every sampled branch-top
state:

    counter += 1
    seedval  = seedval + counter * 90127        # accumulated, not reset
    seedval  = seedval mod 2147483647           # only when it overflows

The accumulator is what makes the reseeding order-dependent, so a caller that
skips a piece would desynchronise every later one.
"""
function next_piece_seed(seedval::Integer, counter::Integer)
    s = Int(seedval) + Int(counter) * 90127
    return s > 2147483647 ? mod(s, 2147483647) : s
end

function _check_stratification(tr::BGTree, strat::Stratification)
    node_times = node_times_bp(tr)
    height = tree_height(tr)
    strat.times[end] > height ||
        throw(ArgumentError("the last time boundary ($(strat.times[end])) must be older " *
                            "than the root ($height)"))
    count(>(height), strat.times) == 1 ||
        throw(ArgumentError("exactly one time boundary must be older than the root " *
                            "($height); got $(strat.times)"))
    for t in strat.times
        for (v, nt) in enumerate(node_times)
            abs(nt - t) < 1e-5 &&
                throw(ArgumentError("node $v has time $nt, too close to the time " *
                                    "boundary $t; move the boundary or the node"))
        end
    end
    return node_times
end

# States allowed in stratum `i`, in the full state-space order.  The null range
# is always kept.  `areas_allowed` uses R's `check_if_state_is_allowed()`, which
# compares the first area of the state against the whole range;
# `areas_adjacency` uses `check_if_state_is_allowed_by_adjacency()`, which
# requires every pair of the state's areas to be flagged adjacent.  When both
# are given a state must pass both, as in R.
function _allowed_states(ss::StateSpace, strat::Stratification, i::Int)
    has_aa = !isempty(strat.areas_allowed)
    has_adj = !isempty(strat.areas_adjacency)
    (has_aa || has_adj) || return collect(1:nstates(ss))
    n = nareas(ss)
    aa = has_aa ? _stratum_matrix(strat.areas_allowed[i], n, "areas_allowed", i) : nothing
    adj = has_adj ? _stratum_matrix(strat.areas_adjacency[i], n, "areas_adjacency", i) : nothing

    keep = Int[]
    for (k, state) in enumerate(ss.states)
        isempty(state) && (push!(keep, k); continue)
        if aa !== nothing
            first_area = state[1]
            all(b -> !iszero(aa[first_area, b]), state) || continue
        end
        if adj !== nothing
            all(a -> all(b -> !iszero(adj[a, b]), state), state) || continue
        end
        push!(keep, k)
    end
    isempty(keep) && throw(ArgumentError("no state is allowed in stratum $i"))
    return keep
end

function _stratum_matrix(mat::AbstractMatrix, n::Int, what::AbstractString, i::Int)
    size(mat) == (n, n) ||
        throw(ArgumentError("$(what)[$i] is $(size(mat)), expected ($n, $n)"))
    return mat
end

function _subset_statespace(ss::StateSpace, keep::Vector{Int})
    length(keep) == nstates(ss) && return ss
    states = [ss.states[k] for k in keep]
    return StateSpace(ss.areanames, states, ss.include_null_range, ss.max_range_size)
end

function _expand(v::AbstractVector, keep::Vector{Int}, ns::Int)
    out = zeros(Float64, ns)
    out[keep] .= v
    return out
end

# Nodes whose incoming branch crosses `boundary`: time[v] <= boundary < time[parent].
function _cut_nodes(tr::BGTree, node_times::AbstractVector{Float64}, boundary::Float64)
    out = Int[]
    for v in 1:nnodes(tr)
        p = parent_node(tr, v)
        tparent = p == 0 ? Inf : node_times[p]
        node_times[v] <= boundary < tparent && push!(out, v)
    end
    return out
end

# One piece of one stratum: either a single branch segment crossing the whole
# period (`:subbranch`) or a subtree rooted at a cut node (`:subtree`).
struct _Piece
    node::Int
    kind::Symbol              # :subbranch or :subtree
    length::Float64           # subbranch length, or the subtree's root edge
    sub::Union{Nothing,BGTree}
    tip_orig::Vector{Int}     # sub-tip index -> original node number
    internal_orig::Vector{Int}
end

# The piece sectioning shared by the downpass and the uppass.  Stratum `i`
# covers `(b[i], b[i+1]]`; the last stratum is never cut and is the leftover
# tree rooted at the root, with no root edge.
function _section_pieces(tr::BGTree, node_times::AbstractVector{Float64},
                         b::AbstractVector{Float64}, nper::Int)
    pieces = Vector{Vector{_Piece}}(undef, nper)
    rank = _postorder_rank(tr)
    for i in 1:nper
        top = b[i]
        cut = i < nper ? _cut_nodes(tr, node_times, b[i + 1]) : [root_node(tr)]
        ps = _Piece[]
        for v in cut
            # R's chainsaw2 works in relative time: every cut tip sits exactly
            # on the stratum's lower boundary, and a tip-branch piece always
            # gets the FULL stratum thickness (chainsaw2 records `timepoint`
            # verbatim for tips, so a newick's small nonzero tip ages are
            # ignored and the lineage is treated as reaching the present;
            # verified against Psychotria_M0strat, whose stratum-1 sections
            # are 0.5 long despite 0.00089 tip ages).  Subtree pieces get the
            # true section from the piece root's own age, because R builds
            # them as `timepoint - get_max_height_tree(subtree)` with the
            # subtree's cut tips sitting on the boundary.
            if is_tip(tr, v) || node_times[v] <= top
                len = i < nper ? b[i + 1] - top : 0.0
                push!(ps, _Piece(v, :subbranch, len, nothing, Int[], Int[]))
            else
                len = i < nper ? b[i + 1] - max(node_times[v], top) : 0.0
                p = _build_piece(tr, v, top, node_times, len; truncate = nper > 1)
                push!(ps, _Piece(v, :subtree, p.root_edge, p.tree, p.tip_orig,
                                 p.internal_orig))
            end
        end
        # R numbers pieces in the order the crossing branches appear in the
        # chopped tree's pruningwise edge matrix - the newick's left-to-right
        # postorder.  The BSM reseeded stream is handed out piece by piece in
        # that order, so the ordering must match R exactly.
        sort!(ps, by = p -> rank[p.node])
        pieces[i] = ps
    end
    return pieces
end

# Left-to-right postorder position of every node (children before parents,
# left before right) - the order ape's pruningwise edge matrix visits them.
function _postorder_rank(tr::BGTree)
    rank = zeros(Int, nnodes(tr))
    k = 0
    visit(v::Int) = begin
        if !is_tip(tr, v)
            visit(left_child(tr, v))
            visit(right_child(tr, v))
        end
        rank[v] = (k += 1)
    end
    visit(root_node(tr))
    return rank
end

# (stratum, original node) -> (piece number, sub-tip index; 0 for a subbranch).
# Every cut node is a piece tip in the stratum above, which is what the uppass
# looks up when it carries probabilities forward.
function _piece_lookup(pieces::Vector{Vector{_Piece}})
    out = [Dict{Int,Tuple{Int,Int}}() for _ in pieces]
    for i in eachindex(pieces)
        for (jj, p) in enumerate(pieces[i])
            if p.kind == :subbranch
                out[i][p.node] = (jj, 0)
            else
                for (k, u) in enumerate(p.tip_orig)
                    out[i][u] = (jj, k)
                end
            end
        end
    end
    return out
end

# The fixed nodes of `fixes` that are root or internal *within* this piece, in
# the piece's own node numbering, with the fixed likelihood vector reduced to
# the stratum's state space.  R does the same mapping through its master table's
# `SUBnode` / `SUBnode.type` columns; a node that is only a piece tip here is
# not fixed (it is fixed in the stratum where it is a subtree root).
function _piece_fixes(p::_Piece, keep::Vector{Int}, fixes)
    fixes === nothing && return nothing
    nodes, mat = fixes
    subnodes = Int[]
    rows = Int[]
    ntip = ntips(p.sub)
    for (k, u) in enumerate(p.internal_orig)
        j = findfirst(==(u), nodes)
        j === nothing && continue
        push!(subnodes, ntip + k)
        push!(rows, j)
    end
    isempty(subnodes) && return nothing
    return (subnodes, mat[rows, keep])
end

# A piece of one stratum: the subtree of T_{i-1} rooted at `v`, cut at `top`.
# Nodes with time > top stay internal; the first node on each path with
# time <= top becomes a tip of the piece (a cut point at the boundary).
struct _PieceTree
    tree::BGTree
    tip_orig::Vector{Int}       # sub-tip index -> original node number
    internal_orig::Vector{Int}  # sub-internal index (minus ntips) -> original node
    root_edge::Float64          # R's chainsaw2 root edge for this piece
end

function _build_piece(tr::BGTree, v::Int, top::Float64,
                      node_times::AbstractVector{Float64}, root_edge::Float64;
                      truncate::Bool = true)
    tips = Int[]
    internals = Int[]

    # An original tip is always a tip of the piece even when rounding leaves its
    # time a hair above the boundary.
    is_piece_tip(u) = is_tip(tr, u) || node_times[u] <= top

    function visit(u)
        if is_piece_tip(u)
            push!(tips, u)
            return
        end
        push!(internals, u)
        visit(left_child(tr, u))
        visit(right_child(tr, u))
    end
    visit(v)

    ntip = length(tips)
    nint = length(internals)
    n = ntip + nint

    tip_index = Dict{Int,Int}(u => k for (k, u) in enumerate(tips))
    int_index = Dict{Int,Int}(u => ntip + k for (k, u) in enumerate(internals))
    sub_index(u) = is_piece_tip(u) ? tip_index[u] : int_index[u]

    parent = zeros(Int, n)
    left = zeros(Int, n)
    right = zeros(Int, n)
    edge = zeros(Float64, n)

    for (k, u) in enumerate(internals)
        iu = ntip + k
        for c in (left_child(tr, u), right_child(tr, u))
            ic = sub_index(c)
            if c == left_child(tr, u)
                left[iu] = ic
            else
                right[iu] = ic
            end
            parent[ic] = iu
            # A cut tip's edge is shortened to the stratum's lower boundary.
            # R's chainsaw2 does this as `edge.length + (time_bp - timepoint)`
            # where prt() has already rounded the tip's time_bp to 7 decimals;
            # for a newick with a small nonzero tip age (Psychotria's
            # 0.0008897866802) that adds ~1.3e-8 to the exact
            # `parent_time - timepoint`.  Reproduce it: DEC+J's split sampling
            # has exact founder-scenario ties that a 1-ulp conditioning
            # difference breaks.  A cut INTERNAL node has no tip age and keeps
            # the exact shortened length; a single-period analysis keeps every
            # original edge.
            edge[ic] = if !truncate || !is_piece_tip(c)
                tr.edge_length[c]
            elseif is_tip(tr, c) && node_times[c] >= 0.001 && node_times[c] > top
                tr.edge_length[c]
            elseif is_tip(tr, c)
                tr.edge_length[c] + round(node_times[c]; digits = 7) - top
            else
                node_times[u] - top
            end
        end
    end

    root_sub = int_index[v]
    edge[root_sub] = 0.0

    # R's chainsaw2 sets the piece's root edge to
    # `timepoint - get_max_height_tree(piece)`, where timepoint is the stratum
    # thickness and get_max_height_tree is the longest root-to-tip path using
    # the edge lengths above (including the rounded tip edges).  With the cut
    # tip edges corrected, the naive `b[i+1] - node_time(v)` is longer by the
    # same ~1.3e-8, so recompute it here to keep the piece geometry exact.
    height = copy(edge)
    for (k, u) in enumerate(tips)
        if is_tip(tr, u) && node_times[u] >= 0.001 && node_times[u] > top
            height[k] += round(node_times[u]; digits = 7) - top
        end
    end
    for k in nint:-1:1
        iu = ntip + k
        height[iu] += max(height[left[iu]], height[right[iu]])
    end
    # Equivalent, and clearer: stratum thickness minus the longest path.
    # `root_edge` as passed in equals `b[i+1] - node_times[v]` for a cut
    # stratum (or 0 for the last one), so `root_edge + node_times[v] - top`
    # is the stratum thickness.
    thickness = root_edge + node_times[v] - top
    corrected_root_edge = thickness - height[root_sub]

    post = Int[]
    function postvisit(x)
        1 <= x <= ntip || (postvisit(left[x]); postvisit(right[x]))
        push!(post, x)
    end
    postvisit(root_sub)
    pop!(post)                     # the root is not part of pruning_order

    sub = BGTree([string(u) for u in tips], [string(u) for u in internals],
                 parent, left, right, edge, post, ntip, nint, corrected_root_edge)
    return _PieceTree(sub, tips, internals, corrected_root_edge)
end
