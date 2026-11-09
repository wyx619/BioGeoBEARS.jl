# Biogeographic stochastic mapping.

"""
    AnaEvent

One anagenetic (along-branch) event, mirroring a row of R's `events_table_for_branch`.
"""
struct AnaEvent
    node::Int              # nodenum_at_top_of_branch
    trynum::Int
    brlen::Float64
    from_state::Int
    to_state::Int
    from_range::String
    to_range::String
    abs_event_time::Float64
    event_time::Float64
    event_type::String     # "d", "e" or "a"
    event_txt::String
    new_area::String       # "-" when not a dispersal
    lost_area::String      # "-" when not an extinction
    dispersal_to::String
    extirpation_from::String
end

"""
    BSMResult

The outcome of one [`stochastic_map`](@ref), laid out like R's `trtable` after
`stochastic_map_given_inputs()` plus the per-branch anagenetic events.

* `node_states` — sampled state at the top of each node's branch
  (R's `sampled_states_AT_nodes`)
* `branch_bottom_states` — sampled state at the bottom of each node's branch
  (R's `sampled_states_AT_brbots`)
* `left_corner` / `right_corner` — the two daughter states right after
  cladogenesis at each internal node
* `branch_events` — the anagenetic events below each node
* `clado_event_type` / `clado_event_txt` / `clado_dispersal_to` — the
  cladogenesis labels for each internal node
"""
struct BSMResult
    node_states::Vector{Int}
    branch_bottom_states::Vector{Int}
    left_corner::Vector{Int}
    right_corner::Vector{Int}
    branch_events::Vector{Vector{AnaEvent}}
    clado_event_type::Vector{String}
    clado_event_txt::Vector{String}
    clado_dispersal_to::Vector{String}
    seedval::Int
end

function Base.show(io::IO, b::BSMResult)
    n = sum(length, b.branch_events)
    print(io, "BSMResult: ", length(b.node_states), " nodes, ", n,
          " anagenetic events, seed ", b.seedval)
end

# ---------------------------------------------------------------------------
# ape scans cladewise edges repeatedly, collecting nodes whose children are
# ready. This ordering is distinct from depth-first postorder on some trees.
function _ape_pruning_nodes(tr::BGTree)
    ready = [is_tip(tr, n) for n in 1:nnodes(tr)]
    order = Int[]
    while length(order) < tr.nnode
        for anc in root_node(tr):nnodes(tr)
            if !ready[anc] && ready[left_child(tr, anc)] && ready[right_child(tr, anc)]
                push!(order, anc)
                ready[anc] = true
            end
        end
    end
    return order
end

# One branch
# ---------------------------------------------------------------------------

"""
    stochastic_map_branch(rng, brlen, st_bottom, st_top, q, ss; kwargs...)
        -> Vector{AnaEvent}

Simulate anagenetic events along one branch of length `brlen`, from state
`st_bottom` (at the bottom) to state `st_top` (at the top).

R's `stochastic_map_branch()` draws an exponential waiting time from the
current state's total rate, picks the next state from the off-diagonal row of
`Q`, records the event, and repeats until either the branch runs out or the
state reaches `st_top`.  A whole attempt is thrown away and restarted if the
endpoint does not match, up to `maxtries`.

`abs_time_bottom` is the absolute (time-before-present) age of the bottom of
the branch; event ages are `abs_time_bottom - event_time`.

`manual_history` enables R's fallback for branches whose endpoint cannot be
reached by forward simulation (e.g. highly constrained state spaces): the set
of area gains/losses is shuffled and given uniformly spaced times.  R prints a
long note when this triggers; here it is silent.
"""
function stochastic_map_branch(rng::RRng, brlen::Real, st_bottom::Integer,
                               st_top::Integer, q::AbstractMatrix, ss::StateSpace;
                               abs_time_bottom::Real = 0.0,
                               maxtries::Integer = 40000,
                               manual_history::Bool = true,
                               sampling_indices::Union{Nothing,Vector{Int}} = nothing,
                               sampling_size::Int = 0,
                               maxrepeats::Integer = 10000,
                               r_float32::Bool = false)
    brlen > 0 || throw(ArgumentError("brlen must be positive, got $brlen"))
    ns = size(q, 1)
    states = ss.states
    areas = ss.areanames
    ranges = [state_text(ss, i; split = false) for i in 1:ns]
    rates = [-q[i, i] for i in 1:ns]

    trynum = 0
    while true
        trynum += 1
        curr_time = 0.0
        current = Int(st_bottom)
        events = AnaEvent[]
        repeatnum = 1

        while true
            rate = rates[current]
            rate == 0 && break                    # null range: nothing happens
            waiting_time = r_rexp(rng, rate)
            curr_time += waiting_time
            if curr_time > brlen
                curr_time = brlen
                break
            end
            if repeatnum >= maxrepeats
                break
            else
                repeatnum += 1
            end

            row = q[current, :]
            probs = [row[k] < 0 ? 0.0 : Float64(row[k]) for k in 1:ns]
            probs ./= r_sum(probs, r_float32)
            if sampling_indices === nothing
                new = r_sample_prob(rng, probs, 1; replace = false)[1]
            else
                full_probs = zeros(Float64, sampling_size)
                full_probs[sampling_indices] .= probs
                sampled = r_sample_prob(rng, full_probs, 1; replace = false)[1]
                new = findfirst(==(sampled), sampling_indices)::Int
            end

            push!(events, _make_event(current, new, brlen, curr_time, trynum,
                                      abs_time_bottom, states, ranges, areas))
            current = new
        end

        current == st_top && return events

        if trynum > maxtries
            manual_history || error("no successful history on this branch after " *
                                    "$maxtries tries")
            return _manual_history(rng, brlen, st_bottom, st_top, states, ranges,
                                   areas, abs_time_bottom; trynum = maxtries + 2)
        end
    end
end

# Classify one anagenetic step and build its event row (R lines 776-851).
function _make_event(cur::Int, new::Int, brlen, curr_time, trynum, abs_time_bottom,
                     states, ranges, areas)
    cur_st = states[cur]
    new_st = states[new]
    event_type = ""
    new_area = "-"
    lost_area = "-"
    dispersal_to = "-"
    extirpation_from = "-"

    if length(new_st) > length(cur_st)
        event_type = "d"
        added = [a for a in new_st if !(a in cur_st)]
        new_area = join((areas[a] for a in added))
        dispersal_to = new_area
    elseif length(new_st) < length(cur_st)
        event_type = "e"
        lost = [a for a in cur_st if !(a in new_st)]
        lost_area = join((areas[a] for a in lost))
        extirpation_from = lost_area
    else
        if ranges[new] == "_"
            event_type = "e"
            lost = copy(cur_st)
            lost_area = join((areas[a] for a in lost))
            extirpation_from = lost_area
        else
            event_type = "a"
            new_area = join((areas[a] for a in new_st))
            lost_area = join((areas[a] for a in cur_st))
            dispersal_to = new_area
            extirpation_from = lost_area
        end
    end

    return AnaEvent(0, Int(trynum), Float64(brlen), cur, new,
                    ranges[cur], ranges[new],
                    abs_time_bottom - curr_time, curr_time,
                    event_type, ranges[cur] * "->" * ranges[new],
                    new_area, lost_area, dispersal_to, extirpation_from)
end

# R's "manual history for difficult branches" fallback (lines 930-1230).
function _manual_history(rng::RRng, brlen, st_bottom, st_top, states, ranges, areas,
                         abs_time_bottom; max_tries::Integer = 10000,
                         trynum::Integer = 1)
    start_areas = states[st_bottom]
    end_areas = states[st_top]

    adds = [a for a in end_areas if !(a in start_areas)]
    subs = [a for a in start_areas if !(a in end_areas)]
    ops = vcat([(a, :add) for a in adds], [(a, :sub) for a in subs])

    # Every range in the state space, for the "intermediate must be allowed" test.
    allowed = Set(ranges)

    local order::Vector{Int}
    local ok::Bool = false
    tries = 0
    while !ok
        tries += 1
        order = r_sample_int(rng, length(ops), length(ops); replace = false)
        tmp = copy(start_areas)
        ok = true
        for (mm, oi) in enumerate(order)
            a, op = ops[oi]
            if op == :add
                tmp = sort(vcat(tmp, a))
            else
                tmp = [x for x in tmp if x != a]
            end
            if isempty(tmp)
                # A null intermediate is only acceptable as the final state,
                # and only when the endpoint really is the null range.
                if !(isempty(end_areas) && mm == length(order))
                    ok = false
                    break
                end
            else
                txt = join((areas[a] for a in tmp))
                if !(txt in allowed)
                    ok = false
                    break
                end
            end
        end
        tries >= max_tries && break
    end

    times = sort([r_unif_rand(rng) * brlen for _ in 1:length(ops)])

    events = AnaEvent[]
    cur = st_bottom
    cur_areas = copy(start_areas)
    for (mm, oi) in enumerate(order)
        a, op = ops[oi]
        if op == :add
            cur_areas = sort(vcat(cur_areas, a))
        else
            cur_areas = [x for x in cur_areas if x != a]
        end
        new = findfirst(==(cur_areas), states)
        new === nothing && error("manual history produced a state outside the state space")
        events = vcat(events, _make_event(cur, new, brlen, times[mm], trynum,
                                          abs_time_bottom, states, ranges, areas))
        cur = new
    end
    return events
end

# ---------------------------------------------------------------------------
# Whole tree
# ---------------------------------------------------------------------------

"""
    stochastic_map(tr, ss, q, w, dp, ap; seedval, kwargs...) -> BSMResult

Draw one biogeographic stochastic map under the model that produced `q` (the
rate matrix), `w` (cladogenesis weights), `dp` (a [`DownpassResult`](@ref)) and
`ap` (an [`AncestralProbs`](@ref)).

`seedval` seeds an [`RRng`](@ref), so a given seed reproduces R's
`stochastic_map_given_inputs(seedval=seedval)` exactly.

Keyword arguments:

* `maxtries` — attempts per branch before falling back (R's default 40000)
* `min_branchlength` — branches shorter than this are hooks (no cladogenesis)
* `include_null_range` — whether state 1 is the null range
* `allow_null_tips` — R's escape hatch for null-range tips; when `false` the
  null range is zeroed out at branch tops
* `manual_history` — enable R's fallback for difficult branches
* `r_float32` — round each cladogenesis weight to single precision before
  summing, as R's `rcpp_calc_rowsums_for_COOweights_columnar()` does (it sums
  C `float`s into a double).  Pass `true` together with
  `qmatrix(...; r_float32=true)` and `cladogenesis_weights(...;
  r_float32=true)` to reproduce R's stream exactly.
"""
function stochastic_map(tr::BGTree, ss::StateSpace, q::AbstractMatrix,
                        w::CladogenesisWeights, dp::DownpassResult,
                        ap::AncestralProbs;
                        seedval::Integer = 0,
                        rng::Union{Nothing,RRng} = nothing,
                        maxtries::Integer = 40000,
                        min_branchlength::Real = 1e-6,
                        include_null_range::Bool = ss.include_null_range,
                        allow_null_tips::Bool = false,
                        manual_history::Bool = true,
                        root_state::Union{Nothing,Integer} = nothing,
                        time_offset::Real = 0.0,
                        renormalize_branch_bottom::Bool = false,
                        sampling_indices::Union{Nothing,Vector{Int}} = nothing,
                        sampling_size::Int = 0,
                        r_float32::Bool = false)
    rng === nothing && (rng = RRng(seedval))
    nn = nnodes(tr)
    ns = nstates(ss)
    rs = row_sums(w, ns; r_float32 = r_float32)
    areas = ss.areanames
    ranges = [state_text(ss, i; split = false) for i in 1:ns]
    pt = prt_table(tr)

    node_states = zeros(Int, nn)
    brbot_states = zeros(Int, nn)
    left_corner = zeros(Int, nn)
    right_corner = zeros(Int, nn)
    branch_events = [AnaEvent[] for _ in 1:nn]

    root = root_node(tr)

    # 1. Global root state.  A stratified caller that already sampled it passes
    # it in, which is what R's `stochastic_mapping_on_stratified()` does: the
    # root draw uses the map's own seed, not the piece's.
    node_states[root] = root_state === nothing ? r_sample_prob(rng, ap.branch_top[root, :]) :
                                                Int(root_state)

    # 2. Reverse ape's pruningwise order for the root-to-tip uppass.
    for anc in Iterators.reverse(_ape_pruning_nodes(tr))
        is_tip(tr, anc) && continue
        l = left_child(tr, anc)
        r = right_child(tr, anc)
        hook = branch_length(tr, l) < min_branchlength ||
               branch_length(tr, r) < min_branchlength

        statenum = node_states[anc]

        if hook
            left_dec = statenum
            right_dec = statenum
        else
            probs_anc = zeros(Float64, ns)
            probs_anc[statenum] = 1.0
            left_likes = dp.branch_bottom[l, :]
            right_likes = dp.branch_bottom[r, :]
            if renormalize_branch_bottom
                # R normalizes the collected stratified branch-bottom table a
                # second time before using it for split sampling.
                left_likes ./= r_sum(left_likes, r_float32)
                right_likes ./= r_sum(right_likes, r_float32)
            end
            left_dec, right_dec = _sample_split(rng, probs_anc, w, rs,
                                                left_likes, right_likes,
                                                r_float32)
        end

        left_corner[anc] = left_dec
        right_corner[anc] = right_dec
        brbot_states[l] = left_dec
        brbot_states[r] = right_dec

        # 3. Push each corner up its branch, combine with the downpass, sample.
        for (child, dec) in ((l, left_dec), (r, right_dec))
            bot = zeros(Float64, ns)
            bot[dec] = 1.0
            top = vec(transition_matrix(q, branch_length(tr, child);
                                        exact = r_float32)' * bot)
            if include_null_range && !allow_null_tips
                top[1] = 0.0
            end
            probs = top .* dp.node_relprobs[child, :]
            # R normalises here and then again inside sample(); the second
            # division is a no-op mathematically but not in the last bit, so
            # keep both to stay bit-identical.
            probs ./= r_sum(probs, r_float32)
            node_states[child] = r_sample_prob(rng, probs)
        end

        # 4. Simulate the anagenetic events on both daughter branches.
        for child in (l, r)
            blen = branch_length(tr, child)
            blen > 0 || continue
            abs_bottom = time_offset + pt.time_bp[child] + blen
            evs = stochastic_map_branch(rng, blen, brbot_states[child],
                                        node_states[child], q, ss;
                                        abs_time_bottom = abs_bottom,
                                        maxtries = maxtries,
                                        manual_history = manual_history,
                                        sampling_indices = sampling_indices,
                                        sampling_size = sampling_size,
                                        r_float32 = r_float32)
            for e in evs
                push!(branch_events[child], _retag(e, child))
            end
        end
    end

    clado_type = fill("", nn)
    clado_txt = fill("", nn)
    clado_disp = fill("", nn)
    for anc in 1:nn
        is_tip(tr, anc) && continue
        t, txt, d = _label_clado(node_states[anc], left_corner[anc],
                                 right_corner[anc], ss)
        clado_type[anc] = t
        clado_txt[anc] = txt
        clado_disp[anc] = d
    end

    return BSMResult(node_states, brbot_states, left_corner, right_corner,
                     branch_events, clado_type, clado_txt, clado_disp,
                     Int(seedval))
end

_retag(e::AnaEvent, node::Int) =
    AnaEvent(node, e.trynum, e.brlen, e.from_state, e.to_state, e.from_range,
             e.to_range, e.abs_event_time, e.event_time, e.event_type,
             e.event_txt, e.new_area, e.lost_area, e.dispersal_to,
             e.extirpation_from)

# Lift a sub-tree event's states back into the full state space, so that a
# stratified map reports one consistent numbering across strata.
_remap(e::AnaEvent, node::Int, keep::Vector{Int}) =
    AnaEvent(node, e.trynum, e.brlen, keep[e.from_state], keep[e.to_state],
             e.from_range, e.to_range, e.abs_event_time, e.event_time,
             e.event_type, e.event_txt, e.new_area, e.lost_area,
             e.dispersal_to, e.extirpation_from)

"""
    stochastic_map(tr, ss, model, dp, ap; seedval, kwargs...) -> BSMResult

Convenience form that builds `Q` and the cladogenesis weights from `model`.
"""
function stochastic_map(tr::BGTree, ss::StateSpace, model::Model,
                        dp::DownpassResult, ap::AncestralProbs; kwargs...)
    # R's runBSM() takes its COO table from
    # get_Qmat_COOmat_from_BioGeoBEARS_run_object(), which does NOT apply the
    # linked-parameter pass: a DEC+J model with j > 0 samples its splits with
    # y = s = v = 1 even though the likelihood (and the dp/ap passed in here)
    # used y = s = v = 2.9/3.  q is anagenetic, so it is unaffected.
    return stochastic_map(tr, ss, qmatrix(ss, model),
                          cladogenesis_weights(ss, model), dp, ap; kwargs...)
end

"""
    run_bsm(tr, ss, q, w, dp, ap; seedval, n_maps = 50, kwargs...)
        -> (Vector{BSMResult}, RRng)

Draw `n_maps` stochastic maps, mirroring R's `runBSM()`.  Map `m` is seeded
with `seedval + m`, and the returned [`RRng`](@ref) is the state left by the
last map — R's global stream continues from there, so
[`simulate_source_areas!`](@ref) must use this object to reproduce R's
source-area draws.
"""
function run_bsm(tr::BGTree, ss::StateSpace, q::AbstractMatrix,
                 w::CladogenesisWeights, dp::DownpassResult, ap::AncestralProbs;
                 seedval::Integer, n_maps::Integer = 50, kwargs...)
    n_maps > 0 || throw(ArgumentError("n_maps must be positive"))
    maps = Vector{BSMResult}(undef, n_maps)
    local rng
    for m in 1:n_maps
        rng = RRng(seedval + m)
        maps[m] = stochastic_map(tr, ss, q, w, dp, ap;
                                 seedval = seedval + m, rng = rng, kwargs...)
    end
    return maps, rng
end

"""
    run_bsm(tr, ss, model, dp, ap; seedval, n_maps = 50, kwargs...)
        -> (Vector{BSMResult}, RRng)

Convenience form that builds `Q` and the cladogenesis weights from `model`.
"""
function run_bsm(tr::BGTree, ss::StateSpace, model::Model,
                 dp::DownpassResult, ap::AncestralProbs; kwargs...)
    # Unlinked split-sampling weights, as in stochastic_map above.
    return run_bsm(tr, ss, qmatrix(ss, model),
                   cladogenesis_weights(ss, model), dp, ap; kwargs...)
end

"""
    run_bsm(tr, ss, model, strat, tipcl; seedval, n_maps=50, kwargs...)
        -> Vector{BSMResult}

Draw time-stratified maps using `seedval + m` for map `m`, as in R's
`runBSM`. Each map independently reconstructs its stratified inputs. With
`return_rng=true`, return `(maps, rng)` with the final map's RNG state for
subsequent source-area sampling. The default returns only the maps.
Errors propagate and no retry or partial-result policy is applied.
"""
function run_bsm(tr::BGTree, ss::StateSpace, model::Model,
                 strat::Stratification, tipcl::AbstractMatrix;
                 seedval::Integer, n_maps::Integer = 50,
                 return_rng::Bool = false, kwargs...)
    n_maps > 0 || throw(ArgumentError("n_maps must be positive"))
    maps = BSMResult[]
    rng = nothing
    for m in 1:n_maps
        map, rng = stochastic_map_stratified(tr, ss, model, strat, tipcl;
            seedval = Base.checked_add(seedval, m), return_rng = true, kwargs...)
        push!(maps, map)
    end
    return return_rng ? (maps, rng) : maps
end

# Sample one cladogenesis scenario given a one-hot ancestral state, exactly as
# R's `calc_uppass_scenario_probs_new2()` + `sample_uppass_split_scenario...`.
function _sample_split(rng::RRng, probs_anc::Vector{Float64},
                       w::CladogenesisWeights, rs, left_likes, right_likes,
                       exact::Bool = false)
    nscen = length(w.weight)
    rel = Vector{Float64}(undef, nscen)
    for k in 1:nscen
        a = w.ancestor[k]
        if rs[a] <= 0
            rel[k] = 0.0
            continue
        end
        rel[k] = probs_anc[a] * left_likes[w.left[k]] * right_likes[w.right[k]] *
                 (w.weight[k] / rs[a])
    end
    total = r_sum(rel, exact)
    total > 0 || error("all split-scenario probabilities are zero")
    rel ./= total
    k = r_sample_prob(rng, rel)
    return w.left[k], w.right[k]
end

# R's `label_table_of_cladogenetic_events()`.
function _label_clado(parent::Int, left::Int, right::Int, ss::StateSpace)
    areas = ss.areanames
    ranges = [state_text(ss, i; split = false) for i in 1:length(ss)]
    txt = ranges[parent] * "->" * ranges[left] * "," * ranges[right]

    left == right && return ("sympatry (y)", txt, "")

    parent_areas = sort(ss.states[parent])
    left_areas = ss.states[left]
    right_areas = ss.states[right]
    merged = sort(vcat(left_areas, right_areas))
    desc = sort(unique(vcat(left_areas, right_areas)))

    if length(desc) > length(parent_areas)
        new_areas = [a for a in desc if !(a in parent_areas)]
        return ("founder (j)", txt, join((areas[a] for a in new_areas)))
    elseif length(parent_areas) == length(merged) && parent_areas == merged
        return ("vicariance (v)", txt, "")
    else
        return ("subset (s)", txt, "")
    end
end

# ---------------------------------------------------------------------------
# Stratified maps
# ---------------------------------------------------------------------------

# A stratum's rate matrix embedded in the full state space, with zeros on the
# rows and columns of states that stratum does not allow.  R builds the same
# `Qmat_big` because `stochastic_map_branch()` indexes by full-space state.
function _expand_q(q::AbstractMatrix, keep::Vector{Int}, ns::Int)
    big = zeros(Float64, ns, ns)
    for (a, ka) in enumerate(keep), (b, kb) in enumerate(keep)
        big[ka, kb] = q[a, b]
    end
    return big
end

# The downpass likelihoods at the top of a piece, restricted to the stratum's
# states.  For an original tip that is the observation; for a cut point it is
# the message its subtree sent up, which the stratified downpass left in
# `all_bottom`.

"""
    stochastic_map_stratified(tr, ss, model, strat, tipcl; seedval, kwargs...)

Draw one time-stratified stochastic map, mirroring R's
`stochastic_mapping_on_stratified()`.

The tree is walked from the oldest stratum to the youngest.  Each stratum has
its own rate matrix, restricted to the states its `areas_allowed` /
`areas_adjacency` blocks permit; every piece is simulated conditional on the
state already known at the bottom of the stratigraphic column, and the states
it ends with become the bottoms of the pieces in the younger stratum.

R reseeds before **every** piece (`seedval + counter * 90127`), so the maps are
not one continuous stream; [`next_piece_seed`](@ref) reproduces that chain.

Returns a [`BSMResult`](@ref) in the original node numbering.  Cladogenesis
labels are only produced inside subtree pieces, since a single-branch segment
never splits.
"""
function stochastic_map_stratified(tr::BGTree, ss::StateSpace, model::Model,
                                   strat::Stratification, tipcl::AbstractMatrix;
                                   seedval::Integer = 0,
                                   maxtries::Integer = 40000,
                                   min_branchlength::Real = 1e-6,
                                   manual_history::Bool = true,
                                   verbose::Bool = false,
                                   r_float32::Bool = true,
                                   return_rng::Bool = false)
    node_times = _check_stratification(tr, strat)
    b = vcat(0.0, strat.times)
    nper = length(strat.times)

    pieces = _section_pieces(tr, node_times, b, nper)
    _, _, _, snaps = _stratified_downpass(tr, ss, tipcl, model, strat;
                                          r_float32 = r_float32)
    ap = stratified_ancestral_probs(tr, ss, tipcl, model, strat;
                                    r_float32 = r_float32)
    bi = stratified_bsm_inputs(tr, ss, model, strat, ap; r_float32 = r_float32)

    nn = nnodes(tr)
    ns = nstates(ss)
    node_states = zeros(Int, nn)
    brbot_states = zeros(Int, nn)
    cut_states = zeros(Int, nn)
    clado_type = fill("", nn)
    clado_txt = fill("", nn)
    clado_disp = fill("", nn)
    branch_events = [AnaEvent[] for _ in 1:nn]

    root = root_node(tr)
    rng = RRng(seedval)
    node_states[root] = r_sample_prob(rng, bi.root_probs)

    counter = 1
    seed = Int(seedval)

    for i in nper:-1:1
        keep = bi.keep[i]
        q = bi.q[i]
        # Unlinked weights drive the split sampling; linked weights drive the
        # per-piece downpass/uppass conditioning (R reads the likelihood's own
        # per-piece tables for every conditioning row — piece root at
        # stochastic_map_given_inputs.R:481, split branch-bottoms at :761,
        # daughter branch-tops at :1169 — and those were computed with the
        # linked parameters).
        w = bi.w[i]
        wl = bi.wl[i]
        qbig = _expand_q(q, keep, ns)
        sub_ss = _subset_statespace(ss, keep)

        ranked_pieces = [(j, p) for (j, p) in enumerate(pieces[i])]
        if i < nper
            rank = _postorder_rank(tr)
            for n in 1:ntips(tr)
                if node_times[n] >= 0.001 && node_times[n] >= b[i + 1] &&
                   node_times[parent_node(tr, n)] > b[i + 1]
                    push!(ranked_pieces, (0, _Piece(n, :phantom, 0.0, nothing, Int[], Int[])))
                end
            end
            sort!(ranked_pieces, by = x -> rank[x[2].node])
        end
        for (j, p) in ranked_pieces
            if p.kind == :phantom
                # R reseeds for both the draw and the skipped phantom history.
                counter += 1
                seed = next_piece_seed(seed, counter)
                rng = RRng(seed)
                counter += 1
                seed = next_piece_seed(seed, counter)
                rng = RRng(seed)
                continue
            end
            if p.kind == :subbranch
                bot = cut_states[p.node]
                bot == 0 &&
                    throw(ArgumentError("no state at the boundary above node " *
                                        "$(p.node) in stratum $i"))
                tl = snaps[i][p.node, keep]
                any(isnan, tl) &&
                    throw(ArgumentError("no downpass message at node $(p.node) " *
                                        "for stratum $i"))
                fossil_in_bin = is_tip(tr, p.node) && node_times[p.node] >= 0.001 &&
                                b[i] <= node_times[p.node] < b[i + 1]
                probs = fossil_in_bin ? tl ./ r_sum(tl, r_float32) :
                        stratified_branch_top_probs(bi, i, j, bot, tl; r_float32 = r_float32)
                # R burns one counter value for the branch-top sample and a
                # second one for the anagenetic history, each on its own stream.
                counter += 1
                seed = next_piece_seed(seed, counter)
                verbose && @info "piece" i j p.node p.kind counter seed
                rng = RRng(seed)
                k = r_sample_prob(rng, probs)
                node_states[p.node] = keep[k]
                cut_states[p.node] = node_states[p.node]
                brbot_states[p.node] == 0 && (brbot_states[p.node] = bot)

                blen = fossil_in_bin ? b[i + 1] - round(node_times[p.node]; digits = 7) : p.length
                if blen > 0
                    counter += 1
                    seed = next_piece_seed(seed, counter)
                    rng = RRng(seed)
                    verbose && @info "  history" i j p.node counter seed
                    # R's abs_time_at_branch_bottom =
                    # `time_top + SUBtime_bp + brlen` = `b[i+1] + SUBtime_bp`.
                    # In the youngest stratum the subbranch's bottom is an
                    # original tip, whose prt()-rounded time_bp is a small
                    # nonzero tip age (Psychotria: 0.0008898); every other
                    # stratum's subbranch bottom is a cut point at b[i] and
                    # has SUBtime_bp = 0.  This only shifts the reported
                    # abs_event_time, not the simulated history.
                    time_offset = !fossil_in_bin && nper > 1 && i == 1 && is_tip(tr, p.node) ?
                                  round(node_times[p.node]; digits = 7) : 0.0
                    evs = stochastic_map_branch(rng, blen, bot,
                                                node_states[p.node], qbig, ss;
                                                abs_time_bottom = b[i + 1] +
                                                                  time_offset,
                                                maxtries = maxtries,
                                                manual_history = manual_history,
                                                r_float32 = r_float32)
                    for e in evs
                        push!(branch_events[p.node], _retag(e, p.node))
                    end
                end
                continue
            end

            counter += 1
            seed = next_piece_seed(seed, counter)
            verbose && @info "piece" i j p.node p.kind counter seed

            # Subtree piece: run the ordinary map on the sub-tree, with the
            # stratum's state space and a fresh stream.
            sub = p.sub
            tipmat = Matrix{Float64}(undef, ntips(sub), length(keep))
            for (k, u) in enumerate(p.tip_orig)
                tipmat[k, :] .= snaps[i][u, keep]
            end
            any(isnan, tipmat) &&
                throw(ArgumentError("no downpass message at a subtree tip of " *
                                    "stratum $i"))
            dp = downpass(sub, sub_ss, tipmat, q, wl;
                          min_branchlength = min_branchlength,
                          rootedge = p.length > 0, r_float32 = r_float32)
            ap_sub = ancestral_probs(sub, sub_ss, tipmat, q, wl;
                                     min_branchlength = min_branchlength,
                                     rootedge = p.length > 0,
                                     r_float32 = r_float32)
            rng = RRng(seed)
            if p.node == root
                # The global root state was sampled once with the base seed
                # before the stratum loop; R's recursion sees that row already
                # set and does not redraw.
                rs_sub = findfirst(==(node_states[root]), keep)
            else
                # R samples a subtree-piece root from the cut state pushed up
                # the piece's root branch, times the piece's own downpass
                # message at the branch top (stochastic_map_given_inputs.R
                # :486-602), then simulates that branch's history on the same
                # stream (R:634).  Each (node, stratum) master-table row
                # starts NA, so a state written for this node by an older
                # stratum's piece never leaks in here.
                bot = cut_states[p.node]
                bot == 0 && throw(ArgumentError(
                    "no state at the boundary above node $(p.node) in " *
                    "stratum $i"))
                # R: `condprobs_branch_bot[starting_state] = 1`, then
                # `condprobs_branch_top = condprobs_branch_bot %*% P` -- the
                # row-vector form pushes the cut state FORWARD along the
                # branch (R's stochastic_map_given_inputs.R:486-538), unlike
                # the likelihood paths, which pre-multiply.  With d != e the
                # two directions differ, so select the row of exp(tQ).
                kbot = findfirst(==(bot), keep)
                topv = vec(_dense_transition(q, p.length)[kbot, :])
                ri = ntips(sub) + findfirst(==(p.node), p.internal_orig)
                drow = dp.node_condlikes[ri, :]
                downs = drow ./ sum(drow)
                if ss.include_null_range
                    topv[1] = 0.0
                    downs[1] = 0.0
                end
                rs_sub = r_sample_prob(rng, topv .* downs)
                if p.length > 0
                    evs = stochastic_map_branch(rng, p.length, bot,
                                                keep[rs_sub], qbig, ss;
                                                abs_time_bottom = b[i + 1],
                                                maxtries = maxtries,
                                                manual_history = manual_history,
                                                r_float32 = r_float32)
                    for e in evs
                        push!(branch_events[p.node], _retag(e, p.node))
                    end
                end
            end
            res = stochastic_map(sub, sub_ss, q, w, dp, ap_sub;
                                 rng = rng,
                                 root_state = rs_sub,
                                 time_offset = b[i] + max(0.0, node_times[p.node] - b[i] - tree_height(sub)),
                                 maxtries = maxtries,
                                 min_branchlength = min_branchlength,
                                 manual_history = manual_history,
                                 renormalize_branch_bottom = true,
                                 sampling_indices = length(keep) == ns ? nothing : keep,
                                 sampling_size = ns,
                                 r_float32 = r_float32)

            ntip = ntips(sub)
            for (k, u) in enumerate(p.tip_orig)
                cut_states[u] = keep[res.node_states[k]]
                brbot_states[u] = keep[res.branch_bottom_states[k]]
                if is_tip(tr, u) && node_times[u] >= 0.001 && b[i] <= node_times[u] < b[i + 1]
                    node_states[u] = keep[res.node_states[k]]
                end
            end
            for (k, u) in enumerate(p.internal_orig)
                node_states[u] = keep[res.node_states[ntip + k]]
                clado_type[u] = res.clado_event_type[ntip + k]
                clado_txt[u] = res.clado_event_txt[ntip + k]
                clado_disp[u] = res.clado_dispersal_to[ntip + k]
            end
            for k in 1:ntip
                bk = res.branch_events[k]
                isempty(bk) || append!(branch_events[p.tip_orig[k]],
                                       (_remap(e, p.tip_orig[k], keep) for e in bk))
            end
            for (k, u) in enumerate(p.internal_orig)
                bk = res.branch_events[ntip + k]
                isempty(bk) || append!(branch_events[u],
                                       (_remap(e, u, keep) for e in bk))
            end
        end
    end

    result = BSMResult(node_states, brbot_states, zeros(Int, nn), zeros(Int, nn),
                       branch_events, clado_type, clado_txt, clado_disp, Int(seedval))
    return return_rng ? (result, rng) : result
end
