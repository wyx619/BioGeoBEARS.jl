function _forward_draw(rng::RRng, weights::AbstractVector{<:Real})
    all(x -> isfinite(x) && x >= 0, weights) || throw(ArgumentError("invalid transition weights"))
    total = sum(weights)
    total > 0 || throw(ArgumentError("no possible transition"))
    u = only(r_runif(rng, 1))
    cumulative = 0.0
    for i in eachindex(weights)
        cumulative += weights[i] / total
        u <= cumulative && return i
    end
    return lastindex(weights)
end

"""
    simulate_biogeog_history(tr, ss, q, weights, root_state; rng=RRng(1))
    simulate_biogeog_history(tr, ss, model, root_state; rng=RRng(1))

Simulate node states on a fixed tree from a known root state, without
conditioning on observed tip ranges. State indices are Julia's 1-based indices.
As in R's `simulate_biogeog_history`, a null range is excluded at branch ends.
Returns node states and the two post-speciation daughter states per node.
The RNG is advanced in place. This simulates endpoints, not along-branch events.
"""
function simulate_biogeog_history(tr::BGTree, ss::StateSpace, q::AbstractMatrix,
                                  w::CladogenesisWeights, root_state::Integer;
                                  rng::RRng = RRng(1))
    ns = nstates(ss)
    size(q) == (ns, ns) || throw(DimensionMismatch("Q and state space must match"))
    1 <= root_state <= ns || throw(ArgumentError("root state is out of bounds"))
    nonnull = findall(!isempty, ss.states)
    root_state in nonnull || throw(ArgumentError("root state must be nonempty"))
    states = zeros(Int, nnodes(tr))
    left = zeros(Int, nnodes(tr)); right = zeros(Int, nnodes(tr))
    states[root_node(tr)] = root_state
    for n in Iterators.reverse(_ape_pruning_nodes(tr))
        events = findall(==(states[n]), w.ancestor)
        isempty(events) && throw(ArgumentError("no cladogenesis events for state $(states[n])"))
        k = events[_forward_draw(rng, w.weight[events])]
        left[n] = w.left[k]; right[n] = w.right[k]
        for (child, start) in ((left_child(tr, n), left[n]), (right_child(tr, n), right[n]))
            start in nonnull || throw(ArgumentError("cladogenesis produced a null or invalid state"))
            p = transition_matrix(q, branch_length(tr, child))
            states[child] = nonnull[_forward_draw(rng, max.(0.0, p[start, nonnull]))]
        end
    end
    return (node_states = states, left_corner = left, right_corner = right)
end

function simulate_biogeog_history(tr::BGTree, ss::StateSpace, model::Model,
                                  root_state::Integer; rng::RRng = RRng(1))
    linked_model = update_linked!(deepcopy(model))
    return simulate_biogeog_history(tr, ss, qmatrix(ss, linked_model; r_float32 = true),
                                    cladogenesis_weights(ss, linked_model; r_float32 = true),
                                    root_state; rng = rng)
end
