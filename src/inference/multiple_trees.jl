"""
    fit_multiple_trees(trees, ss, tipdata, model; stratifications=nothing, kwargs...)

Fit the same model to each tree. `tipdata` is either one `TipRanges` shared by
all trees or a vector containing one `TipRanges` or conditional-likelihood
matrix per tree. `stratifications` is `nothing`, one `Stratification`, or a
vector with one stratification (or `nothing`) per tree. Remaining keywords are
passed to `fit_model`.
"""
function fit_multiple_trees(trees::AbstractVector{<:BGTree}, ss::StateSpace,
                            tipdata, model::Model; stratifications = nothing, kwargs...)
    isempty(trees) && throw(ArgumentError("trees must not be empty"))
    data = tipdata isa AbstractVector ? tipdata : fill(tipdata, length(trees))
    strata = stratifications isa AbstractVector ? stratifications : fill(stratifications, length(trees))
    length(data) == length(trees) || throw(ArgumentError("one tip dataset is required per tree"))
    length(strata) == length(trees) || throw(ArgumentError("one stratification is required per tree"))
    return [strata[i] === nothing ? fit_model(tr, ss, data[i], model; kwargs...) :
            fit_model(tr, ss, data[i], model, strata[i]; kwargs...)
            for (i, tr) in enumerate(trees)]
end

function _descendant_tipsets(tr::BGTree)
    length(unique(tr.tip_labels)) == ntips(tr) || throw(ArgumentError("tree tip labels must be unique"))
    sets = [Set{String}() for _ in 1:nnodes(tr)]
    for i in 1:ntips(tr)
        push!(sets[i], tr.tip_labels[i])
    end
    for n in pruning_order(tr)
        union!(sets[parent_node(tr, n)], sets[n])
    end
    return sets
end

"""
    summarize_tree_ancestry(master, trees, fits; matching=:per_tree)

Average branch-top and branch-bottom ancestral probabilities on `master` by exact descendant
clade among its tips. Extra tips in sampled trees are ignored for matching;
when several nodes reduce to the same clade, use the smallest original clade
in that tree. Each tree contributes at most once per master node. Unmatched
nodes contain `NaN`; `matches` counts matched rows and `bottom_matches` counts
rows with finite branch-bottom probabilities. The root has no branch bottom.
All fits must use the same ordered state space and contain ancestral probabilities.
With `matching=:r`, reproduce R's global selection of one original clade,
including its first-match tie behavior and normalization of column sums.
"""
function summarize_tree_ancestry(master::BGTree, trees::AbstractVector{<:BGTree},
                                 fits::AbstractVector{BGBFit}; matching::Symbol = :per_tree)
    isempty(trees) && throw(ArgumentError("trees must not be empty"))
    matching in (:per_tree, :r) || throw(ArgumentError("matching must be :per_tree or :r"))
    length(trees) == length(fits) || throw(ArgumentError("one fit is required per tree"))
    master_sets = _descendant_tipsets(master)
    master_tips = master_sets[root_node(master)]
    nstates = size(ancprobs(first(fits)).branch_top, 2)
    sums = zeros(nnodes(master), nstates)
    bottom_sums = zeros(nnodes(master), nstates)
    matches = zeros(Int, nnodes(master))
    bottom_matches = zeros(Int, nnodes(master))
    allsets = Vector{Vector{Set{String}}}()
    allprobs = Matrix{Float64}[]
    allbottom = Matrix{Float64}[]
    for (tr, fit) in zip(trees, fits)
        sets = _descendant_tipsets(tr)
        issubset(master_tips, sets[root_node(tr)]) || throw(ArgumentError("sampled tree is missing master-tree tips"))
        probs = ancprobs(fit).branch_top
        bottom = ancprobs(fit).branch_bottom
        size(probs) == (nnodes(tr), nstates) || throw(DimensionMismatch("ancestral probabilities do not match tree or state count"))
        size(bottom) == size(probs) || throw(DimensionMismatch("branch-bottom probabilities do not match tree or state count"))
        push!(allsets, sets); push!(allprobs, probs); push!(allbottom, bottom)
        matching == :r && continue
        for (i, target) in enumerate(master_sets)
            candidates = [n for n in 1:nnodes(tr) if intersect(sets[n], master_tips) == target]
            isempty(candidates) && continue
            n = candidates[argmin(length.(sets[candidates]))]
            sums[i, :] .+= probs[n, :]
            matches[i] += 1
            if !is_root(tr, n) && all(isfinite, bottom[n, :]) && sum(bottom[n, :]) > 0
                bottom_sums[i, :] .+= bottom[n, :]
                bottom_matches[i] += 1
            end
        end
    end
    if matching == :r
        for (i, target) in enumerate(master_sets)
            candidates = Set{String}[]
            for sets in allsets, s in sets
                intersect(s, master_tips) == target && !(s in candidates) && push!(candidates, s)
            end
            isempty(candidates) && continue
            minsize = minimum(length, candidates)
            minima = findall(s -> length(s) == minsize, candidates)
            chosen = candidates[length(minima) == 1 ? only(minima) : 1]
            for (tr, sets, probs, bottom) in zip(trees, allsets, allprobs, allbottom), n in eachindex(sets)
                sets[n] == chosen || continue
                for j in 1:nstates
                    isnan(probs[n, j]) || (sums[i, j] += probs[n, j])
                end
                matches[i] += 1
                if !is_root(tr, n) && all(isfinite, bottom[n, :]) && sum(bottom[n, :]) > 0
                    bottom_sums[i, :] .+= bottom[n, :]
                    bottom_matches[i] += 1
                end
            end
        end
    end
    result = fill(NaN, size(sums))
    bottom_result = fill(NaN, size(sums))
    for i in eachindex(matches)
        denom = matching == :r ? sum(sums[i, :]) : matches[i]
        matches[i] > 0 && denom > 0 && (result[i, :] .= sums[i, :] ./ denom)
        bottom_denom = matching == :r ? sum(bottom_sums[i, :]) : bottom_matches[i]
        bottom_matches[i] > 0 && bottom_denom > 0 && (bottom_result[i, :] .= bottom_sums[i, :] ./ bottom_denom)
    end
    return (branch_top = result, branch_bottom = bottom_result,
            matches = matches, bottom_matches = bottom_matches)
end
