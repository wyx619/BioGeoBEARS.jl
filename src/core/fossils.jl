# Fossil placement and ranges.

"""
    FossilRecord

One row of a fossil table: the fossil's name, its age bracket, its coded
geographic range and the clade it is claimed to belong to.

Fields mirror the columns of R's fossils spreadsheet:

| field              | meaning                                                    |
|:-------------------|:-----------------------------------------------------------|
| `fossil_taxon`     | name; the tip becomes `fossil_<fossil_taxon>`               |
| `stem_or_crown`    | `"stem"`, `"crown"` or `"both"`                             |
| `min_age`, `max_age` | youngest / oldest possible age (Ma)                       |
| `range_coded`      | e.g. `"0011"`, one character per area                       |
| `is_mrca_of`       | first tip of the constraining clade                         |
| `and_mrca_of`      | second tip of the constraining clade                        |
| `geog_constraint`  | `"both"`, `"presence-only"`, `"absence-only"`               |
| `constraint_type`  | `"side_branch"`, `"direct_ancestor"` / `"ancestor"`         |
| `taxon_specifiers` | species / genus / family name, when not using two tips      |
| `specifier_type`   | `"specifiers"`, `"species"`, or a rank (`"genus"`, ...)     |

Only `fossil_taxon`, `min_age`, `max_age` and `range_coded` are required; the
rest default to the values a "both / side_branch / two tips" record would have.
"""
struct FossilRecord
    fossil_taxon::String
    stem_or_crown::String
    min_age::Float64
    max_age::Float64
    range_coded::String
    is_mrca_of::String
    and_mrca_of::String
    geog_constraint::String
    constraint_type::String
    taxon_specifiers::String
    specifier_type::String
end

function FossilRecord(; fossil_taxon::AbstractString,
                      stem_or_crown::AbstractString = "both",
                      min_age::Real,
                      max_age::Real,
                      range_coded::AbstractString = "",
                      is_mrca_of::AbstractString = "",
                      and_mrca_of::AbstractString = "",
                      geog_constraint::AbstractString = "both",
                      constraint_type::AbstractString = "side_branch",
                      taxon_specifiers::AbstractString = "",
                      specifier_type::AbstractString = "specifiers")
    float(max_age) < float(min_age) &&
        throw(ArgumentError("fossil $(repr(fossil_taxon)): max_age ($(max_age)) < min_age ($(min_age))"))
    return FossilRecord(String(fossil_taxon), String(stem_or_crown),
                        float(min_age), float(max_age), String(range_coded),
                        String(is_mrca_of), String(and_mrca_of),
                        String(geog_constraint), String(constraint_type),
                        String(taxon_specifiers), String(specifier_type))
end

const FOSSIL_REQUIRED_FIELDS = (:fossil_taxon, :min_age, :max_age)

"""
    FossilRecord(nt) -> FossilRecord

Build a record from a `NamedTuple` (or anything that supports `getproperty`),
so a row of a `DataFrame` or a literal works directly.
"""
function FossilRecord(nt::NamedTuple)
    getproperty(nt, :fossil_taxon)
    return FossilRecord(;
        fossil_taxon = getproperty(nt, :fossil_taxon),
        stem_or_crown = _ntfield(nt, :stem_or_crown, "both"),
        min_age = getproperty(nt, :min_age),
        max_age = getproperty(nt, :max_age),
        range_coded = _ntfield(nt, :range_coded, ""),
        is_mrca_of = _ntfield(nt, :is_mrca_of, ""),
        and_mrca_of = _ntfield(nt, :and_mrca_of, ""),
        geog_constraint = _ntfield(nt, :geog_constraint, "both"),
        constraint_type = _ntfield(nt, :constraint_type, "side_branch"),
        taxon_specifiers = _ntfield(nt, :taxon_specifiers, ""),
        specifier_type = _ntfield(nt, :specifier_type, "specifiers"))
end

_ntfield(nt::NamedTuple, name::Symbol, default) =
    haskey(nt, name) && nt[name] !== missing && nt[name] !== nothing ? nt[name] : default

"""
    fossil_records(table) -> Vector{FossilRecord}

Normalise a table of fossils: a vector of [`FossilRecord`](@ref), a vector of
`NamedTuple`s, or a `DataFrame` with the documented columns.
"""
fossil_records(records::AbstractVector{<:FossilRecord}) = collect(records)
fossil_records(records::AbstractVector{<:NamedTuple}) = FossilRecord.(records)

function fossil_records(df::DataFrame)
    cols = propertynames(df)
    required = (:fossil_taxon, :min_age, :max_age)
    for c in required
        c in cols || throw(ArgumentError("fossil table is missing the column $(repr(c))"))
    end
    return [FossilRecord((; (c => df[i, c] for c in cols)...)) for i in 1:nrow(df)]
end

"""
    read_fossils(path; delim='\t') -> Vector{FossilRecord}

Read a delimited fossil table with a header row.  Column names are the field
names of [`FossilRecord`](@ref); unknown columns are ignored and missing
optional columns take their defaults.
"""
function read_fossils(path::AbstractString; delim::Char = '\t')
    isfile(path) || throw(ArgumentError("fossil file not found: $(path)"))
    lines = readlines(path)
    idx = findfirst(l -> !isempty(strip(l)), lines)
    idx === nothing && throw(ArgumentError("$(path): empty fossil file"))

    header = String.(split(strip(lines[idx]), delim))
    names = Symbol.(strip.(header))
    records = FossilRecord[]
    for lineno in (idx + 1):length(lines)
        line = strip(lines[lineno])
        isempty(line) && continue
        fields = String.(split(line, delim))
        length(fields) == length(names) ||
            throw(ArgumentError("$(path):$(lineno): expected $(length(names)) fields, got $(length(fields))"))
        vals = Dict{Symbol,Any}()
        for (n, v) in zip(names, fields)
            n === :min_age || n === :max_age ? (vals[n] = parse(Float64, strip(v))) : (vals[n] = strip(v))
        end
        for c in FOSSIL_REQUIRED_FIELDS
            haskey(vals, c) || throw(ArgumentError("$(path):$(lineno): missing column $(repr(c))"))
        end
        push!(records, FossilRecord((; vals...)))
    end
    isempty(records) && throw(ArgumentError("$(path): no fossil rows"))
    return records
end

# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

"""
    add_hook(tr, tipname, depthtime; brlen_of_side_branch=1e-7, newtipname="default")

Insert a new tip as a short side branch ("hook") at `depthtime` below `tipname`,
and return the renumbered tree — R's `add_hook()`.

`depthtime` is the distance from the named tip, measured down the tree.  The
hook hangs off a new internal node spliced into whichever branch contains that
point.  The new tip is appended after the existing tips, and internal nodes are
renumbered in depth-first preorder, so the result follows ape's numbering.

When `depthtime` overshoots the root the input tree is returned unchanged, as R
does (R additionally warns).
"""
function add_hook(tr::BGTree, tipname::AbstractString, depthtime::Real;
                  brlen_of_side_branch::Real = 1e-7,
                  newtipname::AbstractString = "default")
    tipnum = findfirst(==(String(tipname)), tr.tip_labels)
    tipnum === nothing &&
        throw(ArgumentError("add_hook: tip $(repr(tipname)) is not in the tree"))
    float(depthtime) < 0 &&
        throw(ArgumentError("add_hook: depthtime must be non-negative, got $(depthtime)"))

    daughter = _trace_parents_up(tr, tipnum, float(depthtime))
    daughter === nothing && return tr

    dist_to_daughter = _distance_to_ancestor(tr, daughter, tipnum)
    height_below = float(depthtime) - dist_to_daughter
    name = newtipname == "default" ? "hook$(tr.ntip + 1)" : String(newtipname)
    return _splice_hook(tr, daughter, height_below, float(brlen_of_side_branch), name)
end

# Walk rootwards until the remaining distance fits inside the current branch.
# `nothing` means the walk reached the root, i.e. the depth overshoots the tree.
function _trace_parents_up(tr::BGTree, nodenum::Int, depthtime::Float64)
    n = nodenum
    remaining = depthtime
    while true
        len = tr.edge_length[n]
        len > remaining && return n
        remaining -= len
        p = tr.parent[n]
        p == 0 && return nothing
        n = p
    end
end

function _distance_to_ancestor(tr::BGTree, ancestor::Int, descendant::Int)
    total = 0.0
    n = descendant
    while n != ancestor
        total += tr.edge_length[n]
        n = tr.parent[n]
        n == 0 && throw(ArgumentError("node $descendant is not a descendant of $ancestor"))
    end
    return total
end

# Splice a new internal node into `daughter`'s branch, `height_below` above the
# daughter, and hang a new tip off it.
function _splice_hook(tr::BGTree, daughter::Int, height_below::Float64,
                      brlen::Float64, newtipname::String)
    n_old = nnodes(tr)
    new_tip_id = n_old + 1
    new_internal_id = n_old + 2
    total = n_old + 2

    parent = vcat(tr.parent, tr.parent[daughter], tr.parent[daughter])
    edge = vcat(tr.edge_length, 0.0, 0.0)

    parent[new_internal_id] = tr.parent[daughter]
    parent[new_tip_id] = new_internal_id
    parent[daughter] = new_internal_id
    edge[new_internal_id] = tr.edge_length[daughter] - height_below
    edge[daughter] = height_below
    edge[new_tip_id] = brlen

    children = [Int[] for _ in 1:total]
    for n in 1:n_old
        tr.left[n] != 0 && push!(children[n], tr.left[n])
        tr.right[n] != 0 && push!(children[n], tr.right[n])
    end
    old_parent = tr.parent[daughter]
    slot = findfirst(==(daughter), children[old_parent])
    children[old_parent][slot] = new_internal_id
    push!(children[new_internal_id], daughter)
    push!(children[new_internal_id], new_tip_id)

    # R's add_hook() writes the modified tree and reads it back, so tips are
    # renumbered in Newick order.  The hook is the second child of the new node,
    # so the fossil lands immediately after the last tip of the daughter's
    # subtree; later tips shift up by one.
    last_tip = maximum(_descendant_tips(tr, daughter))
    tip_ids = vcat(collect(1:last_tip), new_tip_id, collect(last_tip + 1:tr.ntip))
    tip_labels = vcat(tr.tip_labels[1:last_tip], newtipname, tr.tip_labels[(last_tip + 1):end])

    # Renumber: tips follow Newick order, internals in preorder.
    number = zeros(Int, total)
    for (i, id) in enumerate(tip_ids)
        number[id] = i
    end
    ntip_new = length(tip_labels)
    next_internal = ntip_new + 1
    stack = Int[root_node(tr)]
    while !isempty(stack)
        id = pop!(stack)
        (id <= tr.ntip || id == new_tip_id) && continue   # tips keep their numbers
        number[id] = next_internal
        next_internal += 1
        for c in Iterators.reverse(children[id])
            push!(stack, c)
        end
    end

    nnode_new = tr.nnode + 1
    total_new = ntip_new + nnode_new
    parent_new = zeros(Int, total_new)
    left_new = zeros(Int, total_new)
    right_new = zeros(Int, total_new)
    edge_new = zeros(Float64, total_new)
    node_labels_new = fill("", nnode_new)

    for id in 1:total
        k = number[id]
        k == 0 && continue
        edge_new[k] = edge[id]
        p = parent[id]
        parent_new[k] = p == 0 ? 0 : number[p]
        ch = children[id]
        length(ch) >= 1 && (left_new[k] = number[ch[1]])
        length(ch) >= 2 && (right_new[k] = number[ch[2]])
        if tr.ntip < id <= n_old
            node_labels_new[k - ntip_new] = tr.node_labels[id - tr.ntip]
        end
    end

    postorder = _postorder(left_new, right_new; root_node_number = ntip_new + 1)
    return check_tree(BGTree(tip_labels, node_labels_new, parent_new, left_new,
                             right_new, edge_new, postorder, ntip_new, nnode_new,
                             tr.root_edge))
end

"""
    _mrca(tr, tipnames) -> Int

The most recent common ancestor of the named tips, in node numbers.
"""
function _mrca(tr::BGTree, tipnames::AbstractVector{<:AbstractString})
    nums = Int[]
    for nm in tipnames
        i = findfirst(==(String(nm)), tr.tip_labels)
        i === nothing &&
            throw(ArgumentError("tip $(repr(nm)) is not in the tree"))
        push!(nums, i)
    end
    isempty(nums) && throw(ArgumentError("_mrca needs at least one tip"))
    a = nums[1]
    for b in nums[2:end]
        a = _mrca_pair(tr, a, b)
    end
    return a
end

function _mrca_pair(tr::BGTree, a::Int, b::Int)
    ancestors = Set{Int}()
    x = a
    while x != 0
        push!(ancestors, x)
        x = tr.parent[x]
    end
    y = b
    while !(y in ancestors)
        y = tr.parent[y]
    end
    return y
end

"""
    _descendant_nodes(tr, node) -> Vector{Int}

Every node below `node`, excluding `node` itself.
"""
function _descendant_nodes(tr::BGTree, node::Int)
    out = Int[]
    stack = Int[]
    tr.left[node] != 0 && push!(stack, tr.left[node])
    tr.right[node] != 0 && push!(stack, tr.right[node])
    while !isempty(stack)
        x = pop!(stack)
        push!(out, x)
        tr.left[x] != 0 && push!(stack, tr.left[x])
        tr.right[x] != 0 && push!(stack, tr.right[x])
    end
    return out
end

"""
    _descendant_tips(tr, node) -> Vector{Int}

Every tip at or below `node`.
"""
function _descendant_tips(tr::BGTree, node::Int)
    is_tip(tr, node) && return [node]
    return [n for n in _descendant_nodes(tr, node) if is_tip(tr, n)]
end

# ---------------------------------------------------------------------------
# Fossil placement
# ---------------------------------------------------------------------------

"""
    possible_fossil_branches(tr, tip_specifiers; stem_or_crown="both") -> Vector{Int}

Node numbers of the branches a fossil may attach to, given the tips that
constrain its position — R's `get_possible_branches_to_add_fossils_to()`.

With one specifier the fossil is tied to that terminal branch (so `"crown"` is
invalid: a crown group needs two tips).  With two or more, the constraint is
their most recent common ancestor:

* `"crown"` — every branch inside the clade (the crown node's descendants)
* `"stem"` — only the branch *below* the crown node
* `"both"` — the crown branches plus the stem branch
"""
function possible_fossil_branches(tr::BGTree, tip_specifiers::AbstractVector{<:AbstractString};
                                  stem_or_crown::AbstractString = "both")
    stem_or_crown in ("stem", "crown", "both") ||
        throw(ArgumentError("stem_or_crown must be \"stem\", \"crown\" or \"both\", got $(repr(stem_or_crown))"))
    isempty(tip_specifiers) &&
        throw(ArgumentError("tip_specifiers must not be empty"))

    if length(tip_specifiers) == 1
        stem_or_crown == "crown" &&
            throw(ArgumentError("a single tip cannot define a crown group; use \"stem\" or \"both\""))
        i = findfirst(==(String(tip_specifiers[1])), tr.tip_labels)
        i === nothing &&
            throw(ArgumentError("tip $(repr(tip_specifiers[1])) is not in the tree"))
        return [i]
    end

    crown = _mrca(tr, tip_specifiers)
    nodes = _descendant_nodes(tr, crown)
    if stem_or_crown == "crown"
        return nodes
    elseif stem_or_crown == "both"
        return vcat(nodes, crown)
    else
        return [crown]
    end
end

# Weighted draw of one index; R's sample(..., prob=prop).
function _sample_index(rng::AbstractRNG, w::AbstractVector{<:Real})
    total = sum(w)
    total > 0 || return rand(rng, eachindex(w))
    r = rand(rng) * total
    acc = 0.0
    for (i, x) in enumerate(w)
        acc += x
        r <= acc && return i
    end
    return lastindex(w)
end

"""
    add_fossil_randomly(tr, tip_specifiers, fossil_name, max_age, min_age; kwargs...)

Attach one fossil of uncertain position to `tr` and return the new tree —
R's `add_fossil_randomly()`.

The fossil's age is drawn uniformly from `[min_age, max_age]`, then clamped to
the age span of the branches the clade constraint allows.  If no branch can host
the fossil (or the age bracket is incompatible with the topology), the input
tree is returned unchanged.

Keyword arguments:

* `constraint_type` — `"side_branch"` (default) or `"direct_ancestor"` /
  `"ancestor"` (a 1e-7 hook)
* `stem_or_crown` — passed to [`possible_fossil_branches`](@ref)
* `rng` — the random number generator (use `StableRNGs.StableRNG` for
  reproducible placement across Julia versions)
* `brlen_of_side_branch` — hook length for the direct-ancestor case
* `fossil_age_within_constraints` — clamp the age to the topology (default
  `true`); `false` draws from the raw bracket, as R's pre-2017 behaviour
"""
function add_fossil_randomly(tr::BGTree, tip_specifiers::AbstractVector{<:AbstractString},
                             fossil_name::AbstractString, max_age::Real, min_age::Real;
                             constraint_type::AbstractString = "side_branch",
                             stem_or_crown::AbstractString = "both",
                             rng::AbstractRNG = Random.default_rng(),
                             brlen_of_side_branch::Real = 1e-7,
                             fossil_age_within_constraints::Bool = true)
    lo = float(min_age)
    hi = float(max_age)
    hi < lo && throw(ArgumentError("fossil $(repr(fossil_name)): max_age ($hi) < min_age ($lo)"))

    constraint_type in ("side_branch", "direct_ancestor", "ancestor", "fixnode") ||
        throw(ArgumentError("unknown constraint_type $(repr(constraint_type))"))
    constraint_type == "fixnode" && return tr   # fixnodes are handled by the model, not the tree

    nodes = possible_fossil_branches(tr, tip_specifiers; stem_or_crown = stem_or_crown)
    isempty(nodes) && return tr

    t = prt_table(tr)
    if fossil_age_within_constraints
        bmin = t.time_bp[nodes]
        blen = [isnan(t.edge_length[n]) ? 0.0 : t.edge_length[n] for n in nodes]
        oldest = maximum(bmin .+ blen)
        hi > oldest && (hi = oldest)
        hi < lo && return tr        # R draws runif(min > max) -> NaN -> nothing added
    end
    fossil_age = lo + (hi - lo) * rand(rng)

    if constraint_type == "side_branch"
        return _add_random_side_branch(tr, t, nodes, String(fossil_name), fossil_age, rng)
    end
    return _add_random_direct_ancestor_hook(tr, t, nodes, String(fossil_name),
                                            fossil_age, rng, float(brlen_of_side_branch))
end

function _add_random_side_branch(tr::BGTree, t, nodes::Vector{Int}, fossil_name::String,
                                 fossil_age::Float64, rng::AbstractRNG)
    # R refuses to place a fossil exactly on an internal node.
    for n in 1:nnodes(tr)
        is_tip(tr, n) && continue
        t.time_bp[n] == fossil_age && return tr
    end

    bmin = t.time_bp[nodes]
    blen = [isnan(t.edge_length[n]) ? 0.0 : t.edge_length[n] for n in nodes]
    keep = fossil_age .< (bmin .+ blen)
    any(keep) || return tr

    ns = nodes[keep]
    bmin = bmin[keep]
    blen = blen[keep]

    attach = copy(blen)
    diff = bmin .- fossil_age
    for k in eachindex(attach)
        diff[k] < 0 && (attach[k] += diff[k])
    end
    total = sum(attach)
    total > 0 || return tr

    k = _sample_index(rng, attach ./ total)
    node = ns[k]
    height_above_base = rand(rng) * attach[k]

    tips = _descendant_tips(tr, node)
    hts = t.node_ht[tips]
    tipnum = tips[findfirst(==(maximum(hts)), hts)]

    absolute = t.time_bp[node] + blen[k] - height_above_base
    depthtime = absolute - t.time_bp[tipnum]
    side_length = absolute - fossil_age
    return add_hook(tr, tr.tip_labels[tipnum], depthtime;
                    brlen_of_side_branch = side_length, newtipname = fossil_name)
end

function _add_random_direct_ancestor_hook(tr::BGTree, t, nodes::Vector{Int},
                                          fossil_name::String, fossil_age::Float64,
                                          rng::AbstractRNG, brlen::Float64)
    for n in 1:nnodes(tr)
        is_tip(tr, n) && continue
        t.time_bp[n] == fossil_age && return tr
    end

    bmin = t.time_bp[nodes]
    blen = [isnan(t.edge_length[n]) ? 0.0 : t.edge_length[n] for n in nodes]
    bmax = bmin .+ blen
    keep = (fossil_age .< bmax) .& (fossil_age .> bmin)
    any(keep) || return tr

    ns = nodes[keep]
    bmin = bmin[keep]
    blen = blen[keep]

    total = sum(blen)
    total > 0 || return tr
    k = _sample_index(rng, blen ./ total)
    node = ns[k]

    height_above_base = bmin[k] + blen[k] - fossil_age

    tips = _descendant_tips(tr, node)
    hts = t.node_ht[tips]
    tipnum = tips[findfirst(==(maximum(hts)), hts)]

    absolute = bmin[k] + blen[k] - height_above_base
    depthtime = absolute - t.time_bp[tipnum]
    return add_hook(tr, tr.tip_labels[tipnum], depthtime;
                    brlen_of_side_branch = brlen, newtipname = fossil_name)
end

"""
    add_fossils_randomly(tr, records; rng=..., kwargs...) -> (tree, names)

Add every fossil in `records` (a vector of [`FossilRecord`](@ref), of
`NamedTuple`s, or a `DataFrame`) to `tr`, one at a time — R's
`add_fossils_from_xls_randomly()`.

Returns the tree with all fossils that could be placed, plus the names of the
tips that were added.  Rows with `constraint_type == "fixnode"` are skipped:
those are node constraints for the likelihood, not tips.

The clade of each fossil comes from `specifier_type`:

* `"specifiers"` — the two tips `is_mrca_of` / `and_mrca_of`
* `"species"` — the single tip `taxon_specifiers`
* anything else (a rank such as `"genus"`) — every tip whose name contains
  `taxon_specifiers` followed by `_`
"""
function add_fossils_randomly(tr::BGTree, records; rng::AbstractRNG = Random.default_rng(),
                              brlen_of_side_branch::Real = 1e-7)
    recs = fossil_records(records)
    active = [r for r in recs if r.constraint_type != "fixnode"]

    out = tr
    added = String[]
    for r in active
        specifiers = _tip_specifiers_for(out, r)
        specifiers === nothing && continue
        newtree = add_fossil_randomly(out, specifiers, "fossil_" * r.fossil_taxon,
                                      r.max_age, r.min_age;
                                      constraint_type = r.constraint_type,
                                      stem_or_crown = r.stem_or_crown,
                                      rng = rng,
                                      brlen_of_side_branch = brlen_of_side_branch)
        if ntips(newtree) == ntips(out) + 1
            out = newtree
            push!(added, "fossil_" * r.fossil_taxon)
        end
    end
    return out, added
end

function _tip_specifiers_for(tr::BGTree, r::FossilRecord)
    if r.specifier_type == "specifiers"
        (isempty(r.is_mrca_of) || isempty(r.and_mrca_of)) && return nothing
        for nm in (r.is_mrca_of, r.and_mrca_of)
            nm in tr.tip_labels ||
                throw(ArgumentError("fossil $(repr(r.fossil_taxon)): tip specifier $(repr(nm)) is not in the tree"))
        end
        return [r.is_mrca_of, r.and_mrca_of]
    end

    if r.specifier_type == "species"
        r.taxon_specifiers in tr.tip_labels ||
            throw(ArgumentError("fossil $(repr(r.fossil_taxon)): tip specifier " *
                                "$(repr(r.taxon_specifiers)) is not in the tree"))
        return [r.taxon_specifiers]
    end

    # A rank: match "<name>_" as R does, so names must be non-redundant.
    prefix = r.taxon_specifiers * "_"
    matches = [nm for nm in tr.tip_labels if occursin(prefix, nm)]
    isempty(matches) && return nothing
    return matches
end

"""
    add_fossil_ranges(tips::TipRanges, fossil_names, records) -> TipRanges

Append the fossil rows to a [`TipRanges`](@ref) table — R's
`add_list_of_fossil_tips_to_tipranges()`.

`range_coded` is read one character per area (`"0011"`).  `geog_constraint`
decides how the zeros are treated:

* `"both"` — the coded range is the true range
* `"presence-only"` — `0` becomes `'?'` (absence is unknown)
* `"absence-only"` — `1` becomes `'?'` (presence is unknown)
* `"detection_model"` — left unchanged; R warns that this is unimplemented
"""
function add_fossil_ranges(tips::TipRanges, fossil_names::AbstractVector{<:AbstractString},
                           records)
    isempty(fossil_names) && return tips
    recs = fossil_records(records)

    na = nareas(tips)
    data = Matrix{Char}(undef, length(fossil_names), na)
    for (i, nm) in enumerate(fossil_names)
        rec = _find_fossil(recs, String(nm))
        rec === nothing &&
            throw(ArgumentError("fossil $(repr(nm)) has no row in the fossil table"))
        chars = collect(rec.range_coded)
        length(chars) == na ||
            throw(ArgumentError("fossil $(repr(nm)): range_coded $(repr(rec.range_coded)) " *
                                "has $(length(chars)) characters, expected $(na)"))
        if rec.geog_constraint == "presence-only"
            chars = [c == '0' ? '?' : c for c in chars]
        elseif rec.geog_constraint == "absence-only"
            chars = [c == '1' ? '?' : c for c in chars]
        end
        data[i, :] .= chars
    end

    return TipRanges(vcat(tips.tipnames, String.(fossil_names)), tips.areanames,
                     vcat(tips.data, data))
end

function _find_fossil(records::Vector{FossilRecord}, name::AbstractString)
    for r in records
        ("fossil_" * r.fossil_taxon == name || r.fossil_taxon == name) && return r
    end
    return nothing
end
