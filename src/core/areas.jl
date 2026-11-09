# Geographic states and tip likelihoods.

const NULL_RANGE_TEXT = "_"

"""
    StateSpace

Every geographic range allowed in an analysis, in BioGeoBEARS index order.

`states[i]` is a vector of **1-based** area indices; the null range is the empty
vector `Int[]`.  (R stores the same thing 0-based and flips the meaning of `NA`
between null range and "no match" depending on the function; keeping the null
range as `Int[]` removes that ambiguity.  The 0-based form only exists inside
[`read_lagrange_phylip`](@ref)'s output and is never exposed.)

Construct with [`StateSpace`](@ref); see also [`numstates_from_numareas`](@ref).
"""
struct StateSpace
    areanames::Vector{String}
    states::Vector{Vector{Int}}
    include_null_range::Bool
    max_range_size::Int
end

"""
    StateSpace(areanames; max_range_size=length(areanames), include_null_range=true)

Build the state space for `areanames`.

`max_range_size` is the largest number of areas a single range may occupy; it is
clamped to `length(areanames)` as R does.  With `include_null_range`, the empty
range is state 1 — LAGRANGE's default and the convention the rest of the package
assumes.
"""
function StateSpace(areanames::AbstractVector{<:AbstractString};
                    max_range_size::Integer = length(areanames),
                    include_null_range::Bool = true)
    names = String.(areanames)
    isempty(names) &&
        throw(ArgumentError("StateSpace needs at least one area"))
    length(unique(names)) == length(names) ||
        throw(ArgumentError("area names must be unique, got $(names)"))

    m = min(Int(max_range_size), length(names))
    m >= 1 ||
        throw(ArgumentError("max_range_size must be at least 1, got $max_range_size"))

    states = Vector{Int}[]
    include_null_range && push!(states, Int[])
    for k in 1:m
        append!(states, _combinations(length(names), k))
    end

    return StateSpace(names, states, include_null_range, m)
end

Base.length(ss::StateSpace) = length(ss.states)
Base.getindex(ss::StateSpace, i::Integer) = ss.states[i]
Base.iterate(ss::StateSpace, args...) = iterate(ss.states, args...)
Base.eltype(::Type{StateSpace}) = Vector{Int}

nareas(ss::StateSpace) = length(ss.areanames)
nstates(ss::StateSpace) = length(ss.states)

"""
    numstates_from_numareas(nareas; max_range_size=nareas, include_null_range=true)

Size of the state space without materialising it: `sum(binomial(n, k) for k in
1:max_range_size)`, plus one for the null range.
"""
function numstates_from_numareas(nareas::Integer;
                                 max_range_size::Integer = nareas,
                                 include_null_range::Bool = true)
    n = Int(nareas)
    m = min(Int(max_range_size), n)
    total = include_null_range ? 1 : 0
    for k in 1:m
        total += binomial(n, k)
    end
    return total
end

# All k-subsets of 1:n in lexicographic order — the order `combn` produces and
# the order cladoRcpp enumerates in.
function _combinations(n::Int, k::Int)
    out = Vector{Int}[]
    (k < 1 || k > n) && return out
    idx = collect(1:k)
    while true
        push!(out, copy(idx))
        i = k
        while i >= 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in (i + 1):k
            idx[j] = idx[j - 1] + 1
        end
    end
    return out
end

"""
    state_text(ss::StateSpace, i; split=true) -> String

Text form of state `i`.  The null range is `"_"`.  With `split=true` areas are
joined by commas (`"A,B"`, R's `split_ABC=TRUE`); with `split=false` they are
concatenated (`"AB"`, `split_ABC=FALSE`).
"""
function state_text(ss::StateSpace, i::Integer; split::Bool = true)
    range = ss.states[i]
    isempty(range) && return NULL_RANGE_TEXT
    if split
        return join((ss.areanames[a] for a in range), ",")
    else
        return join((ss.areanames[a] for a in range))
    end
end

"""
    states_list_text(ss; split=true) -> Vector{String}

Text form of every state, in state order.
"""
states_list_text(ss::StateSpace; split::Bool = true) =
    [state_text(ss, i; split = split) for i in 1:length(ss)]

"""
    state_index(ss::StateSpace, range) -> Int

Index of the state holding exactly the areas in `range` (1-based area indices,
any order).  Returns `0` when the range is not part of the state space, which is
the signal callers use to reject a tip state that `max_range_size` excluded.
"""
function state_index(ss::StateSpace, range::AbstractVector{<:Integer})
    key = sort!(collect(Int, range))
    for (i, s) in enumerate(ss.states)
        s == key && return i
    end
    return 0
end

# ---------------------------------------------------------------------------
# Tip ranges
# ---------------------------------------------------------------------------

"""
    TipRanges

The geographic ranges of the tips, as read from a LAGRANGE/PHYLIP geography
file.

`data[i, j]` is `'1'`, `'0'` or `'?'`, where `'?'` means "unknown, possibly
present" — R keeps the character matrix rather than a logical one precisely so
that the ambiguity survives, and so does this type.
"""
struct TipRanges
    tipnames::Vector{String}
    areanames::Vector{String}
    data::Matrix{Char}

    function TipRanges(tipnames, areanames, data)
        tips = String.(tipnames)
        areas = String.(areanames)
        chars = Matrix{Char}(data)
        size(chars, 1) == length(tips) ||
            throw(ArgumentError("TipRanges has $(length(tips)) tip names but $(size(chars, 1)) data rows"))
        size(chars, 2) == length(areas) ||
            throw(ArgumentError("TipRanges has $(length(areas)) area names but $(size(chars, 2)) data columns"))
        length(unique(areas)) == length(areas) ||
            throw(ArgumentError("duplicate area names in TipRanges: $(areas)"))
        bad = setdiff(unique(chars), ['0', '1', '?'])
        isempty(bad) ||
            throw(ArgumentError("TipRanges data may only contain '0', '1' and '?', found $(bad)"))
        return new(tips, areas, chars)
    end
end

ntips(tips::TipRanges) = length(tips.tipnames)
nareas(tips::TipRanges) = length(tips.areanames)

"""
    tip_range(tips::TipRanges, i) -> Vector{Int}

1-based area indices coded `'1'` in row `i`.
"""
tip_range(tips::TipRanges, i::Integer) =
    findall(==('1'), view(tips.data, i, :))

has_ambiguity(tips::TipRanges, i::Integer) = any(==('?'), view(tips.data, i, :))

function Base.show(io::IO, tips::TipRanges)
    print(io, "TipRanges: ", ntips(tips), " tips x ", nareas(tips), " areas")
end

# ---------------------------------------------------------------------------
# Tip conditional likelihoods
# ---------------------------------------------------------------------------

"""
    tip_condlikes(tr::BGTree, tips::TipRanges; max_range_size, include_null_range, use_ambiguities=false)

Conditional likelihood of the observed range of every tip, as an
`ntips x nstates` matrix of zeros and ones, in **tree tip order** (row `i`
belongs to `tr.tip_labels[i]`, not to `tips.tipnames[i]`).

A tip whose range is not in the state space raises an `ArgumentError`; that is
the R behaviour too, and it catches the common mistake of a tip occupying more
areas than `max_range_size` allows.

With `use_ambiguities=true`, `'?'` means "may or may not be present": every
state that includes all `'1'` areas and no `'0'` areas gets likelihood 1.  A row
of all `'?'` therefore allows every state, and a row of all `'0'` is the null
range.
"""
function tip_condlikes(tr::BGTree, tips::TipRanges;
                       max_range_size::Integer = length(tips.areanames),
                       include_null_range::Bool = true,
                       use_ambiguities::Bool = false)
    ss = StateSpace(tips.areanames;
                    max_range_size = max_range_size,
                    include_null_range = include_null_range)

    # Rows must follow the tree, not the geography file.
    rows = Vector{Int}(undef, ntips(tr))
    for (i, label) in enumerate(tr.tip_labels)
        j = findfirst(==(label), tips.tipnames)
        j === nothing &&
            throw(ArgumentError("tip $(repr(label)) is in the tree but not in the geography data"))
        rows[i] = j
    end

    lik = zeros(Float64, ntips(tr), length(ss))
    ambiguities = use_ambiguities && any(==('?'), tips.data)

    for i in 1:ntips(tr)
        lik[i, :] .= _tip_row_likelihood(ss, view(tips.data, rows[i], :), ambiguities, i, tr)
    end
    return lik
end

function _tip_row_likelihood(ss::StateSpace, row, ambiguities::Bool, rownum::Int, tr::BGTree)
    out = zeros(Float64, length(ss))

    if !ambiguities
        # '?' is treated as absent, exactly as R does when useAmbiguities is off.
        range = [j for j in 1:length(row) if row[j] == '1']
        idx = state_index(ss, range)
        idx == 0 && _throw_disallowed_range(ss, range, rownum, tr)
        out[idx] = 1.0
        return out
    end

    required = [j for j in 1:length(row) if row[j] == '1']
    forbidden = [j for j in 1:length(row) if row[j] == '0']
    if isempty(required) && isempty(forbidden)
        # All '?': every state is possible.
        fill!(out, 1.0)
        return out
    end

    for (k, state) in enumerate(ss.states)
        isempty(state) && continue                     # null range is never implied by data
        all(a -> a in state, required) || continue
        any(a -> a in state, forbidden) && continue
        out[k] = 1.0
    end

    # A tip coded entirely as hard absences is the null range (fossils).  A row
    # that still has a '?' is not "all zeros" and gets no null-range mass.
    if isempty(required) && !any(==('?'), row) && ss.include_null_range
        out[state_index(ss, Int[])] = 1.0
    end
    return out
end

function _throw_disallowed_range(ss::StateSpace, range, rownum::Int, tr::BGTree)
    txt = isempty(range) ? NULL_RANGE_TEXT :
          join((ss.areanames[a] for a in range), ",")
    hint = ss.include_null_range ? "" : "  (include_null_range is false)"
    throw(ArgumentError(
        "tip $(repr(tr.tip_labels[rownum])) has range $(repr(txt)), which is not in " *
        "the state space (nareas=$(length(ss.areanames)), max_range_size=$(ss.max_range_size))" * hint))
end
