# Stochastic mapping event summaries.

"""
    CladoEventRow

One row of a BSM's cladogenesis table (R's `trtable` columns that the counting
code uses).  `dispersal_from` is filled in later by
[`simulate_source_areas!`](@ref).
"""
mutable struct CladoEventRow
    node::Int
    time_bp::Float64
    event_type::String
    event_txt::String
    dispersal_to::String
    dispersal_from::String
end

"""
    AnaEventRow

One anagenetic event, with the source area added by
[`simulate_source_areas!`](@ref).
"""
mutable struct AnaEventRow
    node::Int
    trynum::Int
    event_time::Float64
    abs_event_time::Float64
    event_type::String
    from_range::String
    to_range::String
    dispersal_to::String
    dispersal_from::String
    extirpation_from::String
end

"""
    BSMTables

One stochastic map's cladogenesis and anagenetic events, in the layout R's
`summarize_BSM_tables_v1.R` expects.
"""
struct BSMTables
    clado::Vector{CladoEventRow}
    ana::Vector{AnaEventRow}
end

"""
    bsm_tables(b::BSMResult, tr::BGTree) -> BSMTables

Convert a [`BSMResult`](@ref) into the flat tables the counting code uses.
Cladogenesis rows follow node order (tips omitted, since they never speciate);
anagenetic rows follow node order and then event order within a branch, which
is the order R builds `ana_events_table` in.
"""
function bsm_tables(b::BSMResult, tr::BGTree)
    pt = prt_table(tr)
    clado = CladoEventRow[]
    for n in 1:nnodes(tr)
        is_tip(tr, n) && continue
        push!(clado, CladoEventRow(n, pt.time_bp[n], b.clado_event_type[n],
                                   b.clado_event_txt[n], b.clado_dispersal_to[n], ""))
    end
    ana = AnaEventRow[]
    for n in 1:nnodes(tr), e in b.branch_events[n]
        push!(ana, AnaEventRow(n, e.trynum, e.event_time, e.abs_event_time,
                               e.event_type, e.from_range, e.to_range,
                               e.dispersal_to, "", e.extirpation_from))
    end
    return BSMTables(clado, ana)
end

# ---------------------------------------------------------------------------
# Source areas
# ---------------------------------------------------------------------------

"""
    simulate_source_areas!(tables, ss, rng; dmat = nothing, times = nothing)

Fill in `dispersal_from` for every event whose destination is set, exactly as
R's `simulate_source_area_clado()` / `simulate_source_area_ana()` do.

The source range is split into single areas; each candidate is weighted by
`dmat[candidate, destination]` and one is drawn with R's `sample(...,
replace = FALSE, prob = ...)`.  Events are visited destination-area by
destination-area, in table order, because that is the order R consumes the
random stream in.

For time-varying dispersal, pass a vector of matrices and matching strictly
increasing upper boundaries as `times`. A boundary belongs to the older bin.
"""
function simulate_source_areas!(tables::AbstractVector{BSMTables},
                                ss::StateSpace, rng::RRng; dmat = nothing, times = nothing)
    n = nareas(ss)
    varying = dmat isa AbstractVector
    varying && times === nothing && throw(ArgumentError("times are required for time-varying dmat"))
    !varying && times !== nothing && throw(ArgumentError("times require a vector of matrices"))
    bounds = varying ? Float64.(times) : Float64[]
    if varying
        length(dmat) == length(bounds) && !isempty(bounds) ||
            throw(ArgumentError("dmat and times must have the same nonzero length"))
        all(isfinite, bounds) && bounds[1] > 0 && all(diff(bounds) .> 0) ||
            throw(ArgumentError("times must be finite, positive and strictly increasing"))
    end
    matrices = varying ? [Matrix{Float64}(m) for m in dmat] :
               [dmat === nothing ? ones(Float64, n, n) : Matrix{Float64}(dmat)]
    for dm in matrices
        size(dm) == (n, n) || throw(ArgumentError("dmat is $(size(dm)), expected ($n, $n)"))
        all(x -> isfinite(x) && x >= 0, dm) || throw(ArgumentError("dmat entries must be finite and non-negative"))
    end
    for t in tables
        _source_areas!(t.clado, ss, rng, matrices, bounds, :clado)
        _source_areas!(t.ana, ss, rng, matrices, bounds, :ana)
    end
    return tables
end

function _source_areas!(rows, ss::StateSpace, rng::RRng, matrices, bounds,
                        kind::Symbol)
    areas = ss.areanames
    for (i, dest) in enumerate(areas)
        for row in rows
            row.dispersal_to == dest || continue
            src_range = kind === :clado ? _clado_ancestor(row.event_txt) : row.from_range
            src_areas = [string(c) for c in src_range]
            idx = [findfirst(==(a), areas) for a in src_areas]
            any(isnothing, idx) && continue        # null range or unknown area
            event_time = kind === :clado ? row.time_bp : row.abs_event_time
            bin = isempty(bounds) ? 1 : searchsortedlast(bounds, event_time) + 1
            (isempty(bounds) || (0 <= event_time && bin <= length(matrices))) ||
                throw(ArgumentError("event time $event_time is outside the time strata"))
            dm = matrices[bin]
            probs = [dm[j, i] for j in idx]
            s = sum(probs)
            if s <= 0
                # R warns and falls back to uniform weights; keep that.
                probs .= 1.0
                s = Float64(length(probs))
            end
            probs ./= s
            k = r_sample_prob(rng, probs, 1; replace = false)[1]
            row.dispersal_from = string(src_areas[k])
        end
    end
    return rows
end

"Ancestral range of a `parent->left,right` cladogenesis label."
function _clado_ancestor(txt::AbstractString)
    i = findfirst("->", txt)
    i === nothing && return ""
    return txt[1:first(i) - 1]
end

# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

"""
    count_clado_dispersal_events(t::BSMTables, ss::StateSpace) -> Matrix{Int}

Founder/jump dispersal counts by (source, destination) area.
"""
function count_clado_dispersal_events(t::BSMTables, ss::StateSpace)
    n = nareas(ss)
    areas = ss.areanames
    counts = zeros(Int, n, n)
    for row in t.clado
        row.dispersal_to in ("", "-") && continue
        i = findfirst(==(row.dispersal_from), areas)
        j = findfirst(==(row.dispersal_to), areas)
        (i === nothing || j === nothing) && continue
        counts[i, j] += 1
    end
    return counts
end

"""
    count_ana_dispersal_events(t::BSMTables, ss::StateSpace)
        -> NamedTuple

Anagenetic dispersal and extirpation counts.  Returns `counts` (all `a` + `d`),
`a` (range switching), `d` (range expansion) as `nareas × nareas` matrices and
`e` (extirpation, per area) as a vector.
"""
function count_ana_dispersal_events(t::BSMTables, ss::StateSpace)
    n = nareas(ss)
    areas = ss.areanames
    counts = zeros(Int, n, n)
    a = zeros(Int, n, n)
    d = zeros(Int, n, n)
    e = zeros(Int, n)
    for row in t.ana
        if !(row.dispersal_to in ("", "-"))
            i = findfirst(==(row.dispersal_from), areas)
            j = findfirst(==(row.dispersal_to), areas)
            if i !== nothing && j !== nothing
                counts[i, j] += 1
                row.event_type == "a" && (a[i, j] += 1)
                row.event_type == "d" && (d[i, j] += 1)
            end
        end
        if row.event_type == "e" && !(row.extirpation_from in ("", "-"))
            i = findfirst(==(row.extirpation_from), areas)
            i === nothing || (e[i] += 1)
        end
    end
    return (counts = counts, a = a, d = d, e = e)
end

"""
    uniquify_clado_events(t::BSMTables) -> Vector{CladoEventRow}

Drop tip rows and rewrite every `parent->left,right` label so that mirror
images compare equal: the *smaller* daughter goes on the left (R swaps
whenever `nchar(L) > nchar(R)`), and equal-size daughters are sorted
alphabetically (R's `uniquify_clado_events()`).
"""
function uniquify_clado_events(t::BSMTables)
    out = CladoEventRow[]
    for row in t.clado
        isempty(row.event_type) && continue
        txt = row.event_txt
        i = findfirst("->", txt)
        i === nothing && continue
        anc = txt[1:first(i) - 1]
        rest = txt[last(i) + 1:end]
        comma = findfirst(',', rest)
        comma === nothing && continue
        l = rest[1:comma - 1]
        r = rest[comma + 1:end]
        if length(l) > length(r) || (length(l) == length(r) && l > r)
            l, r = r, l
        end
        push!(out, CladoEventRow(row.node, row.time_bp, row.event_type,
                                 anc * "->" * l * "," * r, row.dispersal_to,
                                 row.dispersal_from))
    end
    return out
end

"""
    count_ana_clado_events(tables, ss; timeperiod=nothing) -> NamedTuple

Aggregate event counts over a collection of BSMs, mirroring the parts of R's
`count_ana_clado_events()` that the BSM summaries use.
If `timeperiod` contains two finite ages, count events in the half-open interval
`[minimum(timeperiod), maximum(timeperiod))`, regardless of input order.

Returned fields:

* `summary_counts_BSMs` — 3×12 matrix (`means`, `stdevs`, `sums` ×
  `founder a d e subset vicariance sympatry ALL_disp ana_disp all_ana
  all_clado total_events`)
* `all_dispersals_counts_fromto_means` / `_sds`, `d_counts_fromto_means`,
  `a_counts_fromto_means`, `founder_counts_fromto_means`
* `e_counts_rectangle` — maps × areas
* `unique_vic_counts` / `unique_sub_counts` / `unique_sym_counts` — maps ×
  unique events, with `vic_events` / `sub_events` / `sym_events` names
* the per-map total vectors
"""
function count_ana_clado_events(tables::AbstractVector{BSMTables}, ss::StateSpace;
                                timeperiod = nothing)
    n = nareas(ss)
    nmaps = length(tables)
    nmaps > 0 || throw(ArgumentError("need at least one BSM"))
    if timeperiod !== nothing
        (timeperiod isa Tuple || timeperiod isa AbstractVector) && length(timeperiod) == 2 &&
            all(x -> x isa Real && isfinite(x), timeperiod) ||
            throw(ArgumentError("timeperiod must contain two finite ages"))
        lo, hi = minmax(timeperiod...)
        tables = [BSMTables(filter(r -> lo <= r.time_bp < hi, t.clado),
                            filter(r -> lo <= r.abs_event_time < hi, t.ana)) for t in tables]
    end

    founder_cube = zeros(Int, n, n, nmaps)
    ana_cube = zeros(Int, n, n, nmaps)
    a_cube = zeros(Int, n, n, nmaps)
    d_cube = zeros(Int, n, n, nmaps)
    e_rect = zeros(Int, nmaps, n)

    # First pass: unique cladogenesis labels, per event type.
    uq = [uniquify_clado_events(t) for t in tables]
    vic_names = sort(unique([r.event_txt for rows in uq for r in rows
                             if r.event_type == "vicariance (v)"]))
    sub_names = sort(unique([r.event_txt for rows in uq for r in rows
                             if r.event_type == "subset (s)"]))
    sym_names = sort(unique([r.event_txt for rows in uq for r in rows
                             if r.event_type == "sympatry (y)"]))

    vic_counts = zeros(Int, nmaps, length(vic_names))
    sub_counts = zeros(Int, nmaps, length(sub_names))
    sym_counts = zeros(Int, nmaps, length(sym_names))

    founder_tot = zeros(Int, nmaps)
    ana_tot = zeros(Int, nmaps)
    a_tot = zeros(Int, nmaps)
    d_tot = zeros(Int, nmaps)
    e_tot = zeros(Int, nmaps)
    vic_tot = zeros(Int, nmaps)
    sub_tot = zeros(Int, nmaps)
    sym_tot = zeros(Int, nmaps)

    for (i, t) in enumerate(tables)
        founder_cube[:, :, i] .= count_clado_dispersal_events(t, ss)
        founder_tot[i] = sum(founder_cube[:, :, i])

        ac = count_ana_dispersal_events(t, ss)
        ana_cube[:, :, i] .= ac.counts
        a_cube[:, :, i] .= ac.a
        d_cube[:, :, i] .= ac.d
        e_rect[i, :] .= ac.e
        ana_tot[i] = sum(ac.counts)
        a_tot[i] = sum(ac.a)
        d_tot[i] = sum(ac.d)
        e_tot[i] = sum(ac.e)
        ana_tot[i] += e_tot[i]

        for row in uq[i]
            k = findfirst(==(row.event_txt), vic_names)
            if row.event_type == "vicariance (v)" && k !== nothing
                vic_counts[i, k] += 1
            end
            k = findfirst(==(row.event_txt), sub_names)
            if row.event_type == "subset (s)" && k !== nothing
                sub_counts[i, k] += 1
            end
            k = findfirst(==(row.event_txt), sym_names)
            if row.event_type == "sympatry (y)" && k !== nothing
                sym_counts[i, k] += 1
            end
        end
        vic_tot[i] = sum(vic_counts[i, :])
        sub_tot[i] = sum(sub_counts[i, :])
        sym_tot[i] = sum(sym_counts[i, :])
    end

    all_disp_cube = founder_cube .+ a_cube .+ d_cube
    all_disp_tot = founder_tot .+ a_tot .+ d_tot
    ana_disp_tot = d_tot .+ a_tot
    clado_tot = founder_tot .+ sub_tot .+ vic_tot .+ sym_tot
    all_tot = clado_tot .+ ana_tot

    means = [mean(founder_tot), mean(a_tot), mean(d_tot), mean(e_tot),
             mean(sub_tot), mean(vic_tot), mean(sym_tot), mean(all_disp_tot),
             mean(ana_disp_tot), mean(ana_tot), mean(clado_tot), mean(all_tot)]
    sds = [r_sd(founder_tot), r_sd(a_tot), r_sd(d_tot), r_sd(e_tot),
           r_sd(sub_tot), r_sd(vic_tot), r_sd(sym_tot), r_sd(all_disp_tot),
           r_sd(ana_disp_tot), r_sd(ana_tot), r_sd(clado_tot), r_sd(all_tot)]
    sums = [sum(founder_tot), sum(a_tot), sum(d_tot), sum(e_tot),
            sum(sub_tot), sum(vic_tot), sum(sym_tot), sum(all_disp_tot),
            sum(ana_disp_tot), sum(ana_tot), sum(clado_tot), sum(all_tot)]

    summary = permutedims(hcat(means, sds, sums))   # 3 × 12

    return (
        summary_counts_BSMs = summary,
        all_dispersals_counts_fromto_means = _mean_over_maps(all_disp_cube),
        all_dispersals_counts_fromto_sds = _sd_over_maps(all_disp_cube),
        d_counts_fromto_means = _mean_over_maps(d_cube),
        d_counts_fromto_sds = _sd_over_maps(d_cube),
        a_counts_fromto_means = _mean_over_maps(a_cube),
        a_counts_fromto_sds = _sd_over_maps(a_cube),
        founder_counts_fromto_means = _mean_over_maps(founder_cube),
        founder_counts_fromto_sds = _sd_over_maps(founder_cube),
        ana_counts_fromto_means = _mean_over_maps(ana_cube),
        e_counts_rectangle = e_rect,
        unique_vic_counts = vic_counts, vic_events = vic_names,
        unique_sub_counts = sub_counts, sub_events = sub_names,
        unique_sym_counts = sym_counts, sym_events = sym_names,
        founder_totals_list = founder_tot,
        ana_totals_list = ana_tot,
        a_totals_list = a_tot,
        d_totals_list = d_tot,
        e_totals_list = e_tot,
        subsetSymp_totals_list = sub_tot,
        vicariance_totals_list = vic_tot,
        sympatry_totals_list = sym_tot,
        clado_totals_list = clado_tot,
        all_totals_list = all_tot,
    )
end

# R's sd() is the n-1 sample standard deviation, and returns NA for n = 1.
r_sd(v) = length(v) < 2 ? NaN : std(v)

_mean_over_maps(cube) = dropdims(mean(cube; dims = 3); dims = 3)
_sd_over_maps(cube) = dropdims(mapslices(r_sd, cube; dims = 3); dims = 3)

"""
    calc_bsm_mean_node_states(maps, tr, nstates) -> NamedTuple

R's `calc_BSM_mean_node_states()`: for each node and state, the number of maps
that sampled that state.

Returns

* `state_counts` — `nnodes × nstates × nmaps`, one-hot per node and map
* `mean_bsm_probs` — the per-node state frequencies across maps
* `sum_bsm_counts` — the same counts summed over maps (a matrix)

R takes the `clado_events_tables`; here the [`BSMResult`](@ref)s are enough and
avoid re-deriving the sampled node states.
"""
function calc_bsm_mean_node_states(maps::AbstractVector, tr::BGTree,
                                   nstates::Integer)
    nn = nnodes(tr)
    nmaps = length(maps)
    counts = zeros(Int, nn, nstates, nmaps)
    for (i, b) in enumerate(maps)
        for n in 1:nn
            counts[n, b.node_states[n], i] = 1
        end
    end
    sums = dropdims(sum(counts; dims = 3); dims = 3)
    return (state_counts = counts,
            mean_bsm_probs = sums ./ nmaps,
            sum_bsm_counts = sums)
end

calc_bsm_mean_node_states(maps, tr::BGTree, ss::StateSpace) =
    calc_bsm_mean_node_states(maps, tr, nstates(ss))

const _SUMMARY_ROWS = ["means", "stdevs", "sums"]
const _SUMMARY_COLS = ["founder", "a", "d", "e", "subset", "vicariance",
                       "sympatry", "ALL_disp", "ana_disp", "all_ana",
                       "all_clado", "total_events"]

"""
    print_counts_lists_to_file(c, model_name = "default"; dir = ".")

R's `print_counts_lists_to_file()`: write every two-dimensional table of a
[`count_ana_clado_events`](@ref) result to `dir/<model_name>_<field>.txt` as
tab-delimited text.  Cubes and vectors are skipped, as in R.  Returns the list
of files written.

R also draws a PDF of event-count histograms here; that belongs to the plotting
layer and is not part of this function.
"""
function print_counts_lists_to_file(c, model_name::AbstractString = "default";
                                    dir::AbstractString = ".")
    written = String[]
    for (nm, v) in pairs(c)
        v isa AbstractMatrix || continue
        isempty(v) && continue
        fn = joinpath(dir, string(model_name, "_", nm, ".txt"))
        rows = nm === :summary_counts_BSMs ? _SUMMARY_ROWS : string.(1:size(v, 1))
        cols = nm === :summary_counts_BSMs ? _SUMMARY_COLS : string.(1:size(v, 2))
        open(fn, "w") do io
            println(io, join(vcat("row", cols), '\t'))
            for i in axes(v, 1)
                println(io, join(vcat(rows[i], string.(v[i, :])), '\t'))
            end
        end
        push!(written, fn)
    end
    return written
end
