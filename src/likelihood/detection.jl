# Tip likelihoods with imperfect detection.

"""
    DetectionCounts

A matrix of detection counts, one row per OTU and one column per area, with
the OTU and area names carried alongside — R's `detects_df` / `controls_df`.

Rows are matched to tree tips **by name**, never by position, so the order of
the file does not matter.
"""
struct DetectionCounts
    otunames::Vector{String}
    areanames::Vector{String}
    counts::Matrix{Float64}

    function DetectionCounts(otunames, areanames, counts)
        otu = String.(otunames)
        areas = String.(areanames)
        mat = Matrix{Float64}(counts)
        size(mat, 1) == length(otu) ||
            throw(ArgumentError("DetectionCounts has $(length(otu)) OTU names but $(size(mat, 1)) rows"))
        size(mat, 2) == length(areas) ||
            throw(ArgumentError("DetectionCounts has $(length(areas)) area names but $(size(mat, 2)) columns"))
        length(unique(otu)) == length(otu) ||
            throw(ArgumentError("duplicate OTU names in DetectionCounts: $(otu)"))
        return new(otu, areas, mat)
    end
end

ntips(d::DetectionCounts) = length(d.otunames)
nareas(d::DetectionCounts) = length(d.areanames)

function Base.show(io::IO, d::DetectionCounts)
    print(io, "DetectionCounts: ", ntips(d), " OTUs x ", nareas(d), " areas")
end

"""
    read_detections(path; skip=0) -> DetectionCounts
    read_controls(path; skip=0) -> DetectionCounts

Read a tab- or space-delimited table of counts:

```
	K	O	M	H
P_mariniana_Kokee2	10	1	0	0
```

The first non-blank line is the header of area names; each following line is an
OTU name followed by one count per area.  `skip` drops that many leading lines
(R's `tmpskip`).

R has two readers with identical code, one for detections and one for
taphonomic controls; both are kept here for symmetry with R's input files.
"""
read_detections(path::AbstractString; skip::Integer = 0) = _read_counts(path, skip, "detections")

read_controls(path::AbstractString; skip::Integer = 0) = _read_counts(path, skip, "controls")

function _read_counts(path::AbstractString, skip::Integer, what::AbstractString)
    isfile(path) || throw(ArgumentError("$(what) file not found: $(path)"))
    skip < 0 && throw(ArgumentError("skip must be non-negative, got $(skip)"))

    lines = readlines(path)
    start = skip + 1
    start > length(lines) &&
        throw(ArgumentError("$(path): skip=$(skip) is past the end of the file"))

    idx = findfirst(l -> !isempty(strip(l)), lines[start:end])
    idx === nothing && throw(ArgumentError("$(path): no header line"))
    header = start + idx - 1

    areanames = String.(split(strip(lines[header])))
    isempty(areanames) && throw(ArgumentError("$(path):$(header): missing area names"))

    otunames = String[]
    rows = Vector{Float64}[]
    for lineno in (header + 1):length(lines)
        line = strip(lines[lineno])
        isempty(line) && continue
        fields = split(line)
        length(fields) == length(areanames) + 1 ||
            throw(ArgumentError("$(path):$(lineno): expected an OTU name and " *
                                "$(length(areanames)) counts, got $(length(fields)) fields"))
        push!(otunames, String(fields[1]))
        push!(rows, _parse_row(fields[2:end], path, lineno))
    end
    isempty(rows) && throw(ArgumentError("$(path): no data rows"))
    return DetectionCounts(otunames, areanames, permutedims(reduce(hcat, rows)))
end

"""
    calc_obs_like(truly_present, obs_target, obs_all;
                  mean_frequency=0.1, dp=1.0, fdp=0.0) -> Float64

Log-likelihood of one area's counts given whether the OTU is truly present
there — R's `calc_obs_like()`.

`obs_target` is the count of detections of the OTU of interest, `obs_all` the
count of all detections in that area (target plus taphonomic controls).  With
`obs_all == 0` there is no information at all, so the function returns `0`
(log of probability 1) under either hypothesis.

`0 * log(0)` is treated as `0` (probability 1), exactly as R does with its
`-Inf * 0` traps.
"""
function calc_obs_like(truly_present::Bool, obs_target::Real, obs_all::Real;
                       mean_frequency::Real = 0.1, dp::Real = 1.0, fdp::Real = 0.0)
    obs_all == 0 && return 0.0

    obs_target = float(obs_target)
    obs_non = float(obs_all) - obs_target

    # Sampling: a single sample is the target species only if the OTU is present.
    p_sample_target_given_present = float(mean_frequency)
    p_sample_nontarget_given_present = 1 - p_sample_target_given_present
    p_sample_target_given_absent = 0.0
    p_sample_nontarget_given_absent = 1 - p_sample_target_given_absent

    # Identification.
    p_id_accurate_given_target = float(dp)
    p_id_inaccurate_given_target = 1 - p_id_accurate_given_target
    p_id_accurate_given_nontarget = 1 - float(fdp)
    p_id_inaccurate_given_nontarget = 1 - p_id_accurate_given_nontarget

    if truly_present
        # Ways to record a positive: true positive or false positive.
        p_pos = p_sample_target_given_present * p_id_accurate_given_target +
                p_sample_nontarget_given_present * p_id_inaccurate_given_nontarget
        # Ways to record a negative: true negative (still present) or false negative.
        p_neg = p_sample_nontarget_given_present * p_id_accurate_given_nontarget +
                p_sample_target_given_present * p_id_inaccurate_given_target
        return _logtimes(p_pos, obs_target) + _logtimes(p_neg, obs_non)
    end

    # Truly absent: a positive can only be a misidentification of a non-target.
    p_pos = p_sample_target_given_absent * p_id_accurate_given_target +
            p_sample_nontarget_given_absent * p_id_inaccurate_given_nontarget
    p_neg = p_sample_nontarget_given_absent * p_id_accurate_given_nontarget +
            p_sample_target_given_absent * p_id_accurate_given_target
    return _logtimes(p_pos, obs_target) + _logtimes(p_neg, obs_non)
end

# n * log(p), with 0 * log(0) defined as 0 (R's explicit -Inf * 0 traps).
_logtimes(p::Real, n::Real) = n == 0 ? 0.0 : n * log(p)

"""
    Pdata_given_rangerow(range_as_areas, detects_row, controls_row;
                         mean_frequency=0.1, dp=1.0, fdp=0.0,
                         return_lnls=true) -> Float64

Log-likelihood (or likelihood, with `return_lnls=false`) of one OTU's whole row
of counts under a hypothesised range — R's `Pdata_given_rangerow()`.

`range_as_areas` is a boolean vector, one entry per area: `true` where the range
includes the area.  The result is the sum over areas of
[`calc_obs_like`](@ref), each area conditioned on presence or absence.
"""
function Pdata_given_rangerow(range_as_areas::AbstractVector{Bool},
                              detects_row::AbstractVector{<:Real},
                              controls_row::AbstractVector{<:Real};
                              mean_frequency::Real = 0.1, dp::Real = 1.0,
                              fdp::Real = 0.0, return_lnls::Bool = true)
    length(range_as_areas) == length(detects_row) == length(controls_row) ||
        throw(ArgumentError("range_as_areas, detects_row and controls_row must have the same length"))

    total = 0.0
    for a in eachindex(range_as_areas)
        total += calc_obs_like(range_as_areas[a], detects_row[a], controls_row[a];
                               mean_frequency = mean_frequency, dp = dp, fdp = fdp)
    end
    return return_lnls ? total : exp(total)
end

"""
    tiplikes_wDetectionModel(tr, ss, detects, controls; kwargs...) -> Matrix

`ntips × nstates` matrix of tip conditional likelihoods under the detection
model — R's `tiplikes_wDetectionModel()`.

Row `i` belongs to `tr.tip_labels[i]`, so the result can be passed straight to
[`loglikelihood`](@ref) / [`downpass`](@ref) as `tip_condlikes`.

Keyword arguments (R names in brackets):

* `mean_frequency` [`mean_frequency`], `dp`, `fdp` — the detection model
* `null_range_gets_0_like` — give the null range zero probability, i.e. assume
  the OTU exists somewhere in the study area (default `true`)
* `return_lnls` — work and return log-likelihoods (default `true`)
* `relative_lnls` — subtract each row's maximum (default `true`); only used with
  `return_lnls`
* `exp_lnls` — exponentiate at the end (default `true`); only used with
  `return_lnls`
* `error_check` — throw when a tip's likelihoods sum to zero (default `true`)

The last three reproduce R's default pipeline: compute in log space, scale each
row so the best state has log-likelihood 0, then exponentiate.  That avoids
underflow while returning ordinary likelihoods.
"""
function tiplikes_wDetectionModel(tr::BGTree, ss::StateSpace,
                                  detects::DetectionCounts,
                                  controls::DetectionCounts;
                                  mean_frequency::Real = 0.1, dp::Real = 1.0,
                                  fdp::Real = 0.0,
                                  null_range_gets_0_like::Bool = true,
                                  return_lnls::Bool = true,
                                  relative_lnls::Bool = true,
                                  exp_lnls::Bool = true,
                                  error_check::Bool = true)
    na = nareas(ss)
    nareas(detects) == na ||
        throw(ArgumentError("detections have $(nareas(detects)) areas, state space has $(na)"))
    size(detects.counts) == size(controls.counts) ||
        throw(ArgumentError("detections are $(size(detects.counts)), controls are $(size(controls.counts))"))
    detects.areanames == controls.areanames ||
        throw(ArgumentError("detections and controls use different area names: " *
                            "$(detects.areanames) vs $(controls.areanames)"))
    controls.otunames == detects.otunames ||
        throw(ArgumentError("detections and controls list different OTUs"))

    # Rows follow the tree, not the file.
    rows = Vector{Int}(undef, ntips(tr))
    for (i, label) in enumerate(tr.tip_labels)
        j = findfirst(==(label), detects.otunames)
        j === nothing &&
            throw(ArgumentError("tip $(repr(label)) is in the tree but not in the detection counts"))
        rows[i] = j
    end

    nt = ntips(tr)
    ns = nstates(ss)
    out = zeros(Float64, nt, ns)
    zero_like = return_lnls ? -Inf : 0.0

    for s in 1:ns
        state = ss.states[s]
        if null_range_gets_0_like && isempty(state)
            view(out, :, s) .= zero_like
            continue
        end

        range_as_areas = falses(na)
        for a in state
            range_as_areas[a] = true
        end

        for i in 1:nt
            out[i, s] = Pdata_given_rangerow(range_as_areas,
                                             view(detects.counts, rows[i], :),
                                             view(controls.counts, rows[i], :);
                                             mean_frequency = mean_frequency,
                                             dp = dp, fdp = fdp,
                                             return_lnls = return_lnls)
        end
    end

    if return_lnls && relative_lnls
        for i in 1:nt
            # R subtracts the row max unconditionally; an all -Inf row becomes
            # NaN there too, which is the honest signal that no state fits.
            out[i, :] .-= maximum(view(out, i, :))
        end
    end

    return_lnls && exp_lnls && (out = exp.(out))

    if error_check && ((return_lnls && exp_lnls) || !return_lnls)
        sums = vec(sum(out, dims = 2))
        bad = findall(<=(0), sums)
        if !isempty(bad)
            throw(ArgumentError("tiplikes_wDetectionModel: the likelihoods of tip(s) " *
                                "$(bad) sum to 0; no state can explain the data"))
        end
    end

    return out
end
