# Anagenetic transition rates.

"""
    qmatrix(ss::StateSpace, model::Model;
            dispersal_multipliers=nothing, extinction_modifiers=nothing,
            r_float32=false) -> SparseMatrixCSC

The instantaneous anagenesis rate matrix for `ss`, built from the model's `d`,
`e` and `a`.

`dispersal_multipliers` is the `nareas × nareas` matrix of distance / manual
multipliers applied to `d` and `a`; `extinction_modifiers` is the length-`nareas`
vector of area-dependent extinction multipliers applied to `e`.  Both default to
all ones, which is what a non-stratified analysis without distance matrices
uses.

The diagonal is minus the sum of off-diagonal entries, so each row sums to zero.

`r_float32` reproduces a quirk of R's `cladoRcpp::rcpp_states_list_to_DEmat()`:
it accumulates each dispersal entry in C `float` and stores the matrix in
single precision; the row sum is accumulated in double and the negated total is then
stored as a `float`.  This matters only when the result feeds a fixed-seed
simulation that must match R bit for bit (see [`stochastic_map`](@ref)); for
likelihood work the double version is the accurate one.
"""
function qmatrix(ss::StateSpace, model::Model;
                 dispersal_multipliers = nothing,
                 distances = nothing,
                 extinction_modifiers = nothing,
                 r_float32::Bool = false)
    n = nareas(ss)
    d = model["d"]
    e = model["e"]
    a = model["a"]

    mult = dispersal_multipliers === nothing ? ones(n, n) :
           Matrix{Float64}(dispersal_multipliers)
    mult = _apply_distance_exponent(mult, distances, model["x"], n)
    emod = extinction_modifiers === nothing ? ones(n) :
           Vector{Float64}(extinction_modifiers)
    size(mult) == (n, n) ||
        throw(ArgumentError("dispersal_multipliers is $(size(mult)), expected ($n, $n)"))
    length(emod) == n ||
        throw(ArgumentError("extinction_modifiers has $(length(emod)) entries, expected $n"))

    dmat = d .* mult
    amat = a .* mult
    elist = e .* emod

    ns = length(ss)
    lookup = Dict{Vector{Int},Int}(state => i for (i, state) in enumerate(ss.states))

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for i in 1:ns
        state = ss.states[i]

        # range expansion: add one area
        for x in 1:n
            x in state && continue
            target = sort!(vcat(state, x))
            j = get(lookup, target, 0)
            j == 0 && continue
            w = if r_float32
                total = 0.0f0
                for from in state
                    total += Float32(dmat[from, x])
                end
                Float64(total)
            else
                sum((dmat[from, x] for from in state); init = 0.0)
            end
            w == 0 && continue
            push!(rows, i); push!(cols, j); push!(vals, w)
        end

        # range contraction: drop one area (a size-1 range contracts to the null range)
        for x in state
            target = [y for y in state if y != x]
            j = get(lookup, target, 0)
            j == 0 && continue
            w = elist[x]
            w == 0 && continue
            push!(rows, i); push!(cols, j); push!(vals, w)
        end

        # single-area range switching (the "a" parameter; zero by default)
        if length(state) == 1
            from = state[1]
            for to in 1:n
                to == from && continue
                j = get(lookup, [to], 0)
                j == 0 && continue
                w = amat[from, to]
                w == 0 && continue
                push!(rows, i); push!(cols, j); push!(vals, w)
            end
        end
    end

    # Diagonals last, so the row sum can be taken over the off-diagonal entries
    # only.  With `r_float32` every stored entry is a single-precision value;
    # R's C++ sums those into a double and stores the negated total back as a
    # `float`, so the accumulation is double and only the result is rounded.
    if r_float32
        vals .= Float64.(Float32.(vals))
    end
    for i in 1:ns
        s = 0.0
        for k in eachindex(vals)
            rows[k] == i && (s += vals[k])
        end
        if s >= 1e-15
            push!(rows, i); push!(cols, i)
            push!(vals, r_float32 ? -Float64(Float32(s)) : -s)
        end
    end

    return sparse(rows, cols, vals, ns, ns)
end

"""
    row_sums(q) -> Vector{Float64}

Row sums of a rate matrix; all zero for a well-formed `Q`.
"""
row_sums(q::AbstractMatrix) = vec(sum(q, dims = 2))

# R multiplies the manual dispersal multipliers by `distances^x`, where `x` is
# the model's distance exponent (`calc_transition_matrices_v1.R:114`).  With
# `x = 0` the distances have no effect; `x = -1` is inverse-distance weighting.
function _apply_distance_exponent(mult::Matrix{Float64}, distances, xexp::Float64,
                                  n::Int)
    distances === nothing && return mult
    dmat = Matrix{Float64}(distances)
    size(dmat) == (n, n) ||
        throw(ArgumentError("distances is $(size(dmat)), expected ($n, $n)"))
    any(<(0), dmat) &&
        throw(ArgumentError("distances must be non-negative"))
    return mult .* dmat .^ xexp
end
