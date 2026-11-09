# Model parameters and linked values.

"""
    Param

One row of the parameter table: a name, how it is derived, its starting value,
its bounds, its current estimate, and a human-readable description.
"""
mutable struct Param
    name::String
    kind::String
    init::Float64
    lower::Float64
    upper::Float64
    est::Float64
    desc::String
end

"""
    Model

The parameters of a BioGeoBEARS model, in the same order and with the same
bounds as R's `BioGeoBEARS_model_defaults()`.

Build one with [`Model`](@ref) and inspect or modify it by name:

```julia
m = Model(:DEC)
m["d"]              # current estimate
m["d"] = 0.05       # change the estimate
free_names(m)       # ["d", "e"] for DEC, ["d", "e", "j"] for DEC+J
```
"""
mutable struct Model
    params::Vector{Param}
    index::Dict{String,Int}
    label::String
end

"""
    Model(; minval_anagenesis=1e-12, minval_cladogenesis=1e-5, maxval=5)

The full default parameter table — every parameter BioGeoBEARS knows about,
whether or not the current model estimates it.
"""
function Model(; minval_anagenesis::Real = 1e-12,
               minval_cladogenesis::Real = 1e-5,
               maxval::Real = 5)
    a0 = float(minval_anagenesis)
    c0 = float(minval_cladogenesis)
    hi = float(maxval)

    ps = Param[]
    add!(name, kind, init, lower, upper, desc) =
        push!(ps, Param(name, kind, float(init), float(lower), float(upper),
                        float(init), desc))

    add!("d", "free", 0.01, 0 + a0, hi - a0,
         "anagenesis: rate of 'dispersal' (range expansion)")
    add!("e", "free", 0.01, 0 + a0, hi - a0,
         "anagenesis: rate of 'extinction' (range contraction)")
    add!("a", "fixed", 0.0, 0 + a0, hi - a0,
         "anagenesis: rate of range-switching (i.e. for a standard char.)")
    add!("b", "fixed", 1.0, 0 + a0, 1 - a0,
         "anagenesis: exponent on branch lengths")
    add!("x", "fixed", 0.0, -2.5, 2.5, "exponent on distance (modifies d, j, a)")
    add!("n", "fixed", 0.0, -10.0, 10.0,
         "exponent on environmental distance (modifies d, j, a)")
    add!("w", "fixed", 1.0, -10.0, 10.0,
         "exponent on manual dispersal multipliers (modifies d, j, a)")
    add!("u", "fixed", 0.0, -10.0, 10.0,
         "anagenesis: exponent on extinction risk with area (modifies e)")
    add!("j", "fixed", 0.0, 0 + c0, 3 - c0,
         "cladogenesis: relative per-event weight of jump dispersal")
    add!("ysv", "3-j", 3 - c0, 0 + c0, 3.0, "cladogenesis: y+s+v")
    add!("ys", "ysv*2/3", 2 - c0, 0 + c0, 2.0, "cladogenesis: y+s")
    add!("y", "ysv*1/3", 1.0, 0 + c0, 1.0,
         "cladogenesis: relative per-event weight of sympatry (range-copying)")
    add!("s", "ysv*1/3", 1.0, 0 + c0, 1.0,
         "cladogenesis: relative per-event weight of subset speciation")
    add!("v", "ysv*1/3", 1.0, 0 + c0, 1.0,
         "cladogenesis: relative per-event weight of vicariant speciation")
    add!("mx01", "fixed", 0.0001, 0.0001, 0.9999,
         "cladogenesis: controls range size of smaller daughter")
    add!("mx01j", "mx01", 0.0001, 0.0001, 0.9999,
         "cladogenesis: controls range size of smaller daughter")
    add!("mx01y", "mx01", 0.0001, 0.0001, 0.9999,
         "cladogenesis: controls range size of smaller daughter")
    add!("mx01s", "mx01", 0.0001, 0.0001, 0.9999,
         "cladogenesis: controls range size of smaller daughter")
    add!("mx01v", "mx01", 0.0001, 0.0001, 0.9999,
         "cladogenesis: controls range size of smaller daughter")
    add!("mx01r", "fixed", 0.5, 0.0001, 0.9999,
         "root: controls range size probabilities of root")
    add!("mf", "fixed", 0.1, 0.005, 0.995,
         "mean frequency of truly sampling OTU of interest")
    add!("dp", "fixed", 1.0, 0.005, 0.995,
         "detection probability per true sample of OTU of interest")
    add!("fdp", "fixed", 0.0, 0.005, 0.995,
         "false detection of OTU probability per true taphonomic control sample")

    return Model(ps, Dict(p.name => i for (i, p) in enumerate(ps)), "custom")
end

"""
    Model(spec::Symbol)

A model ready to run, named as in the BioGeoBEARS literature:

| symbol            | sympatry `y` | subset `s` | vicariance `v` | founder `j` |
|:------------------|:-------------|:-----------|:---------------|:------------|
| `:DEC`            | 1            | 1          | 1              | 0           |
| `:DECJ`           | 1            | 1          | 1              | free        |
| `:DIVALIKE`       | 1            | 0          | 1              | 0           |
| `:DIVALIKEJ`      | 1            | 0          | 1              | free        |
| `:BAYAREALIKE`    | 1            | 0          | 0              | 0           |
| `:BAYAREALIKEJ`   | 1            | 0          | 0              | free        |

The four non-DEC models also change a `mx01*` parameter, which is what makes
them differ from merely dropping an event type:

* `DIVALIKE` sets `mx01v = 0.5`, so a widespread ancestor splits into *any* two
  halves (AB|CD as likely as A|BCD), rather than always isolating one area.
* `BAYAREALIKE` sets `mx01y = 0.9999`, so sympatry always copies the ancestral
  range exactly instead of favouring a small daughter.

`j` starts at `0.01` for `:DECJ` (the R tutorial default) and at `0.0001` for
the other two `+J` models, whose upper bounds are narrower (`2` and `1`) because
they have fewer competing cladogenesis events.
"""
const MODEL_LABELS = Dict(:DEC => "DEC", :DECJ => "DEC+J",
                          :DIVALIKE => "DIVALIKE", :DIVALIKEJ => "DIVALIKE+J",
                          :BAYAREALIKE => "BAYAREALIKE",
                          :BAYAREALIKEJ => "BAYAREALIKE+J")

"""
    model_name(m::Model) -> String

The model's short label, e.g. `"DEC+J"`.  `Model()` with a hand-edited table
returns `"custom"`.
"""
model_name(m::Model) = m.label

function Model(spec::Symbol)
    haskey(MODEL_LABELS, spec) ||
        throw(ArgumentError("unknown model $(repr(spec)); expected one of " *
                            ":DEC, :DECJ, :DIVALIKE, :DIVALIKEJ, :BAYAREALIKE, :BAYAREALIKEJ"))
    m = Model()
    m.label = MODEL_LABELS[spec]

    spec === :DEC && return m
    if spec === :DECJ
        set_param!(m, "j"; kind = "free", init = 0.01, est = 0.01)
        return m
    end

    if spec === :DIVALIKE || spec === :DIVALIKEJ
        # No subset speciation; vicariance may split the range anywhere.
        set_param!(m, "s"; kind = "fixed", init = 0.0, est = 0.0)
        set_param!(m, "ysv"; kind = "2-j")
        set_param!(m, "ys"; kind = "ysv*1/2")
        set_param!(m, "y"; kind = "ysv*1/2")
        set_param!(m, "v"; kind = "ysv*1/2")
        set_param!(m, "mx01v"; kind = "fixed", init = 0.5, est = 0.5)
        if spec === :DIVALIKEJ
            set_param!(m, "j"; kind = "free", init = 0.0001, est = 0.0001,
                       lower = 0.00001, upper = 1.99999)
        end
        return m
    end

    if spec === :BAYAREALIKE || spec === :BAYAREALIKEJ
        # Sympatry only, and always an exact copy of the ancestral range.
        set_param!(m, "s"; kind = "fixed", init = 0.0, est = 0.0)
        set_param!(m, "v"; kind = "fixed", init = 0.0, est = 0.0)
        set_param!(m, "ysv"; kind = "1-j")
        set_param!(m, "ys"; kind = "ysv*1/1")
        set_param!(m, "y"; kind = "1-j")
        set_param!(m, "mx01y"; kind = "fixed", init = 0.9999, est = 0.9999)
        if spec === :BAYAREALIKEJ
            set_param!(m, "j"; kind = "free", init = 0.0001, est = 0.0001,
                       lower = 0.00001, upper = 0.99999)
        end
        return m
    end

    throw(ArgumentError("unhandled model $(repr(spec))"))
end

function Base.show(io::IO, m::Model)
    nfree = count(p -> p.kind == "free", m.params)
    print(io, "Model: ", m.label, ", ", length(m.params), " parameters, ", nfree, " free")
end

"""
    Base.getindex(m::Model, name) -> Float64

The current estimate of parameter `name`.
"""
Base.getindex(m::Model, name::AbstractString) = m.params[m.index[name]].est

# Bounds exist for the optimiser, so they are enforced on free parameters only.
# A fixed parameter may legitimately sit outside them: R fixes `s` at 0 in
# DIVALIKE and `j` at 0 in every non-+J model, while their lower bounds are 1e-5.
function Base.setindex!(m::Model, value::Real, name::AbstractString)
    p = m.params[m.index[name]]
    if p.kind == "free"
        value < p.lower - 1e-12 &&
            throw(ArgumentError("$(name)=$(value) is below its lower bound $(p.lower)"))
        value > p.upper + 1e-12 &&
            throw(ArgumentError("$(name)=$(value) is above its upper bound $(p.upper)"))
    end
    p.est = float(value)
    return m
end

Base.haskey(m::Model, name::AbstractString) = haskey(m.index, name)

param(m::Model, name::AbstractString) = m.params[m.index[name]]

"""
    set_param!(m, name; kind, init, lower, upper, est)

Change the metadata of one parameter — used to turn `j` from fixed to free for
DEC+J.  Any keyword left `nothing` keeps its current value.
"""
function set_param!(m::Model, name::AbstractString;
                    kind = nothing, init = nothing, lower = nothing,
                    upper = nothing, est = nothing)
    p = m.params[m.index[name]]
    kind === nothing || (p.kind = String(kind))
    init === nothing || (p.init = float(init))
    lower === nothing || (p.lower = float(lower))
    upper === nothing || (p.upper = float(upper))
    est === nothing || (p.est = float(est))
    return m
end

is_free(m::Model, name::AbstractString) = m.params[m.index[name]].kind == "free"

"""
    free_names(m::Model) -> Vector{String}

Names of the estimated parameters, in table order.
"""
free_names(m::Model) = [p.name for p in m.params if p.kind == "free"]

"""
    free_values(m::Model) -> Vector{Float64}

Current estimates of the free parameters, in the order [`free_names`](@ref)
returns them — the vector the optimiser works on.
"""
free_values(m::Model) = [p.est for p in m.params if p.kind == "free"]

init_params(m::Model) = [p.init for p in m.params if p.kind == "free"]
lower_bounds(m::Model) = [p.lower for p in m.params if p.kind == "free"]
upper_bounds(m::Model) = [p.upper for p in m.params if p.kind == "free"]

"""
    set_free_values!(m::Model, values)

Write a parameter vector from the optimiser back into the model, in
[`free_names`](@ref) order.
"""
function set_free_values!(m::Model, values::AbstractVector{<:Real})
    free = findall(p -> p.kind == "free", m.params)
    length(values) == length(free) ||
        throw(ArgumentError("expected $(length(free)) free parameters, got $(length(values))"))
    for (k, i) in enumerate(free)
        m.params[i].est = float(values[k])
    end
    return m
end

"""
    params_table(m::Model) -> Vector{NamedTuple}

A read-only snapshot of the table, one row per parameter — handy for printing
and for tests that compare against R's `params_table`.
"""
params_table(m::Model) = [
    (name = p.name, kind = p.kind, init = p.init, lower = p.lower,
     upper = p.upper, est = p.est, desc = p.desc)
    for p in m.params
]

# ---------------------------------------------------------------------------
# Linked parameters
# ---------------------------------------------------------------------------

"""
    update_linked!(m::Model) -> Model

Recompute every *derived* parameter from the free and fixed ones, in table
order — R's `calc_linked_params_BioGeoBEARS_model_object()`.

The derived kinds and their formulas are:

| kind        | value                |
|:------------|:---------------------|
| `"3-j"`     | `3 - j`              |
| `"2-j"`     | `2 - j`              |
| `"1-j"`     | `1 - j`              |
| `"ysv*2/3"` | `ysv * 2/3`          |
| `"ysv*1/2"` | `ysv * 1/2`          |
| `"ysv*1/3"` | `ysv * 1/3`          |
| `"ysv*1/1"` | `ysv`                |
| `"mx01"`    | `mx01`               |

Order matters: `ysv` is defined before `ys`, which is defined before `y`, `s`
and `v`, so a single forward pass is enough.  The table is laid out in R's
order for exactly that reason.

**Why this is separate from `qmatrix` / `cladogenesis_weights`:** R applies
these links inside the likelihood (`calc_loglike_for_optim`) but *not* inside
`get_Qmat_COOmat_from_BioGeoBEARS_run_object()`.  So a DEC+J model with `j=0.1`
uses `y=s=v=1` for the exported matrices and `y=s=v=2.9/3` for the likelihood.
Both behaviours are reproduced: the matrix builders take the table as-is, and
[`loglikelihood`](@ref) calls this function first.
"""
function update_linked!(m::Model)
    for p in m.params
        (p.kind == "free" || p.kind == "fixed") && continue
        p.est = _linked_value(m, p.kind)
    end
    return m
end

function _linked_value(m::Model, kind::AbstractString)
    kind == "mx01"    && return m["mx01"]
    kind == "3-j"     && return 3 - m["j"]
    kind == "2-j"     && return 2 - m["j"]
    kind == "1-j"     && return 1 - m["j"]
    kind == "ysv*2/3" && return m["ysv"] * 2 / 3
    kind == "ysv*1/2" && return m["ysv"] / 2
    kind == "ysv*1/3" && return m["ysv"] / 3
    kind == "ysv*1/1" && return m["ysv"]
    throw(ArgumentError("unknown linked parameter type $(repr(kind))"))
end

"""
    linked(m::Model) -> Model

A copy of `m` with [`update_linked!`](@ref) applied, leaving the caller's model
untouched.
"""
linked(m::Model) = update_linked!(deepcopy(m))
