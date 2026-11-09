# BioGeoBEARS.jl

BioGeoBEARS.jl implements model-based historical biogeography in Julia. Given a
phylogenetic tree and geographic ranges at its tips, it estimates model
parameters, reconstructs ancestral ranges, and samples biogeographic histories.
The implementation is based on BioGeoBEARS 1.1.4 for R.

## Features

- DEC, DIVALIKE, and BAYAREALIKE models, with or without founder-event speciation.
- Maximum-likelihood fitting, ancestral-state probabilities, AIC and AICc.
- Time-stratified dispersal and geographic constraints, including fossil tips.
- Biogeographic stochastic mapping, source-area sampling, and event summaries.
- Multi-tree fitting and ancestral-probability summaries.
- Forward simulation of node states on a fixed tree.
- Optional global search followed by local optimization.

Selected likelihoods, ancestral probabilities, fitted models, and stochastic
maps have been validated against the R implementation. Results can differ with
input data, optimization settings, and numerical precision.

## Requirements

Julia 1.10 or later is required. From the repository root, install the project
dependencies with:

```julia
using Pkg
Pkg.add(url = "https://github.com/wyx619/BioGeoBEARS.jl")
```

The package loads MKL.jl and changes the BLAS backend for the Julia process.
Load BioGeoBEARS before other packages that initialize BLAS to avoid potential
OpenMP runtime conflicts.

## Quick Start

The following example uses the test data included in the repository:

```julia
using BioGeoBEARS

data = joinpath("test", "data", "psychotria_m0strat")
tree = read_tree(joinpath(data, "Psychotria_5.2.newick"))
tips = read_lagrange_phylip(joinpath(data, "Psychotria_geog.data"))
states = StateSpace(tips.areanames; max_range_size = 4)
tiplikes = tip_condlikes(tree, tips; max_range_size = 4)

models = (:DEC, :DECJ, :DIVALIKE, :DIVALIKEJ, :BAYAREALIKE, :BAYAREALIKEJ)
fits = [fit_model(tree, states, tiplikes, Model(model); calc_ancprobs = true)
        for model in models]
comparison = summary_table(fits; sort_by = :AICc)
```

## Time-Stratified Mapping

```julia
strata = Stratification(
    timesfn = joinpath(data, "timeperiods.txt"),
    areas_allowed_fn = joinpath(data, "areas_allowed_all1s.txt"),
)
fit = fit_model(tree, states, tiplikes, Model(:DEC), strata; calc_ancprobs = true)
maps, rng = run_bsm(tree, states, fit.model, strata, tiplikes;
                    seedval = 12344, n_maps = 2, return_rng = true)
tables = [bsm_tables(map, tree) for map in maps]
simulate_source_areas!(tables, states, rng)
counts = count_ana_clado_events(tables, states)
```

The returned RNG continues from the last map and can be used for source-area
sampling. To count events in an age interval, pass `timeperiod = (1.0, 2.0)` to
`count_ana_clado_events`; the interval includes its lower bound and excludes its
upper bound.

## Multiple Trees and Simulation

Use `fit_multiple_trees(trees, states, tips, model; calc_ancprobs = true)` to fit
the same model across trees. `summarize_tree_ancestry(master, trees, fits)`
summarizes branch-top and branch-bottom probabilities by descendant clade. Use
`matching = :r` to apply the R implementation's clade-matching rule. All fits
must use the same ordered state space.

`simulate_biogeog_history(tree, states, model, root_state; rng = RRng(1))`
simulates node and daughter states on a fixed tree. State indices are 1-based;
the result does not contain individual events along branches.



## License

GPL-3.0.
