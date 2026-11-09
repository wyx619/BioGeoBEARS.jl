module BioGeoBEARS

# MKL must be loaded before any other package that touches BLAS/LAPACK, so that
# it finds the Intel OpenMP runtime first.  Loading it here means every
# `using BioGeoBEARS` switches the BLAS backend to MKL; callers that need
# OpenBLAS should load their BLAS-dependent packages first.
using MKL
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using DataFrames

using NewickTree
using NLopt

# core: data model and model definition
include("core/tree.jl")     # areas.jl dispatches on BGTree, so the tree comes first
include("core/areas.jl")
include("core/model.jl")
include("core/qmatrix.jl")
include("core/cladogenesis.jl")
include("core/fossils.jl")

# likelihood: the inference core
include("likelihood/expm.jl")
include("likelihood/downpass.jl")
include("likelihood/uppass.jl")
include("likelihood/stratified.jl")
include("likelihood/detection.jl")

# inference: optimisation and results
include("inference/results.jl")
include("inference/optimize.jl")
include("inference/multiple_trees.jl")
include("inference/model_selection.jl")

# io
include("io/newick.jl")
include("io/phylip.jl")
include("io/config.jl")

# simulate
include("simulate/rng_r.jl")
include("simulate/bsm.jl")
include("simulate/forward.jl")
include("simulate/events.jl")

# plot
include("plot/recipes.jl")

# public API
export BGTree, read_tree
export ntips, nnodes, root_node, is_tip, is_internal, is_root
export parent_node, left_child, right_child, branch_length, root_edge, pruning_order
export node_depths, tree_height, node_times_bp, prt_table, check_tree

export StateSpace, TipRanges, nareas, nstates
export numstates_from_numareas, state_text, states_list_text, state_index
export tip_range, has_ambiguity, tip_condlikes

export read_lagrange_phylip, write_lagrange_phylip, read_area_matrix
export read_times, read_matrix_with_names, read_areas_allowed, read_areas_adjacency
export prune_states

export DetectionCounts, read_detections, read_controls
export calc_obs_like, Pdata_given_rangerow, tiplikes_wDetectionModel
export FossilRecord, read_fossils, fossil_records
export add_hook, possible_fossil_branches, add_fossil_randomly, add_fossils_randomly,
       add_fossil_ranges

export Model, Param, param, params_table, set_param!, is_free
export free_names, free_values, init_params, lower_bounds, upper_bounds
export set_free_values!, update_linked!, linked, model_name
export qmatrix, row_sums
export CladogenesisWeights, cladogenesis_weights
export relative_probabilities_of_subsets, relative_probabilities_of_vicariants
export maxent_probs

export transition_matrix, apply_transition, apply_transition_transpose, r_sum
export DownpassResult, downpass, loglikelihood
export Stratification, tree_pieces, stratified_bsm_inputs, StratifiedBSMInputs
export stratified_branch_top_probs, next_piece_seed
export stratified_loglikelihood, stratified_condlikes, stratified_ancestral_probs
export AncestralProbs, ancestral_probs

export BGBFit, fit_model, nparams, paramvalue, ancprobs, fit_summary
export fit_multiple_trees, summarize_tree_ancestry
export aic, aicc, akaike_weights, summary_table

export RRng, r_runif, r_rexp, r_sample_prob, r_sample_int, r_unif_index, r_rbits
export AnaEvent, BSMResult, stochastic_map, stochastic_map_branch, run_bsm
export simulate_biogeog_history
export stochastic_map_stratified
export CladoEventRow, AnaEventRow, BSMTables, bsm_tables, simulate_source_areas!
export uniquify_clado_events, count_clado_dispersal_events
export count_ana_dispersal_events, count_ana_clado_events
export calc_bsm_mean_node_states, print_counts_lists_to_file

end
