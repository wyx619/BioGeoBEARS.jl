# Run with: julia --project=. test/targeted_bsm.jl
using BioGeoBEARS, Test, CSV, DataFrames, LinearAlgebra, Statistics
const DATA = joinpath(@__DIR__, "data")
const BASELINES = joinpath(@__DIR__, "baselines")

include("bsm_unstratified_tests.jl")
include("bsm_stratified_tests.jl")
include("fossil_models.jl")
