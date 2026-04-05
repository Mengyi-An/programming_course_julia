module CarsDataAnalysis

# 1. Import all dependencies
using CSV
using DataFrames
using CategoricalArrays
using Random
using DecisionTree
using GLMakie
using CairoMakie
using Statistics

# 2. Use include statements to load other files from the src
include("data.jl")
include("models.jl")
include("plots.jl")
include("utils.jl")

# 3. Use export to export all public functions
export prepare_data,
       random_forest_price_predict,
       evaluate_model_performance,
       analysis_cols,
       calculate_avg_price_by_age,
       plot_prediction_scatter,
       plot_correlation_heatmap,
       plot_price_age_trend

end # module CarsDataAnalysis
