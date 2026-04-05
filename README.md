# CarsDataAnalysis

## Project Introduction
This project analysis second-hand car market data. The main goals include:
1. Performing correlation analysis to identify relationships between different features(include independent and dependent variables).
2. Analysis how car prices change with car age, grouped by manufacturers.
3. Fitting a Random Forest model to predict car's prices and evaluating the accuracy of the model.

## How to Implement the Project
1. Load Pkg module
```julia
using Pkg
```

2. Activate the project environment
```julia
Pkg.activate(".")
```

3. Install dependencies (execute once)
```julia
Pkg.instantiate()
```

4. Implement the main script
```julia
include("extra/main.jl")
```

## Output
This project will output model performance data,and show all analysis charts.

### Performance:
The R^2, RMSE, MAE in the Random Forest model.

### Visualizations:
1. Correlation Heatmap
2. Price vs. Car Age Trend Plot
3. Price Prediction Scatter Plot