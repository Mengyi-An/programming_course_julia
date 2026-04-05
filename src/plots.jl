using GLMakie
using CairoMakie
using Statistics
using DataFrames

"""
plot_prediction_scatter(y_true::Vector, y_pred::Vector)

# Purpose
-1. The scatter points represent the predicted prices (`y_pred`) against the actual prices (`y_true`).
-2. A red diagonal line (Y=X) as a reference.Scatter points much closer to the red line,more accurate prediction that a model has.
-3. The input vectors `y_true` and `y_pred` are generated from the `random_forest_price_predict` function.

# Arguments
-`y_true::Vector`: The vector of actual car's prices.
-`y_pred::Vector`: The vector of predicted car's prices.

# Examples
-1. Usage with Random Forest output:
plot_prediction_scatter(y_test, pred_test)

-2. Quick test with sample data:
y_true = collect(100:100:1000) 
y_pred = y_true .* 1.05  # A slightly over-predicted set
plot_prediction_scatter(y_true, y_pred)
"""
function plot_prediction_scatter(y_true::Vector, y_pred::Vector)
    # set up the figure and axis
    f = Figure(size = (1200, 800))
    ax = Axis(f[1, 1],
        xlabel = "Actual Price",
        ylabel = "Predicted Price",
        title = "Random Forest Price Prediction",
        titlesize = 24,
        xlabelsize = 20,
        ylabelsize = 20
    )
    # plot the scatter points and the reference line
    scatter!(ax,
        y_true,
        y_pred,
        color = "#6495ED"
    )
    lines!(ax,
        minimum(y_true):maximum(y_true),
        minimum(y_true):maximum(y_true),
        color = :red,
        linewidth = 3
    )
    display(f)
end


"""
plot_correlation_heatmap(cor_matrix::Matrix, cols::Vector{Symbol})

# Purpose
-1. The heatmap plot represents the pairwise correlation coefficients between different variables.
-2. The color intensity indicates the strength and direction of the correlation.
-3. The input matrix `cor_matrix` is generated from `analysis_cols` function.

# Arguments
-`cor_matrix::Matrix`: The square matrix of pairwise correlation coefficients.
-`cols::Vector{Symbol}`: A vector of column names corresponding to the rows and columns of the input matrix `cor_matrix`.

# Examples
-1. Usage with the entire correlation matrix:
num_cols = [:Price, :Mileage, :Engine_size, :Year_of_manufacture, :Car_age]
cor_matrix = analysis_cols(dataset, num_cols)
plot_correlation_heatmap(cor_matrix, num_cols)

-2. Usage with the selected columns of correlation matrix:
sub_cols = [:Price, :Car_age]
sub_matrix = analysis_cols(dataset, sub_cols)
plot_correlation_heatmap(sub_matrix, sub_cols)
"""
function plot_correlation_heatmap(cor_matrix::Matrix, cols::Vector{Symbol})
    # set up the figure and axis
    f = Figure(size = (1200, 800))
    ax = Axis(f[1, 1],
        title = "Correlation Heatmap",
        titlesize = 24,
        xlabelsize = 20,
        ylabelsize = 20,
        xticks = (1:length(cols), string.(cols)),
        yticks = (1:length(cols), string.(cols)),
    )
    # plot the heatmap
    hm = heatmap!(ax,
        cor_matrix,
        colormap = :coolwarm
    )
    Colorbar(f[1, 2],
        hm,
        label = "Correlation"
    )
    display(f)
end


"""
plot_price_age_trend(avg_price_age::Dict)

# Purpose
-1. This function iterates through the input dictionary `avg_price_age` to plot a separate line for each manufacturer.
-2. This plot directly interprets how car's prices change with car's age for different manufacturers.
-3. The input dictionary `avg_price_age` is generated from `calculate_avg_price_by_age` function,which calculates the average price for each car age, grouped by manufacturer.

# Arguments
-`avg_price_age::Dict`: A dictionary containing average prices grouped by manufacturer and car age.

# Examples
-1. Usage with all manufacturers data:
plot_price_age_trend(avg_price_age)

-2. Usage with a selected manufacutrer data:
selected_data = Dict("Toyota" => avg_price_age["Toyota"], "Honda" => avg_price_age["Honda"])
plot_price_age_trend(selected_data)
"""
function plot_price_age_trend(avg_price_age::Dict)
    # set up the figure and axis
    f = Figure(size = (1200, 800))
    max_price = maximum([maximum(price.avg_price) for price in values(avg_price_age)])
    ax = Axis(f[1, 1],
        title = "Average Price vs Car Age by Manufacturer",
        xlabel = "Car Age (Years)",
        ylabel = "Average Price",
        titlesize = 24,
        xlabelsize = 20,
        ylabelsize = 20,
        yticks = 0:5000:max_price
    )
    # iterate through the dictionary and plot each manufacturer's trend line
    for (manufacturer, avg_number) in avg_price_age
        lines!(ax,
            avg_number.Car_age,
            avg_number.avg_price,
            label = string(manufacturer),
            linewidth=3
        )
    end
    # add legend and grid
    axislegend(ax, position=:rt)
    ax.xgridvisible[] = true
    ax.ygridvisible[] = true
    display(f)
end