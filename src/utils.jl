using DataFrames
using Statistics


"""
analysis_cols(dataset::DataFrame, cols::Vector{Symbol})

# Purpose
This function computes the correlation matrix among the listed columns.The result `cor_matrix` is typically used as the input data for creating a correlation heatmap for visualization.

# Arguments
-`df::DataFrame`: The dataset to be used.
-`cols::Vector{Symbol}`: A vector of column names for which the `cor_matrix` will be computed.

# Examples
-1. Usage with the entire dataset and a set of numerical columns:
analysis_cols(dataset, [:Price, :Mileage, :Engine_size, :Year_of_manufacture, :Car_age])

-2. Usage with selected columns:
analysis_cols(dataset, [:Car_age, :Engine_size, :Price])
"""
function analysis_cols(df::DataFrame, cols::Vector{Symbol})
    # initialize an empty correlation matrix
    cor_matrix = Matrix{Float64}(undef, length(cols), length(cols))
    # iterate through each pair of columns to compute the correlation
    for i in 1:length(cols)
        for j in 1:length(cols)
            cor_matrix[i, j] = cor(df[!, cols[i]], df[!, cols[j]])
        end
    end
    return cor_matrix
end

"""
calculate_avg_price_by_age(df::DataFrame)

# Purpose
This function calculates the average price for each car age, and grouped by manufacturer.It performs the following steps:
-1. select the car's manufacturers.
-2. calculate the average price for each car age, separately for each manufacturer.
-3. sort the results by car age in ascending order.The result avg_price_age is a dictionary that used as the input data for plotting the price change trend across car ages for different manufactures.

# Arguments
-`df::DataFrame`: The dataset to be used.

# Examples
-1. Calculate the average price by age for all manufacturers:
avg_prices = calculate_avg_price_by_age(dataset)

-2. Calculate the average price by age for selected manufacturer:
toyota_prices = calculate_avg_price_by_age(dataset)["Toyota"]
"""
function calculate_avg_price_by_age(df::DataFrame)
    # group the dataset by manufacturer
    manufacturer_counts = combine(groupby(df, :Manufacturer), nrow => :count)
    sorted_counts = sort(manufacturer_counts, :count, rev=true)

    # calculate the average price for each car age, separately for each manufacturer
    avg_price_age = Dict()
    for manufacturer_name in sorted_counts.Manufacturer
        current_manufacturer = filter(row -> row.Manufacturer == manufacturer_name, df)
        grouped = combine(groupby(current_manufacturer, :Car_age), :Price => mean => :avg_price)
        grouped = sort(grouped, :Car_age)
        avg_price_age[manufacturer_name] = grouped
    end
    
    return avg_price_age
end