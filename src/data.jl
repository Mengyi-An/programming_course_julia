using CSV
using DataFrames
using CategoricalArrays

"""
prepare_data(path::String; current_year::Int=2025)

# Purpose
This function loads a car sales dataset, cleans the data, and returns a `DataFrame`.It performs the following steps:
-1. load a CSV file.
-2. deal with any missing values by dropping rows.
-3. transform the categorical columns to the `CategoricalArray` type.
-4. calculate a new `Car_age` column based on the `current_year`.

# Arguments
-`path::String`: The CSV dataset file path.
-`current_year::Int=2025`: used to calculate the car age.Default value is 2025.

# Examples
-1. Usage with default current_year:
dataset = prepare_data("car_sales_data.csv")

-2. Specify a different current_year:
dataset = prepare_data("car_sales_data.csv"; current_year=2024)
"""
function prepare_data(path::String; current_year::Int = 2025)
    # load dataset
    dataset = CSV.read(path, DataFrame)

    # deal with the missing values
    dropmissing!(dataset)

    # transform the categorical to CategoricalArray
    dataset.Manufacturer = categorical(dataset.Manufacturer)
    dataset.Model = categorical(dataset.Model)
    dataset.Fuel_type = categorical(dataset.Fuel_type)

    # calculate the car age and create a new column in original dataset
    dataset.Car_age = current_year .- dataset.Year_of_manufacture

    return dataset
end