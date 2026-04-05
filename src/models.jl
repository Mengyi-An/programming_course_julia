using Random
using DecisionTree
using Statistics
using DataFrames
using CategoricalArrays


"""
random_forest_price_predict(df::DataFrame; features::Vector{Symbol}, target::Symbol, test_ratio::Float64 = 0.2, n_trees::Int = 100, max_depth::Int = 10, min_samples_leaf::Int = 1)

# Purpose
This function fits a Random Forest model to predict car's price.It performs the following steps:
-1. convert categorical features to numerical values.
-2. split the dataset into training set and test set. 
-2. fit a random forest model to predict the car's prices.
-3. return the predicted price and the actual price.

# Arguments
- `df::DataFrame`: The dataset to be used. It must contain the features(independent variables) and the target(dependent) variable.
- `features::Vector{Symbol}`: A list of column names to be used as features.
- `target::Symbol`: The name of the target variable column.
- `test_ratio::Float64=0.2`: The ratio of the dataset to be used as the test set.Default ratio is 0.2.
- `n_trees::Int=100`: The number of trees in the random forest.Default value is 100.
- `max_depth::Int=10`: The maximum depth of each tree.Default value is 10.
- `min_samples_leaf::Int=1`: The minimum number of samples required to be at a leaf node.Default value is 1.

# Examples
-1. Usage with the entire dataset that contains all features:
random_forest_price_predict(dataset; features = [:Car_age, :Mileage, :Engine_size, :Manufacturer, :Model, :Fuel_type], target = :Price)

-2. Usage with the selected features and a sepcified test ratio:
random_forest_price_predict(dataset; features = [:Car_age, :Mileage], target = :Price, test_ratio = 0.3)
"""
#random forest function to predict car's prices
function random_forest_price_predict(df::DataFrame;
                                     features::Vector{Symbol},
                                     target::Symbol,
                                     test_ratio::Float64 = 0.2,
                                     n_trees::Int = 100,
                                     max_depth::Int = 10,
                                     min_samples_leaf::Int = 1)

    # make a copy of the dataset,avoiding modify the original dataset
    df_copy = deepcopy(df)

    # iterate through all features and ensure they are numerical types
    processed_features = []
    for f in features
        if eltype(df_copy[!, f]) <: CategoricalValue
            # convert categorical variables to numerical and add to the new list
            push!(processed_features, float.(levelcode.(df_copy[!, f])))
        else
            # if already numerical, add directly to the new list
            push!(processed_features, df_copy[!, f])
        end
    end

    # ensure the vectors in processed_features have the same length
    X = hcat(processed_features...)
    y = convert(Vector{Float64}, df_copy[!, target])


    # split the dataset into training set and test set
    n = size(X, 1)
    n_test = round(Int, n * test_ratio)
    indices = shuffle(1:n)
    test_idx = indices[1:n_test]
    train_idx = indices[(n_test+1):end]

    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    # fit the random forest model
    model = DecisionTree.RandomForestRegressor(n_trees=n_trees,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf)
    DecisionTree.fit!(model, X_train, y_train)

    # predict on the test set
    pred_test = DecisionTree.predict(model, X_test)

    # return the predicted price and the actual price
    return pred_test, y_test
end



"""
evaluate_model_performance(y_true::Vector{Float64}, y_pred::Vector{Float64})

# Purpose
This function evaluates the performance and accuracy of Random Forest.It calculates three common metrics,including:
-1. R²: the coefficient of determination,indicating the propotation of variation in y that can be explained by X.
-2. RMSE: root mean squared error,which measures the average magnitude of the errors between predicted and actual values.
-3. MAE: mean absolute error,which measures the average magnitude of the errors without considering their direction.

# Arguments
-`y_true::Vector{Float64}`: A vector of the true car's prices.
-`y_pred::Vector{Float64}`: A vector of the predicted car's prices.

# Examples
-1. Usage with two vectors that obtained from random_forest_price_predict function and output all three metrics:
r2, rmse, mae = evaluate_model_performance(y_test, pred_test)

-2. Only output a subset of the metrics:
r2, rmse, _ = evaluate_model_performance(y_test, pred_test)
"""
# Function to evaluate model performance
function evaluate_model_performance(y_true::Vector{Float64}, y_pred::Vector{Float64})
    # calculate root mean squared error (RMSE)
    rmse = sqrt(mean((y_true .- y_pred).^2))

    # calculate mean absolute error (MAE)
    mae = mean(abs.(y_true .- y_pred))

    # calculate R-squared (R²)
    ss_total = sum((y_true .- mean(y_true)).^2)
    ss_residual = sum((y_true .- y_pred).^2)
    r2 = 1 - (ss_residual / ss_total)

    return r2, rmse, mae
end