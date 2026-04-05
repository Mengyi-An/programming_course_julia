using CarsDataAnalysis

# 1. prepare the dataset
dataset = CarsDataAnalysis.prepare_data("car_sales_data.csv")

# 2. calcutate the correlation matrix for different columns
num_cols = [:Price, :Mileage, :Engine_size, :Year_of_manufacture, :Car_age]
cor_matrix = CarsDataAnalysis.analysis_cols(dataset, num_cols)

# 2.1 plot heatmap
CarsDataAnalysis.plot_correlation_heatmap(cor_matrix, num_cols)


# 3.calculate the average price of each car age, grouped by manufacturer 
avg_price_age_data = CarsDataAnalysis.calculate_avg_price_by_age(dataset)

# 3.1 plot car's age vs average price, grouped by manufacturer
CarsDataAnalysis.plot_price_age_trend(avg_price_age_data)

# 4. through random forest regression to predict car‘s price
features = [:Car_age, :Mileage, :Engine_size, :Manufacturer, :Model, :Fuel_type]
target = :Price
pred_test, y_test = CarsDataAnalysis.random_forest_price_predict(dataset, features=features, target=target)

# 4.1 evaluate the model performance based on random forest regression
r2, rmse, mae = CarsDataAnalysis.evaluate_model_performance(y_test, pred_test)
println("\nRandom Forest Model Performance:")
println("R-squared:                ", round(r2, digits=4))
println("Root Mean Squared Error:  ", round(rmse, digits=2))
println("Mean Absolute Error:      ", round(mae, digits=2))

# 4.2 plot predicted price vs actual price
CarsDataAnalysis.plot_prediction_scatter(y_test, pred_test)

