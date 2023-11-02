import matplotlib.pyplot as plt
import pandas as pd


#Load the dataset: This code section loads the dataset from a given URL using the pandas library. 
# The dataset contains information about vehicles.
# Plot CYLINDER vs CO2EMISSIONS: This code section creates a scatter plot to visualize the relationship 
# between the "CYLINDERS" feature (number of cylinders in a vehicle) 
# and the "CO2EMISSIONS" feature (carbon dioxide emissions). 
# The scatter plot helps us understand how these two variables are related.

# Load the dataset
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(path)

# Plot CYLINDER vs CO2EMISSIONS
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue')
plt.xlabel("CYLINDER")
plt.ylabel("CO2EMISSIONS")
plt.show()

# Split the data into training and testing sets
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


# Load the dataset (again): This code section reloads the dataset to ensure that it's available 
# for the subsequent steps.
# Split the data into training and testing sets: 
# The dataset is randomly split into two sets: a training set 
# (80% of the data) and a testing set (20% of the data). 
# This is done to train the model on one subset and evaluate its performance on another. 
# The msk variable is used to create a boolean mask for data splitting.

# Create a linear regression model: An instance of a linear regression model 
# is created using scikit-learn's LinearRegression class. 
# This model will be used to find a linear relationship between "FUELCONSUMPTION_COMB" and "CO2EMISSIONS."
# Train the model on FUELCONSUMPTION_COMB: The model is trained using the training data. 
# It learns the relationship between "FUELCONSUMPTION_COMB" (input) and "CO2EMISSIONS" (output).

# Make predictions using the model: The trained model is used to make predictions on the testing data. 
# It predicts "CO2EMISSIONS" based on the "FUELCONSUMPTION_COMB" values in the test set.

# Calculate Mean Absolute Error (MAE): The Mean Absolute Error (MAE) is calculated to 
# measure how well the model's predictions match the actual values in the test set. 
# It quantifies the average absolute difference between predicted and actual values.

# Print MAE: The MAE value is printed to the console, indicating the average error of the model's predictions.

# Create a linear regression model
regr = linear_model.LinearRegression()

# Train the model on FUELCONSUMPTION_COMB
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Make predictions using the model
test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(test_y, test_y_)
print("Mean Absolute Error: %.2f" % mae)

# Performs simple linear regression to predict CO2 emissions based on fuel consumption. 
# It demonstrates data loading, splitting, model creation, training, prediction, and evaluation. 
# The scatter plot earlier helps visualize the relationship between cylinders and emissions, 
# while the MAE quantifies the model's accuracy.