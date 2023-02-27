import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv('ds.salaries.csv')

# Select only remote jobs
df = df[df['remote_ratio'] == 100]

# Select the relevant columns
X = df[['job_title', 'experience_level', 'company_location', 'company_size']]
y = df['salary_in_usd']

# Encode categorical features using OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = ohe.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# Define the models
lr = LinearRegression()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()

# Train the models
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Get the user input
job_title = input("Enter the job title: ")
experience_level = input("Enter the experience level: ")
company_location = input("Enter the company location: ")
company_size = input("Enter the company size: ")
model_type = input("Enter the model type (linear, decision, or random): ")

# Encode the user input
job_title_encoded = ohe.transform([[job_title, 0, 0, 0]])
experience_level_encoded = ohe.transform([[0, experience_level, 0, 0]])
company_location_encoded = ohe.transform([[0, 0, company_location, 0]])
company_size_encoded = ohe.transform([[0, 0, 0, company_size]])

# Concatenate the encoded features into a single array
user_input_encoded = np.concatenate([job_title_encoded, experience_level_encoded, company_location_encoded, company_size_encoded])

# Make the prediction
if model_type == 'linear':
    prediction = lr.predict(user_input_encoded)
elif model_type == 'decision':
    prediction = dt.predict(user_input_encoded)
elif model_type == 'random':
    prediction = rf.predict(user_input_encoded)
else:
    print("Invalid model type")
    prediction = None

# Print the prediction
if prediction is not None:
    print("The predicted salary is:", prediction[0])
