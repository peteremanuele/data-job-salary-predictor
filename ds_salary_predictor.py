import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# Encode categorical features
le = LabelEncoder()
X = X.apply(le.fit_transform)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
job_title = le.transform([job_title])[0]
experience_level = le.transform([experience_level])[0]
company_location = le.transform([company_location])[0]
company_size = le.transform([company_size])[0]

# Make the prediction
if model_type == 'linear':
    prediction = lr.predict([[job_title, experience_level, company_location, company_size]])
elif model_type == 'decision':
    prediction = dt.predict([[job_title, experience_level, company_location, company_size]])
elif model_type == 'random':
    prediction = rf.predict([[job_title, experience_level, company_location, company_size]])
else:
    print("Invalid model type")
    prediction = None

# Print the prediction
if prediction is not None:
    print("The predicted salary is:", prediction[0])
