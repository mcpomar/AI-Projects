#You can download the dataset here (https://www.aihr.com/wp-content/uploads/2019/10/turnover-data-set.csv) from Edward’s Dropbox.

# Import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("data.csv")  # Replace "data.csv" with your actual dataset filename

# Select the relevant columns for prediction
columns = ['experience', 'event', 'gender', 'age', 'industry', 'profession', 'traffic', 'coach', 'head_gender', 'greywage', 'way', 'extraversion', 'independ', 'selfcontrol', 'anxiety', 'novator']
data = dataset[columns]

# Split the dataset into input features (X) and target variable (y)
X = data.drop("event", axis=1)
y = data["event"]

# Convert categorical variables to numeric using one-hot encoding
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest classifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)