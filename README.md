**Documentation for Laptop Price Prediction using Machine Learning**

This documentation provides an overview of the code that performs laptop price prediction using machine learning techniques. The code reads laptop data from a CSV file, preprocesses the data, builds a predictive model, and evaluates its performance.

**1. Importing Required Libraries**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
```

**2. Loading and Preprocessing Data**
The code reads laptop data from a CSV file and performs data preprocessing to prepare it for machine learning.
```python
df = pd.read_csv("laptop_price.csv", encoding="latin-1")
# Various data preprocessing steps such as dropping columns, creating dummy variables, and transforming features.
```

**3. Data Visualization**
The code visualizes the correlation between features using a heatmap.
```python
plt.figure(figsize=(18, 15))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
```

**4. Feature Selection**
The code identifies the features with the highest correlation to the target variable (Price_euros).
```python
target_correlations = df.corr()['Price_euros'].apply(abs).sort_values()
selected_features = target_correlations[-21:].index
```

**5. Feature Scaling and Data Splitting**
The features are scaled using StandardScaler, and the dataset is split into training and testing sets.
```python
X, y = limited_df.drop("Price_euros", axis=1), limited_df["Price_euros"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**6. Building and Evaluating Model**
The code builds a Random Forest Regressor model and evaluates its performance on the test data.
```python
forest = RandomForestRegressor()
forest.fit(X_train_scaled, y_train)
accuracy = forest.score(X_test_scaled, y_test)
```

**7. Prediction and Visualization**
The code makes predictions using the trained model and visualizes the predicted vs. actual prices.
```python
y_pred = forest.predict(X_test_scaled)
plt.figure(figsize=(12, 8))
plt.scatter(y_pred, y_test)
plt.plot(range(0, 6000), range(0, 6000), c="red")
```

**8. Single Prediction**
The code demonstrates how to predict the price of a new laptop using the trained model.
```python
X_new_scaled = scaler.transform([X_test.iloc[0]])
predicted_price = forest.predict(X_new_scaled)
actual_price = y_test.iloc[0]
```

**Running the Code:**
1. Make sure you have the required libraries installed: pandas, seaborn, matplotlib, and scikit-learn.
2. Download the CSV file containing the laptop data (The data used in this code is the following: https://www.kaggle.com/datasets/muhammetvarl/laptop-price).
3. Save the CSV file in the same directory as your code.
4. Copy and paste the provided code into a Jupyter Notebook.
5. Execute the cells in the notebook to see the results.

This code processes laptop data, preprocesses it, builds a predictive model, evaluates its performance, and demonstrates price prediction for a new laptop.
