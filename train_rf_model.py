import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('CPU_DATASET.csv')

# Preprocessing
# Removing outliers
numerical_cols = ['Process Size (nm)', 'TDP (W)', 'Cores', 'Freq (MHz)', 'Benchmark']
df_numerical = df[numerical_cols]
Q1 = df_numerical.quantile(0.25)
Q3 = df_numerical.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df_numerical < (Q1 - 1.5 * IQR)) | (df_numerical > (Q3 + 1.5 * IQR))
df_cleaned = df[~outlier_condition.any(axis=1)].copy()

# Final DataFrame - Remove 'Product', 'Foundry', 'Vendor', 'Release Date', and 'Type'
df_final = df_cleaned.drop(['Product', 'Foundry', 'Vendor', 'Release Date', 'Type'], axis=1)

# Removing rows with NaN values
df_final.dropna(inplace=True)

# Train-test split
features = df_final.drop('Benchmark', axis=1)
target = df_final['Benchmark']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'random_forest_model.pkl')
