import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
gpu_df = pd.read_csv('GPU_DATASET.csv')

# Preprocessing
numerical_cols = ['Process Size (nm)', 'TDP (W)','Die Size (mm^2)', 'Transistors (million)', 'Freq (MHz)'  ]  # Replace with actual numerical columns
df_numerical = gpu_df[numerical_cols]

# Removing outliers
Q1 = df_numerical.quantile(0.25)
Q3 = df_numerical.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df_numerical < (Q1 - 1.5 * IQR)) | (df_numerical > (Q3 + 1.5 * IQR))
df_cleaned = gpu_df[~outlier_condition.any(axis=1)].copy()

df_cleaned.drop(['Product', 'Foundry', 'Vendor', 'Release Date', 'Type'], axis=1, inplace=True)

# Removing rows with NaN values
df_final = df_cleaned.dropna()

# Separating the features and the targets
X = df_final.drop(['2D_Bench', '3D_Bench'], axis=1)
y_2d = df_final['2D_Bench']
y_3d = df_final['3D_Bench']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

# Function to perform grid search and train SVR model
def train_svr(X, y, filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Train SVM model with the best parameters found
    best_params = grid_search.best_params_
    svm_model = SVR(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
    svm_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(svm_model, filename)

# Train and save SVM model for 2D benchmarks
train_svr(X_scaled, y_2d, 'svm_model_2d.pkl')

# Train and save SVM model for 3D benchmarks
train_svr(X_scaled, y_3d, 'svm_model_3d.pkl')

# Save the scaler
joblib.dump(scaler, 'gpu_scaler.pkl')