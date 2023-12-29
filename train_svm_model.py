import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
gpu_df = pd.read_csv('GPU_DATASET.csv')

# Preprocessing
# Assuming the dataset has columns that you want to check for outliers
# Update the column names as per your GPU dataset
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
X = df_final.drop(['2D_Bench', '3D_Bench'], axis=1)  # Replace with your target columns
y_2d = df_final['2D_Bench']
y_3d = df_final['3D_Bench']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset for both 2D and 3D benchmarks
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_scaled, y_2d, test_size=0.2, random_state=42)
X_train_3d, X_test_3d, y_train_3d, y_test_3d = train_test_split(X_scaled, y_3d, test_size=0.2, random_state=42)

# Training SVM for 2D benchmark
svm_model_2d = SVR()
svm_model_2d.fit(X_train_2d, y_train_2d)
joblib.dump(svm_model_2d, 'svm_model_2d.pkl')

# Training SVM for 3D benchmark
svm_model_3d = SVR()
svm_model_3d.fit(X_train_3d, y_train_3d)
joblib.dump(svm_model_3d, 'svm_model_3d.pkl')

# Save the scaler
joblib.dump(scaler, 'gpu_scaler.pkl')
