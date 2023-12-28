import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib

# Load the Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Function to make predictions
def make_prediction():
    input_values = [float(entry.get()) for entry in entries]
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    prediction = rf_model.predict(input_df)
    result_label.config(text=f"Predicted Benchmark: {prediction[0]:.2f}")

# Main window
root = tk.Tk()
root.title("CPU Benchmark Prediction")
root.geometry("600x400")  # Set window size

# Define feature columns
feature_columns = ['Process Size (nm)', 'TDP (W)', 'Cores', 'Freq (MHz)']  # Update as per your features

# Create input fields for features
entries = []
for feature in feature_columns:
    ttk.Label(root, text=feature).pack()
    entry = ttk.Entry(root)
    entry.pack()
    entries.append(entry)

# Prediction button
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.pack()

# Result label
result_label = ttk.Label(root, text="Predicted Benchmark: ")
result_label.pack()

# Run the application
root.mainloop()
