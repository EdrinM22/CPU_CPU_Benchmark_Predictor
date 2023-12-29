import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import pandas as pd

# Function to load models and scaler
def load_models_and_scaler():
    global svm_model_2d, svm_model_3d, scaler
    svm_model_2d = joblib.load('svm_model_2d.pkl')
    svm_model_3d = joblib.load('svm_model_3d.pkl')
    scaler = joblib.load('gpu_scaler.pkl')

# Function to make prediction
def make_prediction():
    input_values = [float(entry.get()) for entry in entries]
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    input_scaled = scaler.transform(input_df)

    if benchmark_var.get() == '2D Benchmark':
        prediction = svm_model_2d.predict(input_scaled)
    else:
        prediction = svm_model_3d.predict(input_scaled)

    result_label.config(text=f"Predicted Score: {prediction[0]:.2f}")

# Initialize Tkinter window
root = tk.Tk()
root.title("GPU Benchmark Prediction")
root.geometry("600x400")
root.iconbitmap("GPU.ico")

# Load models and scaler
load_models_and_scaler()

# Dropdown to select 2D or 3D benchmark
benchmark_var = tk.StringVar()
benchmark_dropdown = ttk.OptionMenu(root, benchmark_var, "2D Benchmark", "2D Benchmark", "3D Benchmark")
benchmark_dropdown.pack()

# Input fields for features
feature_columns = ['Process Size (nm)', 'TDP (W)','Die Size (mm^2)', 'Transistors (million)', 'Freq (MHz)'  ]  # Replace with actual feature names from your dataset
entries = []
for feature in feature_columns:
    ttk.Label(root, text=feature).pack()
    entry = ttk.Entry(root)
    entry.pack()
    entries.append(entry)

# Button for predictions
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.pack()

# Label for showing predictions
result_label = ttk.Label(root, text="Predicted Score: ")
result_label.pack()

# Run the application
root.mainloop()
