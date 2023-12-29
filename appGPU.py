import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import pandas as pd

# Load models and scaler
def load_models_and_scaler():
    global svm_model_2d, svm_model_3d, lasso_model_2d, lasso_model_3d, scaler
    svm_model_2d = joblib.load('svm_model_2d.pkl')
    svm_model_3d = joblib.load('svm_model_3d.pkl')
    lasso_model_2d = joblib.load('lasso_model_2d.pkl')
    lasso_model_3d = joblib.load('lasso_model_3d.pkl')
    scaler = joblib.load('gpu_scaler.pkl')

def make_prediction():
    # Get input values
    input_values = [float(entry.get()) for entry in entries]
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    input_scaled = scaler.transform(input_df)

    model_type = model_var.get()
    benchmark_type = benchmark_var.get()

    if model_type == 'SVM':
        model_2d, model_3d = svm_model_2d, svm_model_3d
    elif model_type == 'Lasso':
        model_2d, model_3d = lasso_model_2d, lasso_model_3d

    prediction = model_2d.predict(input_scaled) if benchmark_type == '2D Benchmark' else model_3d.predict(input_scaled)
    result_label.config(text=f"Predicted Score: {prediction[0]:.2f}")

root = tk.Tk()
root.title("GPU Benchmark Prediction")
root.geometry("600x400")
root.iconbitmap("GPU.ico")

load_models_and_scaler()

# Benchmark selection
benchmark_var = tk.StringVar(value="2D Benchmark")
benchmark_dropdown = ttk.OptionMenu(root, benchmark_var, "2D Benchmark", "2D Benchmark", "3D Benchmark")
benchmark_dropdown.pack()

# Model selection
model_var = tk.StringVar(value="SVM")
model_dropdown = ttk.OptionMenu(root, model_var, "SVM", "SVM", "Lasso")
model_dropdown.pack()

# Feature input fields
feature_columns = ['Process Size (nm)', 'TDP (W)', 'Die Size (mm^2)', 'Transistors (million)', 'Freq (MHz)']  # Replace with actual feature names
entries = []
for feature in feature_columns:
    ttk.Label(root, text=feature).pack()
    entry = ttk.Entry(root)
    entry.pack()
    entries.append(entry)

# Predict button
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.pack()

# Result label
result_label = ttk.Label(root, text="Predicted Score: ")
result_label.pack()

root.mainloop()
