import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib

# Load the models
rf_model = joblib.load('random_forest_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')

# Function to make predictions
def make_prediction():
    input_values = [float(entry.get()) for entry in entries]
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    
    selected_model = model_var.get()
    if selected_model == 'Random Forest':
        prediction = rf_model.predict(input_df)
    else:
        prediction = gb_model.predict(input_df)
    
    result_label.config(text=f"Predicted Benchmark: {prediction[0]:.2f}")

# Main window
root = tk.Tk()
root.title("CPU Benchmark Prediction")
root.geometry("600x400")  # Set window size
root.iconbitmap("CPU.ico")


# Model selection dropdown
model_var = tk.StringVar()
model_var.set("Random Forest")  # default value
model_label = ttk.Label(root, text="Select Model:")
model_label.pack()
model_dropdown = ttk.OptionMenu(root, model_var, "Random Forest", "Random Forest", "Gradient Boosting")
model_dropdown.pack()

# Feature input fields
feature_columns = ['Process Size (nm)', 'TDP (W)', 'Cores', 'Freq (MHz)']

# Input fields for features
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
