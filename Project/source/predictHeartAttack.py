import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load the saved model and scaler
model_filename = 'best_model.pkl'
scaler_filename = 'scaler.pkl'

try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
except FileNotFoundError:
    messagebox.showerror("Error", "Model or scaler file not found. Please ensure 'best_model.pkl' and 'scaler.pkl' are in the directory.")
    exit()

# Create the GUI
def predict():
    try:
        # Get user input
        input_data = [
            float(age_entry.get()),
            float(sex_entry.get()),
            float(cp_entry.get()),
            float(trtbps_entry.get()),
            float(chol_entry.get()),
            float(fbs_entry.get()),
            float(restecg_entry.get()),
            float(thalachh_entry.get()),
            float(exng_entry.get()),
            float(oldpeak_entry.get()),
            float(slp_entry.get()),
            float(caa_entry.get()),
            float(thall_entry.get())
        ]
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for all fields.")
        return

    # Scale the input data
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    output = "Heart Attack Risk: Likely" if prediction[0] == 1 else "Heart Attack Risk: Unlikely"
    messagebox.showinfo("Prediction", output)

# Setup the Tkinter window
root = tk.Tk()
root.title("Heart Attack Prediction")

# Define labels and entries for each feature
features = [
    "Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)", "Resting Blood Pressure",
    "Cholesterol", "Fasting Blood Sugar (1=True, 0=False)", "Resting ECG (0-2)",
    "Max Heart Rate Achieved", "Exercise Induced Angina (1=True, 0=False)", "Oldpeak",
    "Slope (0-2)", "CA (0-3)", "Thal (1-3)"
]

entries = []

for i, feature in enumerate(features):
    tk.Label(root, text=feature).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

age_entry, sex_entry, cp_entry, trtbps_entry, chol_entry, fbs_entry, restecg_entry, \
thalachh_entry, exng_entry, oldpeak_entry, slp_entry, caa_entry, thall_entry = entries

# Create the Predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=len(features), column=0, columnspan=2, pady=10)

# Run the Tkinter main loop
root.mainloop()
