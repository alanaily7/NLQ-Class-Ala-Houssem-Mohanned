import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib
import os
import sys

# Check if the app is frozen (i.e., running as an executable)
if getattr(sys, 'frozen', False):
    # If running as an executable, the base path is the directory of the executable
    base_path = os.path.dirname(sys.executable)
else:
    # If running as a script, the base path is the directory of the script
    base_path = os.path.dirname(__file__)

# Define the paths for the external files
model_path = os.path.join(base_path, 'SVM_Cat_Subcat_model.pkl')
tfidf_vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')

# Load the saved model and TF-IDF vectorizer
try:
    ensemble_model = joblib.load(model_path)  # Load the ensemble model
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)  # Load the TF-IDF vectorizer
except FileNotFoundError as e:
    print(f"Error loading model or vectorizer: {e}")
    sys.exit("Required files are missing. Please ensure all necessary files are in the same directory.")

# Create the main window
window = tk.Tk()
window.title("Class Classification")
window.geometry("500x400")
window.resizable(False, False)

# Set a theme for a modern look
style = ttk.Style(window)
style.theme_use('clam')

# Customize the theme
style.configure(
    "TLabel",
    background="#f8f9fa",
    foreground="#333",
    font=("Arial", 10)
)
style.configure(
    "TButton",
    background="#007bff",
    foreground="#fff",
    font=("Arial", 10, "bold"),
    padding=5
)
style.map(
    "TButton",
    background=[("active", "#0056b3")]
)
style.configure(
    "TEntry",
    padding=5
)

# Set the background color of the main window
window.configure(bg="#f8f9fa")

# Add a title label
title_label = ttk.Label(
    window,
    text="Class Classification Tool",
    font=("Arial", 16, "bold"),
    anchor="center"
)
title_label.pack(pady=20)

# Create a label for instructions
instruction_label = ttk.Label(
    window,
    text="Enter your question below and click 'Classify':",
    wraplength=400,
    anchor="center"
)
instruction_label.pack(pady=10)

# Create a text entry field for the question input
entry_frame = ttk.Frame(window)
entry_frame.pack(pady=10)
entry = ttk.Entry(entry_frame, width=50)
entry.grid(row=0, column=0, padx=5)
clear_button = ttk.Button(
    entry_frame,
    text="Clear",
    command=lambda: entry.delete(0, tk.END),
    style="TButton"
)
clear_button.grid(row=0, column=1, padx=5)

# Create a function to handle the button click
def classify_question():
    question = entry.get().strip()
    if not question:
        messagebox.showerror("Input Error", "Please enter a question.")
        return

    try:
        # Transform the input question using the loaded TF-IDF vectorizer
        question_tfidf = tfidf_vectorizer.transform([question])

        # Make prediction using the loaded ensemble model
        prediction = ensemble_model.predict(question_tfidf)[0]

        # Split the prediction back into category and subcategory
        predicted_category, predicted_subcategory = prediction.split("_")

        # Display the result in a message box
        messagebox.showinfo("Prediction Result", f"The predicted category is: {predicted_category}\nThe predicted subcategory is: {predicted_subcategory}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during classification:\n{e}")

# Create a button to trigger classification
classify_button = ttk.Button(
    window,
    text="Classify",
    command=classify_question
)
classify_button.pack(pady=20)

# Add a footer for credit or help
footer_label = ttk.Label(
    window,
    text="Powered by NAILY ANNENE MALLOULI",
    font=("Arial", 8),
    anchor="center"
)
footer_label.pack(side="bottom", pady=10)

# Run the Tkinter event loop
window.mainloop()
