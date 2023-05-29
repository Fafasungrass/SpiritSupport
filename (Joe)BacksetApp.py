import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from github import Github
from PIL import Image, ImageTk

# Function to handle button click event
def predict_backset():
    try:
        # Read the input values from the entry fields
        fermenter_ph = float(entry_fermenter_ph.get())

        # Define the desired range of post-backset pH values
        desired_post_backset_pH_range = np.arange(4.80, 5.01, 0.01).round(2)

        # Calculate the optimal backset amount for each pH value in the range
        optimal_backset_amounts = []
        for pH in desired_post_backset_pH_range:
            predicted_backset = model.predict([[fermenter_ph, pH]])
            optimal_backset_amounts.append(predicted_backset[0])

        mean_optimal_backset_amount = np.mean(optimal_backset_amounts)
        optimal_pH_index = np.abs(optimal_backset_amounts - mean_optimal_backset_amount).argmin()
        optimal_pH = desired_post_backset_pH_range[optimal_pH_index]

        # Evaluate the model
        mean_backset_added = np.mean(data['Backset Added'])
        mse = mean_squared_error(y, model.predict(X))
        rmse = np.sqrt(mse)
        accuracy = (1 - (rmse / mean_backset_added)) * 100

        messagebox.showinfo("Prediction Result", 
                    f"Given a fermenter pH of {fermenter_ph:.2f}, "  # Round the pH to 2 decimal places
                    f"add {int(np.round(optimal_backset_amounts[optimal_pH_index]))} gallons of backset "
                    f"for a desired post-backset pH of {optimal_pH:.2f}\n"
                    f"\nAccuracy: {accuracy:.2f}%")
    except ValueError:
        messagebox.showerror("Error", "Please enter numeric values for the inputs.")

# Create the Tkinter window
window = tk.Tk()
window.title("Backset Amount Prediction Application")
window.configure(bg="black")
window.geometry("500x300")

# Create label for company name
label_company = tk.Label(window, text="Tennessee Distilling Group", font=("Arial", 20, "bold"), fg="white", bg="black")
label_company.pack(pady=10)

# Create label for application name
label_application = tk.Label(window, text="D1 Backset Calculator", font=("Arial", 16, "bold"), fg="white", bg="black")
label_application.pack(pady=10)

# Integration with Github
access_token = # Access code was emailed to you, place in quotations
repository_owner = "Fafasungrass"
repository_name = "SpiritSupport"
file_path = "D1 Dataset.xlsx"
g = Github(access_token)
repo = g.get_repo(f"{repository_owner}/{repository_name}")
file_content = repo.get_contents(file_path).decoded_content

# Load your data from the file content
data = pd.read_excel(file_content)

# Convert columns with numbers represented as strings to float
data['Backset Added'] = data['Backset Added'].astype(float)

# Separate the features (inputs) and target variable (output)
X = data[['Fermenter pH', 'Post-backset pH']].to_numpy()
y = data['Backset Added'].to_numpy().reshape(-1, 1)

# Create a linear regression model and train it
model = LinearRegression()
model.fit(X, y)

# Create labels and entry fields for inputs
label_fermenter_ph = tk.Label(window, text="Fermenter pH:", font=("Arial", 14, "bold"), fg="white", bg="black")
label_fermenter_ph.pack()
entry_fermenter_ph = tk.Entry(window, font=("Arial", 12))
entry_fermenter_ph.pack()

# Create a button to trigger the prediction
button_predict = tk.Button(window, text="Predict", font=("Arial", 12), command=predict_backset, bg="white", fg="black")
button_predict.pack(pady=20)

image = Image.open(r"C:\Users\Joseph\Pictures\TDG words.png")
image = image.resize((100, 100), Image.ANTIALIAS)  # Resize to 100x100 pixels
logo = ImageTk.PhotoImage(image)

logo_label = tk.Label(window, image=logo, bg="black")
logo_label.pack(pady=10)
logo_label.place(x=25, y=75)  # Adjust x and y values as needed


# Start the Tkinter event loop
window.mainloop()

