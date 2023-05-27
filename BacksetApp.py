import PySimpleGUI as sg
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from github import Github

# Integration with Github
access_token = "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"

repository_owner = "Fafasungrass"
repository_name = "SpiritSupport"
file_path = "D1 Dataset.xlsx"
g = Github(access_token)
repo = g.get_repo(f"{repository_owner}/{repository_name}")

file_content = repo.get_contents(file_path).decoded_content

# Load your data from the file content
data = pd.read_excel(file_content)

# Define the layout for the PySimpleGUI window
layout = [
    [sg.Button('Calculate Backset')]
]

# Create the PySimpleGUI window
window = sg.Window('Backset Amount Prediction Application', layout)

while True:
    event, _ = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == 'Calculate Backset':
        # Remove rows with NaNs
        data = data.dropna()
               # Convert columns with numbers represented as strings to float (if necessary)
        data['Backset Added'] = data['Backset Added'].astype(float)

        # Split the data into training set and test set
        X = data[['Fermenter pH', 'Post-backset pH']]  # Add other columns if required
        y = data['Backset Added']

        # Reshape y to have two dimensions
        y = y.to_numpy().reshape(-1, 1)

        # Convert x from a pandas dataframe to a numpy array for optimal regression
        X = X.to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Linear Regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

         # Evaluate the model
        mean_backset_added = np.mean(data['Backset Added'])
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        accuracy = (1 - (rmse / mean_backset_added)) * 100

        # Ask for user input
        while True:
            try:
                fermenter_pH = float(sg.popup_get_text("Enter the Fermenter pH: "))
                break
            except ValueError:
                sg.popup('Invalid input. Please enter a number.')
                continue

        # Define the desired range of post-backset pH values
        desired_post_backset_pH_range = np.arange(4.80, 5.01, 0.01).round(2)

        # Calculate the optimal backset amount for each pH value in the range
        optimal_backset_amounts = []
        for pH in desired_post_backset_pH_range:
            predicted_backset = model.predict([[fermenter_pH, pH]])
            optimal_backset_amounts.append(predicted_backset[0])

        mean_optimal_backset_amount = np.mean(optimal_backset_amounts)
        optimal_pH_index = np.abs(optimal_backset_amounts - mean_optimal_backset_amount).argmin()
        optimal_pH = desired_post_backset_pH_range[optimal_pH_index]

        sg.popup(f"Given a fermenter pH of {fermenter_pH}, add {int(np.round(optimal_backset_amounts[optimal_pH_index]))} gallons of backset for a desired post-backset pH of {optimal_pH:.2f}\n"
                 f"\nAccuracy: {accuracy:.2f}%")

        sg.popup(f"On average, this model deviates from the true value by {rmse:.2f} gallons")

window.close()
