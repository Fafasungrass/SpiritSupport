import PySimpleGUI as sg
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Define the layout for the PySimpleGUI window
layout = [
    [sg.Text('Load your data file (D1 Dataset.xlsx):'), sg.In(key='FILE'), sg.FileBrowse()],
    [sg.Button('Process Data')]
]

# Create the PySimpleGUI window
window = sg.Window('Backset Amount Prediction Application', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == 'Process Data':
        # Load your data
        try:
            data = pd.read_excel(values['FILE'])
        except FileNotFoundError:
            sg.popup('File not found. Please try again.')
            continue

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
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('R-Squared:', r2_score(y_test, y_pred))

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

        sg.popup(f"Given a fermenter pH of {fermenter_pH}, add {int(np.round(optimal_backset_amounts[optimal_pH_index]))} gallons of backset for a desired post-backset pH of {optimal_pH:.2f}")

window.close()
