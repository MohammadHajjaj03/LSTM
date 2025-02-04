A machine learning project aimed at predicting stock prices using Long Short-Term Memory (LSTM) networks.
The model is built to forecast stock price movements based on historical closing prices.
Here's an overview of the project:
Data Preprocessing: The project loads and processes stock data, including date conversion and feature scaling to normalize the data for training.
Train-Validation-Test Split: The dataset is split into training, validation, and test sets, with customizable ratios defined by the user via an interactive GUI built with Tkinter.
LSTM Model: Builds a deep learning model with two LSTM layers to predict future stock prices.
LSTM Model: Builds a deep learning model with two LSTM layers and one Dense to predict future stock prices.
Model Training & Evaluation: Trains the model, saves the best version, and evaluates it using MSE, RMSE, and R² score.
Results Visualization: Visualizes training loss, predicted vs actual prices, and error distribution, scatter plot comparing actual vs predicted prices, stock price with rolling mean.
User Interface:
•	Select the stock data file in CSV format.
•	Set the ratios for splitting the data into training, validation, and test sets.
•	Submit the settings and initiate the prediction process, ensuring that the system runs in the background using threading to keep the interface responsive during computationally heavy tasks.
Requirements:
•	Python 3.x
•	TensorFlow, Keras
•	Pandas, scikit-learn
•	Matplotlib, Seaborn
•	Tkinter, threading
