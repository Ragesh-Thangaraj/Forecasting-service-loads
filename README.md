# Forecasting-service-loads

# Time-Series Forecasting with LSTM for Hourly Requests

This project demonstrates time-series forecasting of hourly requests using an LSTM (Long Short-Term Memory) neural network. The dataset consists of requests per hour, and the model predicts future requests based on past sequences.

## Project Overview

The primary objective of this project is to predict future hourly requests based on past data using a Recurrent Neural Network (RNN) with an LSTM layer. Additionally, the project demonstrates how to scale the data, visualize trends, and generate future predictions using the trained model.

### Key Features:
- Data Preprocessing: Loading and visualizing the data, scaling the values, and creating sequences for training.
- LSTM Model: Building and training an LSTM model to predict future requests.
- Evaluation & Predictions: Evaluating model performance on test data and making future predictions.
- Visualization: Plotting original data, predictions, and future forecasts.

## Dataset

The dataset consists of hourly request data stored in a CSV file named `requests_every_hour.csv`. This file should contain a column called `Requests` representing the number of requests for each hour.

Sample of the dataset:
| Requests |
|----------|
| 100      |
| 110      |
| ...      |

Ensure that the dataset is placed in the root folder where the Python scripts are located.

## Dependencies

The following libraries are required to run the project:

- Python 3.x
- Numpy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow/Keras

You can install the dependencies using the following command:

pip install numpy pandas matplotlib scikit-learn tensorflow

#Model Architecture
The model consists of the following layers:

LSTM: A Long Short-Term Memory layer with 256 units to process sequential data.
Dense: A Dense output layer with 1 unit to predict the next point in the sequence.
Steps to Run the Project
Load Dataset: The dataset is loaded from the requests_every_hour.csv file.
Data Preprocessing:
Scaling of the data using StandardScaler.
Creating sequences of data for the model using a custom function.
Build LSTM Model:
The model is built using the Keras Sequential API with LSTM layers and Dense layers.
Train the Model:
The model is trained on the first 4 weeks of data with a lookback period of 1 week (168 hours).
Evaluate and Predict:
The model is evaluated on test data and predictions are made for future requests.
Plot Results:
Original, predicted, and forecasted values are plotted for visualization.

#Running the Project
#Clone the repository:
git clone <repository_url>
cd <repository_directory>

## Prepare your dataset:
Ensure the requests_every_hour.csv file is in the project directory.

## Run the Python script:
python lstm_time_series_forecasting.py
View the results:

The script will display plots of daily, weekly, and overall request trends, as well as predictions and forecasts.

## Results
The project provides an LSTM-based model that can accurately predict future requests based on historical data. The results are visualized in plots for better understanding.
