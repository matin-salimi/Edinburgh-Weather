# Edinburgh Weather Forecasting Using LSTM Neural Networks

## 1. Project Overview

This project develops a deep learning model to forecast air temperature in Edinburgh, Scotland, using a Long Short-Term Memory (LSTM) neural network implemented with Python and PyTorch.

The model uses the previous 72 hours of weather information to predict temperature values for the next 12 hours.

The main objective is to investigate how recurrent neural networks can learn temporal patterns in weather data and use them to generate short-term temperature forecasts.

## 2. Project Objectives

- Collect historical hourly weather data for Edinburgh.
- Preprocess and normalize the weather dataset.
- Develop an LSTM-based neural network.
- Train and validate the model using chronological data splits.
- Evaluate forecasting performance using MAE and RMSE.
- Compare LSTM performance against a simple persistence baseline.
- Save the trained model and preprocessing scaler.
- Generate new 12-hour temperature forecasts using recent weather data.

## 3. Data Collection

Historical weather data is collected using the Open-Meteo Historical Forecast API.

**Location:** Edinburgh, Scotland, United Kingdom

**Coordinates:**
- Latitude: 55.9533
- Longitude: -3.1883

**Historical period:** 3 November 2024 – 3 November 2026

**Temporal resolution:** Hourly

The dataset contains the following six meteorological features:

| Feature | Description |
|---|---|
| temperature_2m | Air temperature at 2 meters (°C) |
| relative_humidity_2m | Relative humidity (%) |
| precipitation | Precipitation (mm) |
| wind_speed_10m | Wind speed at 10 meters (km/h) |
| surface_pressure | Surface atmospheric pressure (hPa) |
| cloud_cover | Cloud cover (%) |

The collected dataset contains 17,544 hourly records.

**Data quality checks:**
- Missing values: 0
- Duplicate timestamps: 0
- Non-hourly intervals: 0

Note: Open-Meteo historical forecast data is model-based weather data and should not be interpreted as direct measurements from a local weather station.

## 4. Data Preprocessing

The dataset is sorted chronologically and divided into three subsets:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

A StandardScaler is used to normalize the input features.

To prevent data leakage, the scaler is fitted exclusively on the training data. The same fitted scaler is then applied to the validation and test datasets.

### Sliding Window Approach

A sliding window technique is used to create the input-output sequences.

- Input sequence: Previous 72 hours.
- Number of input features: 6.
- Prediction horizon: Next 12 hours.
- Target variable: Temperature.

Therefore, each input sample has a shape of (72, 6), while its corresponding target contains 12 future temperature values.

The forecasting task is formulated as direct multi-step prediction rather than recursively predicting one hour at a time.

## 5. LSTM Model Architecture

The neural network is implemented using PyTorch.

| Parameter | Value |
|---|---:|
| Input size | 6 |
| Hidden size | 64 |
| Number of LSTM layers | 2 |
| Dropout | 0.2 |
| Output size | 12 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Maximum epochs | 100 |
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |
| Early stopping patience | 10 |

The model processes the 72-hour input sequence and uses a fully connected layer to generate twelve future temperature predictions.

## 6. Model Evaluation

The trained model is evaluated using an independent test dataset.

The following performance metrics are calculated:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual temperatures.
- **Root Mean Squared Error (RMSE):** Measures the square root of the average squared prediction errors.

### Experimental Results

The LSTM model achieved the following results:

| Evaluation Metric | Result |
|---|---:|
| Test samples | 2,621 |
| MAE | 1.2353 °C |
| RMSE | 1.6512 °C |

### Baseline Comparison

To evaluate the effectiveness of the LSTM model, its performance is compared against a Persistence Baseline.

The baseline assumes that the temperature for all twelve future hours will remain equal to the last observed temperature.

Both approaches are evaluated using the same test dataset.

| Evaluation Metric | LSTM | Persistence Baseline |
|---|---:|---:|
| MAE | 1.2353 °C | 2.3522 °C |
| RMSE | 1.6512 °C | 3.1098 °C |

The LSTM model achieves approximately **47.5% lower MAE** than the persistence baseline.

These results demonstrate that the LSTM model learns useful temporal patterns from historical weather data and performs better than simply repeating the last observed temperature.

## 7. Real-Time Prediction

After training, the model and scaler are saved and reused for subsequent predictions.

The prediction pipeline performs the following steps:

1. Retrieve recent hourly weather information from the Open-Meteo Forecast API.
2. Select the latest 72 complete and consecutive hourly records.
3. Normalize the input features using the previously saved scaler.
4. Load the trained LSTM model.
5. Predict temperatures for the next twelve hours.
6. Convert normalized predictions back to degrees Celsius.
7. Display the results using Edinburgh local time.
8. Generate and save a temperature forecast chart.

The model generates its own temperature predictions. It does not simply display the temperature forecasts supplied by the API.

The generated chart is saved in the `outputs/` directory.

## 8. Project Structure

```text
Edinburgh-Weather/
│
├── main.py
├── data.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   └── edinburgh_weather.csv
│
├── models/
│   ├── edinburgh_lstm.pth
│   └── edinburgh_scaler.pkl
│
└── outputs/
    └── edinburgh_weather_forecast.png
```

## 9. Installation and Execution

### Requirements

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Requests
- Joblib
- Matplotlib

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Collect Historical Data

```bash
python3 data.py
```

### Step 2: Train the Model

```bash
python3 train.py
```

### Step 3: Evaluate the Model

```bash
python3 evaluate.py
```

### Step 4: Generate Temperature Predictions

```bash
python3 predict.py
```

The trained model and scaler are reused during prediction, so retraining is not required for every forecast.

## 10. Limitations and Future Improvements

Although the model achieves promising test results, several improvements could be investigated:

- Compare LSTM performance with GRU and other forecasting models.
- Experiment with different sequence lengths and model hyperparameters.
- Evaluate forecasting errors separately for each future prediction horizon.
- Compare predictions against independent weather station observations.
- Retrain the model periodically using newly available historical data.
- Investigate the influence of seasonal weather patterns.

## 11. Conclusion

This project demonstrates the application of an LSTM neural network for short-term temperature forecasting in Edinburgh.

Using six meteorological input features and a 72-hour historical observation window, the model predicts temperatures for the following twelve hours.

The experimental results show an MAE of 1.2353 °C and an RMSE of 1.6512 °C on the test dataset.

The LSTM model also outperforms the Persistence Baseline, achieving approximately 47.5% lower MAE.

The project implements a reusable prediction pipeline that loads the saved model and scaler to generate new forecasts without requiring retraining for every execution.