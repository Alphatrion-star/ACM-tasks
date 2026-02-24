# LSTM-based PM2.5 Forecasting

## 1. Project Overview

This project builds and evaluates LSTM-based deep learning models to forecast hourly PM2.5 concentrations at a selected monitoring station in Beijing. The workflow covers the full forecasting pipeline: loading multisite air quality data, filtering to a single station, transforming the data into a supervised time series format, training baseline and stacked LSTM models, and comparing their performance using regression metrics and visual analysis.

The main objective is to predict the next hour’s PM2.5 value based on the previous 24 hours of measurements, providing a proof-of-concept for short-term air quality forecasting using recurrent neural networks.[file:140]

---

## 2. Dataset

### 2.1 Source

The notebook uses the **“Beijing Multisite Air Quality Data”** hosted on Kaggle, accessed via a CSV file stored in your Kaggle workspace:

- Kaggle path used:  
  `https://www.kaggle.com/code/aravindpcoder/air-quality-monitoring?select=Beijing+Multisite+air+Quality+data.csv

The original dataset includes multiple monitoring stations across Beijing, with hourly pollutant and meteorological measurements over several years.

### 2.2 Structure and Features

The raw dataset has 420,768 rows and 17 columns, with each row representing one hour of measurements at a given station.Key columns include:

- **Time-related**: `year`, `month`, `day`, `hour`
- **Target**: `PM2.5` (fine particulate matter concentration)
- **Other pollutants**: `PM10`, `SO2`, `NO2`, `CO`, `O3`
- **Meteorology**: `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`
- **Categorical**: `wd` (wind direction), `station` (station identifier)[file:140]

### 2.3 Subsetting for the Experiment

To keep the experiment focused and computationally manageable, the code:

1. Prints the list of available stations and selects one station, **Aotizhongxin**, as the study site.[file:140]
2. Combines `year`, `month`, `day`, and `hour` into a proper `datetime` column and sorts the data chronologically.
3. Restricts the analysis to the years **2016–2017**, yielding a contiguous hourly PM2.5 series of 10,200 samples for this station.

In the modeling stage, PM2.5 is treated as a **univariate time series**, where each input window of length 24 hours is used to predict the next hour’s PM2.5 value.

---

## 3. Methodology

### 3.1 Preprocessing and Time Series Construction

**Reproducibility and configuration**

- The notebook sets a fixed random seed (`42`) for NumPy and TensorFlow to ensure consistent results across runs.
- It defines key configuration parameters:
  - `window_size = 24`: the number of past hours used as input.
  - `target_col = 'PM2.5'`: the variable to forecast.
  - `station_name = 'Aotizhongxin'`: station selected for modeling.
  - `max_epochs = 30`: maximum training epochs for each model.

**Data preparation steps**

1. **Load data** from the CSV file and inspect shape, head, and column names to verify correctness and structure.
2. **Filter to one station** (Aotizhongxin) and build a `datetime` index for proper time ordering.
3. **Subset by years** (2016–2017) so the model focuses on a recent, consistent time horizon.
4. **Select the PM2.5 time series** (and optionally related features) to construct a univariate sequence.

**Scaling and supervised sequence creation**

- The PM2.5 series is scaled, typically using MinMax scaling, so that values lie in a normalized range. This is important for stable neural network training.
- The code converts the 1D time series into supervised learning samples by sliding a window of length 24:
  - For each position, the previous 24 hourly PM2.5 values become the input sequence.
  - The PM2.5 value at the next hour becomes the target.
- This yields arrays of shape `(n_samples, window_size)` (or `(n_samples, window_size, n_features)` if more features are used) for inputs and `(n_samples,)` for targets.

**Train/validation/test splitting**

- The sequence data are split into training, validation, and test sets.
- The split respects temporal ordering (i.e., train on earlier time periods, validate and test on later periods) to mimic real forecasting conditions and avoid data leakage.

---

### 3.2 Model Architectures

The notebook defines and trains two main LSTM architectures:

#### 3.2.1 Baseline LSTM (Model A)

- A relatively simple model to act as a baseline:
  - A single LSTM layer with a moderate number of units.
  - A Dropout layer to reduce overfitting.
  - One or two Dense layers, ending with a single output neuron for the PM2.5 prediction.
- Compiled with:
  - Loss: **Mean Squared Error (MSE)**.
  - Metric: **Mean Absolute Error (MAE)**.
  - Optimizer: **Adam** (uned learning rate).

**Purpose:**  
To establish a reference performance level using the simplest reasonable LSTM setup for this time series.

#### 3.2.2 Stacked LSTM (Model B)

- A deeper architecture to explore the benefits of additional capacity:
  - First LSTM layer with `return_sequences=True` to pass full sequence output downstream.
  - Dropout layer after the first LSTM for regularization.
  - A second LSTM layer to further process temporal patterns.
  - Dense layers for final mapping from temporal embedding to scalar prediction.
- Uses the same loss, metric, optimizer, and training configuration as the baseline model.

**Purpose:**  
To investigate whether a stacked LSTM with more parameters and temporal depth improves PM2.5 forecasting accuracy over the baseline.

---

### 3.3 Training Strategy and Evaluation

**Early stopping**

- Both models employ **EarlyStopping**, monitoring validation loss with:
  - A patience window (e.g., 10 epochs) to stop training when validation performance ceases to improve.
  - `restore_best_weights=True` to roll back to the best-performing parameters on the validation set.[file:140]

**Training configuration**

- Each model is trained for up to `max_epochs = 30` with:
  - Batch size: typically 32.
  - Validation data: held-out sequences from the training period.
- The combination of early stopping and moderate epoch limit helps prevent overfitting and limits training time.

**Evaluation**

- After training, the notebook evaluates both models on the test set, reporting:
  - Test MSE (loss).
  - Test MAE (mean absolute error).
- Predictions are transformed back to the original PM2.5 scale using the inverse of the scaler, making errors and visual comparisons interpretable in real units.

**Visualization**

- Training and validation loss curves are plotted to check convergence behavior and overfitting.
- Actual vs. predicted PM2.5 values over the test period are plotted to visually assess how well each model tracks trends, peaks, and troughs.

---

## 4. Results and Analysis

### 4.1 Quantitative Performance

- Both baseline and stacked LSTM models achieve **low MSE and MAE on the test set** (in normalized space), indicating the ability to capture underlying temporal patterns in PM2.5 levels.
- The **stacked LSTM model generally outperforms the baseline** in terms of test loss and MAE, suggesting that additional depth helps model more complex dependencies in the data.
- Early stopping effectively prevents overtraining; the validation curves plateau and the models converge within the allowed epoch range.

### 4.2 Qualitative Behavior

- Forecast plots show that the models are able to:
  - Follow overall trends in PM2.5 concentration over time.
  - Capture moderate peaks and troughs, especially for sustained pollution episodes.
  - However, extreme spikes and very sharp transitions are sometimes smoothed, which is typical for LSTM models trained with MSE loss and relatively short input windows.
- The stacked LSTM’s predictions tend to align slightly better with observed peaks, demonstrating improved temporal expressiveness compared to the baseline architecture.

### 4.3 Interpretation

- An input window of 24 hours is sufficient to capture short-term persistence and daily patterns in PM2.5, but may not fully reflect multi-day or seasonal trends.
- The univariate approach (using only PM2.5 history) performs reasonably well; however, ignoring co-pollutants and meteorological variables may limit performance in more complex weather-driven pollution regimes.
- The models can already serve as a baseline for operational PM2.5 forecasting at a single station and can be extended or refined depending on application needs.

---
