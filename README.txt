# Cup and Handle Pattern Detection using Machine Learning

### Non-Technical Explanation

This project uses machine learning to identify the "cup and handle" chart pattern, a bullish signal, in the daily prices of all S&P 500 stocks. The goal is to predict the probability of this pattern forming, which is a classic imbalanced classification problem because the pattern is very rare. The models are trained on a comprehensive dataset of historical prices and technical indicators to learn what a cup and handle looks like, with the ultimate goal of generating reliable trading signals.

### Data

The dataset consists of daily closing prices for all S&P 500 companies over a period of five years.

* **Source:** All data is dynamically fetched from the Yahoo Finance API using the `yfinance` Python library. The list of S&P 500 tickers is scraped from Wikipedia to ensure it is up-to-date.
* **Features:** From the raw price data, a rich set of features are engineered to describe market conditions. These include:
    * Moving Averages (SMA-10, SMA-50)
    * Relative Strength Index (RSI)
    * Moving Average Convergence Divergence (MACD) and its signal line
    * Bollinger Bands
    * Average True Range (ATR)
    * Volume-based metrics and lagged returns
* **Target Variable:** A custom function, which looks for a preceding uptrend, a U-shaped cup, a short handle, and low volume, generates a binary label of `1` for the presence of a cup and handle and `0` for its absence.

### Model

Two high-performance machine learning models were trained and evaluated on this task:

* **XGBoost Classifier:** A powerful gradient boosting model known for its accuracy and efficiency on structured data. It was chosen to build a strong baseline model.
* **LightGBM Classifier:** Another highly efficient gradient boosting framework designed for speed and scalability on large datasets. It was included to see if it could outperform XGBoost.

Both models were trained to predict a binary outcome (pattern or no pattern) based on the engineered features. The data was split by stock ticker (60% training, 20% validation, 20% testing) to ensure the models were evaluated on entirely unseen stocks.

### Hyperparameter Optimization

`RandomizedSearchCV` was used for hyperparameter optimization for both models to find the best balance between precision and recall. Given the imbalanced nature of the dataset (the pattern is a rare event), the `scale_pos_weight` parameter was included to penalize the misclassification of the positive class more heavily. Key hyperparameters tuned for both models included `n_estimators`, `max_depth`, and `learning_rate`. The `min_child_weight` parameter was also introduced to encourage more conservative, higher-precision predictions.

### Results

Performance is measured using precision, recall, and F1-score on the held-out test set. The results show that the XGBoost model performed better, striking a more effective balance.

| Model | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 0.2678 | 0.7040 | 0.3880 |
| LightGBM | 0.2142 | 0.8899 | 0.3453 |

The XGBoost model's results are a solid foundation. While a precision of ~27% indicates that a significant number of predictions are false alarms, its ability to capture over 70% of the actual patterns is a strong sign. Further work to improve precision is needed before this model could be used for a real-world trading strategy.

### Datasheet

Here is the datasheet for the project's data and methodology.

## Motivation

This project aims to predict a rare chart pattern—the "cup and handle"—in the S&P 500. It uses machine learning models to identify this pattern and provide a probabilistic prediction. The goal is to build a robust model for generating trading signals, a significant step beyond simple price forecasting.

## Composition

* **Dataset:** The dataset contains daily historical prices, volume, and various technical indicators for all companies currently in the S&P 500.
* **Data Points:** Each row in the dataset represents a single trading day for a single stock.
* **Tickers:** The list of S&P 500 tickers is dynamically fetched from Wikipedia, which ensures the list is up-to-date.
* **Confidentiality:** The data is publicly available, so there are no confidentiality concerns.

## Collection Process

* **Data Acquisition:** The data is acquired via the **Yahoo Finance API** using the `yfinance` Python library. The script is designed to handle potential rate-limiting issues.
* **Sampling Strategy:** The data is sampled at the close of every trading day. The project uses 5 years of historical data for each stock.
* **Time Frame:** The data spans the last 5 years up to the date the script is run.

## Preprocessing/Cleaning/Labeling

* **Missing Data:** A fill-forward method is applied to handle missing values, which typically occur on public holidays.
* **Feature Engineering:** A comprehensive set of technical indicators are calculated from the raw price and volume data. These include:
    * Moving Averages (SMA-10, SMA-50)
    * Relative Strength Index (RSI)
    * Moving Average Convergence Divergence (MACD)
    * Bollinger Bands
    * Average True Range (ATR)
* **Labeling:** The target variable, a binary label (`1` for pattern, `0` for no pattern), is generated by a custom function that looks for a preceding uptrend, a U-shaped cup, a short handle, and low volume. This process creates a highly **imbalanced dataset**.

## Uses

* **Primary Use:** Training and evaluating machine learning models to predict the cup and handle pattern.
* **Alternative Uses:**
    * Academic research into chart pattern recognition using machine learning.
    * Backtesting other technical trading strategies.
    * Developing new feature sets for stock market analysis.

## Distribution

* **Current Distribution:** The dataset is created locally from public API calls and is not publicly distributed.
* **IP/Licensing:** All data sourced from Yahoo Finance is subject to their Terms of Service. Non-commercial use is generally permitted, but any commercial application would require additional licensing. Derived datasets should acknowledge Yahoo Finance as the source.

## Maintenance

* **Data Source:** Yahoo Finance maintains the data and provides daily updates.
* **Availability:** The data is dependent on the continued availability of the Yahoo Finance API.
* **Quality Control:** The script includes a data cleaning step, but the quality of the raw data should be regularly checked for anomalies.