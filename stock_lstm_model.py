import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class StockLSTMModel:
    def __init__(self, df, seq_length=30):
        """
        Initializes the StockLSTMModel with the dataset and sequence length.

        Args:
            df: The stock dataset containing historical data.
            seq_length: The number of previous days used to predict the next day's stock price.
        """
        self.df = df
        self.seq_length = seq_length
        self.encoder = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.symbol_columns = []
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None

    def preprocess_data(self):
        """
        Preprocesses the stock data by encoding stock symbols, selecting features, handling missing values,
        and scaling the data.

        Returns:
            np.array: The scaled dataset ready for sequence creation.
        """
        # One-Hot Encoding for stock symbols
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_symbols = self.encoder.fit_transform(self.df[["symbol"]])
        self.symbol_columns = [
            f"symbol_{symbol}" for symbol in self.encoder.categories_[0]
        ]

        encoded_df = pd.DataFrame(
            encoded_symbols, columns=self.symbol_columns, index=self.df.index
        )
        self.df = pd.concat([self.df, encoded_df], axis=1)

        # Feature selection
        features = [
            "open",
            "high",
            "low",
            "volume",
            "sma",
            "ema",
            "rsi",
            "real_upper_band",
            "real_middle_band",
            "real_lower_band",
        ] + self.symbol_columns

        self.df = self.df.dropna(subset=features)
        data = self.df[["close"] + features]

        # Scale the data
        scaled_data = self.scaler.fit_transform(data)

        return scaled_data

    def create_sequences(self, data):
        """
        Converts the scaled data into sequences for LSTM input.

        Args:
            data: The scaled dataset.

        Returns:
            tuple: A tuple containing feature sequences (X) and target values (y).
        """
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i : i + self.seq_length, 1:])  # Features
            y.append(data[i + self.seq_length, 0])  # Target (the 'close' column)
        return np.array(X), np.array(y)

    def prepare_data(self):
        """
        Prepares the data for training and testing by preprocessing, creating sequences,
        and splitting into training and testing sets.
        """
        scaled_data = self.preprocess_data()
        X, y = self.create_sequences(scaled_data)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def build_model(self):
        """
        Builds the LSTM model with two LSTM layers, dropout layers, and dense layers.
        """
        self.model = Sequential(
            [
                LSTM(
                    50,
                    return_sequences=True,
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                ),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1),
            ]
        )

        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train_model(self, epochs=20, batch_size=32):
        """
        Trains the LSTM model using the training data.

        Args:
            epochs: The number of training epochs.
            batch_size: The size of training batches.
        """
        if self.model is None:
            self.build_model()

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
        )

    def evaluate_model(self):
        """
        Evaluates the LSTM model on the test set.

        Returns:
            tuple: A tuple containing the predicted values (y_pred) and the actual test values (y_test).
        """
        y_pred = self.model.predict(self.X_test)
        return y_pred, self.y_test

    def inverse_transform_predictions(self, y_pred, y_test):
        """
        Rescales the predicted and actual values back to their original scale.

        Args:
            y_pred: The predicted values from the model.
            y_test: The actual values from the test set.

        Returns:
            tuple: Rescaled predicted values and actual test values.
        """
        # Reverse scale predictions and true values
        expected_columns = self.scaler.n_features_in_

        # Reverse scale true values
        y_test_rescaled = self.scaler.inverse_transform(
            np.concatenate(
                [y_test.reshape(-1, 1), np.zeros((len(y_test), expected_columns - 1))],
                axis=1,
            )
        )[:, 0]

        # Reverse scale predicted values
        y_pred_rescaled = self.scaler.inverse_transform(
            np.concatenate(
                [y_pred.reshape(-1, 1), np.zeros((len(y_pred), expected_columns - 1))],
                axis=1,
            )
        )[:, 0]

        return y_pred_rescaled, y_test_rescaled

    def calculate_metrics(self, y_pred, y_test):
        """
        Calculates performance metrics for the model.

        Args:
            y_pred: The predicted values from the model.
            y_test: The actual values from the test set.

        Returns:
            tuple: A tuple containing the Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.
        """
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R² Score: {r2}")

        return mse, mae, r2
