import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class StockRandomForestModel:
    def __init__(self, df, seq_length=30):
        """
        Initializes the StockRandomForestModel with the dataset and sequence length.

        Args:
            df: The stock dataset containing historical data.
            seq_length: The number of previous days used to predict the next day's stock price.
        """
        self.df = df
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.close_scaler = MinMaxScaler(feature_range=(0, 5))
        self.encoder = None
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

        self.df = self.df.dropna(
            subset=features
        )  # Drop rows with missing values in features

        # Fit the scaler on the 'close' column
        self.df["close_scaled"] = self.close_scaler.fit_transform(self.df[["close"]])

        # Scaling the features (excluding the scaled 'close' column)
        scaled_features = self.scaler.fit_transform(self.df[features])
        scaled_data = np.hstack((self.df[["close_scaled"]].values, scaled_features))

        return scaled_data

    def create_sequences(self, data):
        """
        Converts the scaled data into sequences for model input.

        Args:
            data: The scaled dataset.

        Returns:
            tuple: A tuple containing feature sequences (X) and target values (y).
        """
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(
                data[i : i + self.seq_length, 1:]
            )  # Features (skip the 'close_scaled' column)
            y.append(data[i + self.seq_length, 0])  # Target (the 'close_scaled' column)
        return np.array(X), np.array(y)

    def prepare_data(self):
        """
        Prepares the data for training and testing by preprocessing, creating sequences,
        flattening features, and splitting into training and testing sets.
        """
        scaled_data = self.preprocess_data()
        X, y = self.create_sequences(scaled_data)

        # Flatten X for machine learning model
        X_flat = X.reshape(X.shape[0], -1)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_flat, y, test_size=0.2, random_state=42
        )

    def build_model(self):
        """
        Builds the Random Forest Regressor model.
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self):
        """
        Trains the Random Forest Regressor model using the training data.
        """
        if self.model is None:
            self.build_model()

        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluates the Random Forest Regressor model on the test set.

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
        # Reverse scale predictions and true values using the close_scaler
        y_test_rescaled = self.close_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        y_pred_rescaled = self.close_scaler.inverse_transform(
            y_pred.reshape(-1, 1)
        ).flatten()

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
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R^2 Score: {r2}")

        return mse, mae, r2
