from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


class StockForecasting:
    def __init__(self, df):
        """
        Initializes the StockForecasting object with the stock data.

        Args:
            df: A DataFrame containing the stock data.
        """
        self.df = df

    def forecast_stock(self, stock_symbol, holding_days):
        """
        Forecasts the future stock prices for a given stock symbol using SARIMAX.

        This method trains a SARIMAX model using the historical stock data and specified features (SMA, EMA, RSI, Bollinger Bands)
        and forecasts the stock price for the given number of holding days.

        Args:
            stock_symbol: The stock symbol to forecast (e.g., 'AAPL', 'GOOGL').
            holding_days: The number of days to forecast the stock price.

        Returns:
            list: A list of forecasted prices with confidence intervals for the specified holding days.
        """
        # Display available stock symbols
        available_symbols = self.df["symbol"].unique()
        print("Available stock symbols:", ", ".join(available_symbols))

        # Check if the stock symbol is valid
        if stock_symbol not in available_symbols:
            return f"Error: '{stock_symbol}' is not a valid stock symbol."

        # Filter the data for the selected stock
        stock_data = self.df[self.df["symbol"] == stock_symbol]

        features = [
            "close",
            "sma",
            "ema",
            "rsi",
            "real_upper_band",
            "real_middle_band",
            "real_lower_band",
        ]

        # Drop missing values for the selected features
        stock_data = stock_data.dropna(subset=features)

        exog = stock_data[features[1:]]

        model = SARIMAX(
            stock_data["close"], exog=exog, order=(5, 1, 3), seasonal_order=(0, 1, 1, 5)
        )
        model_fit = model.fit()

        # Forecast future prices
        forecast = model_fit.get_forecast(
            steps=holding_days, exog=exog.tail(holding_days)
        )

        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Format results
        forecast_results = []
        for day, (price, lower, upper) in enumerate(
            zip(forecast_mean, forecast_ci["lower close"], forecast_ci["upper close"]),
            1,
        ):
            forecast_results.append(
                f"Day {day}: ${price:.2f} (Confidence Interval: ${lower:.2f} - ${upper:.2f})"
            )

        return forecast_results
