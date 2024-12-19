import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
import plotly.express as px

hv.extension("bokeh")


class ExploratoryDataAnalysis:
    def __init__(self, df):
        """
        Initializes the EDA class with a DataFrame.
        Arg: The DataFrame to be explored.
        """
        self.df = df

    def overview(self):
        """Displays basic information about the DataFrame."""
        print("First 5 rows:")
        print(self.df.head())
        print("\nLast 5 rows:")
        print(self.df.tail())
        print("\nDataFrame Info:")
        print(self.df.info())
        print("\nDataFrame Shape:", self.df.shape)
        print("\nColumns:", self.df.columns)
        print("\nSummary Statistics:")
        print(self.df.describe())

    def value_counts(self, column):
        """Shows the count of unique values in a specified column."""
        print(f"\nValue counts for column '{column}':")
        print(self.df[column].value_counts())

    def plot_price_with_moving_average(self, stock_symbol, ma_column):
        """
        Visualizes close prices over time for a specified stock and overlays SMA or EMA.

        Arg 1: Stock ticker symbol to filter data (e.g., 'AAPL').
        Arg 2: The moving average column to overlay (e.g., 'SMA' or 'EMA').
        """
        stock_data = self.df[self.df["symbol"] == stock_symbol]
        if ma_column not in stock_data.columns:
            print(f"Error: Column '{ma_column}' not found in data.")
            return

        plot = (
            stock_data.hvplot.line(
                x="date",
                y="close",
                label="Close Price",
                title=f"{stock_symbol} - Price and {ma_column}",
            )
            * stock_data.hvplot.line(x="date", y=ma_column, label=f"{ma_column}")
        )
        return plot

    def plot_correlation_heatmap_for_symbol(self, symbol):
        """
        Computes pairwise correlations between numeric columns for a specific stock symbol
        and visualizes the correlation matrix using a heatmap.

        Arg: The stock symbol (e.g., 'AAPL').
        """
        stock_data = self.df[self.df["symbol"] == symbol]

        if stock_data.empty:
            print(f"Error: No data found for symbol '{symbol}'.")
            return

        numeric_columns = stock_data.select_dtypes(include=[np.number])

        correlation_matrix = numeric_columns.corr()

        heatmap = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title=f"Correlation Matrix for {symbol}",
            labels=dict(x="Indicators", y="Indicators"),
        ).update_layout(width=800, height=600)

        return heatmap

    def plot_candlestick(self, symbol):
        """
        Plots a candlestick chart for the given stock symbol using OHLC data.

        Arg: The stock symbol (e.g., 'AAPL').
        """
        stock_data = self.df[self.df["symbol"] == symbol]

        if stock_data.empty:
            print(f"Error: No data found for symbol '{symbol}'.")
            return

        ohlc = stock_data[["date", "open", "high", "low", "close"]]

        up = ohlc[ohlc["close"] >= ohlc["open"]]
        down = ohlc[ohlc["close"] < ohlc["open"]]

        up_candles = hv.Curve((up["date"], up["close"]), label="Up Candles")
        down_candles = hv.Curve((down["date"], down["close"]), label="Down Candles")

        up_candles = up_candles.opts(color="green", line_width=2)
        down_candles = down_candles.opts(color="red", line_width=2)

        candlestick_chart = up_candles * down_candles

        candlestick_chart.opts(
            title=f"Candlestick Chart for {symbol}",
            xlabel="Date",
            ylabel="Price",
            width=1000,
            height=400,
            show_legend=True,
        )

        return candlestick_chart

    def overall_eda(self):
        """
        Performs an overall exploratory data analysis by displaying basic information
        about the dataset and value counts of the 'symbol' column.
        """
        self.overview()
        self.value_counts("symbol")
