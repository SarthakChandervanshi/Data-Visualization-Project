import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MonteCarloSimulation:
    def __init__(self, df, selected_symbol, n_simulations=1000, n_days=30):
        """
        Initializes the MonteCarloSimulation class with the stock data and simulation parameters.

        Args:
            df: The DataFrame containing historical stock price data.
            selected_symbol: The stock symbol for which to perform the simulation.
            n_simulations: Number of Monte Carlo simulations to run. Default is 1000.
            n_days: Number of days to simulate into the future. Default is 30.
        """
        self.df = df
        self.selected_symbol = selected_symbol
        self.n_simulations = n_simulations
        self.n_days = n_days
        self.df_symbol = None
        self.simulated_prices = None

    def prepare_data(self):
        """
        Prepares the historical stock data for simulation and visualization.

        Tasks:
            - Filters the data for the selected symbol.
            - Sorts the data by date.
            - Calculates the 20-day Simple Moving Average (SMA) and Exponential Moving Average (EMA).
            - Computes daily returns and 20-day rolling volatility.
        """
        self.df_symbol = self.df[self.df["symbol"] == self.selected_symbol]

        # Ensure data is sorted by date
        self.df_symbol = self.df_symbol.sort_values("date")

        # Calculate Moving Averages
        self.df_symbol["SMA_20"] = self.df_symbol["close"].rolling(window=20).mean()
        self.df_symbol["EMA_20"] = (
            self.df_symbol["close"].ewm(span=20, adjust=False).mean()
        )

        self.df_symbol["daily_return"] = self.df_symbol["close"].pct_change()
        self.df_symbol["volatility"] = (
            self.df_symbol["daily_return"].rolling(window=20).std()
        )

    def run_simulation(self):
        """
        Runs Monte Carlo simulations to generate future stock price paths.

        Steps:
            - Uses the last closing price, mean daily return, and volatility as inputs.
            - Simulates price changes for the specified number of days and simulations.
        """
        last_price = self.df_symbol["close"].iloc[-1]
        mean_return = self.df_symbol["daily_return"].mean()
        volatility = self.df_symbol["daily_return"].std()

        self.simulated_prices = np.zeros((self.n_days, self.n_simulations))
        for sim in range(self.n_simulations):
            prices = [last_price]
            for day in range(self.n_days):
                daily_change = np.random.normal(loc=mean_return, scale=volatility)
                prices.append(prices[-1] * (1 + daily_change))
            self.simulated_prices[:, sim] = prices[1:]

    def plot_historical_prices(self):
        """
        Plots the historical closing prices along with the 20-day SMA and EMA.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.df_symbol["date"],
            self.df_symbol["close"],
            label="Closing Price",
            color="black",
            linewidth=1,
        )
        plt.plot(
            self.df_symbol["date"],
            self.df_symbol["SMA_20"],
            label="20-Day SMA",
            color="blue",
            linestyle="--",
        )
        plt.plot(
            self.df_symbol["date"],
            self.df_symbol["EMA_20"],
            label="20-Day EMA",
            color="orange",
            linestyle="--",
        )
        plt.title(
            f"{self.selected_symbol}: Historical Prices with Moving Averages",
            fontsize=14,
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_volatility(self):
        """
        Plots the 20-day rolling volatility over time.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.df_symbol["date"],
            self.df_symbol["volatility"],
            label="20-Day Rolling Volatility",
            color="purple",
        )
        plt.title(f"{self.selected_symbol}: Volatility Over Time", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Volatility (Standard Deviation)", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_simulation_paths(self):
        """
        Plots several simulated price paths from the Monte Carlo simulation.
        """
        plt.figure(figsize=(12, 6))
        for i in range(50):
            plt.plot(self.simulated_prices[:, i], alpha=0.3, color="blue")
        plt.title(
            f"{self.selected_symbol}: Monte Carlo Simulation of Stock Prices",
            fontsize=14,
        )
        plt.xlabel("Days into the Future", fontsize=12)
        plt.ylabel("Simulated Price", fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()

    def plot_expected_price_range(self):
        """
        Plots the expected price range (mean and 90% confidence interval) from the simulations.
        """
        expected_prices = self.simulated_prices.mean(axis=1)
        lower_bound = np.percentile(self.simulated_prices, 5, axis=1)
        upper_bound = np.percentile(self.simulated_prices, 95, axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(
            expected_prices, label="Expected Price (Mean)", color="blue", linewidth=2
        )
        plt.fill_between(
            range(self.n_days),
            lower_bound,
            upper_bound,
            color="blue",
            alpha=0.2,
            label="90% Confidence Interval",
        )
        plt.title(
            f"{self.selected_symbol}: Expected Price Range (Monte Carlo Simulation)",
            fontsize=14,
        )
        plt.xlabel("Days into the Future", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def analyze(self):
        """
        Performs the complete analysis:
            - Prepares the stock data for the selected symbol.
            - Simulates future stock prices using Monte Carlo methods.
            - Generates plots for historical data, volatility, simulated price paths, and future price ranges.
        """
        self.prepare_data()
        self.run_simulation()
        self.plot_historical_prices()
        self.plot_volatility()
        self.plot_simulation_paths()
        self.plot_expected_price_range()
