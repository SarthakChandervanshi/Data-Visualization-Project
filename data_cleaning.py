import numpy as np
import pandas as pd


class DataCleaner:
    def __init__(self, dataframe):
        """
        Initialize the DataCleaner with a pandas DataFrame.
        Arg: A pandas DataFrame to clean.
        """
        self.df = dataframe

    def drop_missing_values(self):
        """
        Drops rows with missing values from the DataFrame.
        """
        self.df.dropna(inplace=True)
        print("Dropped rows with missing values.")
        return self.df

    def fix_data_type(self, column_name, data_type):
        """
        Fixes the data type of a specific column.
        Accepts callable data types like pd.to_datetime.
        """
        if callable(data_type):
            self.df[column_name] = data_type(self.df[column_name])
        else:
            self.df[column_name] = self.df[column_name].astype(data_type)
        print(f"Converted column '{column_name}' to {data_type}.")
        return self.df

    def clean_column_names(self):
        """
        Converts all column names to lowercase and replaces spaces or other non-alphanumeric characters with underscores.
        """
        self.df.columns = self.df.columns.str.lower().str.replace(
            r"[^a-z0-9]", "_", regex=True
        )
        print(
            "Cleaned column names: converted to lowercase and replaced non-alphanumeric characters with underscores."
        )
        return self.df

    def remove_columns(self, columns):
        """
        Removes unnecessary columns from the DataFrame.
        Arg: list of columns
        """
        self.df.drop(columns=columns, inplace=True)
        print(f"Removed columns: {columns}.")
        return self.df

    def replace_symbols_with_names(self):
        """
        Replace stock symbols in the 'symbol' column with their corresponding names.
        """
        symbol_to_name = {
            "KO": "Coca-Cola",
            "PEP": "PepsiCo",
            "INTC": "Intel",
            "CSCO": "Cisco Systems",
            "XOM": "ExxonMobil",
            "CVX": "Chevron",
            "ORCL": "Oracle",
            "T": "AT&T",
            "MRK": "Merck",
            "UNH": "UnitedHealth Group",
            "MMM": "3M",
            "BA": "Boeing",
            "IBM": "IBM",
            "AMD": "Advanced Micro Devices",
            "GE": "General Electric",
            "TGT": "Target",
            "QCOM": "Qualcomm",
            "COST": "Costco",
            "UPS": "United Parcel Service",
            "PYPL": "PayPal",
            "SBUX": "Starbucks",
            "MDT": "Medtronic",
            "NKE": "Nike",
            "HON": "Honeywell",
            "RTX": "Raytheon Technologies",
        }

        self.df["symbol"] = self.df["symbol"].replace(symbol_to_name)
        print(
            "Replaced stock symbols with their corresponding full names in the 'symbol' column."
        )

        return self.df

    def complete_data_clean(self):
        """
        Runs a sequence of data cleaning operations.
        """

        # Converting column names to lowercase and replacing space with '_'
        self.clean_column_names()

        # Majority values in ATR are null, so dropping it
        remove_columns_list = ["atr"]
        self.remove_columns(remove_columns_list)

        # Dropping any rows having any missing values
        self.drop_missing_values()

        # Converting date column to date time format
        self.fix_data_type("date", pd.to_datetime)

        # Converting symbol with respect to their full names
        self.replace_symbols_with_names()

        return self.df
