import pandas as pd
import holoviews as hv
import panel as pn
import param
from bokeh.io import output_notebook


class StockDashboard:
    def __init__(self, data):
        """
        Initializes the StockDashboard with stock data and sets up widgets, tabs, and dashboard layout.

        Args:
            data: A DataFrame containing stock market data.
        """
        self.data = data

        self.selected_symbol = pn.widgets.Select(
            name="Stock Symbol",
            options=list(data["symbol"].unique()),
            value=data["symbol"].unique()[0],
        )
        self.selected_date_range = pn.widgets.DateRangeSlider(
            name="Date Range",
            start=data["date"].min(),
            end=data["date"].max(),
            value=(data["date"].min(), data["date"].max()),
        )
        self.selected_symbols_multi = pn.widgets.MultiSelect(
            name="Select Multiple Stocks", options=list(data["symbol"].unique()), size=8
        )
        self.intro_tab = self.create_intro_tab()
        self.tabs = self.create_insight_tabs()
        self.dashboard = self.build_dashboard()

    def create_intro_tab(self):
        """
        Creates the introduction tab with an overview of the dashboard and instructions.

        Returns:
            pn.pane.Markdown: Markdown pane containing the introduction.
        """
        intro_text = """
        # Stock Data Dashboard

        This dashboard provides insights into stock prices and associated metrics.
        Explore the data and interact with the visualizations to make data-driven decisions.

        ## Available Tabs:

        **Tab 1: Visualizations**
        
        This tab presents a series of interactive visualizations to help you explore stock price trends over time:

        Line Plot: Displays the stock's closing price over three years, highlighting price trends, peaks, and lows.

        Distribution Plot: Shows the distribution of closing prices, allowing you to easily spot the range of values and their frequency over the years.

        Box Plot: Analyzes the distribution of stock prices by year and helps identify any outliers. Outliers may indicate significant price changes or unusual market activity.
        
        Violin Plot: Visualizes the distribution of closing prices by year, with an additional focus on the shape of the price distribution, even in the presence of outliers.

        **Tab 2: RSI & table**

        This tab provides a detailed table of stock data grouped by symbol and year, allowing you to perform in-depth analysis of stock performance
        over time. Additionally, you can explore the Relative Strength Index (RSI) plot to understand how the stock is performing relative to its 
        historical price movements, helping you identify potential overbought or oversold conditions.

        **Tab 3: Deep Dive**

        In this tab, you can explore additional advanced metrics and technical indicators of latest three months data:

        SMA & EMA: The Simple Moving Average (SMA) and Exponential Moving Average (EMA) are displayed over time to show trends and 
        smooth out price data for better trend analysis.

        Bollinger Bands: View the upper, middle, and lower Bollinger Bands to understand volatility and potential price reversals.

        Volume Over Time: A line plot showing stock volume to understand trading activity during specific periods.

        This tab allows you to analyze the stock in greater detail, using key indicators and metrics commonly used in technical analysis..

        **Tab 4: Multiple Stock Plot**

        This tab enables you to compare the performance of multiple stocks over time. By selecting multiple stocks, you can visualize 
        their closing prices in a single line plot. This is useful for comparing how different stocks performed relative to each other
        within the same date range..

        ## How to Use:

        1. Select a stock symbol from the dropdown every plot of the tab update dynamically.
        2. Navigate to the "Visualizations" and "Deep Dive" tabs to explore dynamic plots.
        3. View the summarized data in the "Data Table" tab 2.
        4. Use the "Multiple Stock Line Plot" tab to compare close prices of multiple stocks.
        """
        return pn.pane.Markdown(intro_text, sizing_mode="stretch_width")

    def filter_data(self):
        """
        Filters the stock data based on the selected symbol and date range.

        Returns:
            pd.DataFrame: Filtered DataFrame based on the current widget selections.
        """
        filtered_data = self.data[
            (self.data["symbol"] == self.selected_symbol.value)
            & (self.data["date"] >= self.selected_date_range.value[0])
            & (self.data["date"] <= self.selected_date_range.value[1])
        ]
        return filtered_data

    def create_insight_tabs(self):
        """
        Creates the interactive tabs for visualizations and data exploration.

        Returns:
            list: A list of Panel layouts, each representing a tab in the dashboard.
        """

        def update_plots():
            """Updates the plots in the first tab based on the selected symbol and date range."""
            filtered_data = self.filter_data()
            plot1 = filtered_data.hvplot.line(
                x="date", y="close", title="Line Plot: Stock Close Price Over Time"
            )
            plot2 = filtered_data.hvplot.hist(
                y="close", title="Hist Plot: Stock Close Price Distribution"
            )
            plot3 = filtered_data.hvplot.box(
                y="close",
                by="date.year",
                title="Box Plot: Stock Price Distribution by Year",
            )
            plot4 = filtered_data.hvplot.violin(
                y="close",
                by="date.year",
                title="Violin Plot: Stock Price Distribution by Year",
            )

            grid[0, 0] = plot1
            grid[0, 1] = plot2
            grid[1, 0] = plot3
            grid[1, 1] = plot4

        def update_table_and_rsi_plot():
            """Updates the data table and RSI plot in the second tab."""
            filtered_data = self.filter_data()
            table.value = filtered_data
            rsi_plot[0, 0] = filtered_data.hvplot.line(
                x="date", y="rsi", title="Line Plot: RSI Over Time"
            )

        def update_extended_plots():
            """Updates advanced metrics plots in the third tab for the latest three months."""
            filtered_data = self.filter_data()[
                (self.filter_data()["date"] >= "2024-10-01")
                & (self.filter_data()["date"] <= "2024-11-30")
            ]
            x_limits = ("2024-10-01", "2024-11-30")
            filtered_data["date"] = pd.to_datetime(filtered_data["date"]).dt.date
            filtered_data = filtered_data.sort_values(by="date")

            plot1 = filtered_data.hvplot.line(
                x="date",
                y=["open", "close", "low", "high"],
                title="Line Plot: Open, Close, Low, High",
                xlim=x_limits,
                legend="top_right",
            ).opts(xrotation=90)
            plot2 = filtered_data.hvplot.line(
                x="date",
                y=["sma", "ema"],
                title="Line Plot: SMA and EMA",
                xlim=x_limits,
                legend="top_right",
            ).opts(xrotation=90)
            plot3 = filtered_data.hvplot.line(
                x="date",
                y=["real_upper_band", "real_middle_band", "real_lower_band"],
                xlim=x_limits,
                legend="top_right",
                title="Line Plot: Bollinger Bands",
            ).opts(xrotation=90)
            plot4 = filtered_data.hvplot.scatter(
                x="date",
                y="volume",
                title="Bar Plot: Volume Over Time",
                xlim=x_limits,
                legend="top_right",
            ).opts(xrotation=90, axiswise=True)

            extended_grid[0, 0] = plot1
            extended_grid[0, 1] = plot2
            extended_grid[1, 0] = plot3
            extended_grid[1, 1] = plot4

        def update_multi_stock_plot():
            """Updates the multi-stock line plot in the fourth tab."""
            selected_stocks = self.selected_symbols_multi.value
            filtered_data = self.data[
                (self.data["date"] >= self.selected_date_range.value[0])
                & (self.data["date"] <= self.selected_date_range.value[1])
            ]
            multi_stock_plot[0, 0] = filtered_data.hvplot.line(
                x="date",
                y="close",
                by="symbol",
                title="Line Plot: Close Prices of Selected Stocks Over Time",
            )

        # Tab 1: Initial visualizations
        grid = pn.GridSpec(sizing_mode="stretch_width", max_width=1200, height=800)
        grid[0, 0] = pn.pane.Markdown("Loading...")
        self.selected_symbol.param.watch(lambda _: update_plots(), "value")
        update_plots()
        tab_1 = pn.Column(pn.Row(self.selected_symbol), grid)

        # Tab 2: Grouped data table with RSI graph
        table = pn.widgets.DataFrame(
            name="Filtered Data Table", sizing_mode="stretch_width"
        )
        rsi_plot = pn.GridSpec(sizing_mode="stretch_width", max_width=1200, height=400)
        rsi_plot[0, 0] = pn.pane.Markdown("Loading RSI Plot...")
        self.selected_symbol.param.watch(lambda _: update_table_and_rsi_plot(), "value")
        update_table_and_rsi_plot()
        tab_2 = pn.Column(pn.Row(self.selected_symbol), rsi_plot, table)

        # Tab 3: Extended visualizations
        extended_grid = pn.GridSpec(
            sizing_mode="stretch_width", max_width=1200, height=800
        )
        extended_grid[0, 0] = pn.pane.Markdown("Loading...")
        self.selected_symbol.param.watch(lambda _: update_extended_plots(), "value")
        update_extended_plots()
        tab_3 = pn.Column(pn.Row(self.selected_symbol), extended_grid)

        # Tab 4: Multiple stock line plot
        multi_stock_plot = pn.GridSpec(
            sizing_mode="stretch_width", max_width=1200, height=400
        )
        multi_stock_plot[0, 0] = pn.pane.Markdown("Select stocks to view line plot...")
        self.selected_symbols_multi.param.watch(
            lambda _: update_multi_stock_plot(), "value"
        )
        update_multi_stock_plot()
        tab_4 = pn.Column(pn.Row(self.selected_symbols_multi), multi_stock_plot)

        return [tab_1, tab_2, tab_3, tab_4]

    def build_dashboard(self):
        """
        Builds the complete dashboard by combining the introduction and insight tabs.

        Returns:
            pn.Tabs: A Panel Tabs layout containing the entire dashboard.
        """
        return pn.Tabs(
            ("Introduction", self.intro_tab),
            *[(f"Tab {i+1}", tab) for i, tab in enumerate(self.tabs)],
        )

    @property
    def dash(self):
        """
        Provides access to the complete dashboard layout.

        Returns:
            pn.Tabs: The dashboard layout.
        """
        return self.dashboard
