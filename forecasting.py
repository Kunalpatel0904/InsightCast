"""forecasting.py — revenue time-series forecasting"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def build_monthly_revenue(df: pd.DataFrame) -> pd.Series:
    df2 = df.copy()
    df2["order_month"] = df2["order_purchase_timestamp"].dt.to_period("M")
    monthly = df2.groupby("order_month")["total_order_value"].sum()
    monthly.index = monthly.index.to_timestamp()
    return monthly.sort_index()


def forecast_revenue(monthly: pd.Series, periods: int = 6):
    """
    Simple trend + seasonality decomposition forecast.
    Returns (forecast_index, forecast_values, lower_bound, upper_bound).
    """
    y = monthly.values.astype(float)
    n = len(y)

    # Linear trend via least-squares
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)

    # Detrend & compute seasonal indices (monthly cycle)
    trend    = slope * x + intercept
    detrend  = y - trend
    # Build seasonal factors per month-of-year
    months   = monthly.index.month
    seas_avg = {m: detrend[months == m].mean() for m in range(1, 13)}

    # Forecast
    last_period = monthly.index[-1]
    fut_dates   = pd.date_range(last_period + pd.DateOffset(months=1),
                                periods=periods, freq="MS")
    fut_x       = np.arange(n, n + periods)
    fut_trend   = slope * fut_x + intercept
    fut_seas    = np.array([seas_avg.get(d.month, 0) for d in fut_dates])
    forecast    = fut_trend + fut_seas
    forecast    = np.clip(forecast, 0, None)

    # Confidence bands (±1 std of residuals)
    residuals   = y - (trend + np.array([seas_avg.get(m, 0) for m in months]))
    sigma       = residuals.std()
    lower       = np.clip(forecast - 1.5 * sigma, 0, None)
    upper       = forecast + 1.5 * sigma

    mape = np.mean(np.abs(residuals / np.clip(y, 1, None))) * 100

    return fut_dates, forecast, lower, upper, round(mape, 2)
