from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from io import BytesIO
import json
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

def make_demo_data():
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=48, freq="MS")
    t = np.arange(len(dates))
    seasonal = 180 * np.sin(2 * np.pi * (t % 12) / 12)
    trend = 18 * t
    promo = np.where(np.isin(dates.month, [3, 6, 11, 12]), 140, 0)
    sales = np.maximum(100, 1100 + trend + seasonal + promo + rng.normal(0, 65, len(dates)))
    return pd.DataFrame({"date": dates, "sales": np.round(sales, 2),
                         "promotion": (promo > 0).astype(int)})

def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or "sales" not in df.columns:
        raise ValueError("Your CSV must include columns named date and sales.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"]).sort_values("date")
    if len(df) < 12:
        raise ValueError("Please provide at least 12 valid historical rows for time-series forecasting.")
    df = df.groupby("date", as_index=False).agg({**{"sales":"sum"}, **({"promotion":"max"} if "promotion" in df.columns else {})})
    df = df.set_index("date").sort_index()
    # Infer daily/monthly frequency; fill missing dates to keep a regular time series.
    freq = "MS" if len(df) > 1 and (df.index.to_series().diff().dropna().median().days > 20) else "D"
    if freq == "MS":
        df.index = df.index.to_period("M").to_timestamp()
        df = df.groupby(level=0).sum(numeric_only=True).asfreq("MS")
    else:
        df = df.asfreq("D")
    df["sales"] = df["sales"].interpolate(limit_direction="both").clip(lower=0)
    if "promotion" not in df.columns:
        df["promotion"] = 0
    df["promotion"] = df["promotion"].fillna(0).astype(int)
    df.attrs["freq"] = freq
    return df

def fit_forecast(df, periods, model_name):
    y = df["sales"].astype(float)
    freq = df.attrs.get("freq", "MS")
    future_dates = pd.date_range(start=y.index[-1], periods=periods + 1, freq=("MS" if freq == "MS" else "D"))[1:]
    if model_name == "ARIMA":
        try:
            fit = ARIMA(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12) if freq == "MS" and len(y) >= 24 else (0,0,0,0)).fit()
            pred = np.asarray(fit.forecast(periods))
        except Exception:
            fit = ARIMA(y, order=(1, 1, 1)).fit()
            pred = np.asarray(fit.forecast(periods))
    elif model_name == "Prophet":
        # Lightweight trend + yearly seasonal approximation, avoiding an extra package.
        x = np.arange(len(y))
        coeff = np.polyfit(x, y.values, 1)
        fitted_trend = np.polyval(coeff, x)
        residual = y.values - fitted_trend
        if freq == "MS" and len(y) >= 24:
            seasonal_profile = pd.Series(residual, index=y.index).groupby(y.index.month).mean()
            future_x = np.arange(len(y), len(y)+periods)
            pred = np.polyval(coeff, future_x) + [seasonal_profile.get(d.month, 0) for d in future_dates]
        else:
            pred = np.polyval(coeff, np.arange(len(y), len(y)+periods))
    else:
        raise ValueError("Unsupported model. Choose ARIMA or Prophet.")
    pred = np.maximum(0, pred)
    return pd.DataFrame({"date": future_dates, "forecast": np.round(pred, 2)})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "SalesCast API"})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/demo")
def demo():
    df = make_demo_data()
    return jsonify({"data": [{"date": d.strftime("%Y-%m-%d"), "sales": float(s), "promotion": int(p)}
                            for d,s,p in zip(df.date, df.sales, df.promotion)]})

@app.route("/api/forecast", methods=["POST"])
def forecast():
    try:
        periods = int(request.form.get("periods", 6))
        model_name = request.form.get("model", "ARIMA")
        if not 1 <= periods <= 36:
            return jsonify({"error": "Forecast horizon must be between 1 and 36 periods."}), 400
        if "file" in request.files and request.files["file"].filename:
            df = pd.read_csv(request.files["file"])
        elif request.form.get("data"):
            df = pd.DataFrame(json.loads(request.form["data"]))
        else:
            df = make_demo_data()
        df = prepare_data(df)
        if model_name not in ("ARIMA", "Prophet"):
            model_name = "ARIMA"
        result = fit_forecast(df, periods, model_name)
        # Hold out the last 20% to report a simple validation score.
        split = max(10, int(len(df) * 0.8))
        if len(df) - split >= 2:
            train = df.iloc[:split].copy()
            train.attrs["freq"] = df.attrs.get("freq", "MS")
            try:
                val_pred = fit_forecast(train, len(df)-split, model_name)["forecast"].values
                actual = df["sales"].iloc[split:].values
                mae = float(mean_absolute_error(actual, val_pred))
                rmse = float(np.sqrt(mean_squared_error(actual, val_pred)))
            except Exception:
                mae = rmse = None
        else:
            mae = rmse = None
        history = [{"date": d.strftime("%Y-%m-%d"), "sales": round(float(s),2),
                    "promotion": int(p) if pd.notna(p) else 0}
                   for d,s,p in zip(df.index, df["sales"], df["promotion"])]
        future = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(v)}
                  for d,v in zip(result.date, result.forecast)]
        return jsonify({
            "history": history, "forecast": future, "model": model_name,
            "frequency": df.attrs.get("freq", "MS"),
            "mae": round(mae, 2) if mae is not None else None,
            "rmse": round(rmse, 2) if rmse is not None else None,
            "last_sales": round(float(df.sales.iloc[-1]), 2),
            "next_forecast": round(float(result.forecast.iloc[0]), 2),
            "total_forecast": round(float(result.forecast.sum()), 2),
            "promotions_in_data": int(df.promotion.sum())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/template.csv")
def template_csv():
    csv = "date,sales,promotion\n2025-01-01,1200,0\n2025-02-01,1250,0\n2025-03-01,1480,1\n2025-04-01,1300,0\n2025-05-01,1360,0\n2025-06-01,1540,1\n2025-07-01,1400,0\n2025-08-01,1450,0\n2025-09-01,1490,0\n2025-10-01,1530,0\n2025-11-01,1780,1\n2025-12-01,1950,1\n"
    return app.response_class(csv, mimetype="text/csv", headers={"Content-Disposition":"attachment; filename=sales_template.csv"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
