"""
Rolling-origin (expanding-window) evaluation.

Answers Reviewer 1, comment 2 -- "use k-fold or repeated cross-validation to
improve reliability and reduce variance in performance estimates" -- without
contradicting Reviewer 1, comment 3, which asks for chronological splitting.

Plain k-fold would shuffle 1993-2020 randomly and let the model interpolate
between neighbouring years, inflating the scores and undermining the temporal
argument. Expanding-window validation instead trains on all data up to a cut
point and tests on the block that follows, repeatedly. That yields the mean
and standard deviation the reviewer wants while every fold still predicts
forward in time.

Scaling is fitted inside each window, on that window's training rows only.

Usage
-----
from rolling_evaluation import rolling_origin_evaluation
summary = rolling_origin_evaluation(df, selected, TARGET)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import xgboost as xgb

from config import RANDOM_SEED, ROLLING_SPLITS, DNN_SEEDS


# ---------------------------------------------------------------------------
# Model factories. Params match the search spaces in models.py; adjust to the
# best_params_ printed by the main run if you prefer to freeze them.
# ---------------------------------------------------------------------------

def _factories():
    return {
        "Linear Regression": (lambda: LinearRegression(), True),
        "Random Forest": (
            lambda: RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_SEED, n_jobs=1),
            True),
        "XGBoost": (
            lambda: xgb.XGBRegressor(
                objective="reg:squarederror", n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_SEED, verbosity=0, n_jobs=1),
            True),
        "SVR": (lambda: SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1), True),
    }


def _score(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def _fit_dnn_fold(X_tr, y_tr, X_te, seed):
    """Single DNN fit matching the architecture in models.py."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_tr.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(), Dropout(0.2),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    model.fit(
        X_tr, y_tr, validation_split=0.15, epochs=200, batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                              min_lr=1e-6),
        ],
        verbose=0,
    )
    return model.predict(X_te, verbose=0).flatten()


# ---------------------------------------------------------------------------

def rolling_origin_evaluation(
    df: pd.DataFrame,
    selected: list,
    target: str,
    n_splits: int = ROLLING_SPLITS,
    include_dnn: bool = True,
    dnn_seeds: list = None,
    out_csv: str = "results_rolling_origin.csv",
) -> pd.DataFrame:
    """
    Expanding-window evaluation over chronologically ordered rows.

    `df` must already be sorted by year (load_and_clean does this) and must
    contain `selected` plus `target`.
    """
    df = df.reset_index(drop=True)
    X_all = df[selected].to_numpy(dtype=float)
    y_all = df[target].to_numpy(dtype=float)

    # Year-boundary windows: cut on whole years, not row positions, so no
    # single year (and therefore no flood event) appears in both train and test.
    years = df["year"].to_numpy()
    uniq = np.sort(np.unique(years))
    # choose n_splits evenly spaced cut years across the second half of the record
    cut_years = uniq[len(uniq) // 3:]  # start after the first third
    edges = np.linspace(0, len(cut_years) - 1, n_splits + 1).astype(int)
    cuts = [int(cut_years[e]) for e in edges]

    folds = []
    for k in range(n_splits):
        tr_end, te_end = cuts[k], cuts[k + 1]
        tr = np.where(years <= tr_end)[0]
        te = np.where((years > tr_end) & (years <= te_end))[0]
        if len(tr) >= 50 and len(te) >= 10:
            folds.append((tr, te))

    print(f"\nRolling-origin evaluation: {n_splits} expanding windows")
    if "year" in df.columns:
        for i, (tr, te) in enumerate(folds, 1):
            print(f"  window {i}: train {len(tr):>5} rows "
                  f"(<= {int(df['year'].iloc[tr[-1]])}) | "
                  f"test {len(te):>5} rows "
                  f"({int(df['year'].iloc[te[0]])}-{int(df['year'].iloc[te[-1]])})")

    per_fold = {name: [] for name in _factories()}
    if include_dnn:
        per_fold["DNN"] = []

    for tr, te in folds:
        scaler = MinMaxScaler().fit(X_all[tr])          # train rows only
        X_tr, X_te = scaler.transform(X_all[tr]), scaler.transform(X_all[te])
        y_tr, y_te = y_all[tr], y_all[te]

        for name, (factory, _) in _factories().items():
            model = factory()
            model.fit(X_tr, y_tr)
            per_fold[name].append(_score(y_te, model.predict(X_te)))

        if include_dnn:
            seeds = list(dnn_seeds or DNN_SEEDS)[:3]   # 3 seeds per window
            preds = np.mean(
                [_fit_dnn_fold(X_tr, y_tr, X_te, s) for s in seeds], axis=0)
            per_fold["DNN"].append(_score(y_te, preds))

    rows = []
    for name, scores in per_fold.items():
        row = {"Model": name, "n_windows": len(scores)}
        for metric in ("rmse", "mae", "r2"):
            vals = np.array([s[metric] for s in scores])
            row[f"{metric.upper()}_mean"] = vals.mean()
            row[f"{metric.upper()}_std"] = vals.std(ddof=1)
        rows.append(row)

    summary = (pd.DataFrame(rows)
               .sort_values("RMSE_mean")
               .reset_index(drop=True))

    print("\nRolling-origin results (mean +/- std across windows)")
    for _, r in summary.iterrows():
        print(f"  {r['Model']:<18} "
              f"RMSE {r['RMSE_mean']:.3f} +/- {r['RMSE_std']:.3f}   "
              f"MAE {r['MAE_mean']:.3f} +/- {r['MAE_std']:.3f}   "
              f"R2 {r['R2_mean']:.3f} +/- {r['R2_std']:.3f}")

    summary.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Does the XGBoost / Random Forest ordering survive fold-to-fold noise?
    if {"XGBoost", "Random Forest"} <= set(summary["Model"]):
        xg = summary.set_index("Model").loc["XGBoost"]
        rf = summary.set_index("Model").loc["Random Forest"]
        gap = abs(xg["RMSE_mean"] - rf["RMSE_mean"])
        noise = max(xg["RMSE_std"], rf["RMSE_std"])
        print(f"\nXGBoost vs Random Forest: RMSE gap {gap:.4f}, "
              f"window-to-window std {noise:.4f}")
        if gap < noise:
            print("  -> gap is smaller than fold-to-fold variation. Section 5.3\n"
                  "     should not claim boosting beats bagging on this dataset;\n"
                  "     report them as statistically indistinguishable.")
        else:
            print("  -> gap exceeds fold-to-fold variation; the ordering holds.")

    return summary
