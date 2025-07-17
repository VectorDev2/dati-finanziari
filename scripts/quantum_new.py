import yfinance as yf
import pandas as pd
import numpy as np
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Ritorni
    df['Return'] = df['Close'].pct_change()

    # 2. RSI (14 giorni)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD e segnale
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bande di Bollinger (20 giorni)
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20

    # 5. Volatilità su 5 giorni
    df['Volatility'] = df['Return'].rolling(window=5).std()

    # 6. Volume (normalizzato)
    df['Volume_Norm'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()

    # 7. Momentum (5 giorni)
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)

    # 8. Media mobile 50 e 200 giorni
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 9. Distanza dalla media mobile 200
    df['Close_SMA_200_diff'] = df['Close'] - df['SMA_200']

    return df


def predict_with_random_forest(ticker: str,
                               start_date: str = "2020-01-01",
                               end_date: str = None):
    df = yf.download(ticker, start=start_date, end=end_date)
    if len(df) < 60:
        raise ValueError("Troppi pochi dati per le feature tecniche.")

    df = compute_technical_indicators(df).dropna()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    feature_cols = ['Return', 'RSI', 'MACD', 'MACD_Signal',
                    'BB_upper', 'BB_lower', 'Volatility',
                    'Volume_Norm', 'Momentum_5',
                    'SMA_50', 'SMA_200', 'Close_SMA_200_diff']
    X = df[feature_cols].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Usa SMOTE per bilanciare il training set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)

    # Grid Search per iperparametri
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    grid = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=0
    )
    grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_

    print(f"\n🏆 Miglior modello trovato: {grid.best_params_}")
    print(f"→ Cross-validation F1-macro: {grid.best_score_:.3f}")

    # Valutazione su test set
    y_pred = best_model.predict(X_test_s)
    print("\n📊 Classification Report su test set:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    print("🔢 Matrice di confusione:")
    print(confusion_matrix(y_test, y_pred))

    # Probabilità di crescita per l’ultimo giorno
    X_latest = scaler.transform(X[-1].reshape(1, -1))
    prob_up = best_model.predict_proba(X_latest)[0, 1]
    print(f"\n📈 Probabilità che {ticker} salga domani: {prob_up * 100:.2f}%")
    print("→ Previsione:", "crescita 📈" if prob_up > 0.5 else "decrescita 📉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    predict_with_random_forest(args.ticker,
                               start_date=args.start_date,
                               end_date=args.end_date)
