import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from xgboost import XGBClassifier

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Ritorni
    df['Return'] = df['Close'].pct_change()

    # 2. RSI (14 giorni)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
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
    ma20  = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20

    # 5. Volatilità su 5 giorni
    df['Volatility'] = df['Return'].rolling(window=5).std()

    # 6. Nuove feature
    df['Momentum_3']      = df['Close'] / df['Close'].shift(3) - 1
    df['Momentum_7']      = df['Close'] / df['Close'].shift(7) - 1
    df['SMA_ratio']       = df['Close'] / ma20
    df['BB_width']        = df['BB_upper'] - df['BB_lower']
    # **** FIX: assicuriamoci di ottenere una Series scalare, non DataFrame ****
    price_pos = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['Price_Position'] = price_pos.astype(float)

    return df

def predict_with_model(ticker: str, start_date: str = "2020-01-01", end_date: str = None):
    # scarica con auto_adjust per chiudere il warning
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if len(df) < 60:
        raise ValueError("Troppi pochi dati per le feature tecniche.")

    # calcola indicatori
    df = compute_technical_indicators(df).dropna()

    # rimuove i giorni "piatti" per ridurre rumore
    df = df[df['Return'].abs() > 0.002]

    # definisci target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    feature_cols = [
        'Return', 'RSI', 'MACD', 'MACD_Signal',
        'BB_upper', 'BB_lower', 'Volatility',
        'Momentum_3', 'Momentum_7', 'SMA_ratio',
        'BB_width', 'Price_Position'
    ]

    X = df[feature_cols].values
    y = df['Target'].values

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # standardizzazione
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # XGBoost con default bilanciamento
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # cross-validation
    cv_bal = cross_val_score(model, X_train_s, y_train,
                             cv=5, scoring='balanced_accuracy')
    cv_f1  = cross_val_score(model, X_train_s, y_train,
                             cv=5, scoring=make_scorer(f1_score))

    print(f"Balanced Accuracy (CV): {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")
    print(f"F1 Score (CV): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # allenamento e valutazione
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    print("\nClassification Report su test set:")
    print(classification_report(y_test, y_pred, target_names=['Down','Up']))
    print("Matrice di confusione:")
    print(confusion_matrix(y_test, y_pred))

    # previsione ultimo giorno
    X_latest = scaler.transform(X[-1].reshape(1, -1))
    prob_up = model.predict_proba(X_latest)[0,1]
    print(f"\nProbabilità che {ticker} salga domani: {prob_up*100:.2f}%")
    print("→ Previsione:", "crescita" if prob_up>0.5 else "decrescita")

    # importanza feature
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    feats = np.array(feature_cols)[idx]

    plt.figure(figsize=(10,5))
    plt.bar(feats, importances[idx])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\n📊 Grafico salvato come feature_importance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    predict_with_model(args.ticker,
                       start_date=args.start_date,
                       end_date=args.end_date)
