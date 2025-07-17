import yfinance as yf
import pandas as pd
import numpy as np
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score,
)

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Ritorni giornalieri
    df['Return'] = df['Close'].pct_change()

    # 2. RSI (14 giorni)
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD e segnale
    ema12             = df['Close'].ewm(span=12, adjust=False).mean()
    ema26             = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bande di Bollinger (20 giorni)
    ma20   = df['Close'].rolling(window=20).mean()
    std20  = df['Close'].rolling(window=20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20

    # 5. Volatilità (5 giorni)
    df['Volatility'] = df['Return'].rolling(window=5).std()

    # 6. Nuove feature
    df['Momentum_3'] = df['Close'].pct_change(periods=3)
    df['Momentum_7'] = df['Close'].pct_change(periods=7)
    df['SMA_ratio']  = df['Close'] / ma20
    df['BB_width']   = df['BB_upper'] - df['BB_lower']

    return df

def predict_with_random_forest(ticker: str,
                               start_date: str = "2020-01-01",
                               end_date: str = None):
    # 1. Scarica dati con auto_adjust per evitare warning
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.shape[0] < 60:
        raise ValueError("Troppi pochi dati per addestrare il modello.")

    # 2. Calcola indicatori
    df = compute_technical_indicators(df)

    # 3. Elimina le righe che ancora hanno NaN nelle FEATURE essenziali
    feature_cols = [
        'Return', 'RSI', 'MACD', 'MACD_Signal',
        'BB_upper', 'BB_lower', 'Volatility',
        'Momentum_3', 'Momentum_7', 'SMA_ratio',
        'BB_width'
    ]
    df.dropna(subset=feature_cols, inplace=True)

    # 4. Filtra i giorni con ritorni troppo bassi (rumore)
    df = df[df['Return'].abs() > 0.002]

    # 5. Definisci target (must essere dopo il dropna su feature)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(subset=['Target'], inplace=True)  # elimina l'ultima riga

    # 6. Ricostruisci la feature Price_Position ORA che tutti i NaN di Bollinger sono rimossi
    df['Price_Position'] = (
        (df['Close'] - df['BB_lower']) /
        (df['BB_upper'] - df['BB_lower'])
    )

    # 7. Preparazione X, y
    feature_cols.append('Price_Position')
    X = df[feature_cols].values
    y = df['Target'].values

    # 8. Train/test split (80/20, senza shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 9. Standardizzazione
    scaler   = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 10. Random Forest bilanciato
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    # 11. Cross-validation bilanciata
    cv_bal = cross_val_score(
        model, X_train_s, y_train,
        cv=5, scoring='balanced_accuracy'
    )
    cv_f1 = cross_val_score(
        model, X_train_s, y_train,
        cv=5, scoring=make_scorer(f1_score)
    )
    print(f"Balanced Accuracy (CV): {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")
    print(f"F1 Score (CV):              {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # 12. Allenamento e valutazione finale
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=['Down','Up']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 13. Previsione ultimo giorno
    latest  = scaler.transform(X[-1].reshape(1, -1))
    prob_up = model.predict_proba(latest)[0, 1]
    print(f"\nProbabilità che {ticker} salga domani: {prob_up*100:.2f}%")
    print("→ Previsione:", "crescita" if prob_up > 0.5 else "decrescita")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",     type=str, required=True)
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--end-date",   type=str, default=None)
    args = parser.parse_args()

    predict_with_random_forest(
        args.ticker,
        start_date=args.start_date,
        end_date=args.end_date
    )
