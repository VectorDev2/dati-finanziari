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
    df['Return'] = df['Close'].pct_change()

    # RSI
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs       = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12             = df['Close'].ewm(span=12, adjust=False).mean()
    ema26             = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    ma20   = df['Close'].rolling(20).mean()
    std20  = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20

    # Volatility
    df['Volatility'] = df['Return'].rolling(5).std()

    # Nuove feature
    df['Momentum_3'] = df['Close'].pct_change(3)
    df['Momentum_7'] = df['Close'].pct_change(7)
    df['SMA_ratio']  = df['Close'] / ma20
    df['BB_width']   = df['BB_upper'] - df['BB_lower']

    return df

def predict_with_random_forest(ticker: str,
                               start_date: str = "2020-01-01",
                               end_date: str = None):
    # 1. Download dati con auto_adjust per evitare il warning
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.shape[0] < 60:
        raise ValueError("Troppi pochi dati per addestrare il modello.")

    # 2. Calcola indicatori
    df = compute_technical_indicators(df)

    # 3. Rimuovi tutte le righe con almeno un NaN nelle feature calcolate
    feature_base = [
        'Return', 'RSI', 'MACD', 'MACD_Signal',
        'BB_upper', 'BB_lower', 'Volatility',
        'Momentum_3', 'Momentum_7', 'SMA_ratio', 'BB_width'
    ]
    df.dropna(subset=feature_base, inplace=True)

    # 4. Ora che BB_upper e BB_lower sono pulite, calcola Price_Position
    df['Price_Position'] = (
        (df['Close'] - df['BB_lower']) /
        (df['BB_upper'] - df['BB_lower'])
    )

    # 5. Filtra i giorni con ritorno ≤ 0.2% per ridurre rumore
    df = df[df['Return'].abs() > 0.002]

    # 6. Costruisci target (shift dopo dropna)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(subset=['Target'], inplace=True)  # rimuove l'ultima riga senza target

    # 7. Prepara feature e label
    feature_cols = feature_base + ['Price_Position']
    X = df[feature_cols].values
    y = df['Target'].values

    # 8. Split train/test (80/20, no shuffle)
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

    # 11. Cross-validation con metriche bilanciate
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

    # 12. Allenamento finale e valutazione test set
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=['Down','Up']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 13. Previsione per l'ultimo giorno disponibile
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
