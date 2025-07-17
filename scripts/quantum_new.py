import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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

    return df

def predict_with_random_forest(ticker: str,
                               start_date: str = "2020-01-01",
                               end_date: str = None):
    # Scarica dati
    df = yf.download(ticker, start=start_date, end=end_date)
    if len(df) < 60:
        raise ValueError("Troppi pochi dati per le feature tecniche.")

    # Calcola indicatori
    df = compute_technical_indicators(df).dropna()

    # Crea target: 1 se close_next > close, 0 altrimenti
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    # Seleziona feature
    feature_cols = ['Return', 'RSI', 'MACD', 'MACD_Signal',
                    'BB_upper', 'BB_lower', 'Volatility']
    X = df[feature_cols].values
    y = df['Target'].values

    # Split train / test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Standardizza
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation su train set
    cv_scores = cross_val_score(model, X_train_s, y_train,
                                cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Allena su tutto il train
    model.fit(X_train_s, y_train)

    # Valutazione su test
    y_pred = model.predict(X_test_s)
    print("\nClassification Report su test set:")
    print(classification_report(y_test, y_pred, target_names=['Down','Up']))

    print("Matrice di confusione:")
    print(confusion_matrix(y_test, y_pred))

    # Probabilità di crescita per l’ultimo giorno disponibile
    X_latest = scaler.transform(X[-1].reshape(1, -1))
    prob_up = model.predict_proba(X_latest)[0, 1]
    print(f"\nProbabilità che {ticker} salga domani: {prob_up*100:.2f}%")
    print("→ Previsione:", "crescita" if prob_up > 0.5 else "decrescita")

if __name__ == "__main__":
    ticker = input("Ticker (es. AAPL): ").upper().strip()
    predict_with_random_forest(ticker,
                               start_date="2020-01-01")
