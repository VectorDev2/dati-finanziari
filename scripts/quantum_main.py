import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend      import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands

from sklearn.decomposition        import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.svm                  import SVC
from sklearn.model_selection      import StratifiedKFold
from sklearn.metrics              import accuracy_score

import pennylane as qml


# ────────────────────────────────────────────────────────────────
# 1) FETCH + FEATURE ENGINEERING (con .squeeze() su ogni indicatore)
# ────────────────────────────────────────────────────────────────
def fetch_features(symbol):
    data = yf.download(symbol, period="5y", interval="1d", auto_adjust=False).dropna()
    rp = yf.Ticker(symbol).info.get("regularMarketPrice")
    if rp is not None:
        data.at[data.index[-1], "Close"] = rp

    close = pd.Series(data["Close"].values.flatten(), index=data.index)
    high  = pd.Series(data["High"].values.flatten(), index=data.index)
    low   = pd.Series(data["Low"].values.flatten(), index=data.index)
    vol   = pd.Series(data["Volume"].values.flatten(), index=data.index)
    
    pct   = close.pct_change().squeeze()
    ema10 = (EMAIndicator(close, window=10).ema_indicator().squeeze() - close) / close
    rsi   = RSIIndicator(close).rsi().squeeze() / 100.0
    macd  = MACD(close).macd_diff().squeeze()
    stoch = StochasticOscillator(high, low, close).stoch().squeeze() / 100.0
    cci   = CCIIndicator(high, low, close).cci().squeeze() / 200.0
    willr = -WilliamsRIndicator(high, low, close).williams_r().squeeze() / 100.0
    bbw   = BollingerBands(close).bollinger_wband().squeeze() / close
    atr   = (high - low).rolling(14).mean().squeeze() / close
    voln  = ((vol - vol.mean()) / vol.std()).squeeze()

    df = pd.DataFrame({
        "pct":   pct,
        "ema10": ema10,
        "rsi":   rsi,
        "macd":  macd,
        "stoch": stoch,
        "cci":   cci,
        "willr": willr,
        "bbw":   bbw,
        "atr":   atr,
        "voln":  voln,
    }).dropna()
    return df


# ────────────────────────────────────────────────────────────────
# 2) PCA → fino a 5 COMPONENTI (con controlli di shape)
# ────────────────────────────────────────────────────────────────
def apply_pca(X, n_components=5):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"apply_pca: array non 2-D (shape {X.shape})")
    max_comp = min(n_components, X.shape[0], X.shape[1])
    if max_comp < 1:
        raise ValueError("apply_pca: non ci sono abbastanza dati per costruire nemmeno 1 componente")
    pca = PCA(n_components=max_comp, random_state=42)
    return pca.fit_transform(X), pca


# ────────────────────────────────────────────────────────────────
# 3) QUANTUM KERNEL
# ────────────────────────────────────────────────────────────────

def make_quantum_kernel(wires):
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def quantum_circuit(x):
        for i, v in enumerate(x):
            qml.RY(v, wires=i)
        return qml.probs(wires=range(wires))  # ritorna distribuzione di probabilità

    def qkernel(x, y):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        probs_x = quantum_circuit(x)
        probs_y = quantum_circuit(y)
        return float(np.dot(probs_x, probs_y))  # similarità tra le distribuzioni

    return qkernel


# ────────────────────────────────────────────────────────────────
# 4) TRAIN ENSEMBLE CON KERNEL (CV solo se n_samples ≥ 3)
# ────────────────────────────────────────────────────────────────
def train_ensemble(X, y, kernels, n_landmarks=200):
    n_samples = X.shape[0]
    maps = [Nystroem(kernel=k, n_components=n_landmarks, random_state=42)
            for k in kernels]

    if n_samples < 3:
        print(f"Solo {n_samples} campioni: nessuna CV, fit diretto")
        fitted_maps, fitted_svcs = [], []
        for kmap in maps:
            feat  = kmap.fit_transform(X)
            model = SVC(kernel="linear", random_state=42)
            model.fit(feat, y)
            fitted_maps.append(kmap)
            fitted_svcs.append(model)
        return fitted_maps, fitted_svcs

    kf   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in kf.split(X, y):
        preds = np.zeros((len(test_idx), len(kernels)))
        for i, kmap in enumerate(maps):
            X_tr = kmap.fit_transform(X[train_idx])
            X_te = kmap.transform(X[test_idx])
            model = SVC(kernel="linear", random_state=42)
            model.fit(X_tr, y[train_idx])
            preds[:, i] = model.predict(X_te)
        maj_vote = (preds.sum(axis=1) >= len(kernels)/2).astype(int)
        accs.append(accuracy_score(y[test_idx], maj_vote))

    print(f"3-fold CV ensemble acc: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

    fitted_maps, fitted_svcs = [], []
    for kmap in maps:
        feat  = kmap.fit_transform(X)
        model = SVC(kernel="linear", random_state=42)
        model.fit(feat, y)
        fitted_maps.append(kmap)
        fitted_svcs.append(model)
    return fitted_maps, fitted_svcs


# ────────────────────────────────────────────────────────────────
# 5) MAIN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assets = os.getenv("ASSETS")
    if not assets:
        print("Errore: definisci ASSETS")
        sys.exit(1)

    window = 5
    for symbol in assets.split(","):
        symbol = symbol.strip().upper()
        print(f"\n🔍 Analisi per: {symbol}")

        # 1) fetch delle feature
        df = fetch_features(symbol)
        vals = df.values

        # 2) costruzione di X_list e y_list
        X_list, y_list = [], []
        for i in range(window, len(vals) - 1):
            X_list.append(vals[i - window : i].flatten())
            y_list.append(int(vals[i + 1, 0] > 0))

        # 3) controllo numero di campioni
        if len(X_list) < 2:
            print(f"Skip {symbol}: campioni insufficienti ({len(X_list)}) per PCA/allenamento")
            continue

        # 4) trasformo in array 2D
        X_all = np.vstack(X_list)
        y_all = np.array(y_list)

        # 5) PCA con fallback
        try:
            X_pca, pca = apply_pca(X_all, n_components=5)
        except ValueError as e:
            print(f"Impossibile PCA per {symbol} ({e}), uso X_all raw")
            X_pca, pca = X_all, None

        # 6) preparo i quantum-kernels
        k1 = make_quantum_kernel(wires=5)
        k2 = make_quantum_kernel(wires=5)

        # 7) allena ensemble
        t0        = time.time()
        maps, svcs = train_ensemble(
            X_pca, y_all,
            kernels     =[k1, k2],
            n_landmarks =200,
        )
        print(f"Training totale: {time.time() - t0:.1f}s")

        # 8) inference ultimo giorno
        last       = vals[-window:].flatten().reshape(1, -1)
        last_feats = pca.transform(last) if pca is not None else last
        votes      = [
            svc.predict(m.transform(last_feats))[0]
            for m, svc in zip(maps, svcs)
        ]
        pred = int(np.sum(votes) >= len(votes)/2)
        print(f"Previsione {symbol}: {'Rialzo' if pred==1 else 'Ribasso'}")
        
        
        

'''import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands

import numpy as np
import pennylane as qml
import time

# ────────────────────────────────────────────────────────────────
# Funzioni di utilità
# ────────────────────────────────────────────────────────────────

def fetch_and_prepare_data_all_days(symbol):
    symbol = symbol.upper()
    # ultimi 5 anni
    data = yf.download(symbol, period="5y", interval="1d", auto_adjust=False)
    if data.empty:
        raise ValueError(f"Nessun dato disponibile per {symbol}.")
    data.dropna(inplace=True)

    info = yf.Ticker(symbol).info
    rp = info.get("regularMarketPrice", None)
    if rp is not None:
        data.at[data.index[-1], "Close"] = rp

    close, high, low = data["Close"].squeeze(), data["High"].squeeze(), data["Low"].squeeze()
    open_, volume    = data["Open"].squeeze(), data["Volume"].squeeze()

    ema10     = EMAIndicator(close, window=10).ema_indicator().squeeze()
    rsi       = RSIIndicator(close).rsi().squeeze()
    macd_obj  = MACD(close)
    macd_line = macd_obj.macd().squeeze()
    macd_sig  = macd_obj.macd_signal().squeeze()
    stoch     = StochasticOscillator(high, low, close)
    stoch_k   = stoch.stoch().squeeze()
    stoch_d   = stoch.stoch_signal().squeeze()
    cci       = CCIIndicator(high, low, close).cci().squeeze()
    willr     = WilliamsRIndicator(high, low, close).williams_r().squeeze()
    bb        = BollingerBands(close)
    bb_up     = bb.bollinger_hband().squeeze()
    bb_w      = bb.bollinger_wband().squeeze()

    vol_mean = volume.mean()
    bbw_mean = bb_w.mean()

    df = pd.DataFrame({
        "f1":  (close > open_).astype(int),
        "f2":  (volume > vol_mean).astype(int),
        "f3":  (ema10 > close).astype(int),
        "f4":  (rsi > 50).astype(int),
        "f5":  (macd_line > macd_sig).astype(int),
        "f6":  (stoch_k > stoch_d).astype(int),
        "f7":  (cci > 0).astype(int),
        "f8":  (willr > -50).astype(int),
        "f9":  (close > bb_up).astype(int),
        "f10": (bb_w > bbw_mean).astype(int),
    }, index=data.index)
    df.dropna(inplace=True)
    return df

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize(x):
    mu, sigma = np.mean(x), np.std(x)
    return (x - mu) / sigma

def encode_qubit(x):
    return np.pi * x  # x in [-1,1]

# ────────────────────────────────────────────────────────────────
# Modello ibrido quantistico-classico
# ────────────────────────────────────────────────────────────────

class QuantumSimModel:
    def __init__(
        self,
        n_features,
        hidden_size=10,
        lr=0.01,
        reg=1e-4,
        batch_size=32,
        epochs=30,
        patience=7,
        tol=1e-3,
        n_rotations=2,
        window=3
    ):
        self.n = n_features
        self.k = n_rotations
        self.window = window
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.tol = tol

        # Backend quantistico C++
        self.dev = qml.device("lightning.qubit", wires=self.n)

        # Parametri quantistici: una matrice (layers, wires)
        self.thetas = np.random.uniform(0, 2*np.pi, (self.k, self.n))

        # MLP classica
        self.W1 = np.random.randn(hidden_size, self.n) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size) * 0.1
        self.b2 = 0.0

        # Stato Adam
        self.m = {p: 0 for p in ["W1","b1","W2","b2"]}
        self.v = {p: 0 for p in ["W1","b1","W2","b2"]}
        self.beta1, self.beta2 = 0.9, 0.999
        self.epsilon = 1e-8
        self.iteration = 0

        # QNode + jacobian (parameter-shift)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
        self.grad_qnode = qml.jacobian(self.qnode, argnum=1)

    def _circuit(self, x, thetas):
        # Encoding dati
        for i, v in enumerate(x):
            qml.RY(encode_qubit(v), wires=i)

        # BasicEntanglerLayers: rotazioni + CNOT in un template
        qml.templates.BasicEntanglerLayers(weights=thetas, wires=range(self.n))

        # Misure e stack
        meas = [qml.expval(qml.PauliZ(i)) for i in range(self.n)]
        return qml.math.stack(meas, axis=0)

    def _simulate(self, x):
        return np.array(self.qnode(x, self.thetas))

    def _forward(self, p):
        z1 = self.W1 @ p + self.b1
        a1 = relu(z1)
        z2 = self.W2 @ a1 + self.b2
        return sigmoid(z2), a1

    def _loss(self, y, yhat):
        return - (y * np.log(yhat+1e-9) + (1-y)*np.log(1-yhat+1e-9))

    def _adam_step(self, name, grad):
        self.iteration += 1
        m = self.beta1*self.m[name] + (1-self.beta1)*grad
        v = self.beta2*self.v[name] + (1-self.beta2)*(grad**2)
        m_hat = m/(1-self.beta1**self.iteration)
        v_hat = v/(1-self.beta2**self.iteration)
        update = self.lr * m_hat/(np.sqrt(v_hat)+self.epsilon)
        self.m[name], self.v[name] = m, v
        return update

    def fit(self, X, y):
        X = normalize(np.array(X))
        y = np.array(y)
        best, wait = float("inf"), 0
        t0 = time.time()

        for ep in range(1, self.epochs + 1):
            idx = np.random.permutation(len(X))
            total_loss = 0.0

            for start in range(0, len(X), self.batch_size):
                batch = idx[start:start + self.batch_size]
                gW1 = np.zeros_like(self.W1)
                gb1 = np.zeros_like(self.b1)
                gW2 = np.zeros_like(self.W2)
                gb2 = 0.0
                gT  = np.zeros_like(self.thetas)

                for i in batch:
                    xi, yi = X[i], y[i]
                    # simulazione + forward
                    p = self._simulate(xi)
                    out, a1 = self._forward(p)
                    loss = self._loss(yi, out)
                    total_loss += loss

                    # gradiente MLP
                    dL_do = -(yi/(out+1e-9)) + ((1-yi)/(1-out+1e-9))
                    d_out = out * (1 - out)
                    d2    = dL_do * d_out

                    gW2 += d2 * a1
                    gb2 += d2
                    d1 = (self.W2 * d2) * (a1 > 0)
                    gW1 += np.outer(d1, p)
                    gb1 += d1

                    # **gradiente quantistico corretto**
                    dL_dp = d1 @ self.W1                         # shape (n,)
                    grad_q = self.grad_qnode(xi, self.thetas)    # shape (n, layers, wires)
                    # sommiamo lungo il primo asse di entrambi
                    gT   += np.tensordot(dL_dp, grad_q, axes=([0], [0]))  # result shape (layers, wires)

                # regolarizzazione
                gW1 += self.reg * self.W1
                gW2 += self.reg * self.W2
                gT  += self.reg * self.thetas

                # aggiornamento parametri
                self.W1     -= self._adam_step("W1", gW1 / len(batch))
                self.b1     -= self._adam_step("b1", gb1 / len(batch))
                self.W2     -= self._adam_step("W2", gW2 / len(batch))
                self.b2     -= self._adam_step("b2", gb2 / len(batch))
                self.thetas -= self.lr * (gT / len(batch))

            avg = total_loss / len(X)
            print(f"Epoch {ep}/{self.epochs} — Loss: {avg:.5f}")

            if avg + self.tol < best:
                best, wait = avg, 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {ep}")
                    break

        print(f"Training completed in {time.time() - t0:.1f}s")

    def predict_proba(self, data):
        data = normalize(np.array(data))
        x_w = data[-self.window:]
        return self._forward(self._simulate(x_w))[0]

    def predict(self, data):
        return int(self.predict_proba(data) >= 0.5)


# ────────────────────────────────────────────────────────────────
# Script principale
# ────────────────────────────────────────────────────────────────

import os
import sys

if __name__ == "__main__":
    assets_env = os.getenv("ASSETS")
    if not assets_env:
        print("Errore: nessun asset fornito. Definisci la variabile ASSETS.")
        sys.exit(1)

    symbols = [s.strip().upper() for s in assets_env.split(",")]

    for symbol in symbols:
        print(f"\n🔍 Analisi per: {symbol}")
        try:
            df = fetch_and_prepare_data_all_days(symbol)
            df = df[["f1", "f2", "f4", "f5", "f10"]]
            vals = df.values

            window = 3
            X, y = [], []
            for i in range(window, len(vals)-1):
                win = vals[i-window:i]
                X.append(win.flatten())
                y.append(int(vals[i][0]))

            model = QuantumSimModel(n_features=window * vals.shape[1])
            model.fit(X, y)

            last = vals[-window:]
            inp = last.flatten()
            proba = model.predict_proba(inp)
            pred = model.predict(inp)

            print(f"Probabilità rialzo per {symbol}: {proba:.3f}")
            print(f"Previsione {symbol}: {'Rialzo' if pred else 'Ribasso'}")

        except Exception as e:
            print(f"Errore per {symbol}: {e}")'''
