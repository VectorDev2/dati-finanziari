from github import Github, GithubException
import re
import feedparser
import os
from datetime import datetime, timedelta
import math
import spacy
#Librerie per ottenere dati storici e calcolare indicatori
import yfinance as yf
import ta
import pandas as pd
import random
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
from urllib.parse import quote_plus
from collections import defaultdict
# ————————————————————————————————
# import per il nuovo modello
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# carica il modello e il tokenizer una sola volta
MODEL_NAME = "NCFM/fin-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# mappa id→punteggio numerico
ID_TO_SCORE = {0: -1, 1: 0, 2: 1}
# ————————————————————————————————


# Carica il modello linguistico per l'inglese
nlp = spacy.load("en_core_web_sm")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "VecorDEV/dati-finanziari"

# Salva il file HTML nella cartella 'results'
file_path = "results/classifica.html"
news_path = "results/news.html"
    
# Salva il file su GitHub
github = Github(GITHUB_TOKEN)
repo = github.get_repo(REPO_NAME)

'''
        
        
        '''

# Lista dei simboli azionari da cercare
symbol_list = ["AAPL", "MSFT"]  # Puoi aggiungere altri simboli

'''
    

    
    '''
symbol_list_for_yfinance = [
    # Stocks (unchanged)
    "AAPL", "MSFT"
]

symbol_name_map = {
    # Stocks
    "AAPL": ["Apple", "Apple Inc."],
    "MSFT": ["Microsoft", "Microsoft Corporation"],
}

indicator_data = {}
fundamental_data = {}

def generate_query_variants(symbol):
    base_variants = [
        f"{symbol} stock",
        f"{symbol} investing",
        f"{symbol} earnings",
        f"{symbol} news",
        f"{symbol} financial results",
        f"{symbol} analysis",
        f"{symbol} quarterly report",
        f"{symbol} Wall Street",
    ]
    
    name_variants = symbol_name_map.get(symbol.upper(), [])
    for name in name_variants:
        base_variants += [
            f"{name} stock",
            f"{name} investing",
            f"{name} earnings",
            f"{name} news",
            f"{name} financial results",
            f"{name} analysis",
            f"{name} quarterly report",
        ]
    
    return list(set(base_variants))  # Rimuove duplicati
   

MAX_ARTICLES_PER_SYMBOL = 500  # Limite massimo per asset

def get_stock_news(symbol):
    """Recupera titoli, date e link delle notizie per un determinato simbolo, includendo varianti di nome."""
    query_variants = generate_query_variants(symbol)

    base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"

    now = datetime.utcnow()
    days_90 = now - timedelta(days=90)
    days_30 = now - timedelta(days=30)
    days_7  = now - timedelta(days=7)

    news_90_days = []
    news_30_days = []
    news_7_days  = []

    seen_titles = set()
    total_articles = 0

    for raw_query in query_variants:
        if total_articles >= MAX_ARTICLES_PER_SYMBOL:
            break  # Fermati se superato il limite

        query = quote_plus(raw_query)
        url = base_url.format(query)
        feed = feedparser.parse(url)

        for entry in feed.entries:
            if total_articles >= MAX_ARTICLES_PER_SYMBOL:
                break

            try:
                title = entry.title.strip()
                link = entry.link.strip()

                # Evita titoli duplicati
                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                news_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")

                if news_date >= days_90:
                    news_90_days.append((title, news_date, link))
                if news_date >= days_30:
                    news_30_days.append((title, news_date, link))
                if news_date >= days_7:
                    news_7_days.append((title, news_date, link))

                total_articles += 1

            except (ValueError, AttributeError):
                continue

    return {
        "last_90_days": news_90_days,
        "last_30_days": news_30_days,
        "last_7_days":  news_7_days
    }





#Normalizza il testo della notizia, rimuovendo impurità
def normalize_text(text):
    #Pulisce e normalizza il testo per una migliore corrispondenza.
    
    text = re.sub(r'\s-\s[^-]+$', '', text)    # Rimuove la parte dopo l'ultimo " - " (se presente)
    text = text.lower()    # Converti tutto in minuscolo
    text = re.sub(r'[-_/]', ' ', text)    # Sostituisci trattini e underscore con spazi
    text = re.sub(r'\s+', ' ', text).strip()    # Rimuovi spazi multipli e spazi iniziali/finali
    
    return text



#Trova i lemmi delle parole per una ricerca più completa
def lemmatize_words(words):
    """Lemmatizza le parole usando spaCy e restituisce una lista di lemmi."""
    doc = nlp(" ".join(words))  # Analizza le parole con spaCy
    return [token.lemma_ for token in doc]



#Calcola il sentiment basato sulle notizie del singolo asset
def calculate_sentiment(news, decay_factor=0.03):
    """
    Calcola il sentiment medio ponderato di una lista di titoli di notizie
    usando il modello NCFM/fin-sentiment.
    news: lista di tuple (title, date) o (title, date, link)
    """
    total_sentiment = 0.0
    total_weight    = 0.0
    now             = datetime.utcnow()

    for item in news:
        # estrai title e date (ignora eventuale link/extra)
        if len(item) == 3:
            title, date, _ = item
        elif len(item) == 2:
            title, date = item
        else:
            continue

        # calcola il peso esponenziale in base all'età
        days_old = (now - date).days
        weight   = math.exp(-decay_factor * days_old)

        # inferenza singola (potresti in futuro ottimizzare in batch)
        inputs = tokenizer(str(title), return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            score   = ID_TO_SCORE[pred_id]  # -1, 0, +1

        total_sentiment += score * weight
        total_weight    += weight

    if total_weight > 0:
        return total_sentiment / total_weight
    else:
        return 0.0  # neutro se nessuna notizia




# Funzione per calcolare la percentuale in base agli indicatori
def calcola_punteggio(indicatori, close_price, bb_upper, bb_lower):
    punteggio = 0

    if indicatori["RSI (14)"] > 70:
        punteggio -= 8
    elif indicatori["RSI (14)"] < 30:
        punteggio += 8
    else:
        punteggio += 4

    if indicatori["MACD Line"] > indicatori["MACD Signal"]:
        punteggio += 8
    else:
        punteggio -= 6

    if indicatori["Stochastic %K"] > 80:
        punteggio -= 6
    elif indicatori["Stochastic %K"] < 20:
        punteggio += 6

    if indicatori["EMA (10)"] < close_price:
        punteggio += 7

    if indicatori["CCI (14)"] > 0:
        punteggio += 6
    else:
        punteggio -= 4

    if indicatori["Williams %R"] > -20:
        punteggio -= 4
    else:
        punteggio += 4

    # Bollinger Bands
    if close_price > bb_upper:
        punteggio -= 5
    elif close_price < bb_lower:
        punteggio += 5

    return round(((punteggio + 44) * 100) / 88, 2)  # normalizzazione 0-100



#Inserisce tutti i risultati nel file html
def get_sentiment_for_all_symbols(symbol_list):
    sentiment_results = {}
    percentuali_tecniche = {}
    percentuali_combine = {}
    all_news_entries = []

    
    for symbol, adjusted_symbol in zip(symbol_list, symbol_list_for_yfinance):
        news_data = get_stock_news(symbol)  # Ottieni le notizie divise per periodo

        # Calcola il sentiment per ciascun intervallo di tempo
        sentiment_90_days = calculate_sentiment(news_data["last_90_days"])  
        sentiment_30_days = calculate_sentiment(news_data["last_30_days"])  
        sentiment_7_days = calculate_sentiment(news_data["last_7_days"])  

        sentiment_results[symbol] = {
            "90_days": sentiment_90_days,
            "30_days": sentiment_30_days,
            "7_days": sentiment_7_days
        }

        # Prepara i dati relativi agli indicatori
        tabella_indicatori = None  # Inizializza la variabile tabella_indicatori
        try:
            # 1. Scarica dati per un solo ticker → niente MultiIndex
            ticker = str(adjusted_symbol).strip().upper()
            data = yf.download(ticker, period="3mo", auto_adjust=False, progress=False)
            
            # 2. Check: dataset vuoto?
            if data.empty:
                raise ValueError(f"Nessun dato disponibile per {symbol} ({adjusted_symbol})")
            
            # 3. (Facoltativo) Normalizza eventuale MultiIndex legacy
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.xs(ticker, axis=1, level=1)
                except KeyError:
                    raise ValueError(f"Ticker {ticker} non trovato nel MultiIndex: {data.columns}")
            
            # 4. Estrazione sicura delle colonne
            try:
                close = data['Close']
                high  = data['High']
                low   = data['Low']
            except KeyError as e:
                raise ValueError(f"Colonna mancante per {symbol}: {e}")
            
            # 5. Stampa debug
            try:
                ultimo_close = float(close.iloc[-1])
                print(f"DEBUG: {symbol} ({adjusted_symbol}) → Ultimo Close: {ultimo_close}")
            except Exception as e:
                print(f"DEBUG ERROR: impossibile ricavare close per {symbol} → {e}")
                
        
            # Indicatori tecnici
            rsi = RSIIndicator(close).rsi().iloc[-1]
            macd = MACD(close)
            macd_line = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            stoch = StochasticOscillator(high, low, close)
            stoch_k = stoch.stoch().iloc[-1]
            stoch_d = stoch.stoch_signal().iloc[-1]
            ema_10 = EMAIndicator(close, window=10).ema_indicator().iloc[-1]
            cci = CCIIndicator(high, low, close).cci().iloc[-1]
            will_r = WilliamsRIndicator(high, low, close).williams_r().iloc[-1]
    
            bb = BollingerBands(close)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_width = bb.bollinger_wband().iloc[-1]
    
            indicators = {
                "RSI (14)": round(rsi, 2),
                "MACD Line": round(macd_line, 2),
                "MACD Signal": round(macd_signal, 2),
                "Stochastic %K": round(stoch_k, 2),
                "Stochastic %D": round(stoch_d, 2),
                "EMA (10)": round(ema_10, 2),
                "CCI (14)": round(cci, 2),
                "Williams %R": round(will_r, 2),
                "BB Upper": round(bb_upper, 2),
                "BB Lower": round(bb_lower, 2),
                "BB Width": round(bb_width, 4),
            }

            # CREA LA TABELLA HTML DEGLI INDICATORI TECNICI
            tabella_indicatori = pd.DataFrame(indicators.items(), columns=["Indicatore", "Valore"]).to_html(index=False, border=0)

            percentuale = calcola_punteggio(indicators, close.iloc[-1], bb_upper, bb_lower)



            # ────────────────────────────────────────
            # 1) RECUPERO DATI FONDAMENTALI DA yfinance
            # ────────────────────────────────────────
            ticker_obj = yf.Ticker(adjusted_symbol)
            try:
                info = ticker_obj.info or {}
            except Exception as e:
                print(f"Errore nel recupero dati fondamentali per {symbol}: {e}")
                info = {}
            
            # Funzione helper per validare numeri
            def safe_value(key):
                value = info.get(key)
                if isinstance(value, (int, float)):
                    return round(value, 4)
                return "N/A"
            
            # Costruisci il dizionario con valori sicuri
            fondamentali = {
                "Trailing P/E": safe_value("trailingPE"),
                "Forward P/E": safe_value("forwardPE"),
                "EPS Growth (YoY)": safe_value("earningsQuarterlyGrowth"),
                "Revenue Growth (YoY)": safe_value("revenueGrowth"),
                "Profit Margins": safe_value("profitMargins"),
                "Debt to Equity": safe_value("debtToEquity"),
                "Dividend Yield": safe_value("dividendYield")
            }
            
            # Costruisci la tabella HTML
            tabella_fondamentali = pd.DataFrame(
                fondamentali.items(), columns=["Fundamentale", "Valore"]
            ).to_html(index=False, border=0)

            
            #percentuale = calcola_punteggio(indicators, close.iloc[-1], bb_upper, bb_lower)
            percentuali_tecniche[symbol] = percentuale
            
            # Crea tabella dei dati storici (ultimi 90 giorni)
            dati_storici = data.tail(90).copy()
            dati_storici['Date'] = dati_storici.index.strftime('%Y-%m-%d')  # Aggiungi la colonna Date
            dati_storici_html = dati_storici[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].to_html(index=False, border=1)

        
            #Salvo in variabili globali per generare il daily brief
            indicator_data[symbol] = indicators
            fundamental_data[symbol] = fondamentali

        except Exception as e:
            print(f"Errore durante l'analisi di {symbol}: {e}")

        # GENERA FILE HTML INDIVIDUALE
        file_path = f"results/{symbol.upper()}_RESULT.html"

        html_content = [
            f"<html><head><title>Previsione per {symbol}</title></head><body>",
            f"<h1>Previsione per: ({symbol})</h1>",
            "<table border='1'><tr><th>Probability</th></tr>",
            f"<tr><td>{sentiment_90_days * 100}</td></tr>",
            "</table>",
            "<table border='1'><tr><th>Probability30</th></tr>",  # Nuova riga per 30 giorni
            f"<tr><td>{sentiment_30_days * 100}</td></tr>",
            "</table>",
            "<table border='1'><tr><th>Probability7</th></tr>",  # Nuova riga per 7 giorni
            f"<tr><td>{sentiment_7_days * 100}</td></tr>",
            "</table>",
            
            # Aggiunta della nuova sezione con gli indicatori tecnici e la probabilità calcolata
            "<hr>",
            "<h2>Indicatori Tecnici</h2>",
        ]

        if percentuale is not None:
            html_content.append(f"<p><strong>Probabilità calcolata sugli indicatori tecnici:</strong> {percentuale}%</p>")
        else:
            html_content.append("<p><strong>Impossibile calcolare la probabilità sugli indicatori tecnici.</strong></p>")
        
        # Aggiungi gli indicatori tecnici alla tabella
        if tabella_indicatori:
            html_content.append(tabella_indicatori)
        else:
            html_content.append("<p>No technical indicators available.</p>")

        # Aggiungi i dati fondamentali
        html_content.append("<h2>Dati Fondamentali</h2>")
        if tabella_fondamentali:
            html_content.append(tabella_fondamentali)
        else:
            html_content.append("<p>Nessun dato fondamentale disponibile.</p>")
        
        # Aggiungi i dati storici degli ultimi 90 giorni
        if dati_storici_html:
            html_content += [
                "<h2>Dati Storici (ultimi 90 giorni)</h2>",
                dati_storici_html,  # Usa il DataFrame formattato
                "</body></html>"
            ]
        else:
            html_content.append("<p>No historical data available.</p>")
        
        html_content.append("</body></html>")
        
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, f"Updated probability for {symbol}", "\n".join(html_content), contents.sha)
        except GithubException:
            repo.create_file(file_path, f"Created probability for {symbol}", "\n".join(html_content))

        # Aggiungi le notizie e i sentimenti alla lista per il file `news.html` (solo le notizie degli ultimi 90 giorni)
        for title, news_date, link in news_data["last_90_days"]:
            title_sentiment = calculate_sentiment([(title, news_date)])  # Se la tua funzione ha bisogno della data
            all_news_entries.append((symbol, title, title_sentiment, link))

    # CALCOLA MEDIA PONDERATA (fuori dal ciclo principale)
    w7 = 0.5
    w30 = 0.3
    w90 = 0.2
    
    for symbol in sentiment_results:
        if symbol in percentuali_tecniche:
            sentiment_7 = sentiment_results[symbol]["7_days"] * 100
            sentiment_30 = sentiment_results[symbol]["30_days"] * 100
            sentiment_90 = sentiment_results[symbol]["90_days"] * 100
    
            # Nuovo sentiment combinato (invece di usare solo quello a 90 giorni)
            sentiment_combinato = (w7 * sentiment_7) + (w30 * sentiment_30) + (w90 * sentiment_90)
    
            tecnica = percentuali_tecniche[symbol]
            
            # Combinazione sentiment + tecnica
            combinata = (sentiment_combinato * 0.6) + (tecnica * 0.4)
            percentuali_combine[symbol] = combinata

    #return sentiment_results, percentuali_combine, all_news_entries
    return sentiment_results, percentuali_combine, all_news_entries, indicator_data, fundamental_data






# Calcolare il sentiment medio per ogni simbolo
sentiment_for_symbols, percentuali_combine, all_news_entries, indicator_data, fundamental_data = get_sentiment_for_all_symbols(symbol_list)



#PER CREARE LA CLASSIFICA NORMALE-------------------------------------------------------------------------
# Ordinare i simboli in base al sentiment medio (decrescente)
sorted_symbols = sorted(sentiment_for_symbols.items(), key=lambda x: x[1]["90_days"], reverse=True)

# Crea il contenuto del file classifica.html
html_classifica = ["<html><head><title>Classifica dei Simboli</title></head><body>",
                   "<h1>Classifica dei Simboli in Base alla Probabilità di Crescita</h1>",
                   "<table border='1'><tr><th>Simbolo</th><th>Probabilità</th></tr>"]

# Aggiungere i simboli alla classifica con la probabilità calcolata
for symbol, sentiment_dict in sorted_symbols:
    # Estrai il sentiment per i 90 giorni
    probability = sentiment_dict["90_days"]
    
    # Aggiungi la riga alla classifica
    html_classifica.append(f"<tr><td>{symbol}</td><td>{probability*100:.2f}%</td></tr>")

html_classifica.append("</table></body></html>")

try:
    contents = repo.get_contents(file_path)
    repo.update_file(contents.path, "Updated classification", "\n".join(html_classifica), contents.sha)
except GithubException:
    repo.create_file(file_path, "Created classification", "\n".join(html_classifica))

print("Classifica aggiornata con successo!")



#PER CREARE LA CLASSIFICA PRO----------------------------------------------------------------------------
# Ordinare i simboli in base alla percentuale combinata (decrescente)
sorted_symbols_pro = sorted(percentuali_combine.items(), key=lambda x: x[1], reverse=True)

# Crea il contenuto del file classificaPRO.html
html_classifica_pro = ["<html><head><title>Classifica Combinata</title></head><body>",
                       "<h1>Classifica Combinata (60% Sentiment + 40% Indicatori Tecnici)</h1>",
                       "<table border='1'><tr><th>Simbolo</th><th>Media Ponderata</th><th>Sentiment 90g</th><th>Indicatori Tecnici</th></tr>"]

# Aggiungi le righe
for symbol, media in sorted_symbols_pro:
    html_classifica_pro.append(
        f"<tr><td>{symbol}</td><td>{media:.2f}%</td></tr>"
    )

html_classifica_pro.append("</table></body></html>")

# Scrivi il file su GitHub
pro_file_path = "results/classificaPRO.html"
try:
    contents = repo.get_contents(pro_file_path)
    repo.update_file(contents.path, "Updated combined classification", "\n".join(html_classifica_pro), contents.sha)
except GithubException:
    repo.create_file(pro_file_path, "Created combined classification", "\n".join(html_classifica_pro))

print("Classifica PRO aggiornata con successo!")




# Creazione del file news.html con solo 5 notizie positive e 5 negative per simbolo
html_news = ["<html><head><title>Notizie e Sentiment</title></head><body>",
             "<h1>Notizie Finanziarie con Sentiment</h1>",
             "<table border='1'><tr><th>Simbolo</th><th>Notizia</th><th>Sentiment</th><th>Link</th></tr>"]

# Raggruppa le notizie per simbolo
news_by_symbol = defaultdict(list)
for symbol, title, sentiment, url in all_news_entries:
    news_by_symbol[symbol].append((title, sentiment, url))

# Per ogni simbolo, prendi le 5 notizie col sentiment più basso e le 5 col più alto
for symbol, entries in news_by_symbol.items():
    # Ordina per sentiment (dal più negativo al più positivo)
    sorted_entries = sorted(entries, key=lambda x: x[1])

    # Prendi le 5 peggiori e le 5 migliori
    selected_entries = sorted_entries[:5] + sorted_entries[-5:]

    # Rimuove eventuali duplicati (se ci sono meno di 10 notizie)
    selected_entries = list(dict.fromkeys(selected_entries))

    for title, sentiment, url in selected_entries:
        html_news.append(
            f"<tr><td>{symbol}</td><td>{title}</td><td>{sentiment:.2f}</td>"
            f"<td><a href='{url}' target='_blank'>Leggi</a></td></tr>"
        )

html_news.append("</table></body></html>")

try:
    contents = repo.get_contents(news_path)
    repo.update_file(contents.path, "Updated news sentiment", "\n".join(html_news), contents.sha)
except GithubException:
    repo.create_file(news_path, "Created news sentiment", "\n".join(html_news))

print("News aggiornata con successo!")







def generate_fluid_market_summary_v2(
    sentiment_results, percentuali_combine, all_news_entries, 
    symbol_name_map, indicator_data, fundamental_data
):
    
    # Raggruppa notizie per simbolo
    news_by_symbol = defaultdict(list)
    for symbol, title, sentiment, url in all_news_entries:
        news_by_symbol[symbol].append((title, sentiment, url))

    # Seleziona 2 top e 2 bottom performer
    ranked = sorted(percentuali_combine.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in ranked[:2]]
    bottom_symbols = [s for s, _ in ranked[-2:] if s not in top_symbols]
    selected_symbols = (top_symbols + bottom_symbols)[:4]

    # Seleziona notizia più significativa, se molto positiva/negativa
    top_news = None
    sorted_news = sorted(all_news_entries, key=lambda x: abs(x[2]), reverse=True)
    for symbol, title, sentiment, url in sorted_news:
        if abs(sentiment) > 0.45 and symbol in selected_symbols:
            top_news = (symbol, title.strip().rstrip("."))
            break

    # Frasi variabili per ciascun tipo di segnale
    templates = {
        "strong_positive": [
            "{name} is expected to rise ({score}%)",
            "{name} showing upward momentum ({score}%)",
            "Bullish signals on {name} ({score}% upside)",
        ],
        "strong_negative": [
            "{name} under pressure ({risk}% downside risk)",
            "Bearish outlook for {name} ({risk}% probability of losses)",
            "{name} facing negative momentum ({risk}% risk)",
        ],
        "rsi_oversold": [
            "{name} looks oversold (RSI {rsi})",
            "{name} may rebound from oversold levels (RSI {rsi})",
        ],
        "rsi_overbought": [
            "{name} appears overbought (RSI {rsi})",
            "{name} might face selling pressure (RSI {rsi})",
        ],
        "mild_positive": [
            "{name} slightly bullish (+{delta:.1f}%)",
            "{name} trading modestly higher (+{delta:.1f}%)",
        ],
        "mild_negative": [
            "{name} trading lower ({delta:.1f}%)",
            "{name} facing mild selling (-{delta:.1f}%)",
        ],
        "neutral": [
            "{name} little changed",
            "{name} stable on the day",
        ]
    }

    phrases = []

    for symbol in selected_symbols:
        name = symbol_name_map.get(symbol, [symbol])[0]
        score = percentuali_combine.get(symbol, 0)
        rsi = indicator_data.get(symbol, {}).get("RSI (14)")
        delta = score - 50
        phrase = ""

        if score > 70:
            phrase = random.choice(templates["strong_positive"]).format(name=name, score=int(score))
        elif score < 30:
            phrase = random.choice(templates["strong_negative"]).format(name=name, risk=int(100 - score))
        elif rsi is not None and rsi < 30:
            phrase = random.choice(templates["rsi_oversold"]).format(name=name, rsi=int(rsi))
        elif rsi is not None and rsi > 70:
            phrase = random.choice(templates["rsi_overbought"]).format(name=name, rsi=int(rsi))
        elif delta > 10:
            phrase = random.choice(templates["mild_positive"]).format(name=name, delta=delta)
        elif delta < -10:
            phrase = random.choice(templates["mild_negative"]).format(name=name, delta=delta)
        else:
            phrase = random.choice(templates["neutral"]).format(name=name)

        # Se è presente una notizia rilevante per questo asset
        if top_news and top_news[0] == symbol:
            phrase += f" after news: \"{top_news[1]}\""

        phrases.append(phrase)

    # Se nessuna frase valida
    if not phrases:
        return "📰 <b>Market Today:</b><br>No major market developments."

    # Costruzione fluida del mini paragrafo
    summary = "📰 <b>Market Today:</b><br>"
    if len(phrases) == 1:
        summary += phrases[0] + "."
    else:
        summary += phrases[0] + ", "
        summary += ", ".join(phrases[1:-1])
        summary += f", and {phrases[-1]}."

    return summary



brief_text = generate_fluid_market_summary_v2(
    sentiment_for_symbols,
    percentuali_combine,
    all_news_entries,
    symbol_name_map,
    indicator_data,
    fundamental_data
)

# Salva il brief in HTML
html_content = f"""
<html>
  <head><title>Market Brief</title></head>
  <body>
    <h1>📊 Daily Market Summary</h1>
    <p style='font-family: Arial; font-size: 16px;'>{brief_text}</p>
  </body>
</html>
"""

file_path = "results/daily_brief.html"
try:
    contents = repo.get_contents(file_path)
    repo.update_file(file_path, "Updated daily brief", html_content, contents.sha)
except GithubException:
    repo.create_file(file_path, "Created daily brief", html_content)
