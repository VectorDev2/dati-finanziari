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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# —————————————
MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

ID_TO_SCORE = {0: -1, 1: 0, 2: 1}
# —————————————


# Carica il modello linguistico per l'inglese
nlp = spacy.load("en_core_web_sm")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "VectorDev2/dati-finanziari"

# Salva il file HTML nella cartella 'results'
file_path = "results/classifica.html"
news_path = "results/news.html"
    
# Salva il file su GitHub
github = Github(GITHUB_TOKEN)
repo = github.get_repo(REPO_NAME)

# Lista dei simboli azionari da cercare
symbol_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "JPM", "JNJ", "WMT",
        "NVDA", "PYPL", "DIS", "NFLX", "NIO", "NRG", "ADBE", "INTC", "CSCO", "PFE",
        "KO", "PEP", "MRK", "ABT", "XOM", "CVX", "T", "MCD", "NKE", "HD",
        "IBM", "CRM", "BMY", "ORCL", "ACN", "LLY", "QCOM", "HON", "COST", "SBUX",
        "CAT", "LOW", "MS", "GS", "AXP", "INTU", "AMGN", "GE", "FIS", "CVS",
        "DE", "BDX", "NOW", "SCHW", "LMT", "ADP", "C", "PLD", "NSC", "TMUS",
        "ITW", "FDX", "PNC", "SO", "APD", "ADI", "ICE", "ZTS", "TJX", "CL",
        "MMC", "EL", "GM", "CME", "EW", "AON", "D", "PSA", "AEP", "TROW", 
        "LNTH", "HE", "BTDR", "NAAS", "SCHL", "TGT", "SYK", "BKNG", "DUK", "USB",
        "ARM", "BABA", "BIDU", "COIN", "DDOG", "HTZ", "JD", "LCID", "LYFT", "NET", "PDD", #NEW
        "PLTR", "RIVN", "ROKU", "SHOP", "SNOW", "SQ", "TWLO", "UBER", "ZI", "ZM", "DUOL",    #NEW
        
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
        "AUDJPY", "CADJPY", "CHFJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPCHF", "AUDCAD",

        "SPX500", "DJ30", "NAS100", "NASCOMP", "RUS2000", "VIX", "EU50", "GER40", "UK100",
        "FRA40", "SWI20", "ESP35", "NETH25", "JPN225", "HKG50", "CHN50", "IND50", "KOR200",
               
        "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD", "EOSUSD", "XLMUSD", "ADAUSD", "TRXUSD", "NEOUSD",
        "DASHUSD", "XMRUSD", "ETCUSD", "ZECUSD", "BNBUSD", "DOGEUSD", "USDTUSD", "LINKUSD", "ATOMUSD", "XTZUSD",
        "COCOA", "XAUUSD", "GOLD", "XAGUSD", "SILVER", "OIL", "NATGAS"]  # Puoi aggiungere altri simboli

'''
    

    
    '''
symbol_list_for_yfinance = [
    # Stocks (unchanged)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "JPM", "JNJ", "WMT",
    "NVDA", "PYPL", "DIS", "NFLX", "NIO", "NRG", "ADBE", "INTC", "CSCO", "PFE",
    "KO", "PEP", "MRK", "ABT", "XOM", "CVX", "T", "MCD", "NKE", "HD",
    "IBM", "CRM", "BMY", "ORCL", "ACN", "LLY", "QCOM", "HON", "COST", "SBUX",
    "CAT", "LOW", "MS", "GS", "AXP", "INTU", "AMGN", "GE", "FIS", "CVS",
    "DE", "BDX", "NOW", "SCHW", "LMT", "ADP", "C", "PLD", "NSC", "TMUS",
    "ITW", "FDX", "PNC", "SO", "APD", "ADI", "ICE", "ZTS", "TJX", "CL",
    "MMC", "EL", "GM", "CME", "EW", "AON", "D", "PSA", "AEP", "TROW", 
    "LNTH", "HE", "BTDR", "NAAS", "SCHL", "TGT", "SYK", "BKNG", "DUK", "USB",
    "ARM", "BABA", "BIDU", "COIN", "DDOG", "HTZ", "JD", "LCID", "LYFT", "NET", "PDD",
    "PLTR", "RIVN", "ROKU", "SHOP", "SNOW", "SQ", "TWLO", "UBER", "ZI", "ZM", "DUOL",

    # Forex (with =X)
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    "EURCHF=X", "GBPCHF=X", "AUDCAD=X",

    # Global Indices
    "^GSPC", "^DJI", "^NDX", "^IXIC", "^RUT", "^VIX", "^STOXX50E", "^GDAXI", "^FTSE",
    "^FCHI", "^SSMI", "^IBEX", "^AEX", "^N225", "^HSI", "000001.SS", "^NSEI", "^KS200",

    # Crypto (with -USD)
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD", "EOS-USD", "XLM-USD", "ADA-USD",
    "TRX-USD", "NEO-USD", "DASH-USD", "XMR-USD", "ETC-USD", "ZEC-USD", "BNB-USD", "DOGE-USD",
    "USDT-USD", "LINK-USD", "ATOM-USD", "XTZ-USD",

    # Commodities (correct tickers)
    "CC=F",       # Cocoa
    "GC=F",   # Gold spot
    "GC=F",   # Gold spot
    "SI=F",   # Silver spot
    "SI=F",   # Silver spot
    "CL=F",        # Crude oil
    "NG=F"        # Natural gas
]

symbol_name_map = {
    # Stocks
    "AAPL": ["Apple", "Apple Inc."],
    "MSFT": ["Microsoft", "Microsoft Corporation"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc."],
    "AMZN": ["Amazon", "Amazon.com"],
    "META": ["Meta", "Facebook", "Meta Platforms"],
    "TSLA": ["Tesla", "Tesla Inc."],
    "V": ["Visa", "Visa Inc."],
    "JPM": ["JPMorgan", "JPMorgan Chase"],
    "JNJ": ["Johnson & Johnson", "JNJ"],
    "WMT": ["Walmart"],
    "NVDA": ["NVIDIA", "Nvidia Corp."],
    "PYPL": ["PayPal"],
    "DIS": ["Disney", "The Walt Disney Company"],
    "NFLX": ["Netflix"],
    "NIO": ["NIO Inc."],
    "NRG": ["NRG Energy"],
    "ADBE": ["Adobe", "Adobe Inc."],
    "INTC": ["Intel", "Intel Corporation"],
    "CSCO": ["Cisco", "Cisco Systems"],
    "PFE": ["Pfizer"],
    "KO": ["Coca-Cola", "The Coca-Cola Company"],
    "PEP": ["Pepsi", "PepsiCo"],
    "MRK": ["Merck"],
    "ABT": ["Abbott", "Abbott Laboratories"],
    "XOM": ["ExxonMobil", "Exxon"],
    "CVX": ["Chevron"],
    "T": ["AT&T"],
    "MCD": ["McDonald's"],
    "NKE": ["Nike"],
    "HD": ["Home Depot"],
    "IBM": ["IBM", "International Business Machines"],
    "CRM": ["Salesforce"],
    "BMY": ["Bristol-Myers", "Bristol-Myers Squibb"],
    "ORCL": ["Oracle"],
    "ACN": ["Accenture"],
    "LLY": ["Eli Lilly"],
    "QCOM": ["Qualcomm"],
    "HON": ["Honeywell"],
    "COST": ["Costco"],
    "SBUX": ["Starbucks"],
    "CAT": ["Caterpillar"],
    "LOW": ["Lowe's"],
    "MS": ["Morgan Stanley", "Morgan Stanley Bank", "MS bank", "MS financial"],
    "GS": ["Goldman Sachs"],
    "AXP": ["American Express"],
    "INTU": ["Intuit"],
    "AMGN": ["Amgen"],
    "GE": ["General Electric"],
    "FIS": ["Fidelity National Information Services"],
    "CVS": ["CVS Health"],
    "DE": ["Deere", "John Deere"],
    "BDX": ["Becton Dickinson"],
    "NOW": ["ServiceNow"],
    "SCHW": ["Charles Schwab"],
    "LMT": ["Lockheed Martin"],
    "ADP": ["ADP", "Automatic Data Processing"],
    "C": ["Citigroup"],
    "PLD": ["Prologis"],
    "NSC": ["Norfolk Southern"],
    "TMUS": ["T-Mobile"],
    "ITW": ["Illinois Tool Works"],
    "FDX": ["FedEx"],
    "PNC": ["PNC Financial"],
    "SO": ["Southern Company"],
    "APD": ["Air Products & Chemicals"],
    "ADI": ["Analog Devices"],
    "ICE": ["Intercontinental Exchange"],
    "ZTS": ["Zoetis"],
    "TJX": ["TJX Companies"],
    "CL": ["Colgate-Palmolive"],
    "MMC": ["Marsh & McLennan"],
    "EL": ["Estée Lauder"],
    "GM": ["General Motors"],
    "CME": ["CME Group"],
    "EW": ["Edwards Lifesciences"],
    "AON": ["Aon plc"],
    "D": ["Dominion Energy"],
    "PSA": ["Public Storage"],
    "AEP": ["American Electric Power"],
    "TROW": ["T. Rowe Price"],
    "LNTH": ["Lantheus"],
    "HE": ["Hawaiian Electric"],
    "BTDR": ["Bitdeer"],
    "NAAS": ["NaaS Technology"],
    "SCHL": ["Scholastic"],
    "TGT": ["Target"],
    "SYK": ["Stryker"],
    "BKNG": ["Booking Holdings", "Booking.com"],
    "DUK": ["Duke Energy"],
    "USB": ["U.S. Bancorp"],
    "BABA": ["Alibaba", "Alibaba Group", "阿里巴巴"],
    "HTZ": ["Hertz", "Hertz Global", "Hertz Global Holdings"],
    "UBER": ["Uber", "Uber Technologies", "Uber Technologies Inc."],
    "LYFT": ["Lyft", "Lyft Inc."],
    "PLTR": ["Palantir", "Palantir Technologies", "Palantir Technologies Inc."],
    "SNOW": ["Snowflake", "Snowflake Inc."],
    "ROKU": ["Roku", "Roku Inc."],
    "TWLO": ["Twilio", "Twilio Inc."],
    "SQ": ["Block", "Square", "Block Inc.", "Square Inc."],
    "COIN": ["Coinbase", "Coinbase Global", "Coinbase Global Inc."],
    "RIVN": ["Rivian", "Rivian Automotive", "Rivian Automotive Inc."],
    "LCID": ["Lucid", "Lucid Motors", "Lucid Group", "Lucid Group Inc."],
    "DDOG": ["Datadog", "Datadog Inc."],
    "NET": ["Cloudflare", "Cloudflare Inc."],
    "SHOP": ["Shopify", "Shopify Inc."],
    "ZI": ["ZoomInfo", "ZoomInfo Technologies", "ZoomInfo Technologies Inc."],
    "ZM": ["Zoom", "Zoom Video", "Zoom Video Communications", "Zoom Video Communications Inc."],
    "BIDU": ["Baidu", "百度"],
    "PDD": ["Pinduoduo", "PDD Holdings", "Pinduoduo Inc.", "拼多多"],
    "JD": ["JD.com", "京东"],
    "ARM": ["Arm", "Arm Holdings", "Arm Holdings plc"],
    "DUOL": ["Duolingo", "Duolingo Inc.", "DUOL"],

    # Forex
    "EURUSD": ["EUR/USD", "Euro Dollar", "Euro vs USD"],
    "USDJPY": ["USD/JPY", "Dollar Yen", "USD vs JPY"],
    "GBPUSD": ["GBP/USD", "British Pound", "Sterling", "GBP vs USD"],
    "AUDUSD": ["AUD/USD", "Australian Dollar", "Aussie Dollar"],
    "USDCAD": ["USD/CAD", "US Dollar vs Canadian Dollar", "Loonie"],
    "USDCHF": ["USD/CHF", "US Dollar vs Swiss Franc"],
    "NZDUSD": ["NZD/USD", "New Zealand Dollar"],
    "EURGBP": ["EUR/GBP", "Euro vs Pound"],
    "EURJPY": ["EUR/JPY", "Euro vs Yen"],
    "GBPJPY": ["GBP/JPY", "Pound vs Yen"],
    "AUDJPY": ["AUD/JPY", "Aussie vs Yen"],
    "CADJPY": ["CAD/JPY", "Canadian Dollar vs Yen"],
    "CHFJPY": ["CHF/JPY", "Swiss Franc vs Yen"],
    "EURAUD": ["EUR/AUD", "Euro vs Aussie"],
    "EURNZD": ["EUR/NZD", "Euro vs Kiwi"],
    "EURCAD": ["EUR/CAD", "Euro vs Canadian Dollar"],
    "EURCHF": ["EUR/CHF", "Euro vs Swiss Franc"],
    "GBPCHF": ["GBP/CHF", "Pound vs Swiss Franc"],
    "AUDCAD": ["AUD/CAD", "Aussie vs Canadian Dollar"],

    #Index
    "SPX500": ["S&P 500", "SPX", "S&P", "S&P 500 Index", "Standard & Poor's 500"],
    "DJ30": ["Dow Jones", "DJIA", "Dow Jones Industrial", "Dow 30", "Dow Jones Industrial Average"],
    "NAS100": ["Nasdaq 100", "NDX", "Nasdaq100", "NASDAQ 100 Index"],
    "NASCOMP": ["Nasdaq Composite", "IXIC", "Nasdaq", "Nasdaq Composite Index"],
    "RUS2000": ["Russell 2000", "RUT", "Russell Small Cap", "Russell 2K"],
    "VIX": ["VIX", "Volatility Index", "Fear Gauge", "CBOE Volatility Index"],
    "EU50": ["Euro Stoxx 50", "Euro Stoxx", "STOXX50", "Euro Stoxx 50 Index"],
    "GER40": ["DAX", "DAX 40", "German DAX", "Frankfurt DAX"],
    "UK100": ["FTSE 100", "FTSE", "UK FTSE 100", "FTSE Index"],
    "FRA40": ["CAC 40", "CAC", "France CAC 40", "CAC40 Index"],
    "SWI20": ["Swiss Market Index", "SMI", "Swiss SMI", "Swiss Market"],
    "SPA35": ["IBEX 35", "IBEX", "Spanish IBEX", "IBEX 35 Index"],
    "NETH25": ["AEX", "Dutch AEX", "Amsterdam Exchange", "AEX Index"],
    "JPN225": ["Nikkei 225", "Nikkei", "Japan Nikkei", "Nikkei Index"],
    "HKG50": ["Hang Seng", "Hong Kong Hang Seng", "Hang Seng Index"],
    "CHN50": ["Shanghai Composite", "SSEC", "China Shanghai", "Shanghai Composite Index"],
    "IND50": ["Nifty 50", "Nifty", "India Nifty", "Nifty 50 Index"],
    "KOR200": ["KOSPI", "KOSPI 200", "Korea KOSPI", "KOSPI Index"],
    
    # Crypto
    "BTCUSD": ["Bitcoin", "BTC"],
    "ETHUSD": ["Ethereum", "ETH"],
    "LTCUSD": ["Litecoin", "LTC"],
    "XRPUSD": ["Ripple", "XRP"],
    "BCHUSD": ["Bitcoin Cash", "BCH"],
    "EOSUSD": ["EOS"],
    "XLMUSD": ["Stellar", "XLM"],
    "ADAUSD": ["Cardano", "ADA"],
    "TRXUSD": ["Tron", "TRX"],
    "NEOUSD": ["NEO"],
    "DASHUSD": ["Dash crypto", "Dash cryptocurrency", "DASH coin", "DASH token", "Digital Cash", "Dash blockchain", "Dash digital currency"],
    "XMRUSD": ["Monero", "XMR"],
    "ETCUSD": ["Ethereum Classic", "ETC"],
    "ZECUSD": ["Zcash", "ZEC"],
    "BNBUSD": ["Binance Coin", "BNB"],
    "DOGEUSD": ["Dogecoin", "DOGE"],
    "USDTUSD": ["Tether", "USDT"],
    "LINKUSD": ["Chainlink", "LINK"],
    "ATOMUSD": ["Cosmos", "ATOM"],
    "XTZUSD": ["Tezos", "XTZ"],

    # Commodities
    "COCOA": ["Cocoa", "Cocoa Futures"],
    "XAUUSD": ["Gold", "XAU/USD", "Gold price", "Gold spot"],
    "GOLD": ["Gold", "XAU/USD", "Gold price", "Gold spot"],
    "XAGUSD": ["Silver", "XAG/USD", "Silver price", "Silver spot"],
    "SILVER": ["Silver", "XAG/USD", "Silver price", "Silver spot"],
    "OIL": ["Crude oil", "Oil price", "WTI", "Brent", "Brent oil", "WTI crude"],
    "NATGAS": ["Natural gas", "Gas price", "Natgas", "Henry Hub", "NG=F", "Natural gas futures"]
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
    total_sentiment = 0.0
    total_weight    = 0.0
    now             = datetime.utcnow()

    for item in news:
        if len(item) == 3:
            title, date, _ = item
        elif len(item) == 2:
            title, date = item
        else:
            continue

        days_old = (now - date).days
        weight   = math.exp(-decay_factor * days_old)

        inputs = tokenizer(str(title), return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            score = (probs[0][0] * 0.0 + 
                     probs[0][1] * 0.5 + 
                     probs[0][2] * 1.0).item()

        total_sentiment += score * weight
        total_weight    += weight

    return (total_sentiment / total_weight) if total_weight > 0 else 0.0




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
