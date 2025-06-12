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
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
from urllib.parse import quote_plus


# Carica il modello linguistico per l'inglese
nlp = spacy.load("en_core_web_sm")

#GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "VectorDev2/dati-finanziari"

# Salva il file HTML nella cartella 'results'
file_path = "results/classifica.html"
news_path = "results/news.html"
    
# Salva il file su GitHub
#github = Github(GITHUB_TOKEN)
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
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
        "AUDJPY", "CADJPY", "CHFJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPCHF", "AUDCAD",
        "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD", "EOSUSD", "XLMUSD", "ADAUSD", "TRXUSD", "NEOUSD",
        "DASHUSD", "XMRUSD", "ETCUSD", "ZECUSD", "BNBUSD", "DOGEUSD", "USDTUSD", "LINKUSD", "ATOMUSD", "XTZUSD",
        "COCOA", "XAUUSD", "XAGUSD", "OIL"]  # Puoi aggiungere altri simboli

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

    # Forex (with =X)
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    "EURCHF=X", "GBPCHF=X", "AUDCAD=X",

    # Crypto (with -USD)
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD", "EOS-USD", "XLM-USD", "ADA-USD",
    "TRX-USD", "NEO-USD", "DASH-USD", "XMR-USD", "ETC-USD", "ZEC-USD", "BNB-USD", "DOGE-USD",
    "USDT-USD", "LINK-USD", "ATOM-USD", "XTZ-USD",

    # Commodities (correct tickers)
    "CC=F",       # Cocoa
    "GC=F",   # Gold spot
    "SI=F",   # Silver spot
    "CL=F"        # Crude oil
]



def get_stock_news(symbol):
    """Recupera molti più titoli e date delle notizie per un determinato simbolo negli ultimi 90, 30 e 7 giorni."""
    query_variants = [
        f"{symbol} stock",
        f"{symbol} investing",
        f"{symbol} earnings",
        f"{symbol} news",
        f"{symbol} financial results",
        f"{symbol} analysis",
        f"{symbol} quarterly report",
        f"{symbol} Wall Street"
    ]

    base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"

    now = datetime.utcnow()
    days_90 = now - timedelta(days=90)
    days_30 = now - timedelta(days=30)
    days_7  = now - timedelta(days=7)

    news_90_days = []
    news_30_days = []
    news_7_days  = []

    seen_titles = set()

    for raw_query in query_variants:
        query = quote_plus(raw_query)
        url = base_url.format(query)
        feed = feedparser.parse(url)

        for entry in feed.entries:
            try:
                title = entry.title.strip()
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                news_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")

                if news_date >= days_90:
                    news_90_days.append((title, news_date))
                if news_date >= days_30:
                    news_30_days.append((title, news_date))
                if news_date >= days_7:
                    news_7_days.append((title, news_date))

            except (ValueError, AttributeError):
                continue

    return {
        "last_90_days": news_90_days,
        "last_30_days": news_30_days,
        "last_7_days":  news_7_days
    }


# Lista di negazioni da considerare
negation_words = {"not", "never", "no", "without", "don't", "dont", "doesn't", "doesnt"}


# Dizionario di parole chiave con il loro punteggio di sentiment
sentiment_dict = {
    "ai": 0.6,
    "analyst rating": 0.6,
    "acquisition": 0.7,
    "acquisitions": 0.7,
    "appreciation": 0.9,
    "advance": 0.8,
    "advanced": 0.8,
    "agreement": 0.7,
    "agreements": 0.7,
    "agree": 0.7,
    "agreed": 0.6,
    "allocation": 0.6,
    "augmented": 0.7,
    "augment": 0.7,
    "augments": 0.7,
    "attraction": 0.85,
    "attractions": 0.85,
    "attractive": 0.85,
    "attractives": 0.85,
    "affluence": 0.9,
    "accelerator": 0.7,
    "ascend": 0.8,
    "advantage": 0.8,
    "advantaged": 0.8,
    "amplification": 0.7,
    "abundance": 0.9,
    "amendment": 0.6,
    "allowance": 0.6,
    "achievement": 0.8,
    "accession": 0.7,
    "ascension": 0.8,
    "allocation": 0.6,
    "acceptance": 0.7,
    "accreditation": 0.6,
    "authorized": 0.6,
    "approval": 0.6,
    "approved": 0.7,
    "assurance": 0.7,
    "advancement": 0.8,
    "aspiration": 0.7,
    "adoption": 0.6,
    "achievement": 0.8,
    "acceleration": 0.8,
    "appraisal": 0.6,
    "amortization": 0.4,
    "arrest": 0.1,
    "arrested": 0.1,
    "arrests": 0.1,
    "adversity": 0.2,
    "anomaly": 0.3,
    "attrition": 0.2,
    "antitrust": 0.2,
    "aversion": 0.2,
    "arrears": 0.1,
    "abandonment": 0.1,
    "alienation": 0.2,
    "asymmetry": 0.3,
    "ambiguity": 0.3,
    "anxiety": 0.2,
    "adjustment": 0.6,
    "adjusted earnings": 0.7,
    "algorithmic trading": 0.6,
    "austerity": 0.3,
    "audit": 0.4,
    "amendment": 0.6,
    "apprehension": 0.2,
    "abrogation": 0.1,
    "annulment": 0.1,
    "arrogation": 0.1,
    "admonition": 0.2,
    "antagonism": 0.2,
    "abysmal": 0.1,
    "accountability": 0.6,
    "arrestment": 0.2,
    "attrition": 0.3,
    "aftermath": 0.2,

    "bad": 0.1,
    "badly": 0.1,
    "bull": 0.9,
    "bullish": 0.9,
    "bully": 0.7,
    "bear": 0.3,
    "bear market": 0.2,
    "bankruptcy": 0.1,
    "bankrupt": 0.1,
    "balanced": 0.6,
    "bomb": 0.65,
    "boom": 0.9,
    "booms": 0.9,
    "buy": 0.9,
    "buys": 0.9,
    "bought": 0.85,
    "boost": 0.8,
    "boosts": 0.8,
    "boosted": 0.8,
    "benefit": 0.8,
    "benefits": 0.8,
    "billion": 0.6,
    "billions": 0.6,
    "bonds": 0.7,
    "breakthrough": 0.7,
    "benchmark": 0.7,
    "bust": 0.1,
    "bargain": 0.8,
    "bid": 0.7,
    "bailout": 0.3,
    "beneficiary": 0.7,
    "blockchain": 0.6,
    "bail": 0.3,
    "barrier": 0.4,
    "bottom line": 0.6,
    "balance sheet": 0.6,
    "backlog": 0.4,
    "backer": 0.65,
    "brisk": 0.7,
    "burnout": 0.2,
    "blockbuster": 0.8,
    "balance of payments": 0.6,
    "breach": 0.2,
    "blowout": 0.1,
    "bribe": 0.0,
    "brutal": 0.1,
    "bust up": 0.2,
    "bank run": 0.1,
    "bubble": 0.2,

    "capital": 0.7,
    "cancellation": 0.2,
    "cancellations": 0.2,
    "cash": 0.65,
    "crash": 0.1,
    "crashes": 0.1,
    "crashed": 0.1,
    "cautious": 0.35,
    "caution": 0.35,
    "cautiously": 0.65,
    "climb": 0.8,
    "climbed": 0.7,
    "climbs": 0.8,
    "close to get": 0.8,
    "coup": 0.2,
    "couch potato portfolio": 0.75,
    "credit": 0.7,
    "credit crunch": 0.2,
    "cut": 0.2,
    "cuts": 0.2,
    "collapse": 0.1,
    "collapsed": 0.1,
    "creditor": 0.65,
    "correction": 0.3,
    "commodities": 0.7,
    "change": 0.65,
    "competition": 0.3,
    "coupon": 0.6,
    "contribution": 0.7,
    "crisis": 0.1,
    "consolidation": 0.7,
    "capitalization": 0.8,
    "collateral damage": 0.3,
    "compliance": 0.65,
    "collaboration": 0.7,
    "consumer confidence": 0.8,
    "credibility": 0.8,
    "closure": 0.3,
    "commitment": 0.7,
    "clawback": 0.3,
    "cutback": 0.3,
    "come to an end": 0.3,
    "comes to an end": 0.3,
    "came to an end": 0.3,
    "coming to an end": 0.3,
    "contraction": 0.3,
    "conservative": 0.4,
    "corruption": 0.0,
    "corrupted": 0.0,
    "concerns": 0.2,
    "concern": 0.2,
    "capital gains": 0.8,
    "cash flow": 0.7,
    "credit rating": 0.65,
    "contribution margin": 0.7,
    "crisis management": 0.3,
    "capital raise": 0.7,
    "counterfeit": 0.1,
    "convergence": 0.65,
    "compensation package": 0.7,
    "compensation": 0.7,
    "capital flow": 0.6,
    "corruption scandal": 0.1,

    "damage": 0.1,
    "damages": 0.1,
    "damaged": 0.1,
    "damaging": 0.1,
    "debt": 0.2,
    "deal": 0.8,
    "deals": 0.8,
    "delay": 0.2,
    "delays": 0.2,
    "delayed": 0.2,
    "dividend": 0.8,
    "deficit": 0.1,
    "decline": 0.2,
    "declines": 0.2,
    "depreciation": 0.3,
    "drop": 0.2,
    "drops": 0.2,
    "downturn": 0.2,
    "down": 0.3,
    "downgrade": 0.25,
    "devaluation": 0.3,
    "disruption": 0.3,
    "discount": 0.6,
    "dilution": 0.4,
    "die": 0.2,
    "dies": 0.2,
    "died": 0.2,
    "dead": 0.2,
    "death": 0.2,
    "development": 0.7,
    "declining": 0.3,
    "delisting": 0.2,
    "delisted": 0.2,
    "distribution": 0.65,
    "dissatisfaction": 0.2,
    "dissatisfactions": 0.2,
    "debt ceiling": 0.2,
    "decline rate": 0.2,
    "dominance": 0.7,
    "distressed": 0.2,
    "downsize": 0.3,
    "drain": 0.2,
    "delisting": 0.1,
    "doubt": 0.2,
    "diminish": 0.3,
    "declining market": 0.1,
    "deterioration": 0.2,
    "diversification": 0.7,
    "direct investment": 0.7,
    "downward": 0.2,
    "danger": 0.1,
    "decline in sales": 0.1,
    "debt reduction": 0.65,
    "discrepancy": 0.3,
    "debt to equity": 0.4,
    "dismantling": 0.3,
    "deflation": 0.2,
    "debtor": 0.3,
    "debt servicing": 0.3,
    "dominant": 0.7,
    "diversified": 0.7,
    "dormant": 0.3,
    "downward spiral": 0.2,
    "dysfunction": 0.1,

    "equity": 0.8,
    "earning": 0.7,
    "earnings": 0.7,
    "emerging": 0.7,
    "expansion": 0.8,
    "efficiency": 0.7,
    "exit": 0.4,
    "estimation": 0.7,
    "expenditure": 0.3,
    "enterprise": 0.7,
    "evercore isi": 0.75,
    "equilibrium": 0.65,
    "economic slowdown": 0.15,
    "endowment": 0.7,
    "elevate": 0.8,
    "erode": 0.2,
    "erodes": 0.2,
    "eroding": 0.2,
    "eroded": 0.2,
    "exceed": 0.8,
    "expectation": 0.7,
    "excess": 0.35,
    "enrichment": 0.8,
    "encouragement": 0.7,
    "enterprise value": 0.7,
    "equity market": 0.7,
    "exclusivity": 0.7,
    "escrow": 0.65,
    "exodus": 0.2,
    "evasion": 0.1,
    "equitable": 0.7,
    "equilibrium price": 0.65,
    "empowerment": 0.7,
    "effort": 0.7,
    "elasticity": 0.7,
    "enforce": 0.65,
    "enforcing": 0.65,
    "establishment": 0.7,
    "enlightenment": 0.7,
    "equity fund": 0.7,

    "financial crisis": 0.1,
    "fund": 0.8,
    "failure": 0.1,
    "fail": 0.1,
    "fails": 0.1,
    "failed": 0.1,
    "fluctuation": 0.3,
    "funding": 0.7,
    "flexibility": 0.7,
    "favorable": 0.8,
    "fall": 0.15,
    "fell": 0.15,
    "fraud": 0.0,
    "flow": 0.7,
    "fintech": 0.7,
    "finance": 0.7,
    "flourish": 0.7,
    "fast track": 0.7,
    "foreclosure": 0.1,
    "failing": 0.1,
    "frenzy": 0.3,
    "fallout": 0.2,
    "failure rate": 0.2,
    "fundamentals": 0.7,
    "freeze": 0.3,
    "flare": 0.2,
    "forecasting": 0.65,
    "fraudulent": 0.1,
    "favorable outlook": 0.8,
    "favorable": 0.8,
    "financing": 0.7,
    "flow through": 0.7,
    "forward looking": 0.7,
    "fledgling": 0.4,
    "fire sale": 0.1,
    "full disclosure": 0.7,
    "financial innovation": 0.7,
    "free market": 0.7,
    "falling prices": 0.2,
    "falling price": 0.2,
    "falling": 0.2,

    "growth": 0.9,
    "growths": 0.9,
    "gain": 0.9,
    "gains": 0.9,
    "growth rate": 0.9,
    "growing dissatisfaction": 0.1,
    "guarantee": 0.8,
    "gross": 0.6,
    "green": 0.8,
    "grants": 0.7,
    "guidance": 0.7,
    "glut": 0.2,
    "gap": 0.3,
    "gloom": 0.2,
    "grave": 0.1,
    "gridlock": 0.3,
    "grind": 0.3,
    "gross margin": 0.7,
    "gross product": 0.7,
    "gains per share": 0.8,
    "greenfield": 0.7,
    "garnishment": 0.2,
    "growing market": 0.9,
    "government debt": 0.2,
    "garnishee": 0.2,
    "globalization": 0.65,
    "grip": 0.4,
    "global demand": 0.7,
    "gross revenue": 0.7,
    "goodwill": 0.65,
    "graft": 0.1,
    "guarantor": 0.65,
    "growth stock": 0.9,
    "good debt": 0.7,
    "good": 0.9,
    "goodly": 0.9,
    "global recession": 0.2,

    "hike": 0.8,
    "high": 0.8,
    "highs": 0.8,
    "holding company": 0.7,
    "holding companies": 0.7,
    "holding structure": 0.6,
    "holding structures": 0.6,
    "holding pattern": 0.4,
    "holding fund": 0.7,
    "holding funds": 0.7,
    "holding patterns": 0.4,
    "holding losses": 0.2,
    "holding loss": 0.2,
    "holding steady": 0.8,
    "holding off": 0.4,
    "hold off": 0.4,
    "holds off": 0.4,
    "holding gains": 0.9,
    "holding back": 0.3,
    "hold back": 0.3,
    "holds back": 0.3,
    "holding onto": 0.6,
    "hit": 0.3,
    "hurdle": 0.3,
    "healthy": 0.8,
    "hoarding": 0.2,
    "headwind": 0.3,
    "headwinds": 0.3,
    "hyperinflation": 0.1,
    "high risk": 0.2,
    "high risks": 0.2,
    "hedge fund": 0.4,
    "holding company": 0.7,
    "harmonic": 0.6,
    "high yield": 0.8,
    "healthy growth": 0.8,
    "haircut": 0.2,
    "high performance": 0.9,
    "high performances": 0.9,
    "high value": 0.9,
    "hollow": 0.2,
    "high headcount": 0.65,
    "high impact": 0.7,
    "hasty": 0.3,
    "healthcare": 0.7,
    "hustle": 0.4,
    "hardship": 0.2,
    "hard asset": 0.7,
    "hollowed out": 0.2,
    "hollow out": 0.2,
    "heavily indebted": 0.0,

    "ia": 0.6,
    "income": 0.8,
    "incomes": 0.8,
    "investment": 0.8,
    "inflation": 0.3,
    "inflate": 0.3,
    "increase": 0.8,
    "increased": 0.8,
    "increased competition": 0.2,
    "improvement": 0.8,
    "interest": 0.7,
    "insight": 0.7,
    "insights": 0.7,
    "inflationary": 0.3,
    "innovative": 0.8,
    "insurance": 0.7,
    "integrity": 0.8,
    "investment grade": 0.7,
    "intelligence": 0.7,
    "increase rate": 0.7,
    "increased rate": 0.7,
    "indebtedness": 0.2,
    "interest rate": 0.6,
    "impactful": 0.7,
    "incursion": 0.3,
    "illiquid": 0.2,
    "illiquidity": 0.2,
    "impairment": 0.2,
    "insolvency": 0.1,
    "income tax": 0.4,
    "incentive": 0.7,
    "issue": 0.1,
    "issues": 0.1,
    "inflated": 0.3,
    "inflating": 0.3,
    "insider trading": 0.1,
    "increased demand": 0.7,
    "increase demand": 0.7,
    "increased competition": 0.25,
    "increase competition": 0.25,
    "independent": 0.7,
    "invest": 0.65,
    "incorporation": 0.6,
    "illegal": 0.0,
    "investing": 0.65,
    "impaired": 0.2,
    "interest bearing": 0.7,
    "interest": 0.7,
    "interesting": 0.7,
    "interested": 0.7,

    "job": 0.7,
    "jobs": 0.7,
    "joint": 0.7,
    "jump": 0.9,
    "jumps": 0.9,
    "junks": 0.2,
    "junk": 0.2,
    "justice": 0.8,
    "jittery": 0.3,
    "jackpot": 0.9,
    "jackpots": 0.9,
    "jockey": 0.65,
    "jumpstart": 0.8,
    "juggernaut": 0.7,
    "jockeying": 0.6,
    "justice system": 0.65,
    "job market": 0.7,
    "jack up": 0.3,
    "judicious": 0.7,
    "jobless": 0.2,

    "kicker": 0.6,
    "knockout": 0.7,
    "keep growing": 0.9,
    "keep increasing": 0.9,
    "keep holding": 0.7,
    "keep outperforming": 0.9,
    "keep strengthening": 0.9,
    "keep stable": 0.6,
    "keep waiting": 0.4,
    "keep struggling": 0.3,
    "keep losing": 0.2,
    "knot": 0.3,
    "kickback": 0.2,
    "keen on": 0.7,
    "kill": 0.2,
    "key player": 0.8,
    "kind": 0.6,
    "kerfuffle": 0.3,
    "kickstart": 0.7,
    "kudos": 0.8,

    "layoff": 0.2,
    "loss": 0.1,
    "losses": 0.1,
    "lost": 0.1,
    "liquidity": 0.65,
    "loan": 0.7,
    "loans": 0.7,
    "liability": 0.3,
    "liquid": 0.7,
    "long": 0.7,
    "lift": 0.8,
    "low": 0.3,
    "lows": 0.3,
    "leading": 0.8,
    "lending": 0.7,
    "lead": 0.8,
    "lag": 0.35,
    "liquidation": 0.15,
    "low risk": 0.7,
    "low risks": 0.7,
    "low cost": 0.7,
    "low costs": 0.7,
    "lower than anticipated": 0.25,
    "lower than expected": 0.25,
    "lucrative": 0.9,
    "late": 0.3,
    "lack": 0.2,
    "long term": 0.7,
    "long terms": 0.7,
    "large cap": 0.7,
    "long position": 0.7,
    "long positions": 0.7,
    "leading indicator": 0.7,
    "liquid assets": 0.7,
    "lull": 0.3,
    "leveraged buyout": 0.7,
    "low growth": 0.2,
    "loss making": 0.1,
    "leveraging": 0.6,
    "launch": 0.7,
    "low value": 0.3,
    "lifetime value": 0.7,
    "liquidate": 0.1,

    "market risk": 0.4,
    "market risks": 0.4,
    "market crisis": 0.1,
    "market share": 0.7,
    "market downturn": 0.2,
    "merger": 0.8,
    "magnitude": 0.7,
    "momentum": 0.8,
    "maturity": 0.7,
    "million": 0.65,
    "millions": 0.65,
    "master": 0.8,
    "markup": 0.7,
    "minimize": 0.7,
    "miss": 0.2,
    "missing": 0.2,
    "missed": 0.2,
    "maximization": 0.85,
    "maximizations": 0.85,
    "multiplier": 0.7,
    "modest": 0.4,
    "manipulation": 0.2,
    "margin call": 0.3,
    "move up": 0.85,
    "moves up": 0.85,
    "moved up": 0.75,
    "money market": 0.7,
    "manufacture": 0.65,
    "markup pricing": 0.7,
    "money supply": 0.65,
    "move forward": 0.7,
    "mining": 0.6,
    "multinational": 0.7,
    "multinationals": 0.7,
    "most attractive": 0.8,
    "magnificent": 0.75,
    "magnific": 0.75,

    "niche": 0.8,
    "non performing": 0.1,
    "narrow": 0.4,
    "new": 0.65,
    "negative": 0.1,
    "negative earnings": 0.0,
    "new product": 0.8,
    "new products": 0.8,
    "net profit": 0.7,
    "net profits": 0.7,
    "note worthy": 0.7,
    "non essential": 0.3,
    "net worth": 0.7,
    "nurture": 0.7,
    "non compliant": 0.2,
    "nervous": 0.3,
    "no growth": 0.2,
    "no growths": 0.2,
    "new investment": 0.8,
    "new investments": 0.8,
    "non cyclical": 0.6,
    "noteworthy": 0.7,
    "normalization": 0.65,
    "net gain": 0.8,
    "net gains": 0.8,
    "not reachable": 0.15,
    "not reached": 0.1,

    "offer": 0.65,
    "offers": 0.65,
    "overperform": 0.9,
    "outperform": 0.9,
    "optimistic": 0.8,
    "opportunity": 0.8,
    "opportunities": 0.8,
    "organic": 0.7,
    "overdue": 0.3,
    "overhead": 0.7,
    "overvalued": 0.2,
    "offset": 0.65,
    "outflow": 0.3,
    "overleveraged": 0.2,
    "overestimate": 0.3,
    "outstanding": 0.8,
    "overcapacity": 0.3,
    "overreaction": 0.3,
    "overexposure": 0.3,
    "overperformance": 0.8,
    "obsolescence": 0.2,
    "overfunded": 0.7,
    "optimization": 0.7,
    "optimizations": 0.7,
    "operating profit": 0.8,
    "overstretch": 0.3,
    "oversupply": 0.2,
    "offerings": 0.7,
    "on track": 0.7,
    "overcome": 0.8,
    "oscillation": 0.35,
    "overproduction": 0.3,
    "organic growth": 0.8,
    "organic growths": 0.8,

    "panic": 0.1,
    "panic selling": 0.1,
    "profits": 0.8,
    "profit": 0.8,
    "profit margin": 0.8,
    "positive": 0.9,
    "positively": 0.9,
    "premium": 0.8,
    "predict": 0.6,
    "prediction": 0.6,
    "predictions": 0.6,
    "pioneer": 0.8,
    "purchasing": 0.7,
    "prosper": 0.9,
    "prospered": 0.9,
    "prospers": 0.9,
    "plan": 0.8,
    "plans": 0.8,
    "positive growth": 0.9,
    "positive growths": 0.9,
    "payoff": 0.8,
    "peak": 0.65,
    "peaking": 0.7,
    "price increase": 0.65,
    "power": 0.7,
    "price cut": 0.4,
    "plunge": 0.2,
    "plunges": 0.2,
    "plunged": 0.2,
    "plummeted": 0.2,
    "pressure": 0.3,
    "pressures": 0.3,
    "pressured": 0.3,
    "pandemic": 0.2,
    "pessimistic": 0.2,
    "plentiful": 0.8,
    "penetrant": 0.65,
    "premium rate": 0.8,
    "plunge risk": 0.3,
    "poor performance": 0.2,
    "poor": 0.1,
    "progress": 0.8,
    "problem": 0.15,
    "problems": 0.1,
    "product release": 0.7,
    "product releases": 0.7,
    "product released": 0.7,
    "pull back": 0.2,
    "pulls back": 0.2,
    "pulling back": 0.2,
    "pulled back": 0.2,

    "quality": 0.65,
    "quick": 0.8,
    "quarantine": 0.2,
    "questionable": 0.3,
    "quiet": 0.4,
    "quick turnaround": 0.65,
    "quality control": 0.65,
    "quaint": 0.65,

    "rapid": 0.7,
    "rattle": 0.3,
    "revenue": 0.8,
    "rebound": 0.8,
    "rebounds": 0.8,
    "revenues": 0.8,
    "recovery": 0.7,
    "reinvestment": 0.6,
    "reduction": 0.3,
    "reductions": 0.3,
    "resilience": 0.8,
    "risk": 0.2,
    "risks": 0.2,
    "robust": 0.8,
    "recession": 0.1,
    "rebalancing": 0.65,
    "revenue growth": 0.9,
    "revenue growths": 0.9,
    "reliable": 0.9,
    "raise": 0.8,
    "rise": 0.8,
    "rises": 0.8,
    "rising price": 0.3,
    "rising prices": 0.3,
    "rising debt": 0.2,
    "refinancing": 0.6,
    "reduction in force": 0.3,
    "risk aversion": 0.3,
    "rally": 0.8,
    "recovery plan": 0.75,
    "reliable performance": 0.8,
    "reinforcement": 0.7,
    "reinvestment strategy": 0.8,
    "risky": 0.2,
    "repayment": 0.6,
    "recessionary": 0.2,
    "redemption": 0.65,
    "revenue stream": 0.75,
    "revenue model": 0.8,
    "reserves": 0.6,
    "revenue per share": 0.8,
    
    "share": 0.8,
    "shares": 0.8,
    "shared": 0.8,
    "shocking": 0.2,
    "shocked": 0.2,
    "shock": 0.2,
    "surge": 0.8,
    "surges": 0.8,
    "strong": 0.8,
    "strategy": 0.75,
    "strategic": 0.75,
    "strategies": 0.75,
    "successful": 0.9,
    "savings": 0.7,
    "sustainability": 0.8,
    "sustainable": 0.8,
    "stability": 0.8,
    "securities": 0.7,
    "security": 0.7,
    "secure": 0.9,
    "security breach": 0.1,
    "skepticism": 0.3,
    "steady": 0.7,
    "subsidy": 0.7,
    "startup": 0.65,
    "startups": 0.65,
    "solid": 0.8,
    "sell": 0.3,
    "sell off": 0.2,
    "sells": 0.3,
    "setback": 0.2,
    "setbacks": 0.2,
    "sold": 0.3,
    "sold out": 0.7,
    "spend on growth": 0.9,
    "spends on growth": 0.9,
    "spend efficiently": 0.8,
    "spend more than": 0.4,
    "spends more than": 0.4,
    "surplus": 0.8,
    "stimulus": 0.7,
    "short": 0.3,
    "shrinking": 0.2,
    "shrink": 0.2,
    "slow": 0.3,
    "slows": 0.3,
    "slower": 0.3,
    "slowing": 0.3,
    "slowdown": 0.25,
    "slash": 0.3,
    "slashing": 0.3,
    "slashed": 0.3,
    "slide": 0.3,
    "slides": 0.3,
    "savings plan": 0.7,
    "stagnation": 0.2,
    "stagnant": 0.2,
    "stagflation": 0.2,
    "steep": 0.7,
    "scalability": 0.7,
    "softening": 0.3,
    "soar": 0.8,
    "soars": 0.8,
    "saturation": 0.35,
    "shutdown": 0.2,
    "squeeze": 0.3,
    "sale": 0.65,
    "sales": 0.65,
    "synergy": 0.75,
    "share price": 0.65,
    "spin off": 0.7,
    "stimulation": 0.7,
    "speed": 0.7,
    "stock market crash": 0.1,
    "simply wall st": 0.7,
    "suffered": 0.25,
    "suffer": 0.25,
    "suffers": 0.25,
    "suffering": 0.25,
    
    "tax": 0.3,
    "taxes": 0.3,
    "taxed": 0.2,
    "tangible": 0.65,
    "treasury": 0.7,
    "tension": 0.3,
    "tensions": 0.3,
    "trust": 0.8,
    "technologic": 0.7,
    "technology stock": 0.8,
    "tactical": 0.7,
    "takeover": 0.7,
    "tailwind": 0.8,
    "taxation": 0.35,
    "tighten": 0.3,
    "thrive": 0.7,
    "thrives": 0.7,
    "thriving": 0.7,
    "thrived": 0.7,
    "trade off": 0.65,
    "tactical position": 0.7,
    "targeted": 0.65,
    "tangible asset": 0.8,
    "turbulence": 0.2,
    "trouble": 0.2,
    "troubles": 0.15,
    "transparency": 0.8,
    "total return": 0.65,
    "top line": 0.7,
    "turnkey": 0.7,
    "turmoil": 0.2,
    "takedown": 0.3,
    "toxic": 0.2,
    "toxic asset": 0.1,
    "toxic assets": 0.1,
    "threaten": 0.1,
    "trade war": 0.2,
    "too much": 0.1,
    
    "underperform": 0.2,
    "unemployment": 0.1,
    "upturn": 0.8,
    "unsecured": 0.3,
    "unforeseen": 0.3,
    "utility": 0.7,
    "utilities": 0.7,
    "unified": 0.65,
    "unfavorable": 0.2,
    "unpredictable": 0.3,
    "usury": 0.1,
    "upside": 0.8,
    "up": 0.9,
    "upgrade": 0.8,
    "upgraded": 0.8,
    "upgrades": 0.8,
    "underutilized": 0.35,
    "unavailable": 0.3,
    "unrealized": 0.35,
    "uncovered": 0.3,
    "unsustainable": 0.2,
    "unprofitable": 0.1,
    "unfavorable trend": 0.1,
    "uncertainty": 0.3,
    "uncertain": 0.3,
    "underperformance": 0.2,
    "under": 0.2,
    "upward": 0.8,
    "uncapped": 0.7,
    "unquestionable": 0.65,
    "unlimited": 0.8,
    "unpredictability": 0.25,

    "volatility": 0.3,
    "viability": 0.65,
    "viable": 0.65,
    "vulnerable": 0.2,
    "victory": 0.9,
    "victories": 0.9,
    "violation": 0.2,
    "violations": 0.2,
    "vacancy": 0.3,
    "verifiable": 0.7,
    "venture capital": 0.4,
    "visibility": 0.65,
    "visible": 0.65,
    "vanguard": 0.8,
    "valuation risk": 0.3,
    "vigor": 0.8,
    "vantage": 0.8,
    "vantages": 0.8,
    "volatile": 0.2,
    "victimized": 0.1,
    "vicious": 0.1,
    "valiant": 0.8,
    "verification": 0.65,
    "void": 0.2,
    "vulnerability": 0.2,
    "vulnerabilities": 0.2,
    "volume trading": 0.6,
    "value creation": 0.75,
    "vertical": 0.65,

    "war": 0.2,
    "wars": 0.15,
    "wealth": 0.9,
    "win": 0.9,
    "wins": 0.9,
    "won": 0.9,
    "weakness": 0.2,
    "weaknesses": 0.2,
    "weak": 0.2,
    "weaker": 0.2,
    "weaker than expected": 0.15,
    "withdraw": 0.3,
    "withdrawal": 0.3,
    "withdrawals": 0.3,
    "wave": 0.6,
    "waves": 0.6,
    "wealthy": 0.8,
    "widening": 0.7,
    "wide": 0.7,
    "wholesale": 0.7,
    "well being": 0.8,
    "workforce": 0.7,
    "worst case": 0.2,
    "warning": 0.2,
    "warnings": 0.2,
    "winners": 0.9,
    "win win": 0.9,
    "worth": 0.8,
    "write off": 0.2,
    "wage growth": 0.7,
    "waterfall": 0.6,
    "waterfalls": 0.6,
    "worsen": 0.1,
    "worse": 0.1,
    "worst": 0.1,
    "weaken": 0.2,
    "waiting": 0.4,
    "widen": 0.6,
    "worry": 0.2,
    "worried": 0.2,
    "welfare": 0.8,
    "whipsaw": 0.2,
    "wild": 0.3,
    "winds": 0.6,
    "wind": 0.6,
    
    "x efficiency": 0.7,
    "x factor": 0.8,
    "xenocurrency": 0.6,
    "xenophobic": 0.2,
    "xerox effect": 0.5,
    "xit": 0.3,

    "yield": 0.7,
    "yields curve": 0.6,
    "young market": 0.4,
    "young": 0.4,
    "yellow flag": 0.3,
    "yield spread": 0.7,
    "yield growth": 0.8,
    "yield risk": 0.3,
    "yield risks": 0.3,

    "z score": 0.7,
    "zombie company": 0.2,
    "zombie bank": 0.2,
    "zigzag market": 0.35,
    "zig zag": 0.35,
    "zigzag": 0.35,
    "zenith": 0.8,
    "zero coupon": 0.6,
    "zero inflation": 0.7,
    "zero sum": 0.4,

    #Figure importanti
    "warren buffett": 0.80,
    "elon musk": 0.65,
    "musk": 0.65,
    "donald trump": 0.3,
    "trump": 0.3,
    "jim cramer": 0.65,
    "cathie wood": 0.65,
    "jerome powell": 0.65,
    "jamie dimon": 0.65,
    "ray dalio": 0.65,
    "peter thiel": 0.65,
    "bill ackman": 0.60,
    "charlie munger": 0.65,
    "larry fink": 0.65,
    "michael burry": 0.65,
    "ken griffin": 0.65,
    "david tepper": 0.65,
    "george soros": 0.65,
    "jeff bezos": 0.65,
    "mark zuckerberg": 0.65,
    "tim cook": 0.65,
    "sundar pichai": 0.65,
    "satya nadella": 0.65,
    "sam altman": 0.65,
    "kathy jones": 0.65,
    "liz ann sonders": 0.65,
    "paul tudor jones": 0.65
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
def calculate_sentiment(news, decay_factor=0.03):    #Prima era 0.06
    """Calcola il sentiment medio ponderato di una lista di titoli di notizie."""
    total_sentiment = 0
    total_weight = 0
    now = datetime.utcnow()

    for title, date in news:
        days_old = (now - date).days  # Calcola l'età della notizia in giorni
        weight = math.exp(-decay_factor * days_old)  # Applica il decadimento esponenziale

        normalized_title = normalize_text(title)  # Normalizza il titolo
        sentiment_score = 0
        count = 0

        words = normalized_title.split()  # Parole del titolo
        lemmatized_words = lemmatize_words(words)  # Lemmatizza le parole

        for i, word in enumerate(lemmatized_words):
            if word in sentiment_dict:
                score = sentiment_dict[word]

                if i > 0 and lemmatized_words[i - 1] in negation_words:
                    score = 1 - score  # Inverto il punteggio

                sentiment_score += score
                count += 1

        if count != 0:
            sentiment_score /= count  # Normalizza il punteggio
        else:
            sentiment_score = 0.5  # Sentiment neutro se nessuna parola è trovata

        total_sentiment += sentiment_score * weight
        total_weight += weight

    if total_weight > 0:
        average_sentiment = total_sentiment / total_weight
    else:
        average_sentiment = 0.5  # Sentiment neutro se non ci sono notizie

    return average_sentiment




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
            # Scarica i dati storici per l'asset
            data = yf.download(adjusted_symbol, period="3mo", interval="1d", auto_adjust=True)
            if data.empty:
                raise ValueError(f"Nessun dato disponibile per {symbol}.")
            
            data.dropna(inplace=True)

            close = data['Close'].squeeze()
            high = data['High'].squeeze()
            low = data['Low'].squeeze()
    
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
            info = ticker_obj.info  # dizionario con decine di campi

            # Scegli i campi che ti interessano, ad es.:
            fondamentali = {
                "Trailing P/E": info.get("trailingPE", "N/A"),
                "Forward P/E": info.get("forwardPE", "N/A"),
                "EPS Growth (YoY)": info.get("earningsQuarterlyGrowth", "N/A"),
                "Revenue Growth (YoY)": info.get("revenueGrowth", "N/A"),
                "Profit Margins": info.get("profitMargins", "N/A"),
                "Debt to Equity": info.get("debtToEquity", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A")
            }


            # Costruisci la tabella HTML
            tabella_fondamentali = (
                pd.DataFrame(fondamentali.items(), columns=["Fundamentale", "Valore"])
                  .to_html(index=False, border=0, float_format="%.4f")
            )


            
            #percentuale = calcola_punteggio(indicators, close.iloc[-1], bb_upper, bb_lower)
            percentuali_tecniche[symbol] = percentuale
            
            # Crea tabella dei dati storici (ultimi 90 giorni)
            dati_storici = data.tail(90)
            dati_storici['Date'] = dati_storici.index.strftime('%Y-%m-%d')  # Aggiungi la colonna Date
            dati_storici_html = dati_storici[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].to_html(index=False, border=1)

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
        for title, news_date in news_data["last_90_days"]:
            title_sentiment = calculate_sentiment([(title, news_date)])  # Passa (titolo, data)
            all_news_entries.append((symbol, title, title_sentiment))

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

    return sentiment_results, percentuali_combine, all_news_entries






# Calcolare il sentiment medio per ogni simbolo
sentiment_for_symbols, percentuali_combine, all_news_entries = get_sentiment_for_all_symbols(symbol_list)



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



# Creazione del file news.html con i titoli e il sentiment
html_news = ["<html><head><title>Notizie e Sentiment</title></head><body>",
             "<h1>Notizie Finanziarie con Sentiment</h1>",
             "<table border='1'><tr><th>Simbolo</th><th>Notizia</th><th>Sentiment</th></tr>"]

for symbol, title, sentiment in all_news_entries:
    html_news.append(f"<tr><td>{symbol}</td><td>{title}</td><td>{sentiment:.2f}</td></tr>")

html_news.append("</table></body></html>")

try:
    contents = repo.get_contents(news_path)
    repo.update_file(contents.path, "Updated news sentiment", "\n".join(html_news), contents.sha)
except GithubException:
    repo.create_file(news_path, "Created news sentiment", "\n".join(html_news))

print("News aggiornata con successo!")
