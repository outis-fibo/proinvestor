import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
import hashlib
from pathlib import Path
import time
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

# Environment variables'ı yükle
load_dotenv()

# ============================================
# 🚀 HYBRID CACHE SİSTEMİ - Streamlit + Disk
# ============================================
import streamlit as st

# Disk cache için SmartCache
class DiskCache:
    """Disk tabanlı kalıcı cache"""
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key):
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key, ttl_seconds=300):
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            timestamp = cached_data.get('timestamp', 0)
            if time.time() - timestamp > ttl_seconds:
                cache_path.unlink()
                return None
            return cached_data.get('data')
        except:
            return None
    
    def set(self, key, data):
        cache_path = self._get_cache_path(key)
        cached_data = {'timestamp': time.time(), 'data': data}
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            return True
        except:
            return False

_disk_cache = DiskCache()

@st.cache_data(ttl=300, show_spinner=False)
def calculate_technical_indicators_optimized(symbol, market_type):
    """
    TÜM teknik göstergeleri hesapla ve cache'le
    PERFORMANS: Bu fonksiyon sayesinde TEKNİK sekmesi %80 daha hızlı
    """
    # DataFrame'i session state'den al
    if 'cached_df_long' not in st.session_state:
        return None
    
    df_long = st.session_state.cached_df_long
    curr_price = st.session_state.cached_info.get('regularMarketPrice', 0)
    
    if len(df_long) < 200:
        return None
    
    # === EMA Hesaplamaları ===
    ema10 = df_long['Close'].ewm(span=10, adjust=False).mean().iloc[-1]
    ema20 = df_long['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema100 = df_long['Close'].ewm(span=100, adjust=False).mean().iloc[-1]
    ema200 = df_long['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
    
    # === RSI ===
    delta = df_long['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # === MACD ===
    ema12 = df_long['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_long['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    current_macd_val = macd_hist.iloc[-1]
    
    # === Bollinger Bands ===
    bb_period = 20
    bb_std = 2
    bb_middle = df_long['Close'].rolling(window=bb_period).mean()
    bb_std_dev = df_long['Close'].rolling(window=bb_period).std()
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    bb_position = (curr_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100
    
    # === Stochastic ===
    period = 14
    low_min = df_long['Low'].rolling(window=period).min()
    high_max = df_long['High'].rolling(window=period).max()
    stoch_k = 100 * ((df_long['Close'] - low_min) / (high_max - low_min))
    current_stoch = stoch_k.iloc[-1]
    
    # === ATR ve ADX ===
    tr = pd.concat([df_long['High'] - df_long['Low'], 
                   abs(df_long['High'] - df_long['Close'].shift()), 
                   abs(df_long['Low'] - df_long['Close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    dx = 100 * abs( (100*(df_long['High'].diff().where((df_long['High'].diff() > -df_long['Low'].diff()) & (df_long['High'].diff() > 0), 0).rolling(window=14).mean() / atr)) - (100*(-df_long['Low'].diff().where((-df_long['Low'].diff() > df_long['High'].diff()) & (-df_long['Low'].diff() > 0), 0).rolling(window=14).mean() / atr)) ) / ( (100*(df_long['High'].diff().where((df_long['High'].diff() > -df_long['Low'].diff()) & (df_long['High'].diff() > 0), 0).rolling(window=14).mean() / atr)) + (100*(-df_long['Low'].diff().where((-df_long['Low'].diff() > df_long['High'].diff()) & (-df_long['Low'].diff() > 0), 0).rolling(window=14).mean() / atr)) )
    adx = dx.rolling(window=14).mean().iloc[-1]
    
    # === Diğer Metrikler ===
    atr_percent = (atr.iloc[-1] / curr_price) * 100 if curr_price > 0 else 0
    avg_volume = df_long['Volume'].rolling(window=20).mean().iloc[-1]
    volume_ratio = df_long['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
    
    # === Pivot Points ===
    pivot = (df_long['High'].iloc[-1] + df_long['Low'].iloc[-1] + df_long['Close'].iloc[-1]) / 3
    r1 = 2 * pivot - df_long['Low'].iloc[-1]
    r2 = pivot + (df_long['High'].iloc[-1] - df_long['Low'].iloc[-1])
    s1 = 2 * pivot - df_long['High'].iloc[-1]
    s2 = pivot - (df_long['High'].iloc[-1] - df_long['Low'].iloc[-1])
    
    return {
        'ema10': ema10, 'ema20': ema20, 'ema100': ema100, 'ema200': ema200,
        'current_rsi': current_rsi, 'current_macd_val': current_macd_val,
        'bb_position': bb_position, 'bb_upper': bb_upper.iloc[-1], 'bb_lower': bb_lower.iloc[-1],
        'current_stoch': current_stoch, 'adx': adx,
        'atr_percent': atr_percent, 'volume_ratio': volume_ratio,
        'pivot': pivot, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2
    }

# ============================================
# 📥 EXPORT FONKSİYONLARI
# ============================================

def generate_pdf_report(symbol, info, curr_price, df_long, indicators=None):
    """PDF raporu oluştur"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Başlık
    title_text = f"{symbol} - Hisse Analiz Raporu"
    title = Paragraph(title_text, styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Tarih
    date_text = f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
    date_para = Paragraph(date_text, styles['Normal'])
    story.append(date_para)
    story.append(Spacer(1, 20))
    
    # Şirket Bilgileri Tablosu
    company_data = [
        ['ŞİRKET BİLGİLERİ', ''],
        ['Şirket Adı:', info.get('longName', 'N/A')],
        ['Sembol:', symbol],
        ['Sektör:', info.get('sector', 'N/A')],
        ['Endüstri:', info.get('industry', 'N/A')],
        ['Borsa:', info.get('exchange', 'N/A')],
    ]
    
    company_table = Table(company_data, colWidths=[150, 350])
    company_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(company_table)
    story.append(Spacer(1, 20))
    
    # Fiyat Bilgileri
    reg_change = info.get('regularMarketChange', 0)
    reg_pct = info.get('regularMarketChangePercent', 0)
    
    price_data = [
        ['FİYAT BİLGİLERİ', ''],
        ['Güncel Fiyat:', f"${curr_price:.2f}"],
        ['Değişim:', f"${reg_change:+.2f} ({reg_pct:+.2f}%)"],
        ['52 Hafta Düşük:', f"${info.get('fiftyTwoWeekLow', 0):.2f}"],
        ['52 Hafta Yüksek:', f"${info.get('fiftyTwoWeekHigh', 0):.2f}"],
        ['Market Cap:', f"${info.get('marketCap', 0):,.0f}"],
    ]
    
    price_table = Table(price_data, colWidths=[150, 350])
    price_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(price_table)
    story.append(Spacer(1, 20))
    
    # Teknik Göstergeler (varsa)
    if indicators:
        tech_data = [
            ['TEKNİK GÖSTERGELER', ''],
            ['RSI (14):', f"{indicators.get('current_rsi', 0):.2f}"],
            ['MACD:', f"{indicators.get('current_macd_val', 0):.2f}"],
            ['Stochastic:', f"{indicators.get('current_stoch', 0):.2f}"],
            ['ADX:', f"{indicators.get('adx', 0):.2f}"],
            ['EMA 200:', f"${indicators.get('ema200', 0):.2f}"],
        ]
        
        tech_table = Table(tech_data, colWidths=[150, 350])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(tech_table)
    
    # PDF oluştur
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_excel_report(symbol, info, df_long, indicators=None):
    """Excel raporu oluştur"""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet 1: Genel Bilgiler
        overview_data = {
            'Metrik': [
                'Şirket Adı', 'Sembol', 'Sektör', 'Endüstri', 'Borsa',
                'Güncel Fiyat', 'Market Cap', '52W Düşük', '52W Yüksek',
                'P/E Ratio', 'P/B Ratio', 'Dividend Yield'
            ],
            'Değer': [
                info.get('longName', 'N/A'),
                symbol,
                info.get('sector', 'N/A'),
                info.get('industry', 'N/A'),
                info.get('exchange', 'N/A'),
                info.get('regularMarketPrice', 0),
                info.get('marketCap', 0),
                info.get('fiftyTwoWeekLow', 0),
                info.get('fiftyTwoWeekHigh', 0),
                info.get('trailingPE', 0),
                info.get('priceToBook', 0),
                info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            ]
        }
        pd.DataFrame(overview_data).to_excel(writer, sheet_name='Genel Bilgiler', index=False)
        
        # Sheet 2: Fiyat Geçmişi (Son 500 gün)
        price_history = df_long[['Open', 'High', 'Low', 'Close', 'Volume']].tail(500).copy()
        price_history.to_excel(writer, sheet_name='Fiyat Geçmişi')
        
        # Sheet 3: Teknik Göstergeler
        if indicators:
            tech_data = {
                'Gösterge': [],
                'Değer': []
            }
            for key, value in indicators.items():
                tech_data['Gösterge'].append(key)
                tech_data['Değer'].append(value)
            
            pd.DataFrame(tech_data).to_excel(writer, sheet_name='Teknik Göstergeler', index=False)
        
        # Sheet 4: Finansal Metrikler
        financial_data = {
            'Metrik': [
                'Revenue', 'Gross Profit', 'EBITDA', 'Net Income',
                'Total Cash', 'Total Debt', 'Revenue Growth', 'Earnings Growth'
            ],
            'Değer': [
                info.get('totalRevenue', 0),
                info.get('grossProfits', 0),
                info.get('ebitda', 0),
                info.get('netIncomeToCommon', 0),
                info.get('totalCash', 0),
                info.get('totalDebt', 0),
                info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
            ]
        }
        pd.DataFrame(financial_data).to_excel(writer, sheet_name='Finansal Metrikler', index=False)
    
    buffer.seek(0)
    return buffer

@st.cache_data(ttl=300, show_spinner=False)
def calculate_technical_indicators(df_long_hash, curr_price):
    """Tüm teknik göstergeleri bir kerede hesapla - CACHE'Lİ
    Not: df_long_hash DataFrame'in hash'i, gerçek DataFrame session_state'den alınacak
    """
    # DataFrame'i session_state'den al
    df_long = st.session_state.get('_df_long_temp')
    if df_long is None or len(df_long) < 200:
        return None
    
    # EMA'lar
    ema10 = df_long['Close'].ewm(span=10, adjust=False).mean().iloc[-1]
    ema20 = df_long['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema100 = df_long['Close'].ewm(span=100, adjust=False).mean().iloc[-1]
    ema200 = df_long['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
    
    # RSI
    delta = df_long['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # MACD
    ema12 = df_long['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_long['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    current_macd_val = macd_hist.iloc[-1]
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_middle = df_long['Close'].rolling(window=bb_period).mean()
    bb_std_dev = df_long['Close'].rolling(window=bb_period).std()
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    bb_position = (curr_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100
    
    # Stochastic
    period = 14
    low_min = df_long['Low'].rolling(window=period).min()
    high_max = df_long['High'].rolling(window=period).max()
    stoch_k = 100 * ((df_long['Close'] - low_min) / (high_max - low_min))
    current_stoch = stoch_k.iloc[-1]
    
    # ATR ve ADX
    tr = pd.concat([df_long['High'] - df_long['Low'], 
                   abs(df_long['High'] - df_long['Close'].shift()), 
                   abs(df_long['Low'] - df_long['Close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    dx = 100 * abs( (100*(df_long['High'].diff().where((df_long['High'].diff() > -df_long['Low'].diff()) & (df_long['High'].diff() > 0), 0).rolling(window=14).mean() / atr)) - (100*(-df_long['Low'].diff().where((-df_long['Low'].diff() > df_long['High'].diff()) & (-df_long['Low'].diff() > 0), 0).rolling(window=14).mean() / atr)) ) / ( (100*(df_long['High'].diff().where((df_long['High'].diff() > -df_long['Low'].diff()) & (df_long['High'].diff() > 0), 0).rolling(window=14).mean() / atr)) + (100*(-df_long['Low'].diff().where((-df_long['Low'].diff() > df_long['High'].diff()) & (-df_long['Low'].diff() > 0), 0).rolling(window=14).mean() / atr)) )
    adx = dx.rolling(window=14).mean().iloc[-1]
    
    return {
        'ema10': ema10, 'ema20': ema20, 'ema100': ema100, 'ema200': ema200,
        'current_rsi': current_rsi, 'current_macd_val': current_macd_val,
        'bb_position': bb_position, 'bb_upper': bb_upper.iloc[-1], 'bb_lower': bb_lower.iloc[-1],
        'current_stoch': current_stoch, 'adx': adx
    }

# --- 1. GÖRSEL TASARIM (CORE PROTOCOL 19 / OUTIS RESET) ---
st.set_page_config(page_title="ProInvestor AI Terminal", layout="wide")

st.markdown("""
    <style>
    /* ============================================
       KOMPAKT & PROFESYONEL UI OPTİMİZASYONU
       ============================================ */
    
    /* Ana sayfa padding'i azalt */
    .block-container { 
        padding-top: 2rem !important; 
        padding-bottom: 1rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    /* Streamlit elementleri arası boşlukları azalt */
    .stVerticalBlock { gap: 0.5rem !important; }
    .element-container { margin-bottom: 0.3rem !important; }
    
    /* Hisse başlık kartı - kompakt */
    .stock-tab-container {
        background-color: #161b22; 
        border: 1px solid #30363d;
        border-radius: 8px; 
        padding: 10px 20px;
        display: flex; 
        align-items: center; 
        gap: 20px;
        margin-bottom: 0.5rem;
    }
    
    /* Başlık fontları - biraz küçültüldü */
    .stock-title { 
        font-size: 2.2rem; 
        font-weight: 800; 
        color: #ffffff; 
        line-height: 1; 
        margin-right: 10px; 
    }
    .price-hero { 
        font-size: 1.8rem; 
        font-weight: 700; 
        line-height: 1; 
    }
    
    .price-up { color: #238636 !important; }
    .price-down { color: #da3633 !important; }
    
    /* Extended hours box - kompakt */
    .extended-hours-box {
        background-color: #1c2128; 
        border-left: 3px solid #f69e5d;
        padding: 6px 12px; 
        margin-top: 3px; 
        margin-bottom: 3px;
        border-radius: 4px;
        font-size: 0.85rem; 
        color: #adbac7; 
        display: inline-flex; 
        align-items: center; 
        gap: 10px;
    }
    
    .status-open { color: #238636; font-weight: 800; }
    .status-closed { color: #da3633; font-weight: 800; }
    .ext-up { color: #238636; font-weight: bold; }
    .ext-down { color: #da3633; font-weight: bold; }

    /* Info tag'ler - kompakt */
    .info-tag { 
        background-color: #1e293b; 
        color: #818cf8; 
        padding: 4px 12px; 
        border-radius: 6px; 
        font-size: 0.9rem; 
        font-weight: 500; 
        display: inline-block; 
        margin-right: 4px;
        margin-top: 2px;
        margin-bottom: 2px;
    }
    
    .update-text-inline { 
        color: #8b949e; 
        font-size: 0.85rem; 
        margin-left: 8px; 
        border-left: 1px solid #30363d; 
        padding-left: 12px; 
    }
    
    /* Toolbar metrics - kompakt */
    .toolbar-metrics {
        display: flex; 
        align-items: center; 
        gap: 15px; 
        font-size: 0.9rem; 
        color: #ffffff; 
        border-left: 1px solid #30363d; 
        padding-left: 15px;
    }
    .data-value { 
        color: #00f2ff !important; 
        font-weight: 800; 
        text-shadow: 0 0 8px rgba(0, 242, 255, 0.4); 
    }

    /* Section divider - daha ince */
    .section-divider {
        border: 0;
        height: 1px;
        background: #30363d;
        margin: 10px 0 !important;
    }

    /* Tabs - kompakt */
    .stTabs { 
        background-color: #161b22; 
        padding: 8px !important;
        border-radius: 8px; 
        border: 1px solid #30363d;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 6px 12px;
        font-size: 0.9rem;
    }
    
    /* Card'lar - kompakt */
    .tech-card, .mini-card-vertical {
        background: #0d1117; 
        border: 1px solid #30363d;
        border-radius: 6px; 
        padding: 8px; 
        text-align: center;
        height: 100%;
        margin: 2px 0;
    }
    
    .tech-card small {
        font-size: 0.7rem;
        display: block;
        margin-bottom: 4px;
    }
    
    .tech-card b {
        font-size: 1.1rem;
    }
    
    /* DataFrame - kompakt */
    [data-testid="stDataFrame"] {
        font-size: 0.85rem;
    }
    
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        text-align: left !important;
        padding: 4px 8px !important;
    }
    
    /* Markdown başlıkları - kompakt */
    h1 { 
        font-size: 1.8rem !important; 
        margin-top: 0.5rem !important; 
        margin-bottom: 0.5rem !important; 
    }
    h2 { 
        font-size: 1.5rem !important; 
        margin-top: 0.5rem !important; 
        margin-bottom: 0.5rem !important; 
    }
    h3 { 
        font-size: 1.2rem !important; 
        margin-top: 0.5rem !important; 
        margin-bottom: 0.5rem !important; 
    }
    
    /* Buttons - kompakt */
    .stButton button {
        padding: 6px 12px;
        font-size: 0.9rem;
        border-radius: 6px;
    }
    
    /* Metric cards - kompakt */
    [data-testid="stMetric"] {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    
    /* Expander - kompakt */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        padding: 6px 12px;
    }
    
    /* Text input - kompakt */
    .stTextInput input {
        padding: 6px 12px;
        font-size: 0.9rem;
    }
    
    /* Selectbox - kompakt */
    .stSelectbox select {
        padding: 6px 12px;
        font-size: 0.9rem;
    }
    
    /* Radio buttons - kompakt */
    .stRadio > div {
        gap: 8px;
    }
    
    /* Plotly grafikleri - kompakt margin */
    .js-plotly-plot {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Column gaps - azaltıldı */
    [data-testid="column"] {
        padding: 0 4px;
    }
    
    /* Info/warning/error boxes - kompakt */
    .stAlert {
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    
    /* Sidebar - kompakt */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Widget label gizle */
    [data-testid="stWidgetLabel"] { 
        display: none !important; 
    }
    
    /* Horizontal line - kompakt */
    hr {
        margin: 8px 0 !important;
        border-color: #30363d !important;
    }
    
    /* Spinner - kompakt */
    .stSpinner > div {
        border-width: 2px;
        width: 30px;
        height: 30px;
    }
    
    /* Progress bar - kompakt */
    .stProgress > div > div {
        height: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. STATE ---
if 'menu_secim' not in st.session_state:
    st.session_state.menu_secim = "GRAFİK"

# --- 3. VERİ ÇEKME & HİBRİT API SİSTEMİ ---
import requests
import json

# API Yardımcı Fonksiyonları
def get_fmp_data(symbol, api_key=None):
    """Financial Modeling Prep API - Alternatif veri kaynağı"""
    if not api_key or api_key == "demo":
        return None  # API key yoksa veya demo ise skip
    
    try:
        # Profile endpoint
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
        profile_response = requests.get(profile_url, timeout=5)
        
        # Key Metrics endpoint
        metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?apikey={api_key}"
        metrics_response = requests.get(metrics_url, timeout=5)
        
        # Financial Ratios endpoint
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?apikey={api_key}"
        ratios_response = requests.get(ratios_url, timeout=5)
        
        if profile_response.status_code == 200:
            profile_data = profile_response.json()
            metrics_data = metrics_response.json() if metrics_response.status_code == 200 else []
            ratios_data = ratios_response.json() if ratios_response.status_code == 200 else []
            
            return {
                'profile': profile_data[0] if profile_data else {},
                'metrics': metrics_data[0] if metrics_data else {},
                'ratios': ratios_data[0] if ratios_data else {}
            }
    except (requests.RequestException, requests.Timeout) as e:
        # Network hataları, timeout vb.
        return None
    except (ValueError, KeyError, IndexError) as e:
        # JSON parse hataları veya veri erişim hataları
        return None
    return None

def merge_data_sources(yf_info, fmp_data):
    """İki veri kaynağını birleştir - öncelik yfinance'de ama eksikler FMP'den doldurulur"""
    merged = yf_info.copy()
    
    if fmp_data:
        profile = fmp_data.get('profile', {})
        metrics = fmp_data.get('metrics', {})
        ratios = fmp_data.get('ratios', {})
        
        # Eksik verileri FMP'den doldur
        if not merged.get('trailingPE') and profile.get('pe'):
            merged['trailingPE'] = profile.get('pe')
        
        if not merged.get('priceToBook') and profile.get('priceToBook'):
            merged['priceToBook'] = profile.get('priceToBook')
        
        if not merged.get('returnOnEquity') and ratios.get('returnOnEquity'):
            merged['returnOnEquity'] = ratios.get('returnOnEquity')
        
        if not merged.get('returnOnAssets') and ratios.get('returnOnAssets'):
            merged['returnOnAssets'] = ratios.get('returnOnAssets')
        
        if not merged.get('debtToEquity') and ratios.get('debtEquityRatio'):
            merged['debtToEquity'] = ratios.get('debtEquityRatio') * 100
        
        if not merged.get('currentRatio') and ratios.get('currentRatio'):
            merged['currentRatio'] = ratios.get('currentRatio')
        
        if not merged.get('quickRatio') and ratios.get('quickRatio'):
            merged['quickRatio'] = ratios.get('quickRatio')
        
        if not merged.get('profitMargins') and ratios.get('netProfitMargin'):
            merged['profitMargins'] = ratios.get('netProfitMargin')
        
        if not merged.get('operatingMargins') and ratios.get('operatingProfitMargin'):
            merged['operatingMargins'] = ratios.get('operatingProfitMargin')
        
        if not merged.get('revenueGrowth') and metrics.get('revenuePerShareTTM'):
            # FMP'den büyüme hesapla
            pass
        
        if not merged.get('beta') and profile.get('beta'):
            merged['beta'] = profile.get('beta')
    
    return merged

def calculate_data_quality(info_dict):
    """Veri kalitesi skorunu hesapla"""
    critical_fields = {
        'trailingPE': 10,
        'forwardPE': 5,
        'priceToBook': 8,
        'trailingEps': 10,
        'bookValue': 8,
        'returnOnEquity': 10,
        'returnOnAssets': 7,
        'profitMargins': 10,
        'operatingMargins': 7,
        'revenueGrowth': 9,
        'earningsGrowth': 9,
        'debtToEquity': 8,
        'currentRatio': 7,
        'quickRatio': 5,
        'beta': 5,
        'targetMeanPrice': 8,
        'numberOfAnalystOpinions': 5,
        'freeCashflow': 7,
        'sharesOutstanding': 5,
    }
    
    total_possible = sum(critical_fields.values())
    score = 0
    missing_fields = []
    available_fields = []
    
    for field, weight in critical_fields.items():
        value = info_dict.get(field)
        if value and value != 0:
            score += weight
            available_fields.append(field)
        else:
            missing_fields.append(field)
    
    quality_percentage = (score / total_possible) * 100
    
    return {
        'score': score,
        'total': total_possible,
        'percentage': quality_percentage,
        'missing': missing_fields,
        'available': available_fields,
        'level': 'YÜKSEK' if quality_percentage >= 80 else 'ORTA' if quality_percentage >= 50 else 'DÜŞÜK'
    }

with st.sidebar:
    st.title("⚙️ Ayarlar")
    market_type = st.selectbox("Borsa Bölgesi", ["ABD (Global)", "Türkiye (BIST)", "İngiltere (LSE)"])
    
    # Daha önce seçilmiş bir symbol varsa onu göster
    default_symbol = st.session_state.get('current_symbol', '')
    symbol = st.text_input("Sembol", value=default_symbol, placeholder="Örn: AAPL, TSLA, MSFT").upper().strip()
    
    if symbol:
        ticker_symbol = symbol + (".IS" if market_type == "Türkiye (BIST)" else ".L" if market_type == "İngiltere (LSE)" else "")
        st.session_state['current_symbol'] = symbol  # Session'da sakla
    
    # Son kullanılan hisseler (geçmiş)
    st.markdown("---")
    st.markdown("**🕐 Son Kullanılan Hisseler:**")
    
    # Session state'de son kullanılanları sakla
    if 'recent_stocks' not in st.session_state:
        st.session_state.recent_stocks = []
    
    # Mevcut sembolü son kullanılanlara ekle (tekrar yoksa)
    if symbol and symbol not in st.session_state.recent_stocks:
        st.session_state.recent_stocks.insert(0, symbol)
        # Maksimum 6 hisse sakla
        st.session_state.recent_stocks = st.session_state.recent_stocks[:6]
    
    # Son kullanılanları göster
    if st.session_state.recent_stocks:
        cols = st.columns(3)
        for idx, stock in enumerate(st.session_state.recent_stocks[:6]):
            col_idx = idx % 3
            if cols[col_idx].button(stock, key=f"recent_{stock}", use_container_width=True):
                st.session_state['selected_symbol'] = stock
                st.rerun()
    else:
        st.markdown("<small style='color:#6e7681;'>Henüz hisse analiz etmediniz</small>", unsafe_allow_html=True)
    
    # API Ayarları (arka planda)
    use_hybrid = True  # Hibrit mod her zaman aktif
    fmp_api_key = os.getenv("FMP_API_KEY", "demo")  # .env dosyasından API key oku
    
    # === MODERN WATCHLIST ===
    st.markdown("---")
    
    import json
    import os
    
    WATCHLIST_FILE = "watchlist_data.json"
    
    def load_watchlist():
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def save_watchlist(watchlist):
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(watchlist, f, indent=2)
    
    watchlist = load_watchlist()
    
    # Modern başlık
    col_title, col_count = st.columns([3, 1])
    with col_title:
        st.markdown('<div style="color:#0969da; font-size:1.1rem; font-weight:700; margin-bottom:8px;">⭐ İzleme Listesi</div>', unsafe_allow_html=True)
    with col_count:
        if watchlist:
            st.markdown(f'<div style="background:#0969da; color:#fff; border-radius:12px; padding:2px 8px; text-align:center; font-size:0.75rem; font-weight:700; margin-top:2px;">{len(watchlist)}</div>', unsafe_allow_html=True)
    
    # Hisse ekleme
    new_symbol = st.text_input("", placeholder="Sembol ekle...", key="watch_input", label_visibility="collapsed").upper()
    if new_symbol and new_symbol not in watchlist:
        try:
            test = yf.Ticker(new_symbol)
            if not test.history(period="1d").empty:
                watchlist.append(new_symbol)
                save_watchlist(watchlist)
                st.rerun()
        except (ConnectionError, TimeoutError, ValueError):
            # Hisse bulunamazsa veya bağlantı hatas varsa sessizce geç
            pass
    
    # Watchlist kartları
    if watchlist:
        for sym in watchlist:
            try:
                stock_data = yf.Ticker(sym)
                hist = stock_data.history(period="5d")
                
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
                    change_pct = (change / hist['Close'].iloc[0]) * 100
                    
                    color = "#26a641" if change >= 0 else "#f85149"
                    arrow = "↗" if change >= 0 else "↘"
                    
                    # Tek satır kompakt kart
                    st.markdown(f"""
                        <div style='background:linear-gradient(135deg, #161b22 0%, #0d1117 100%); 
                                    border:1px solid #30363d; border-radius:6px; 
                                    padding:8px 10px; margin-bottom:4px;
                                    display:flex; justify-content:space-between; align-items:center;'>
                            <span style='color:#e6edf3; font-size:1rem; font-weight:700;'>{sym}</span>
                            <div style='display:flex; gap:10px; align-items:center;'>
                                <span style='color:#8b949e; font-size:1.2rem; font-weight:700;'>${price:.2f}</span>
                                <span style='color:{color}; font-size:1.2rem; font-weight:700;'>{arrow} {abs(change_pct):.1f}%</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Gizli butonlar - seçim ve silme için
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        if st.button(f"Analiz et", key=f"w_{sym}", use_container_width=True):
                            st.session_state['watch_selected'] = sym
                            st.rerun()
                    with col2:
                        if st.button("×", key=f"del_{sym}"):
                            watchlist.remove(sym)
                            save_watchlist(watchlist)
                            st.rerun()
            except (ConnectionError, TimeoutError, ValueError, IndexError, KeyError):
                # Veri çekilemezse veya boşsa sessizce atla
                pass
        
        # Temizle butonu
        if len(watchlist) > 1:
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            if st.button("🗑 Tümünü Temizle", key="clear_all", use_container_width=True):
                watchlist.clear()
                save_watchlist(watchlist)
                st.rerun()
    else:
        st.markdown("""
            <div style='text-align:center; padding:20px 10px; 
                        background:#0d1117; border:1px dashed #30363d; 
                        border-radius:8px; margin-top:10px;'>
                <div style='font-size:2rem; margin-bottom:8px; opacity:0.5;'>📊</div>
                <div style='color:#6e7681; font-size:0.85rem;'>Liste boş</div>
                <div style='color:#6e7681; font-size:0.7rem; margin-top:4px;'>Yukarıdan sembol ekleyin</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol, market_type):
    """Hisse verilerini çek ve cache'le - HYBRID CACHE"""
    import hashlib
    
    # Önce disk cache'e bak
    cache_key = f"stock_{symbol}_{market_type}"
    cached = _disk_cache.get(cache_key, ttl_seconds=300)
    
    if cached is not None:
        # Disk cache'den bulundu - SÜPER HIZLI
        return cached
    
    # Cache'de yok, yfinance'den çek
    ticker_symbol = symbol + (".IS" if market_type == "Türkiye (BIST)" else ".L" if market_type == "İngiltere (LSE)" else "")
    
    stock = yf.Ticker(ticker_symbol)
    df_long = stock.history(period="2y")
    info = stock.info
    
    result = {
        'df_long': df_long,
        'info': dict(info),
        'ticker_symbol': ticker_symbol,
        'symbol': symbol
    }
    
    # Disk cache'e kaydet
    _disk_cache.set(cache_key, result)
    
    return result

@st.cache_data(ttl=600, show_spinner=False)
def fetch_peers_data(symbol, sector, industry, api_key):
    """Peer şirketleri çek ve cache'le - ÖNCELİK: Manuel > FMP API > Industry > Sector"""
    import requests
    
    peers_list = []
    
    # ÖNCELİK 1: MANUEL PEER MAPPING (En doğru - uzman seçimi)
    if symbol in MANUAL_PEERS:
        peers_list = MANUAL_PEERS[symbol].copy()
        # Seçili hisseyi başa ekle
        if symbol not in peers_list:
            peers_list.insert(0, symbol)
        return peers_list[:15]
    
    # ÖNCELİK 2: FMP API'den peers al
    try:
        peers_url = f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={symbol}&apikey={api_key}"
        response = requests.get(peers_url, timeout=10)
        
        if response.status_code == 200:
            fmp_peers = response.json()
            if fmp_peers and len(fmp_peers) > 0:
                peers_data = fmp_peers[0] if isinstance(fmp_peers, list) else fmp_peers
                if 'peersList' in peers_data:
                    peers_list = peers_data['peersList'][:15]
                elif isinstance(peers_data, list):
                    peers_list = peers_data[:15]
    except (requests.RequestException, requests.Timeout, ValueError, KeyError):
        # API hatası veya veri parse edilemezse sessizce geç
        pass
    
    # 2. Eğer FMP'den peer gelmezse, aynı industry'den şirketleri bul
    if len(peers_list) < 3:
        # Industry bazlı manuel peer grupları
        industry_peers = {
            # Technology
            'Semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC'],
            'Software—Application': ['MSFT', 'ORCL', 'CRM', 'ADBE', 'INTU', 'NOW', 'WDAY', 'SNOW', 'DDOG', 'ZM', 'TEAM', 'HUBS'],
            'Software—Infrastructure': ['MSFT', 'ORCL', 'IBM', 'CSCO', 'VMW', 'PANW', 'CRWD', 'NET', 'ZS', 'S', 'OKTA', 'MDB'],
            'Consumer Electronics': ['AAPL', 'SONY', 'DELL', 'HPQ', 'LOGI', 'SONO'],
            'Internet Content & Information': ['GOOGL', 'META', 'NFLX', 'DIS', 'PINS', 'SNAP', 'SPOT', 'RBLX'],
            
            # Financial
            'Banks—Diversified': ['JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'SCHW'],
            'Investment Banking & Brokerage': ['GS', 'MS', 'SCHW', 'IBN', 'CS'],
            'Asset Management': ['BLK', 'BX', 'KKR', 'TROW', 'IVZ', 'BEN'],
            'Insurance—Property & Casualty': ['BRK.B', 'PGR', 'ALL', 'TRV', 'CB', 'AIG'],
            
            # Healthcare
            'Drug Manufacturers—General': ['JNJ', 'PFE', 'ABBV', 'LLY', 'MRK', 'BMY', 'GILD', 'AMGN', 'GSK', 'NVO'],
            'Biotechnology': ['AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'BNTX', 'SGEN'],
            'Medical Devices': ['ABT', 'TMO', 'DHR', 'ISRG', 'SYK', 'EW', 'ZBH', 'BSX'],
            'Health Care Plans': ['UNH', 'CVS', 'CI', 'HUM', 'ELV', 'CNC'],
            
            # Consumer
            'Auto Manufacturers': ['TSLA', 'F', 'GM', 'TM', 'HMC', 'STLA', 'RIVN', 'LCID'],
            'Restaurants': ['MCD', 'SBUX', 'YUM', 'CMG', 'QSR', 'DPZ', 'WING'],
            'Specialty Retail': ['HD', 'LOW', 'TGT', 'COST', 'BBY', 'ULTA', 'AZO', 'ORLY'],
            'Apparel Retail': ['NKE', 'LULU', 'TJX', 'ROST', 'GPS', 'UAA'],
            
            # Energy
            'Oil & Gas Integrated': ['XOM', 'CVX', 'COP', 'BP', 'SHEL', 'TTE'],
            'Oil & Gas E&P': ['EOG', 'PXD', 'DVN', 'FANG', 'MRO', 'APA', 'OXY'],
            'Oil & Gas Equipment & Services': ['SLB', 'HAL', 'BKR', 'FTI', 'NOV'],
            
            # E-commerce & Retail
            'Internet Retail': ['AMZN', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY'],
            'Discount Stores': ['WMT', 'COST', 'TGT', 'DG', 'DLTR', 'BIG'],
        }
        
        peers_list = industry_peers.get(industry, [])
        
        # Hala peer bulunamadıysa, aynı sector'dan popüler hisseleri al
        if len(peers_list) < 3:
            sector_fallback = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NOW', 'SNOW', 'PANW', 'CRWD'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB'],
                'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'ABT', 'ISRG', 'AMGN'],
                'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG'],
                'Communication Services': ['META', 'GOOGL', 'DIS', 'NFLX', 'T', 'VZ', 'CMCSA', 'TMUS'],
                'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'VLO'],
                'Industrials': ['BA', 'UNP', 'HON', 'UPS', 'CAT', 'LMT', 'RTX', 'GE', 'DE', 'MMM'],
                'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'MDLZ', 'CL', 'KMB'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB'],
                'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'DOW', 'PPG'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED']
            }
            peers_list = sector_fallback.get(sector, [])
    
    # 3. Seçili hisseyi listeye ekle
    if symbol not in peers_list:
        peers_list = [symbol] + peers_list
    
    return peers_list[:15]  # Maksimum 15 peer

@st.cache_data(ttl=300, show_spinner=False)
def fetch_period_history(symbol, market_type, period):
    """Belirli bir period için history çek - HYBRID CACHE"""
    # Önce disk cache'e bak
    cache_key = f"history_{symbol}_{market_type}_{period}"
    cached = _disk_cache.get(cache_key, ttl_seconds=300)
    
    if cached is not None:
        return cached
    
    # Cache'de yok, çek
    ticker_symbol = symbol + (".IS" if market_type == "Türkiye (BIST)" else ".L" if market_type == "İngiltere (LSE)" else "")
    stock = yf.Ticker(ticker_symbol)
    result = stock.history(period=period)
    
    # Disk cache'e kaydet
    _disk_cache.set(cache_key, result)
    
    return result

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_data(symbol):
    """Haberleri çek ve cache'le"""
    import requests
    import xml.etree.ElementTree as ET
    from datetime import datetime
    
    try:
        rss_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        news_list = []
        for item in items[:10]:
            title_elem = item.find('title')
            link_elem = item.find('link')
            pub_date_elem = item.find('pubDate')
            source_elem = item.find('source')
            
            title = title_elem.text if title_elem is not None else 'Başlık yok'
            link = link_elem.text if link_elem is not None else '#'
            pub_date = pub_date_elem.text if pub_date_elem is not None else ''
            source = source_elem.text if source_elem is not None else 'Google News'
            
            # Tarih formatla
            try:
                pub_datetime = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                time_ago = datetime.now() - pub_datetime
                if time_ago.days > 0:
                    time_str = f"{time_ago.days} gün önce"
                elif time_ago.seconds // 3600 > 0:
                    time_str = f"{time_ago.seconds // 3600} saat önce"
                else:
                    time_str = f"{time_ago.seconds // 60} dakika önce"
            except (ValueError, AttributeError):
                # Tarih parse edilemezse orijinal metni kullan
                time_str = pub_date
            
            news_list.append({
                'title': title,
                'link': link,
                'source': source,
                'time': time_str
            })
        
        return news_list
    except (requests.RequestException, requests.Timeout) as e:
        # HTTP request hataları
        return []
    except ET.ParseError as e:
        # XML parsing hataları
        return []
    except (KeyError, AttributeError) as e:
        # Veri erişim hataları
        return []

def fetch_single_peer_sync(peer):
    """Tek bir peer için veri çek - thread-safe"""
    try:
        peer_stock = yf.Ticker(peer)
        peer_info = peer_stock.info
        
        market_cap = peer_info.get('marketCap', 0)
        if market_cap > 0:
            return {
                'Sembol': peer,
                'Şirket': peer_info.get('shortName', peer)[:30],
                'Fiyat': peer_info.get('regularMarketPrice', 0),
                'P/E': peer_info.get('trailingPE', 0),
                'P/B': peer_info.get('priceToBook', 0),
                'ROE': peer_info.get('returnOnEquity', 0) * 100 if peer_info.get('returnOnEquity') else 0,
                'EPS': peer_info.get('trailingEps', 0),  # Earnings Per Share
                'Market Cap': market_cap / 1e9,
            }
    except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError):
        # Peer verisi çekilemezse None dön
        pass
    return None

def fetch_peers_parallel(peers_list, max_workers=10):
    """Paralel olarak peer verilerini çek - ThreadPoolExecutor kullanarak"""
    comparison_data = []
    
    # ThreadPoolExecutor ile paralel çalıştır
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Tüm peer'ları aynı anda başlat
        futures = {executor.submit(fetch_single_peer_sync, peer): peer for peer in peers_list}
        
        # Sonuçları topla (tamamlanma sırasına göre)
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            result = future.result()
            if result:
                comparison_data.append(result)
    
    return comparison_data

# ===== MANUEL PEER MAPPING - UZMAN SEÇİMİ =====
# Gerçek analistlerin kullandığı peer grupları
MANUAL_PEERS = {
    # Technology - Consumer Electronics & Ecosystems
    'AAPL': ['MSFT', 'GOOGL', 'SONY', 'DELL'],
    'MSFT': ['AAPL', 'GOOGL', 'ORCL', 'CRM', 'IBM'],
    'GOOGL': ['MSFT', 'AAPL', 'META', 'AMZN'],
    'GOOG': ['MSFT', 'AAPL', 'META', 'AMZN'],
    
    # Technology - Semiconductors
    'NVDA': ['AMD', 'INTC', 'QCOM', 'AVGO', 'MU'],
    'AMD': ['NVDA', 'INTC', 'QCOM', 'MU'],
    'INTC': ['AMD', 'NVDA', 'QCOM', 'TXN', 'AVGO'],
    'QCOM': ['NVDA', 'AMD', 'AVGO', 'MRVL'],
    'AVGO': ['NVDA', 'QCOM', 'INTC', 'TXN'],
    'MU': ['NVDA', 'AMD', 'WDC', 'STX'],
    'TXN': ['INTC', 'AVGO', 'ADI', 'MCHP'],
    
    # Automotive - EV & Traditional
    'TSLA': ['RIVN', 'LCID', 'F', 'GM', 'NIO'],
    'RIVN': ['TSLA', 'LCID', 'F', 'GM'],
    'LCID': ['TSLA', 'RIVN', 'FSR', 'NKLA'],
    'F': ['GM', 'TSLA', 'STLA', 'TM', 'HMC'],
    'GM': ['F', 'TSLA', 'STLA', 'TM'],
    'NIO': ['TSLA', 'XPEV', 'LI'],
    
    # Social Media & Digital Platforms
    'META': ['GOOGL', 'SNAP', 'PINS', 'RBLX', 'MTCH'],
    'SNAP': ['META', 'PINS', 'RBLX'],
    'PINS': ['META', 'SNAP', 'ETSY'],
    'TWTR': ['META', 'SNAP', 'PINS'],
    
    # E-commerce & Retail
    'AMZN': ['WMT', 'TGT', 'COST', 'SHOP'],
    'SHOP': ['AMZN', 'EBAY', 'ETSY', 'W', 'BIGC'],
    'EBAY': ['AMZN', 'SHOP', 'ETSY', 'W'],
    'ETSY': ['SHOP', 'EBAY', 'W', 'PINS'],
    'WMT': ['AMZN', 'TGT', 'COST', 'KR', 'DG'],
    'TGT': ['WMT', 'COST', 'DG', 'DLTR'],
    'COST': ['WMT', 'TGT', 'BJ', 'PSMT'],
    
    # Streaming & Entertainment
    'NFLX': ['DIS', 'PARA', 'WBD', 'SPOT'],
    'DIS': ['NFLX', 'PARA', 'WBD', 'CMCSA'],
    'SPOT': ['NFLX', 'AAPL', 'GOOGL', 'AMZN'],
    'RBLX': ['U', 'EA', 'TTWO', 'ATVI'],
    
    # Cloud & Enterprise Software
    'CRM': ['MSFT', 'ORCL', 'NOW', 'ADBE', 'INTU'],
    'ORCL': ['MSFT', 'CRM', 'IBM', 'SAP'],
    'NOW': ['CRM', 'MSFT', 'WDAY', 'SNOW'],
    'SNOW': ['NOW', 'DDOG', 'MDB', 'PLTR'],
    'DDOG': ['SNOW', 'SPLK', 'ESTC', 'NET'],
    'ADBE': ['CRM', 'INTU', 'MSFT', 'ADSK'],
    
    # Cybersecurity
    'CRWD': ['PANW', 'ZS', 'FTNT', 'S'],
    'PANW': ['CRWD', 'ZS', 'FTNT', 'CHKP'],
    'ZS': ['CRWD', 'PANW', 'OKTA', 'NET'],
    
    # Payment Processors
    'V': ['MA', 'PYPL', 'SQ', 'AXP'],
    'MA': ['V', 'PYPL', 'SQ', 'AXP'],
    'PYPL': ['V', 'MA', 'SQ', 'COIN'],
    'SQ': ['PYPL', 'V', 'MA', 'AFRM'],
    
    # Financial Services - Banks
    'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS'],
    'BAC': ['JPM', 'WFC', 'C', 'USB'],
    'WFC': ['JPM', 'BAC', 'C', 'USB'],
    'C': ['JPM', 'BAC', 'WFC', 'GS'],
    'GS': ['MS', 'JPM', 'C', 'SCHW'],
    'MS': ['GS', 'JPM', 'C', 'SCHW'],
    
    # Biotech & Pharma
    'JNJ': ['PFE', 'ABBV', 'MRK', 'LLY', 'BMY'],
    'PFE': ['JNJ', 'ABBV', 'MRK', 'LLY', 'GSK'],
    'ABBV': ['JNJ', 'PFE', 'GILD', 'AMGN', 'BMY'],
    'LLY': ['JNJ', 'PFE', 'MRK', 'ABBV', 'NVO'],
    'MRNA': ['BNTX', 'NVAX', 'PFE'],
    'GILD': ['ABBV', 'AMGN', 'REGN', 'VRTX'],
    
    # Healthcare - Insurance & Services
    'UNH': ['CVS', 'CI', 'HUM', 'ELV', 'CNC'],
    'CVS': ['UNH', 'CI', 'WBA', 'HUM'],
    
    # Energy - Oil & Gas
    'XOM': ['CVX', 'COP', 'BP', 'SHEL', 'TTE'],
    'CVX': ['XOM', 'COP', 'BP', 'SHEL'],
    'COP': ['XOM', 'CVX', 'EOG', 'PXD'],
    'SLB': ['HAL', 'BKR', 'NOV', 'FTI'],
    
    # Consumer - Food & Beverage
    'KO': ['PEP', 'DPS', 'MNST', 'CELH'],
    'PEP': ['KO', 'DPS', 'MNST', 'KDP'],
    'SBUX': ['MCD', 'CMG', 'YUM', 'DPZ'],
    'MCD': ['SBUX', 'YUM', 'QSR', 'WEN'],
    
    # Consumer - Apparel & Retail
    'NKE': ['LULU', 'ADDYY', 'UAA', 'DECK'],
    'LULU': ['NKE', 'UAA', 'PTON', 'GOOS'],
    
    # Airlines
    'DAL': ['UAL', 'AAL', 'LUV', 'JBLU'],
    'UAL': ['DAL', 'AAL', 'LUV', 'ALK'],
    
    # Hotels & Travel
    'MAR': ['HLT', 'H', 'IHG', 'BKNG'],
    'BKNG': ['EXPE', 'ABNB', 'TRIP'],
    'ABNB': ['BKNG', 'EXPE', 'VRBO'],
    
    # Real Estate
    'AMT': ['CCI', 'SBAC', 'EQIX', 'DLR'],
    'PLD': ['DRE', 'FR', 'EGP', 'STAG'],
    
    # Utilities
    'NEE': ['DUK', 'SO', 'D', 'AEP'],
    'DUK': ['NEE', 'SO', 'D', 'AEP'],
}

# Watchlist'ten seçim kontrolü
# Popüler hisseden seçim yapıldıysa
if 'selected_symbol' in st.session_state and st.session_state.get('selected_symbol'):
    symbol = st.session_state['selected_symbol']
    st.session_state['selected_symbol'] = None
    st.session_state['current_symbol'] = symbol  # Session'da sakla

# Watchlist'ten seçim kontrolü
if 'watch_selected' in st.session_state and st.session_state.get('watch_selected'):
    symbol = st.session_state['watch_selected']
    st.session_state['watch_selected'] = None
    st.session_state['current_symbol'] = symbol  # Session'da sakla

# Eğer daha önce seçilmiş bir symbol varsa onu kullan
if not symbol and 'current_symbol' in st.session_state:
    symbol = st.session_state['current_symbol']

# KARŞILAMA EKRANI - Hisse seçilmediyse
if not symbol:
    st.markdown("<h1 style='text-align:center; color:#00f2ff; text-shadow: 0 0 15px rgba(0, 242, 255, 0.5);'>📈 ProInvestor AI Terminal</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align:center; margin:50px 0;'>
            <h2 style='color:#8b949e;'>Hisse Senedi Analizine Hoş Geldiniz</h2>
            <p style='color:#6e7681; font-size:1.2rem; margin:20px 0;'>
                Başlamak için sol taraftan bir hisse senedi sembolü girin veya popüler hisselerden birini seçin
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Özellikler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background:#161b22; padding:30px; border-radius:12px; border:1px solid #30363d; text-align:center;'>
                <h3 style='color:#00f2ff;'>📊 Gerçek Zamanlı Veriler</h3>
                <p style='color:#8b949e;'>Yahoo Finance ve FMP API ile canlı piyasa verileri</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background:#161b22; padding:30px; border-radius:12px; border:1px solid #30363d; text-align:center;'>
                <h3 style='color:#00f2ff;'>📈 Teknik Analiz</h3>
                <p style='color:#8b949e;'>RSI, MACD, Bollinger Bands ve daha fazlası</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background:#161b22; padding:30px; border-radius:12px; border:1px solid #30363d; text-align:center;'>
                <h3 style='color:#00f2ff;'>🏭 Peers Karşılaştırma</h3>
                <p style='color:#8b949e;'>Rakip şirketlerle detaylı karşılaştırma</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hızlı başlangıç örnekleri
    st.markdown("""
        <div style='text-align:center; margin:40px 0;'>
            <h3 style='color:#8b949e;'>🚀 Hızlı Başlangıç</h3>
            <p style='color:#6e7681;'>Popüler hisse örnekleri:</p>
        </div>
    """, unsafe_allow_html=True)
    
    example_cols = st.columns(6)
    examples = [
        ("AAPL", "Apple"),
        ("TSLA", "Tesla"),
        ("MSFT", "Microsoft"),
        ("GOOGL", "Google"),
        ("NVDA", "NVIDIA"),
        ("META", "Meta")
    ]
    
    for idx, (sym, name) in enumerate(examples):
        with example_cols[idx]:
            if st.button(f"**{sym}**\n{name}", key=f"example_{sym}", use_container_width=True):
                st.session_state['selected_symbol'] = sym
                st.rerun()
    
    st.stop()  # Hisse seçilmediyse burada dur

# HİSSE SEÇİLDİYSE DEVAM ET
try:
    # API Ayarları
    use_hybrid = True
    fmp_api_key = os.getenv("FMP_API_KEY", "demo")
    
    # ============================================
    # 🚀 SESSION STATE CACHE - SEKMEler arası ANLIK geçiş
    # ============================================
    # Eğer aynı hisse için zaten veri varsa, TEKRAR ÇEKME!
    
    current_cache_key = f"{symbol}_{market_type}"
    
    # Session state'de bu hisse için veri var mı kontrol et
    if 'cached_symbol_key' not in st.session_state or st.session_state.cached_symbol_key != current_cache_key:
        # Yeni hisse - veri çek
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📊 Temel veriler yükleniyor...")
        progress_bar.progress(20)
        
        # Threading ile paralel veri çekimi
        results = {'stock_data': None, 'fmp_data': None}
        
        def fetch_main_data():
            results['stock_data'] = fetch_stock_data(symbol, market_type)
        
        def fetch_fmp_data():
            if use_hybrid and market_type == "ABD (Global)":
                results['fmp_data'] = get_fmp_data(symbol, fmp_api_key)
        
        # İki işlemi paralel başlat
        thread1 = threading.Thread(target=fetch_main_data)
        thread2 = threading.Thread(target=fetch_fmp_data)
        
        thread1.start()
        thread2.start()
        
        status_text.text("⚡ Veriler çekiliyor...")
        progress_bar.progress(50)
        
        # İkisinin de bitmesini bekle
        thread1.join()
        thread2.join()
        
        status_text.text("✅ Veriler hazırlanıyor...")
        progress_bar.progress(80)
        
        # Veriyi çıkar
        stock_data = results['stock_data']
        fmp_data = results['fmp_data']
        
        df_long = stock_data['df_long']
        info = stock_data['info']
        ticker_symbol = stock_data['ticker_symbol']
        
        # FMP verisi varsa birleştir
        if fmp_data:
            info = merge_data_sources(info, fmp_data)
        
        progress_bar.progress(100)
        status_text.text("✅ Yükleme tamamlandı!")
        
        # SESSION STATE'e kaydet - sekmeler arası anında geçiş için
        st.session_state.cached_symbol_key = current_cache_key
        st.session_state.cached_df_long = df_long
        st.session_state.cached_info = info
        st.session_state.cached_ticker_symbol = ticker_symbol
        st.session_state.cached_fmp_data = fmp_data  # fmp_data'yı da kaydet
        
        # Progress bar'ı temizle
        import time
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
    else:
        # AYNI HİSSE - Session state'den al (ANINDA!)
        df_long = st.session_state.cached_df_long
        info = st.session_state.cached_info
        ticker_symbol = st.session_state.cached_ticker_symbol
        fmp_data = st.session_state.get('cached_fmp_data', None)  # fmp_data'yı da al
    
    # 3. Veri kalitesi kontrolü
    data_quality = calculate_data_quality(info)
    
    # 4. Temel fiyat verileri
    curr_price = info.get('regularMarketPrice') or (df_long['Close'].iloc[-1] if not df_long.empty else 0)
    reg_change = info.get('regularMarketChange', 0) or 0
    reg_pct = info.get('regularMarketChangePercent', 0) or 0
    p_class = "price-up" if reg_change >= 0 else "price-down"


    # --- HEADER ---
    last_day = df_long.iloc[-1] if not df_long.empty else None
    
    main_header = f'<span class="stock-title">{info.get("longName", symbol)}</span>'
    price_section = f'<div><span class="price-hero {p_class}">{curr_price:.2f} {info.get("currency", "$")}</span><span style="font-size:1.2rem; margin-left:10px;" class="{p_class}">{reg_change:+.2f} ({reg_pct:+.2f}%)</span></div>'
    
    metrics_section = ""
    if last_day is not None:
        metrics_section = f"""
        <div class="toolbar-metrics">
            <span>AÇILIŞ: <b class="data-value">{last_day['Open']:.2f}</b></span>
            <span>DÜŞÜK: <b class="data-value">{last_day['Low']:.2f}</b></span>
            <span>YÜKSEK: <b class="data-value">{last_day['High']:.2f}</b></span>
        </div>
        """

    st.markdown(f"""
        <div class="stock-tab-container">
            {main_header}
            {price_section}
            {metrics_section}
        </div>
    """, unsafe_allow_html=True)
    
    # Watchlist ekle/çıkar butonu
    watchlist = load_watchlist()
    is_in_watchlist = symbol in watchlist
    
    # Market Status & Info
    m_state = info.get('marketState', 'UNKNOWN')
    status_class = "status-open" if m_state == "REGULAR" else "status-closed"
    ext_price = info.get('preMarketPrice') if m_state == "PRE" else info.get('postMarketPrice')
    ext_html = ""
    if ext_price:
        ext_change = ext_price - curr_price
        ext_pct = (ext_change / curr_price) * 100
        e_class = "ext-up" if ext_change >= 0 else "ext-down"
        ext_label = "PRE-MARKET" if m_state == "PRE" else "AFTER HOURS"
        ext_html = f"<b>{ext_label}:</b> {ext_price:.2f} <span class='{e_class}'>{ext_change:+.2f} ({ext_pct:+.2f}%)</span> | "
    
    last_update_time = datetime.now().strftime('%H:%M:%S')
    
    # Veri kalitesi rengi
    dq_color = "#238636" if data_quality['level'] == 'YÜKSEK' else "#f69e5d" if data_quality['level'] == 'ORTA' else "#da3633"
    dq_icon = "✅" if data_quality['level'] == 'YÜKSEK' else "⚠️" if data_quality['level'] == 'ORTA' else "🔴"
    
    # API kaynağı bilgisi
    api_source = "📡 Yahoo Finance"
    if use_hybrid and fmp_data and market_type == "ABD (Global)":
        api_source = "🔄 Hibrit Mod (Yahoo + FMP)"
    
    st.markdown(f"""
        <div class='extended-hours-box'>
            {ext_html}STATUS: <span class='{status_class}'>{m_state}</span>
            <span class='update-text-inline'>🕒 Son Güncelleme: {last_update_time}</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Veri Kalitesi Göstergesi
    st.markdown(f"""
        <div style='background:#0d1117; border:1px solid {dq_color}; border-radius:8px; padding:10px; margin-top:10px; margin-bottom:15px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='color:#8b949e; font-size:0.85rem;'>{api_source}</span>
                    <span style='color:#8b949e; margin:0 10px;'>|</span>
                    <span style='color:#8b949e; font-size:0.85rem;'>VERİ KALİTESİ:</span>
                    <b style='color:{dq_color}; margin-left:8px; font-size:1.1rem;'>{dq_icon} {data_quality['level']}</b>
                    <span style='color:#00f2ff; margin-left:8px; font-size:0.9rem;'>(%{data_quality['percentage']:.0f})</span>
                </div>
                <div style='color:#8b949e; font-size:0.85rem;'>
                    <span>Mevcut: {len(data_quality['available'])}/20 metrik</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Eğer veri kalitesi düşükse uyarı göster
    if data_quality['level'] == 'DÜŞÜK' or data_quality['percentage'] < 60:
        with st.expander("⚠️ EKSIK VERİLER - Detayları Göster", expanded=False):
            st.warning(f"**Dikkat:** Bazı önemli finansal veriler eksik. Analiz sonuçları tam doğru olmayabilir.")
            
            missing_critical = [f for f in data_quality['missing'] if f in ['trailingPE', 'trailingEps', 'returnOnEquity', 'profitMargins', 'revenueGrowth']]
            if missing_critical:
                st.error(f"**Kritik Eksik Veriler:** {', '.join(missing_critical)}")
            
            st.info(f"**Tüm Eksik Veriler ({len(data_quality['missing'])}):** {', '.join(data_quality['missing'][:10])}{'...' if len(data_quality['missing']) > 10 else ''}")
            
            if market_type == "ABD (Global)" and not use_hybrid:
                st.info("💡 **İpucu:** Sidebar'dan 'Hibrit API Kullan' seçeneğini aktif ederek daha fazla veriye erişebilirsiniz.")
            elif use_hybrid and fmp_api_key == "demo":
                st.info("💡 **İpucu:** Ücretsiz FMP API key alarak (financialmodelingprep.com) veri kalitesini artırabilirsiniz.")

    
    exchange_map = {"NMS": "NASDAQ", "NYQ": "NYSE", "ISE": "BORSA İSTANBUL", "LSE": "LONDON STOCK EXCH."}
    exch = exchange_map.get(info.get('exchange'), info.get('exchange', 'N/A'))
    st.markdown(f"""
        <div style='margin-top:15px;'>
            <span class='info-tag'>🏛️ {exch}</span>
            <span class='info-tag'>📂 {info.get('sector', 'N/A')}</span>
            <span class='info-tag'>📑 {info.get('industry', 'N/A')}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # --- NAVİGASYON ---
    n_cols = st.columns(6)
    if n_cols[0].button("📊 GRAFİK PANELİ", use_container_width=True): st.session_state.menu_secim = "GRAFİK"
    if n_cols[1].button("🛠️ TEKNİK ANALİZ", use_container_width=True): st.session_state.menu_secim = "TEKNİK"
    if n_cols[2].button("⚖️ FINANSAL ANALIZ", use_container_width=True): st.session_state.menu_secim = "ADİL"
    if n_cols[3].button("🏭 SEKTÖR ANALİZİ", use_container_width=True): st.session_state.menu_secim = "SEKTÖR"
    if n_cols[4].button("📰 HABERLER", use_container_width=True): st.session_state.menu_secim = "HABERLER"
    if n_cols[5].button("🏢 PROFİL", use_container_width=True): st.session_state.menu_secim = "PROFİL"

    # --- MODÜLLER ---
    if st.session_state.menu_secim == "GRAFİK":
        tab_main, = st.tabs(["📉 GRAFİK PANELİ"])
        with tab_main:
            t_col1, t_col2 = st.columns([1, 3])
            with t_col1:
                period_map = {"1H": "5d", "3A": "3mo", "1Y": "1y", "3Y": "3y", "ALL": "max"}
                selected_label = st.radio("", list(period_map.keys()), horizontal=True)
            hist = fetch_period_history(symbol, market_type, period_map[selected_label])
            if not hist.empty:
                g_col_l, g_col_r = st.columns([4, 1])
                with g_col_l:
                    p_perf = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    c_color = "#238636" if p_perf >= 0 else "#da3633"
                    fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], line=dict(color=c_color, width=3), fill='tozeroy', fillcolor=f"rgba({ '35,134,54' if p_perf >= 0 else '218,54,51' },0.05)")])
                    fig.update_layout(height=320, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(side="right"))
                    st.plotly_chart(fig, use_container_width=True)
                with g_col_r:
                    st.markdown(f"<div class='mini-card-vertical' style='padding:10px;'><small>EN YÜKSEK</small><br><b class='price-up' style='font-size:1.1rem;'>{hist['High'].max():.2f}</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='mini-card-vertical' style='padding:10px;'><small>EN DÜŞÜK</small><br><b class='price-down' style='font-size:1.1rem;'>{hist['Low'].min():.2f}</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='mini-card-vertical' style='padding:10px;'><small>PERF.</small><br><b style='color:{c_color}; font-size:1.1rem;'>%{p_perf:+.2f}</b></div>", unsafe_allow_html=True)

    elif st.session_state.menu_secim == "TEKNİK":
        tab_tech, = st.tabs(["🛠️ TEKNİK GÖSTERGELER & AI ANALİZ"])
        with tab_tech:
            if len(df_long) > 200:
                # === CACHE'Lİ GÖSTERGE HESAPLAMALARI (SÜPER HIZLI!) ===
                indicators = calculate_technical_indicators_optimized(symbol, market_type)
                
                if indicators:
                    # Değişkenleri çıkar
                    ema10 = indicators['ema10']
                    ema20 = indicators['ema20']
                    ema100 = indicators['ema100']
                    ema200 = indicators['ema200']
                    current_rsi = indicators['current_rsi']
                    current_macd_val = indicators['current_macd_val']
                    bb_position = indicators['bb_position']
                    current_stoch = indicators['current_stoch']
                    adx = indicators['adx']
                    atr_percent = indicators['atr_percent']
                    volume_ratio = indicators['volume_ratio']
                    pivot = indicators['pivot']
                    r1 = indicators['r1']
                    r2 = indicators['r2']
                    s1 = indicators['s1']
                    s2 = indicators['s2']

                # === AI SCORE HESAPLAMA (DEĞİŞTİRİLMEDİ) ===
                signals = []
                score = 0
                
                ema_signals = 0
                if curr_price > ema10: ema_signals += 2
                if curr_price > ema20: ema_signals += 2
                if curr_price > ema100: ema_signals += 3
                if curr_price > ema200: ema_signals += 3
                if ema10 > ema20: ema_signals += 2
                if ema20 > ema100: ema_signals += 2
                if ema100 > ema200: ema_signals += 2
                if curr_price < ema10: ema_signals -= 2
                if curr_price < ema20: ema_signals -= 2
                if curr_price < ema100: ema_signals -= 3
                if curr_price < ema200: ema_signals -= 3
                score += ema_signals
                signals.append(("EMA Sistemi", "TREND ANALİZİ", ema_signals, "#238636" if ema_signals > 0 else "#da3633"))

                if current_rsi > 70: score -= 8; signals.append(("RSI", "AŞIRI ALIM", -8, "#da3633"))
                elif current_rsi < 30: score += 8; signals.append(("RSI", "AŞIRI SATIM", 8, "#238636"))
                
                if current_macd_val > 0: score += 5; signals.append(("MACD", "BULLISH", 5, "#238636"))
                else: score -= 5; signals.append(("MACD", "BEARISH", -5, "#da3633"))
                
                if bb_position < 20: score += 7; signals.append(("Bollinger", "ALT BANT", 7, "#238636"))
                elif bb_position > 80: score -= 7; signals.append(("Bollinger", "ÜST BANT", -7, "#da3633"))

                # === GÜNCELLENMİŞ NİHAİ KARAR MEKANİZMASI ===
                if score >= 40:
                    decision = "GÜÇLÜ AL"
                    decision_color = "#238636"
                    risk = "DÜŞÜK"
                    confidence = min(98, 75 + (score - 40) / 2)
                elif score >= 15:
                    decision = "AL"
                    decision_color = "#2cbb4d"
                    risk = "DÜŞÜK/ORTA"
                    confidence = 60 + (score - 15)
                elif score <= -40:
                    decision = "GÜÇLÜ SAT"
                    decision_color = "#da3633"
                    risk = "YÜKSEK"
                    confidence = min(98, 75 + (abs(score) - 40) / 2)
                elif score <= -15:
                    decision = "SAT"
                    decision_color = "#ff4b4b"
                    risk = "ORTA/YÜKSEK"
                    confidence = 60 + (abs(score) - 15)
                else:
                    decision = "BEKLE / NÖTR"
                    decision_color = "#f69e5d"
                    risk = "ORTA"
                    confidence = 50 + (score / 2)

                # AI Karar Paneli
                st.markdown(f"""
                    <div style='background:linear-gradient(135deg, #1c2128, #0d1117); border:2px solid {decision_color}; 
                                border-radius:6px; padding:9px; margin-bottom:9px; box-shadow:0 0 8px {decision_color}40;'>
                        <div style='text-align:center;'>
                            <div style='color:#ffffff; font-size:1.5rem; margin-bottom:4px;'> AI KARAR SİSTEMİ</div>
                            <div style='color:{decision_color}; font-size:3rem; font-weight:900; margin:6px 0; 
                                        text-shadow:0 0 15px {decision_color};'>{decision}</div>
                            <div style='display:flex; justify-content:center; gap:15px; margin-top:8px;'>
                                <div style='text-align:center;'>
                                    <div style='color:#8b949e; font-size:0.85rem;'>TOPLAM SKOR</div>
                                    <div style='color:#00f2ff; font-size:2rem; font-weight:800;'>{score:+d}</div>
                                </div>
                                <div style='text-align:center;'>
                                    <div style='color:#8b949e; font-size:0.85rem;'>GÜVEN</div>
                                    <div style='color:#00f2ff; font-size:2rem; font-weight:800;'>%{confidence:.0f}</div>
                                </div>
                                <div style='text-align:center;'>
                                    <div style='color:#8b949e; font-size:0.85rem;'>RİSK</div>
                                    <div style='color:{"#238636" if risk=="DÜŞÜK" else "#da3633" if risk=="YÜKSEK" else "#f69e5d"}; 
                                                font-size:2rem; font-weight:800;'>{risk}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Gösterge Kartları - Satır 1
                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                row1 = st.columns(4, gap="small")
                row1[0].markdown(f"<div class='tech-card'><small>EMA 10</small><br><b style='color:{'#238636' if curr_price > ema10 else '#da3633'}; font-size:1.2rem;'>{ema10:.2f}</b></div>", unsafe_allow_html=True)
                row1[1].markdown(f"<div class='tech-card'><small>EMA 20</small><br><b style='color:{'#238636' if curr_price > ema20 else '#da3633'}; font-size:1.2rem;'>{ema20:.2f}</b></div>", unsafe_allow_html=True)
                row1[2].markdown(f"<div class='tech-card'><small>EMA 100</small><br><b style='color:{'#238636' if curr_price > ema100 else '#da3633'}; font-size:1.2rem;'>{ema100:.2f}</b></div>", unsafe_allow_html=True)
                row1[3].markdown(f"<div class='tech-card'><small>EMA 200</small><br><b style='color:{'#238636' if curr_price > ema200 else '#da3633'}; font-size:1.2rem;'>{ema200:.2f}</b></div>", unsafe_allow_html=True)
                
                # Gösterge Kartları - Satır 2
                row2 = st.columns(5, gap="small")
                rsi_color = "#da3633" if current_rsi > 70 else "#238636" if current_rsi < 30 else "#f69e5d"
                row2[0].markdown(f"<div class='tech-card'><small>RSI (14)</small><br><b style='color:{rsi_color}; font-size:1.3rem;'>{current_rsi:.1f}</b></div>", unsafe_allow_html=True)
                row2[1].markdown(f"<div class='tech-card'><small>MACD</small><br><b style='color:{'#238636' if current_macd_val > 0 else '#da3633'}; font-size:1rem;'>{'BULL' if current_macd_val > 0 else 'BEAR'}</b></div>", unsafe_allow_html=True)
                row2[2].markdown(f"<div class='tech-card'><small>ADX</small><br><b style='color:#00f2ff; font-size:1.3rem;'>{adx:.1f}</b><br><small>{'GÜÇLÜ' if adx > 40 else 'ORTA' if adx > 25 else 'ZAYIF'}</small></div>", unsafe_allow_html=True)
                row2[3].markdown(f"<div class='tech-card'><small>STOCHASTIC</small><br><b style='color:{'#238636' if current_stoch < 20 else '#da3633' if current_stoch > 80 else '#f69e5d'}; font-size:1.3rem;'>{current_stoch:.1f}</b></div>", unsafe_allow_html=True)
                row2[4].markdown(f"<div class='tech-card'><small>VOLATİLİTE (ATR)</small><br><b style='color:#00f2ff; font-size:1.3rem;'>%{atr_percent:.2f}</b></div>", unsafe_allow_html=True)
                
                # Gösterge Kartları - Satır 3 (Pivot)
                row3 = st.columns(6, gap="small")
                row3[0].markdown(f"<div class='tech-card'><small>R2</small><br><b style='color:#da3633; font-size:1.1rem;'>{r2:.2f}</b></div>", unsafe_allow_html=True)
                row3[1].markdown(f"<div class='tech-card'><small>R1</small><br><b style='color:#da3633; font-size:1.1rem;'>{r1:.2f}</b></div>", unsafe_allow_html=True)
                row3[2].markdown(f"<div class='tech-card'><small>PIVOT</small><br><b style='color:#00f2ff; font-size:1.1rem;'>{pivot:.2f}</b></div>", unsafe_allow_html=True)
                row3[3].markdown(f"<div class='tech-card'><small>S1</small><br><b style='color:#238636; font-size:1.1rem;'>{s1:.2f}</b></div>", unsafe_allow_html=True)
                row3[4].markdown(f"<div class='tech-card'><small>S2</small><br><b style='color:#238636; font-size:1.1rem;'>{s2:.2f}</b></div>", unsafe_allow_html=True)
                row3[5].markdown(f"<div class='tech-card'><small>VOLUME</small><br><b style='color:{'#00f2ff' if volume_ratio > 1.2 else '#f69e5d'}; font-size:1rem;'>{volume_ratio:.1f}x</b></div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
                
            else:
                st.warning("⚠️ Teknik analiz için en az 200 günlük veri gerekli.")

    elif st.session_state.menu_secim == "ADİL":
        tab_fair, = st.tabs(["⚖️ FİNANSAL ANALİZ"])
        with tab_fair:
            # Veri kalitesi uyarısı
            if data_quality['percentage'] < 70:
                st.warning(f"""
                    ⚠️ **VERİ KALİTESİ UYARISI:** Mevcut veri kalitesi **{data_quality['level']}** (%{data_quality['percentage']:.0f}). 
                    Bazı finansal analizler eksik veriler nedeniyle tam doğru olmayabilir. 
                    {'Hibrit API modunu aktif ederek daha fazla veriye erişebilirsiniz.' if not use_hybrid and market_type == 'ABD (Global)' else ''}
                """)
            
            # === FİNANSAL VERİLER ===
            pe = info.get('trailingPE', 0) or 0
            forward_pe = info.get('forwardPE', 0) or 0
            bv = info.get('bookValue', 0) or 0
            eps = info.get('trailingEps', 0) or 0
            pb = info.get('priceToBook', 0) or 0
            ps = info.get('priceToSalesTrailing12Months', 0) or 0
            
            # Karlılık
            roe = info.get('returnOnEquity', 0) or 0
            roa = info.get('returnOnAssets', 0) or 0
            profit_margin = info.get('profitMargins', 0) or 0
            operating_margin = info.get('operatingMargins', 0) or 0
            
            # Büyüme
            revenue_growth = info.get('revenueGrowth', 0) or 0
            earnings_growth = info.get('earningsGrowth', 0) or 0
            
            # Bilanço
            debt_to_equity = info.get('debtToEquity', 0) or 0
            current_ratio = info.get('currentRatio', 0) or 0
            quick_ratio = info.get('quickRatio', 0) or 0
            
            # Diğer
            beta = info.get('beta', 1) or 1
            dividend_yield = info.get('dividendYield', 0) or 0
            payout_ratio = info.get('payoutRatio', 0) or 0
            
            # === DEĞERLEME MODELLERİ ===
            # 1. Graham Formülü
            graham = math.sqrt(22.5 * eps * bv) if (eps > 0 and bv > 0) else 0
            
            # 2. PEG Ratio ve değerleme
            peg = pe / (earnings_growth * 100) if (pe > 0 and earnings_growth > 0) else 0
            
            # 3. Analist konsensüsü
            analyst_target = info.get('targetMeanPrice', 0) or 0
            analyst_count = info.get('numberOfAnalystOpinions', 0) or 0
            
            # 4. DCF yaklaşımı (basitleştirilmiş)
            fcf = info.get('freeCashflow', 0) or 0
            shares = info.get('sharesOutstanding', 0) or 0
            growth_rate = max(0.05, min(0.15, earnings_growth)) if earnings_growth > 0 else 0.08
            discount_rate = 0.10 + (beta - 1) * 0.05
            
            dcf_value = 0
            if fcf > 0 and shares > 0:
                terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
                dcf_value = terminal_value / shares
            
            # 5. Sektör P/E çarpanı (varsayılan değerler - genellikle API'den gelir)
            sector_pe = 20  # Ortalama piyasa P/E'si
            pe_based_value = eps * sector_pe if eps > 0 else 0
            
            # === NİHAİ ADİL DEĞER HESABI ===
            values = []
            weights = []
            
            if graham > 0:
                values.append(graham)
                weights.append(0.25)
            
            if analyst_target > 0:
                values.append(analyst_target)
                weights.append(0.30)
            
            if dcf_value > 0:
                values.append(dcf_value)
                weights.append(0.25)
            
            if pe_based_value > 0:
                values.append(pe_based_value)
                weights.append(0.20)
            
            # Ağırlıklı ortalama
            if values and weights:
                total_weight = sum(weights)
                final_fair_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            else:
                final_fair_value = curr_price
            
            upside = ((final_fair_value - curr_price) / curr_price) * 100 if curr_price > 0 else 0
            margin_of_safety = ((final_fair_value - curr_price) / final_fair_value) * 100 if final_fair_value > 0 else 0
            
            # === AI SKORLAMA SİSTEMİ ===
            valuation_score = 0
            valuation_signals = []
            
            # Değerleme Skorları
            if upside > 30:
                valuation_score += 20
                valuation_signals.append(("Adil Değer", "AŞIRI UCUZ", 20, "#238636"))
            elif upside > 15:
                valuation_score += 12
                valuation_signals.append(("Adil Değer", "UCUZ", 12, "#238636"))
            elif upside < -30:
                valuation_score -= 20
                valuation_signals.append(("Adil Değer", "AŞIRI PAHALI", -20, "#da3633"))
            elif upside < -15:
                valuation_score -= 12
                valuation_signals.append(("Adil Değer", "PAHALI", -12, "#da3633"))
            else:
                valuation_signals.append(("Adil Değer", "MAKUL", 0, "#f69e5d"))
            
            # PEG Ratio
            if 0 < peg < 1:
                valuation_score += 10
                valuation_signals.append(("PEG Ratio", "MÜKEMMEL", 10, "#238636"))
            elif 1 <= peg < 1.5:
                valuation_score += 5
                valuation_signals.append(("PEG Ratio", "İYİ", 5, "#238636"))
            elif peg >= 2:
                valuation_score -= 8
                valuation_signals.append(("PEG Ratio", "YÜKSEK", -8, "#da3633"))
            
            # P/E Ratio
            if 0 < pe < 15:
                valuation_score += 8
                valuation_signals.append(("P/E Ratio", "DÜŞÜK", 8, "#238636"))
            elif pe > 30:
                valuation_score -= 8
                valuation_signals.append(("P/E Ratio", "YÜKSEK", -8, "#da3633"))
            
            # P/B Ratio
            if 0 < pb < 1:
                valuation_score += 7
                valuation_signals.append(("P/B Ratio", "DEFTER DEĞERİ ALTI", 7, "#238636"))
            elif pb > 5:
                valuation_score -= 5
                valuation_signals.append(("P/B Ratio", "YÜKSEK", -5, "#da3633"))
            
            # Karlılık Skorları
            if roe > 0.20:
                valuation_score += 10
                valuation_signals.append(("ROE", "YÜKSEK KARLILILIK", 10, "#238636"))
            elif roe < 0.08:
                valuation_score -= 8
                valuation_signals.append(("ROE", "DÜŞÜK KARLILILIK", -8, "#da3633"))
            
            if profit_margin > 0.15:
                valuation_score += 7
                valuation_signals.append(("Kar Marjı", "GÜÇLÜ", 7, "#238636"))
            elif profit_margin < 0.05:
                valuation_score -= 5
                valuation_signals.append(("Kar Marjı", "ZAYIF", -5, "#da3633"))
            
            # Büyüme Skorları
            if revenue_growth > 0.15:
                valuation_score += 10
                valuation_signals.append(("Gelir Büyümesi", "HIZLI", 10, "#238636"))
            elif revenue_growth < 0:
                valuation_score -= 10
                valuation_signals.append(("Gelir Büyümesi", "NEGATİF", -10, "#da3633"))
            
            if earnings_growth > 0.15:
                valuation_score += 8
                valuation_signals.append(("Kazanç Büyümesi", "GÜÇLÜ", 8, "#238636"))
            elif earnings_growth < 0:
                valuation_score -= 8
                valuation_signals.append(("Kazanç Büyümesi", "NEGATİF", -8, "#da3633"))
            
            # Bilanço Sağlığı
            if debt_to_equity < 50:
                valuation_score += 8
                valuation_signals.append(("Borç/Özkaynak", "DÜŞÜK BORÇ", 8, "#238636"))
            elif debt_to_equity > 200:
                valuation_score -= 10
                valuation_signals.append(("Borç/Özkaynak", "YÜKSEK BORÇ", -10, "#da3633"))
            
            if current_ratio > 2:
                valuation_score += 5
                valuation_signals.append(("Likidite", "GÜÇLÜ", 5, "#238636"))
            elif current_ratio < 1:
                valuation_score -= 8
                valuation_signals.append(("Likidite", "ZAYIF", -8, "#da3633"))
            
            # Temettü
            if dividend_yield > 0.03:
                valuation_score += 5
                valuation_signals.append(("Temettü", "CAZIP", 5, "#238636"))
            
            # === NİHAİ KARAR ===
            if valuation_score >= 60:
                fair_decision = "GÜÇLÜ YATIRIM YAPI"
                fair_color = "#238636"
                fair_risk = "DÜŞÜK"
                fair_confidence = min(98, 75 + (valuation_score - 60) / 2)
            elif valuation_score >= 30:
                fair_decision = "YATIRIM YAP"
                fair_color = "#2cbb4d"
                fair_risk = "DÜŞÜK/ORTA"
                fair_confidence = 60 + (valuation_score - 30)
            elif valuation_score <= -60:
                fair_decision = "GÜÇLÜ KAÇIN"
                fair_color = "#da3633"
                fair_risk = "YÜKSEK"
                fair_confidence = min(98, 75 + (abs(valuation_score) - 60) / 2)
            elif valuation_score <= -30:
                fair_decision = "KAÇININ"
                fair_color = "#ff4b4b"
                fair_risk = "ORTA/YÜKSEK"
                fair_confidence = 60 + (abs(valuation_score) - 30)
            else:
                fair_decision = "NÖTR / BEKLE"
                fair_color = "#f69e5d"
                fair_risk = "ORTA"
                fair_confidence = 50 + (valuation_score / 2)
            
            # === AI KARAR PANELİ ===
            st.markdown(f"""
                <div style='background:linear-gradient(135deg, #1c2128, #0d1117); border:2px solid {fair_color}; 
                            border-radius:6px; padding:9px; margin-bottom:9px; box-shadow:0 0 8px {fair_color}40;'>
                    <div style='text-align:center;'>
                        <div style='color:#ffffff; font-size:1.5rem; margin-bottom:4px;'>💰 FİNANSAL ANALİZ AI KARAR</div>
                        <div style='color:{fair_color}; font-size:3rem; font-weight:900; margin:6px 0; 
                                    text-shadow:0 0 15px {fair_color};'>{fair_decision}</div>
                        <div style='display:flex; justify-content:center; gap:15px; margin-top:8px;'>
                            <div style='text-align:center;'>
                                <div style='color:#8b949e; font-size:0.85rem;'>TOPLAM SKOR</div>
                                <div style='color:#00f2ff; font-size:2rem; font-weight:800;'>{valuation_score:+d}</div>
                            </div>
                            <div style='text-align:center;'>
                                <div style='color:#8b949e; font-size:0.85rem;'>GÜVEN</div>
                                <div style='color:#00f2ff; font-size:2rem; font-weight:800;'>%{fair_confidence:.0f}</div>
                            </div>
                            <div style='text-align:center;'>
                                <div style='color:#8b949e; font-size:0.85rem;'>RİSK</div>
                                <div style='color:{"#238636" if fair_risk=="DÜŞÜK" else "#da3633" if fair_risk=="YÜKSEK" else "#f69e5d"}; 
                                            font-size:2rem; font-weight:800;'>{fair_risk}</div>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # === DEĞERLEME MODELLERİ ===
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>📊 DEĞERLEME MODELLERİ</div>", unsafe_allow_html=True)
            
            val_row1 = st.columns(5, gap="small")
            val_row1[0].markdown(f"<div class='tech-card'><small>MEVCUT FİYAT</small><br><b style='color:#00f2ff; font-size:1.3rem;'>{curr_price:.2f}</b></div>", unsafe_allow_html=True)
            val_row1[1].markdown(f"<div class='tech-card'><small>ADİL DEĞER</small><br><b style='color:#f69e5d; font-size:1.3rem;'>{final_fair_value:.2f}</b></div>", unsafe_allow_html=True)
            val_row1[2].markdown(f"<div class='tech-card'><small>POTANSİYEL</small><br><b style='color:{'#238636' if upside > 0 else '#da3633'}; font-size:1.3rem;'>%{upside:+.1f}</b></div>", unsafe_allow_html=True)
            val_row1[3].markdown(f"<div class='tech-card'><small>GÜVENLİK MARJI</small><br><b style='color:{'#238636' if margin_of_safety > 20 else '#f69e5d'}; font-size:1.3rem;'>%{margin_of_safety:.1f}</b></div>", unsafe_allow_html=True)
            val_row1[4].markdown(f"<div class='tech-card'><small>ANALIST HEDEF</small><br><b style='color:#818cf8; font-size:1.3rem;'>{analyst_target:.2f}</b><br><small>{analyst_count} analist</small></div>", unsafe_allow_html=True)
            
            val_row2 = st.columns(4, gap="small")
            val_row2[0].markdown(f"<div class='tech-card'><small>GRAHAM</small><br><b style='color:#238636; font-size:1.2rem;'>{graham:.2f}</b></div>", unsafe_allow_html=True)
            val_row2[1].markdown(f"<div class='tech-card'><small>DCF DEĞER</small><br><b style='color:#238636; font-size:1.2rem;'>{dcf_value:.2f}</b></div>", unsafe_allow_html=True)
            val_row2[2].markdown(f"<div class='tech-card'><small>P/E BAZLI</small><br><b style='color:#238636; font-size:1.2rem;'>{pe_based_value:.2f}</b></div>", unsafe_allow_html=True)
            val_row2[3].markdown(f"<div class='tech-card'><small>PEG RATIO</small><br><b style='color:{'#238636' if 0 < peg < 1.5 else '#da3633' if peg >= 2 else '#f69e5d'}; font-size:1.2rem;'>{peg:.2f}</b></div>", unsafe_allow_html=True)
            
            # === DEĞERLEME ORANLARI ===
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>📈 DEĞERLEME ORANLARI</div>", unsafe_allow_html=True)
            
            ratio_row = st.columns(6, gap="small")
            ratio_row[0].markdown(f"<div class='tech-card'><small>P/E</small><br><b style='color:{'#238636' if pe < 15 else '#da3633' if pe > 30 else '#f69e5d'}; font-size:1.2rem;'>{pe:.2f}</b></div>", unsafe_allow_html=True)
            ratio_row[1].markdown(f"<div class='tech-card'><small>Forward P/E</small><br><b style='color:#00f2ff; font-size:1.2rem;'>{forward_pe:.2f}</b></div>", unsafe_allow_html=True)
            ratio_row[2].markdown(f"<div class='tech-card'><small>P/B</small><br><b style='color:{'#238636' if pb < 1 else '#da3633' if pb > 5 else '#f69e5d'}; font-size:1.2rem;'>{pb:.2f}</b></div>", unsafe_allow_html=True)
            ratio_row[3].markdown(f"<div class='tech-card'><small>P/S</small><br><b style='color:#00f2ff; font-size:1.2rem;'>{ps:.2f}</b></div>", unsafe_allow_html=True)
            ratio_row[4].markdown(f"<div class='tech-card'><small>EPS</small><br><b style='color:#00f2ff; font-size:1.2rem;'>{eps:.2f}</b></div>", unsafe_allow_html=True)
            ratio_row[5].markdown(f"<div class='tech-card'><small>DEFTER DEĞER</small><br><b style='color:#00f2ff; font-size:1.2rem;'>{bv:.2f}</b></div>", unsafe_allow_html=True)
            
            # === KARLILK & BÜYÜME ===
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>💎 KARLILIK & BÜYÜME</div>", unsafe_allow_html=True)
            
            profit_row = st.columns(6, gap="small")
            profit_row[0].markdown(f"<div class='tech-card'><small>ROE</small><br><b style='color:{'#238636' if roe > 0.20 else '#da3633' if roe < 0.08 else '#f69e5d'}; font-size:1.2rem;'>%{roe*100:.1f}</b></div>", unsafe_allow_html=True)
            profit_row[1].markdown(f"<div class='tech-card'><small>ROA</small><br><b style='color:#00f2ff; font-size:1.2rem;'>%{roa*100:.1f}</b></div>", unsafe_allow_html=True)
            profit_row[2].markdown(f"<div class='tech-card'><small>KAR MARJI</small><br><b style='color:{'#238636' if profit_margin > 0.15 else '#da3633' if profit_margin < 0.05 else '#f69e5d'}; font-size:1.2rem;'>%{profit_margin*100:.1f}</b></div>", unsafe_allow_html=True)
            profit_row[3].markdown(f"<div class='tech-card'><small>OPER. MARJ</small><br><b style='color:#00f2ff; font-size:1.2rem;'>%{operating_margin*100:.1f}</b></div>", unsafe_allow_html=True)
            profit_row[4].markdown(f"<div class='tech-card'><small>GELİR BÜY.</small><br><b style='color:{'#238636' if revenue_growth > 0.10 else '#da3633' if revenue_growth < 0 else '#f69e5d'}; font-size:1.2rem;'>%{revenue_growth*100:+.1f}</b></div>", unsafe_allow_html=True)
            profit_row[5].markdown(f"<div class='tech-card'><small>KAZANÇ BÜY.</small><br><b style='color:{'#238636' if earnings_growth > 0.10 else '#da3633' if earnings_growth < 0 else '#f69e5d'}; font-size:1.2rem;'>%{earnings_growth*100:+.1f}</b></div>", unsafe_allow_html=True)
            
            # === BİLANÇO & RİSK ===
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>🛡️ BİLANÇO SAĞLIĞI & RİSK</div>", unsafe_allow_html=True)
            
            balance_row = st.columns(6, gap="small")
            balance_row[0].markdown(f"<div class='tech-card'><small>BORÇ/ÖZK.</small><br><b style='color:{'#238636' if debt_to_equity < 50 else '#da3633' if debt_to_equity > 200 else '#f69e5d'}; font-size:1.2rem;'>{debt_to_equity:.1f}</b></div>", unsafe_allow_html=True)
            balance_row[1].markdown(f"<div class='tech-card'><small>CARİ ORAN</small><br><b style='color:{'#238636' if current_ratio > 2 else '#da3633' if current_ratio < 1 else '#f69e5d'}; font-size:1.2rem;'>{current_ratio:.2f}</b></div>", unsafe_allow_html=True)
            balance_row[2].markdown(f"<div class='tech-card'><small>ASİT TEST</small><br><b style='color:#00f2ff; font-size:1.2rem;'>{quick_ratio:.2f}</b></div>", unsafe_allow_html=True)
            balance_row[3].markdown(f"<div class='tech-card'><small>BETA</small><br><b style='color:{'#238636' if beta < 1 else '#da3633' if beta > 1.5 else '#f69e5d'}; font-size:1.2rem;'>{beta:.2f}</b></div>", unsafe_allow_html=True)
            balance_row[4].markdown(f"<div class='tech-card'><small>TEMETTÜ VER.</small><br><b style='color:{'#238636' if dividend_yield > 0.03 else '#f69e5d'}; font-size:1.2rem;'>%{dividend_yield*100:.2f}</b></div>", unsafe_allow_html=True)
            balance_row[5].markdown(f"<div class='tech-card'><small>ÖDEME ORANI</small><br><b style='color:#00f2ff; font-size:1.2rem;'>%{payout_ratio*100:.1f}</b></div>", unsafe_allow_html=True)
            
            # === SİNYAL DETAYLARI ===
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>📋 FİNANSAL SİNYALLER</div>", unsafe_allow_html=True)
            
            col_count = 0
            cols = None
            for indicator, status, points, color in valuation_signals:
                if col_count % 2 == 0: cols = st.columns(2)
                with cols[col_count % 2]:
                    st.markdown(f"""
                        <div style='background:#0d1117; border-left:3px solid {color}; padding:10px; border-radius:8px; margin-bottom:8px;'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <div><div style='color:#8b949e; font-size:0.85rem;'>{indicator}</div><div style='color:{color}; font-size:1.05rem; font-weight:700;'>{status}</div></div>
                                <div style='color:#00f2ff; font-size:1.25rem; font-weight:900;'>{points:+d}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                col_count += 1
            
            # === KALİTE SKORLARI ===
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>⭐ KALİTE SKORLARI</div>", unsafe_allow_html=True)
            
            # Piotroski F-Score (0-9)
            f_score = 0
            if profit_margin > 0: f_score += 1  # Pozitif kar
            if operating_margin > 0: f_score += 1  # Pozitif operasyonel nakit akışı
            if roa > 0: f_score += 1  # ROA artışı (basitleştirilmiş)
            if current_ratio > 1.5: f_score += 1  # Likidite
            if debt_to_equity < 100: f_score += 1  # Borç azalması
            if roe > 0.10: f_score += 1  # ROE iyileşmesi
            if revenue_growth > 0: f_score += 1  # Gelir artışı
            if profit_margin > 0.10: f_score += 1  # Marj iyileşmesi
            if shares > 0: f_score += 1  # Hisse arzı kontrolü
            
            f_score_color = "#238636" if f_score >= 7 else "#f69e5d" if f_score >= 4 else "#da3633"
            f_score_label = "ÇOK GÜÇLÜ" if f_score >= 7 else "İYİ" if f_score >= 4 else "ZAYIF"
            
            # Altman Z-Score (İflas riski)
            working_capital = current_ratio * curr_price * shares if shares > 0 else 0
            market_cap = curr_price * shares if shares > 0 else 1
            retained_earnings = roe * bv if bv > 0 else 0
            ebit = profit_margin * market_cap * 0.1  # Yaklaşık
            
            z_score = 0
            if market_cap > 0:
                z_score = (1.2 * working_capital / market_cap + 
                          1.4 * retained_earnings / market_cap + 
                          3.3 * ebit / market_cap + 
                          0.6 * market_cap / max(debt_to_equity * market_cap / 100, 1) + 
                          1.0)
            
            z_score_color = "#238636" if z_score > 2.99 else "#f69e5d" if z_score > 1.81 else "#da3633"
            z_score_label = "GÜVENLİ" if z_score > 2.99 else "GRİ BÖLGE" if z_score > 1.81 else "RİSKLİ"
            
            # Beneish M-Score (Manipülasyon riski - basitleştirilmiş)
            m_score = -2.5  # Varsayılan düşük risk
            if revenue_growth > 0.50: m_score += 1  # Anormal gelir artışı
            if debt_to_equity > 150: m_score += 0.5  # Yüksek kaldıraç
            if current_ratio < 1: m_score += 1  # Likidite sorunu
            
            m_score_color = "#238636" if m_score < -2 else "#f69e5d" if m_score < -1 else "#da3633"
            m_score_label = "DÜŞÜK RİSK" if m_score < -2 else "ORTA RİSK" if m_score < -1 else "YÜKSEK RİSK"
            
            quality_row = st.columns(3, gap="small")
            quality_row[0].markdown(f"""
                <div class='tech-card'>
                    <small>PIOTROSKI F-SCORE</small><br>
                    <b style='color:{f_score_color}; font-size:1.5rem;'>{f_score}/9</b><br>
                    <small style='color:{f_score_color};'>{f_score_label}</small>
                </div>
            """, unsafe_allow_html=True)
            quality_row[1].markdown(f"""
                <div class='tech-card'>
                    <small>ALTMAN Z-SCORE</small><br>
                    <b style='color:{z_score_color}; font-size:1.5rem;'>{z_score:.2f}</b><br>
                    <small style='color:{z_score_color};'>{z_score_label}</small>
                </div>
            """, unsafe_allow_html=True)
            quality_row[2].markdown(f"""
                <div class='tech-card'>
                    <small>BENEISH M-SCORE</small><br>
                    <b style='color:{m_score_color}; font-size:1.5rem;'>{m_score:.2f}</b><br>
                    <small style='color:{m_score_color};'>{m_score_label}</small>
                </div>
            """, unsafe_allow_html=True)
            
            # === SENARYO ANALİZİ ===
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>🎯 SENARYO ANALİZİ</div>", unsafe_allow_html=True)
            
            # Senaryolar
            optimistic_return = upside * 1.8  # İyimser: mevcut potansiyelin 1.8x
            base_return = upside  # Temel senaryo
            pessimistic_return = upside * 0.3 - 15  # Kötümser: düşük potansiyel + kayıp
            
            # Olasılıklar
            opt_prob = 0.25 if upside > 20 else 0.15
            base_prob = 0.50
            pess_prob = 1 - opt_prob - base_prob
            
            expected_return = (optimistic_return * opt_prob + 
                             base_return * base_prob + 
                             pessimistic_return * pess_prob)
            
            scenario_row = st.columns(4, gap="small")
            scenario_row[0].markdown(f"""
                <div class='tech-card'>
                    <small>🎯 İYİMSER</small><br>
                    <b style='color:#238636; font-size:1.3rem;'>%{optimistic_return:+.1f}</b><br>
                    <small style='color:#8b949e;'>Olasılık: %{opt_prob*100:.0f}</small>
                </div>
            """, unsafe_allow_html=True)
            scenario_row[1].markdown(f"""
                <div class='tech-card'>
                    <small>😐 TEMEL</small><br>
                    <b style='color:#00f2ff; font-size:1.3rem;'>%{base_return:+.1f}</b><br>
                    <small style='color:#8b949e;'>Olasılık: %{base_prob*100:.0f}</small>
                </div>
            """, unsafe_allow_html=True)
            scenario_row[2].markdown(f"""
                <div class='tech-card'>
                    <small>📉 KÖTÜMSER</small><br>
                    <b style='color:#da3633; font-size:1.3rem;'>%{pessimistic_return:+.1f}</b><br>
                    <small style='color:#8b949e;'>Olasılık: %{pess_prob*100:.0f}</small>
                </div>
            """, unsafe_allow_html=True)
            scenario_row[3].markdown(f"""
                <div class='tech-card'>
                    <small>📊 BEKLENEN</small><br>
                    <b style='color:{'#238636' if expected_return > 10 else '#f69e5d'}; font-size:1.3rem;'>%{expected_return:+.1f}</b><br>
                    <small style='color:#8b949e;'>Ağırlıklı Ort.</small>
                </div>
            """, unsafe_allow_html=True)
            
            # === YATIRIM STRATEJİSİ ===
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>📋 YATIRIM STRATEJİSİ ÖNERİLERİ</div>", unsafe_allow_html=True)
            
            # Hedef fiyat (adil değer + güvenlik payı)
            target_price = final_fair_value * 1.05 if upside > 0 else final_fair_value * 0.95
            stop_loss = curr_price * 0.88  # %12 stop loss
            position_size = "Yüksek (15-20%)" if valuation_score > 60 else "Orta (8-12%)" if valuation_score > 30 else "Düşük (3-5%)" if valuation_score > 0 else "YOK"
            entry_timing = "Agresif (Hemen)" if valuation_score > 60 else "Kademeli (3 ay)" if valuation_score > 30 else "Bekle" if valuation_score > 0 else "GİRME"
            
            strategy_cols = st.columns(2, gap="medium")
            with strategy_cols[0]:
                st.markdown(f"""
                    <div style='background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px;'>
                        <div style='color:#00f2ff; font-size:1rem; font-weight:700; margin-bottom:8px;'>🎯 FİYAT HEDEFLERİ</div>
                        <div style='margin-bottom:6px;'><span style='color:#8b949e;'>Hedef Fiyat:</span> <b style='color:#238636;'>${target_price:.2f}</b> <small style='color:#8b949e;'>(Potansiyel: %{((target_price-curr_price)/curr_price*100):+.1f})</small></div>
                        <div style='margin-bottom:6px;'><span style='color:#8b949e;'>Stop Loss:</span> <b style='color:#da3633;'>${stop_loss:.2f}</b> <small style='color:#8b949e;'>(Risk: %{((stop_loss-curr_price)/curr_price*100):+.1f})</small></div>
                        <div><span style='color:#8b949e;'>Risk/Ödül:</span> <b style='color:#00f2ff;'>1:{abs((target_price-curr_price)/(curr_price-stop_loss)):.2f}</b></div>
                    </div>
                """, unsafe_allow_html=True)
            
            with strategy_cols[1]:
                st.markdown(f"""
                    <div style='background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px;'>
                        <div style='color:#00f2ff; font-size:1rem; font-weight:700; margin-bottom:8px;'>⚙️ POZİSYON YÖNETİMİ</div>
                        <div style='margin-bottom:6px;'><span style='color:#8b949e;'>Giriş Zamanlaması:</span> <b style='color:#f69e5d;'>{entry_timing}</b></div>
                        <div style='margin-bottom:6px;'><span style='color:#8b949e;'>Portföy Ağırlığı:</span> <b style='color:#00f2ff;'>{position_size}</b></div>
                        <div><span style='color:#8b949e;'>Vade:</span> <b style='color:#818cf8;'>{'Uzun (12+ ay)' if valuation_score > 40 else 'Orta (6-12 ay)' if valuation_score > 20 else 'Kısa (3-6 ay)'}</b></div>
                    </div>
                """, unsafe_allow_html=True)
            
            # === AKILLI UYARILAR ===
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>⚠️ AKILLI UYARILAR & RİSKLER</div>", unsafe_allow_html=True)
            
            warnings = []
            
            # Risk flagleri
            if debt_to_equity > 150:
                warnings.append(("🔴 YÜKSEK RİSK", f"Borç/Özkaynak çok yüksek: {debt_to_equity:.1f}", "#da3633"))
            elif debt_to_equity > 100:
                warnings.append(("🟡 DİKKAT", f"Borç/Özkaynak yükseliyor: {debt_to_equity:.1f}", "#f69e5d"))
            
            if current_ratio < 1:
                warnings.append(("🔴 LİKİDİTE RİSKİ", f"Cari oran düşük: {current_ratio:.2f}", "#da3633"))
            
            if pe > 40:
                warnings.append(("🟡 DEĞERLEME", f"P/E çok yüksek: {pe:.1f} (Balon riski)", "#f69e5d"))
            
            if revenue_growth < 0:
                warnings.append(("🔴 BÜYÜME SORUNU", f"Gelir düşüyor: %{revenue_growth*100:.1f}", "#da3633"))
            
            if profit_margin < 0.05:
                warnings.append(("🟡 MARJ BASKISI", f"Kar marjı düşük: %{profit_margin*100:.1f}", "#f69e5d"))
            
            if beta > 1.5:
                warnings.append(("🟡 VOLATİLİTE", f"Yüksek Beta: {beta:.2f} (Piyasadan %{(beta-1)*100:.0f} daha volatil)", "#f69e5d"))
            
            if peg > 2:
                warnings.append(("🟡 BÜYÜME PAHALI", f"PEG Ratio yüksek: {peg:.2f}", "#f69e5d"))
            
            if f_score < 4:
                warnings.append(("🔴 KALİTE DÜŞÜK", f"Piotroski F-Score: {f_score}/9", "#da3633"))
            
            if z_score < 1.81:
                warnings.append(("🔴 İFLAS RİSKİ", f"Altman Z-Score: {z_score:.2f} (Riskli bölge)", "#da3633"))
            
            # Pozitif sinyaller
            if not warnings or len(warnings) < 3:
                if roe > 0.25:
                    warnings.append(("✅ GÜÇLÜ KARLILIK", f"ROE mükemmel: %{roe*100:.1f}", "#238636"))
                if margin_of_safety > 30:
                    warnings.append(("✅ BÜYÜK FIRSATI", f"Güvenlik marjı: %{margin_of_safety:.1f}", "#238636"))
                if f_score >= 7:
                    warnings.append(("✅ YÜKSEK KALİTE", f"Piotroski: {f_score}/9", "#238636"))
            
            if not warnings:
                warnings.append(("✅ UYARI YOK", "Önemli risk tespit edilmedi", "#238636"))
            
            for warning_type, warning_msg, warning_color in warnings:
                st.markdown(f"""
                    <div style='background:#0d1117; border-left:4px solid {warning_color}; padding:10px; border-radius:6px; margin-bottom:6px;'>
                        <b style='color:{warning_color};'>{warning_type}</b>
                        <span style='color:#8b949e; margin-left:10px;'>{warning_msg}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # === İÇERİDEN İŞLEMLER & KURUMSAL SAHİPLİK ===
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>👥 SAHİPLİK YAPISI & SENTIMENT</div>", unsafe_allow_html=True)
            
            # Sahiplik verileri
            institutional = info.get('heldPercentInstitutions', 0) or 0
            insider_held = info.get('heldPercentInsiders', 0) or 0
            short_percent = info.get('shortPercentOfFloat', 0) or 0
            
            # Analist önerileri
            rec_buy = info.get('recommendationKey', 'hold')
            rec_mean = info.get('recommendationMean', 3) or 3  # 1=Strong Buy, 5=Strong Sell
            
            # Sentiment yorumlama
            if rec_mean < 2:
                analyst_sentiment = "GÜÇLÜ AL"
                analyst_color = "#238636"
            elif rec_mean < 2.5:
                analyst_sentiment = "AL"
                analyst_color = "#2cbb4d"
            elif rec_mean < 3.5:
                analyst_sentiment = "TUT"
                analyst_color = "#f69e5d"
            elif rec_mean < 4:
                analyst_sentiment = "SAT"
                analyst_color = "#ff4b4b"
            else:
                analyst_sentiment = "GÜÇLÜ SAT"
                analyst_color = "#da3633"
            
            ownership_row = st.columns(3, gap="small")
            ownership_row[0].markdown(f"""
                <div class='tech-card'>
                    <small>KURUMSAL SAHİPLİK</small><br>
                    <b style='color:{'#238636' if institutional > 0.6 else '#f69e5d'}; font-size:1.4rem;'>%{institutional*100:.1f}</b><br>
                    <small style='color:#8b949e;'>{'Güçlü' if institutional > 0.6 else 'Orta' if institutional > 0.3 else 'Düşük'}</small>
                </div>
            """, unsafe_allow_html=True)
            ownership_row[1].markdown(f"""
                <div class='tech-card'>
                    <small>İÇERİDEN SAHİPLİK</small><br>
                    <b style='color:#00f2ff; font-size:1.4rem;'>%{insider_held*100:.1f}</b><br>
                    <small style='color:#8b949e;'>Yönetici İnancı</small>
                </div>
            """, unsafe_allow_html=True)
            ownership_row[2].markdown(f"""
                <div class='tech-card'>
                    <small>SHORT ORANI</small><br>
                    <b style='color:{'#da3633' if short_percent > 0.10 else '#238636'}; font-size:1.4rem;'>%{short_percent*100:.1f}</b><br>
                    <small style='color:#8b949e;'>{'Yüksek Baskı' if short_percent > 0.10 else 'Normal'}</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Analist konsensüsü
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style='background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px;'>
                    <div style='color:#00f2ff; font-size:1rem; font-weight:700; margin-bottom:8px;'>📊 ANALİST KONSENSÜSÜ</div>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <span style='color:#8b949e;'>Ortalama Öneri:</span> 
                            <b style='color:{analyst_color}; font-size:1.2rem; margin-left:10px;'>{analyst_sentiment}</b>
                        </div>
                        <div>
                            <span style='color:#8b949e;'>Skor:</span> 
                            <b style='color:#00f2ff; font-size:1.2rem; margin-left:5px;'>{rec_mean:.2f}/5.0</b>
                        </div>
                        <div>
                            <span style='color:#8b949e;'>Analist Sayısı:</span> 
                            <b style='color:#818cf8; font-size:1.2rem; margin-left:5px;'>{analyst_count}</b>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # === TARİHSEL DEĞERLEME TRENDİ ===
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#00f2ff; font-size:1.2rem; font-weight:700; margin-bottom:8px;'>📈 TARİHSEL DEĞERLEME TRENDİ</div>", unsafe_allow_html=True)
            
            # Son 52 hafta verileri
            week52_high = info.get('fiftyTwoWeekHigh', curr_price) or curr_price
            week52_low = info.get('fiftyTwoWeekLow', curr_price) or curr_price
            week52_change = info.get('52WeekChange', 0) or 0
            
            # Mevcut pozisyon (52 hafta aralığında)
            if week52_high != week52_low:
                position_in_range = ((curr_price - week52_low) / (week52_high - week52_low)) * 100
            else:
                position_in_range = 50
            
            trend_row = st.columns(4, gap="small")
            trend_row[0].markdown(f"""
                <div class='tech-card'>
                    <small>52 HAFTA DÜŞÜK</small><br>
                    <b style='color:#238636; font-size:1.3rem;'>${week52_low:.2f}</b><br>
                    <small style='color:#8b949e;'>%{((curr_price-week52_low)/week52_low*100):+.1f} üstte</small>
                </div>
            """, unsafe_allow_html=True)
            trend_row[1].markdown(f"""
                <div class='tech-card'>
                    <small>52 HAFTA YÜKSEK</small><br>
                    <b style='color:#da3633; font-size:1.3rem;'>${week52_high:.2f}</b><br>
                    <small style='color:#8b949e;'>%{((week52_high-curr_price)/curr_price*100):+.1f} aşağıda</small>
                </div>
            """, unsafe_allow_html=True)
            trend_row[2].markdown(f"""
                <div class='tech-card'>
                    <small>ARALIK POZİSYONU</small><br>
                    <b style='color:#00f2ff; font-size:1.3rem;'>%{position_in_range:.0f}</b><br>
                    <small style='color:#8b949e;'>{'Zirveye yakın' if position_in_range > 80 else 'Dibe yakın' if position_in_range < 20 else 'Orta bölge'}</small>
                </div>
            """, unsafe_allow_html=True)
            trend_row[3].markdown(f"""
                <div class='tech-card'>
                    <small>52 HAFTA DEĞİŞİM</small><br>
                    <b style='color:{'#238636' if week52_change > 0 else '#da3633'}; font-size:1.3rem;'>%{week52_change*100:+.1f}</b><br>
                    <small style='color:#8b949e;'>Yıllık performans</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Değerleme trend yorumu
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            trend_interpretation = ""
            if position_in_range < 30 and upside > 20:
                trend_interpretation = "✅ Hisse 52 hafta düşüğüne yakın ve adil değerin altında → GÜÇLÜ ALIM FIRSATI"
                trend_color = "#238636"
            elif position_in_range > 80 and upside < -10:
                trend_interpretation = "⚠️ Hisse 52 hafta yükseğine yakın ve adil değerin üstünde → RİSK YÜKSEK"
                trend_color = "#da3633"
            elif position_in_range < 50 and upside > 10:
                trend_interpretation = "💡 Hisse aralığın altı yarısında ve potansiyel var → ALIM FIRSATı OLABİLİR"
                trend_color = "#2cbb4d"
            else:
                trend_interpretation = "📊 Hisse tarihi aralıkta dengeli seviyelerde işlem görüyor"
                trend_color = "#f69e5d"
            
            st.markdown(f"""
                <div style='background:#0d1117; border-left:4px solid {trend_color}; padding:12px; border-radius:6px;'>
                    <b style='color:{trend_color};'>TARİHSEL ANALİZ:</b>
                    <span style='color:#8b949e; margin-left:10px;'>{trend_interpretation}</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    elif st.session_state.menu_secim == "SEKTÖR":
        tab_peers, tab_custom = st.tabs(["📊 Rakip Şirketler", "🔬 Özel Karşılaştırma"])
        
        # TAB 1: RAKİP ŞİRKETLER (Otomatik) - LAZY LOADING
        with tab_peers:
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            if sector != 'N/A':
                st.markdown(f"**Sektör:** {sector}")
                st.markdown(f"**Endüstri:** {industry}")
                
                # LAZY LOADING: Sadece bu sekmeye girildiğinde veriler çekilecek
                st.markdown("### 📊 Rakip Şirketler (Peers)")
                
                # Hangi kaynak kullanıldığını göster
                if symbol in MANUAL_PEERS:
                    st.info("✨ **Uzman Seçimi:** Bu hisse için dikkatle seçilmiş rakipler gösteriliyor")
                
                # Cache'li peers listesi al (hızlı, API çağrısı yok)
                peers_list = fetch_peers_data(symbol, sector, industry, fmp_api_key)
                
                # BURADA LAZY LOADING: Peer verilerini sadece gerektiğinde çek
                with st.spinner(f'📊 {len(peers_list[:15])} şirket verisi çekiliyor...'):
                    try:
                        # Paralel veri çekme - bu satır artık sadece bu sekme açıldığında çalışacak
                        comparison_data = fetch_peers_parallel(peers_list[:15], max_workers=10)
                        
                        # Seçili hissenin kesinlikle listede olmasını garanti et
                        symbol_exists = any(stock['Sembol'] == symbol for stock in comparison_data)
                        if not symbol_exists:
                            comparison_data.append({
                                'Sembol': symbol,
                                'Şirket': info.get('shortName', symbol)[:30],
                                'Fiyat': info.get('regularMarketPrice', 0),
                                'P/E': info.get('trailingPE', 0),
                                'P/B': info.get('priceToBook', 0),
                                'ROE': (info.get('returnOnEquity', 0) or 0) * 100,
                                'Market Cap': info.get('marketCap', 0) / 1e9 if info.get('marketCap', 0) > 0 else 0,
                            })
                        
                        # Eğer hiç peer bulunamadıysa uyar
                        if len(comparison_data) < 2:
                            # Son çare: En azından aynı sektörden göster
                            if len(peers_list) > 0:
                                st.info(f"ℹ️ {symbol} için spesifik rakipler bulunamadı. Sektör: {sector}, Industry: {industry}")
                            else:
                                st.warning(f"⚠️ {symbol} için rakip şirketler bulunamadı. Sadece seçili hisse gösteriliyor.")
                        elif len(peers_list) >= 3:
                            st.info(f"💡 Industry bazlı karşılaştırma: {industry}")
                            
                    except (ConnectionError, TimeoutError) as e:
                        st.error(f"🌐 Bağlantı Hatası: Rakip şirket verileri alınamadı.")
                        comparison_data = [{
                            'Sembol': symbol,
                            'Şirket': info.get('shortName', symbol)[:30],
                            'Fiyat': info.get('regularMarketPrice', 0),
                            'P/E': info.get('trailingPE', 0),
                            'P/B': info.get('priceToBook', 0),
                            'ROE': (info.get('returnOnEquity', 0) or 0) * 100,
                            'Market Cap': info.get('marketCap', 0) / 1e9 if info.get('marketCap', 0) > 0 else 0,
                        }]
                
                if comparison_data:
                    import pandas as pd
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Market Cap'e göre sırala ve en yüksek 10'u al
                    df_comparison = df_comparison.sort_values('Market Cap', ascending=False).head(10)
                    
                    st.success(f"✅ {len(df_comparison)} rakip şirket karşılaştırılıyor (market cap'e göre sıralı)")
                    
                    # Mevcut hisseyi vurgula
                    def highlight_row(row):
                        if row['Sembol'] == symbol:
                            return ['background-color: #0d1117; font-weight: bold'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        df_comparison.style.apply(highlight_row, axis=1).format({
                            'Fiyat': '${:.2f}',
                            'P/E': '{:.2f}',
                            'P/B': '{:.2f}',
                            'ROE': '{:.1f}%',
                            'Market Cap': '${:.2f}B'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Peer ortalamaları
                    st.markdown("### 📈 Rakip Analizi")
                    
                    avg_pe = df_comparison['P/E'].mean()
                    avg_pb = df_comparison['P/B'].mean()
                    avg_roe = df_comparison['ROE'].mean()
                    
                    current_pe = info.get('trailingPE', 0) or 0
                    current_pb = info.get('priceToBook', 0) or 0
                    current_roe = (info.get('returnOnEquity', 0) or 0) * 100
                    
                    metrics_cols = st.columns(3)
                    
                    # P/E Karşılaştırması
                    pe_status = "UCUZ" if current_pe < avg_pe else "PAHALI"
                    pe_color = "#238636" if current_pe < avg_pe else "#da3633"
                    metrics_cols[0].markdown(f"""
                        <div class='tech-card'>
                            <small>P/E ORANI</small><br>
                            <b style='font-size:1.3rem;'>{current_pe:.1f}</b>
                            <small style='color:#8b949e;'> vs {avg_pe:.1f}</small><br>
                            <small style='color:{pe_color};'>{pe_status}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # ROE Karşılaştırması
                    roe_status = "GÜÇLÜ" if current_roe > avg_roe else "ZAYIF"
                    roe_color = "#238636" if current_roe > avg_roe else "#da3633"
                    metrics_cols[1].markdown(f"""
                        <div class='tech-card'>
                            <small>ROE</small><br>
                            <b style='font-size:1.3rem;'>{current_roe:.1f}%</b>
                            <small style='color:#8b949e;'> vs {avg_roe:.1f}%</small><br>
                            <small style='color:{roe_color};'>{roe_status}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # P/B Karşılaştırması
                    pb_status = "UCUZ" if current_pb < avg_pb else "PAHALI"
                    pb_color = "#238636" if current_pb < avg_pb else "#da3633"
                    metrics_cols[2].markdown(f"""
                        <div class='tech-card'>
                            <small>P/B ORANI</small><br>
                            <b style='font-size:1.3rem;'>{current_pb:.1f}</b>
                            <small style='color:#8b949e;'> vs {avg_pb:.1f}</small><br>
                            <small style='color:{pb_color};'>{pb_status}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Genel yorum
                    st.markdown("---")
                    advantages = 0
                    if current_pe < avg_pe: advantages += 1
                    if current_pb < avg_pb: advantages += 1
                    if current_roe > avg_roe: advantages += 1
                    
                    if advantages >= 2:
                        overall = "✅ RAKİPLERİNDEN DAHA İYİ"
                        overall_color = "#238636"
                    elif advantages == 1:
                        overall = "📊 RAKİPLERİ İLE DENGEDE"
                        overall_color = "#f69e5d"
                    else:
                        overall = "⚠️ RAKİPLERİNİN GERİSİNDE"
                        overall_color = "#da3633"
                    
                    st.markdown(f"""
                        <div style='background:#0d1117; border-left:4px solid {overall_color}; padding:15px; border-radius:6px; margin-bottom:25px;'>
                            <b style='color:{overall_color}; font-size:1.1rem;'>{overall}</b><br>
                            <span style='color:#8b949e; margin-top:5px; display:block;'>
                                {symbol} rakip şirketlere göre {advantages}/3 metrikte daha iyi performans gösteriyor.
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("📊 Rakip şirketler için veri bulunamadı.")
            else:
                st.warning("⚠️ Sektör bilgisi mevcut değil.")
        
        # TAB 2: ÖZEL KARŞILAŞTIRMA (Kullanıcı seçer)
        with tab_custom:
            st.markdown("### 🔬 Özel Karşılaştırma")
            st.markdown("İstediğiniz hisseleri karşılaştırın - sektör fark etmez!")
            
            # Ana hisse (otomatik dolu)
            st.markdown(f"**Ana Hisse:** `{symbol}`")
            
            # Seçili hisseleri session'da sakla
            if 'custom_compare_list' not in st.session_state:
                st.session_state.custom_compare_list = []
            
            # Karşılaştırılacak hisseler - Enter ile otomatik ekle
            st.markdown("**Karşılaştırmak İstediğiniz Hisseler:** *(Enter ile ekleyin)*")
            
            compare_symbol = st.text_input("", placeholder="Hisse sembolü yazın ve Enter'a basın...", 
                                          label_visibility="collapsed", key="custom_compare_input").upper().strip()
            
            # Enter ile otomatik ekleme
            if compare_symbol:
                if compare_symbol not in st.session_state.custom_compare_list and compare_symbol != symbol:
                    st.session_state.custom_compare_list.append(compare_symbol)
                    st.rerun()
            
            st.markdown("---")
            
            # Otomatik karşılaştırma (liste boş değilse)
            if len(st.session_state.custom_compare_list) > 0:
                # Ana hisse + seçili hisseler
                all_symbols = [symbol] + st.session_state.custom_compare_list
                
                with st.spinner(f'📊 {len(all_symbols)} hisse verisi çekiliyor...'):
                    # Paralel olarak veriyi çek
                    custom_comparison_data = fetch_peers_parallel(all_symbols, max_workers=10)
                
                if custom_comparison_data:
                    import pandas as pd
                    df_custom = pd.DataFrame(custom_comparison_data)
                    
                    # Market Cap'e göre sırala
                    df_custom = df_custom.sort_values('Market Cap', ascending=False)
                    
                    st.success(f"✅ {len(df_custom)} hisse karşılaştırılıyor")
                    
                    # Boş satırları temizle ve sıfır değerleri kontrol et
                    df_custom = df_custom.dropna(subset=['Sembol'])
                    df_custom = df_custom[df_custom['Sembol'].str.strip() != '']
                    df_custom = df_custom.fillna(0)  # NaN'ları 0 yap
                    
                    # Ana hisseyi vurgula
                    def highlight_main(row):
                        if row['Sembol'] == symbol:
                            return ['background-color: #1a4d2e; font-weight: bold; color: #ffffff'] * len(row)
                        return [''] * len(row)
                    
                    # Streamlit dataframe ile tam kontrol
                    st.dataframe(
                        df_custom.style.apply(highlight_main, axis=1).format({
                            'Sembol': '{}',
                            'Fiyat': '${:.2f}',
                            'P/E': '{:.2f}',
                            'P/B': '{:.2f}',
                            'ROE': '{:.1f}%',
                            'EPS': '${:.2f}',
                            'Market Cap': '${:.2f}B'
                        }),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Sembol": st.column_config.TextColumn("Sembol", width="small"),
                            "Fiyat": st.column_config.NumberColumn("Fiyat", format="$%.2f"),
                            "P/E": st.column_config.NumberColumn("P/E", format="%.2f"),
                            "P/B": st.column_config.NumberColumn("P/B", format="%.2f"),
                            "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
                            "EPS": st.column_config.NumberColumn("EPS", format="$%.2f"),
                            "Market Cap": st.column_config.NumberColumn("Market Cap", format="$%.2fB")
                        }
                    )
                    
                    # Silme butonları (tablonun altında)
                    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                    if len(st.session_state.custom_compare_list) > 0:
                        st.markdown("**Hisse Çıkar:**")
                        remove_cols = st.columns(min(len(st.session_state.custom_compare_list), 6))
                        for idx, sym in enumerate(st.session_state.custom_compare_list):
                            with remove_cols[idx % 6]:
                                if st.button(f"🗑 {sym}", key=f"remove_custom_{sym}", use_container_width=True):
                                    st.session_state.custom_compare_list.remove(sym)
                                    st.rerun()
                    
                    # 5 Yıllık Zaman Serisi Grafiği
                    st.markdown("---")
                    st.markdown("### 📈 5 Yıllık Tarihsel Gelişim")
                    
                    # Metrik seçimi (Market Cap hariç, diğer metrikleri ekle)
                    time_series_metric = st.selectbox(
                        "Zaman serisi için metrik seçin:",
                        options=['Fiyat', 'P/E', 'P/B', 'ROE', 'EPS', 'Revenue', 'Net Income'],
                        help="5 yıllık tarihsel veriler için metrik"
                    )
                    
                    with st.spinner('📊 5 yıllık veriler çekiliyor...'):
                        try:
                            # Her hisse için 5 yıllık fiyat verisi çek
                            fig_timeline = go.Figure()
                            
                            # Renk paleti
                            colors = ['#00f2ff', '#ff006e', '#8338ec', '#3a86ff', '#fb5607', '#06ffa5', '#ffbe0b']
                            
                            for idx, sym in enumerate(all_symbols):
                                try:
                                    stock_hist = yf.Ticker(sym)
                                    hist_data = stock_hist.history(period='5y')
                                    
                                    if not hist_data.empty:
                                        # Seçilen metriğe göre veri
                                        if time_series_metric == 'Fiyat':
                                            y_data = hist_data['Close']
                                            y_label = 'Fiyat ($)'
                                        elif time_series_metric == 'P/E':
                                            # P/E için quarterly info gerekir, basit yaklaşım: mevcut P/E'yi kullan
                                            try:
                                                current_pe = stock_hist.info.get('trailingPE', None)
                                                if current_pe:
                                                    # P/E yaklaşık olarak fiyat ile orantılı varsayalım
                                                    y_data = hist_data['Close'] * (current_pe / hist_data['Close'].iloc[-1]) if hist_data['Close'].iloc[-1] != 0 else hist_data['Close']
                                                    y_label = 'P/E Oranı (Yaklaşık)'
                                                else:
                                                    y_data = hist_data['Close']
                                                    y_label = 'Fiyat ($) - P/E verisi yok'
                                            except (ValueError, KeyError, ZeroDivisionError):
                                                # P/E hesaplanamıyorsa fiyat kullan
                                                y_data = hist_data['Close']
                                                y_label = 'Fiyat ($)'
                                        elif time_series_metric == 'P/B':
                                            try:
                                                current_pb = stock_hist.info.get('priceToBook', None)
                                                if current_pb:
                                                    y_data = hist_data['Close'] * (current_pb / hist_data['Close'].iloc[-1]) if hist_data['Close'].iloc[-1] != 0 else hist_data['Close']
                                                    y_label = 'P/B Oranı (Yaklaşık)'
                                                else:
                                                    y_data = hist_data['Close']
                                                    y_label = 'Fiyat ($) - P/B verisi yok'
                                            except (ValueError, KeyError, ZeroDivisionError):
                                                # P/B hesaplanamıyorsa fiyat kullan
                                                y_data = hist_data['Close']
                                                y_label = 'Fiyat ($)'
                                        elif time_series_metric == 'ROE':
                                            try:
                                                current_roe = stock_hist.info.get('returnOnEquity', None)
                                                if current_roe:
                                                    # ROE sabit varsayalım (değişimi görmek için daha detaylı veri gerekir)
                                                    y_data = pd.Series([current_roe * 100] * len(hist_data), index=hist_data.index)
                                                    y_label = 'ROE (%)'
                                                else:
                                                    y_data = hist_data['Close']
                                                    y_label = 'Fiyat ($) - ROE verisi yok'
                                            except (ValueError, KeyError, AttributeError):
                                                # ROE hesaplanamıyorsa fiyat kullan
                                                y_data = hist_data['Close']
                                                y_label = 'Fiyat ($)'
                                        elif time_series_metric == 'EPS':
                                            try:
                                                current_eps = stock_hist.info.get('trailingEps', None)
                                                if current_eps:
                                                    # EPS yaklaşık fiyat ile orantılı
                                                    y_data = hist_data['Close'] * (current_eps / hist_data['Close'].iloc[-1]) if hist_data['Close'].iloc[-1] != 0 else hist_data['Close']
                                                    y_label = 'EPS ($) (Yaklaşık)'
                                                else:
                                                    y_data = hist_data['Close']
                                                    y_label = 'Fiyat ($) - EPS verisi yok'
                                            except (ValueError, KeyError, ZeroDivisionError):
                                                # EPS hesaplanamıyorsa fiyat kullan
                                                y_data = hist_data['Close']
                                                y_label = 'Fiyat ($)'
                                        elif time_series_metric == 'Revenue':
                                            try:
                                                # Revenue için quarterly veya annual data gerekir
                                                financials = stock_hist.quarterly_financials
                                                if not financials.empty and 'Total Revenue' in financials.index:
                                                    revenue_data = financials.loc['Total Revenue']
                                                    # Tarihsel fiyata göre normalize et
                                                    y_data = hist_data['Close'] * (revenue_data.iloc[0] / hist_data['Close'].iloc[-1] / 1e9) if hist_data['Close'].iloc[-1] != 0 else hist_data['Close']
                                                    y_label = 'Revenue (Milyar $, Yaklaşık)'
                                                else:
                                                    y_data = hist_data['Close']
                                                    y_label = 'Fiyat ($) - Revenue verisi yok'
                                            except (ValueError, KeyError, IndexError, ZeroDivisionError):
                                                # Revenue hesaplanamıyorsa fiyat kullan
                                                y_data = hist_data['Close']
                                                y_label = 'Fiyat ($)'
                                        elif time_series_metric == 'Net Income':
                                            try:
                                                # Net Income için quarterly veya annual data gerekir
                                                financials = stock_hist.quarterly_financials
                                                if not financials.empty and 'Net Income' in financials.index:
                                                    income_data = financials.loc['Net Income']
                                                    # Tarihsel fiyata göre normalize et
                                                    y_data = hist_data['Close'] * (income_data.iloc[0] / hist_data['Close'].iloc[-1] / 1e9) if hist_data['Close'].iloc[-1] != 0 else hist_data['Close']
                                                    y_label = 'Net Income (Milyar $, Yaklaşık)'
                                                else:
                                                    y_data = hist_data['Close']
                                                    y_label = 'Fiyat ($) - Net Income verisi yok'
                                            except (ValueError, KeyError, IndexError, ZeroDivisionError):
                                                # Net Income hesaplanamıyorsa fiyat kullan
                                                y_data = hist_data['Close']
                                                y_label = 'Fiyat ($)'
                                        else:
                                            y_data = hist_data['Close']  # Default
                                            y_label = time_series_metric
                                        
                                        # Ana hisse kalın çizgi, diğerleri ince
                                        line_width = 3 if sym == symbol else 1.5
                                        
                                        fig_timeline.add_trace(go.Scatter(
                                            x=hist_data.index,
                                            y=y_data,
                                            mode='lines',
                                            name=sym,
                                            line=dict(
                                                color=colors[idx % len(colors)],
                                                width=line_width
                                            ),
                                            hovertemplate=f'<b>{sym}</b><br>' +
                                                        'Tarih: %{x|%Y-%m-%d}<br>' +
                                                        f'{time_series_metric}: %{{y:.2f}}<br>' +
                                                        '<extra></extra>'
                                        ))
                                except (ConnectionError, TimeoutError, ValueError, KeyError, IndexError):
                                    # Hisse verisi çekilemezse veya boşsa atla
                                    pass
                            
                            fig_timeline.update_layout(
                                title=f'5 Yıllık {time_series_metric} Gelişimi',
                                xaxis_title='Tarih',
                                yaxis_title=y_label,
                                plot_bgcolor='#0d1117',
                                paper_bgcolor='#0d1117',
                                font=dict(color='#c9d1d9'),
                                height=500,
                                hovermode='x unified',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                xaxis=dict(
                                    gridcolor='#30363d',
                                    showgrid=True
                                ),
                                yaxis=dict(
                                    gridcolor='#30363d',
                                    showgrid=True
                                )
                            )
                            
                            st.plotly_chart(fig_timeline, use_container_width=True)
                            
                            # Analiz özeti
                            st.markdown("""
                                <div style='background:#0d1117; border-left:4px solid #00f2ff; padding:15px; border-radius:6px; margin-top:15px;'>
                                    <b style='color:#00f2ff;'>💡 Grafik İpuçları</b><br>
                                    <span style='color:#8b949e; font-size:0.9rem; margin-top:5px; display:block;'>
                                        • Kalın çizgi ana hisseyi gösterir<br>
                                        • Fareyi grafiğin üzerine getirerek detaylı bilgi alın<br>
                                        • Legend'dan hisselere tıklayarak göster/gizle yapabilirsiniz
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        except (ConnectionError, TimeoutError) as e:
                            st.error(f"🌐 Bağlantı Hatası: Tarihsel veriler alınamadı.")
                        except (ValueError, KeyError, IndexError) as e:
                            st.error(f"⚠️ Veri Hatası: Tarihsel veri işlenirken hata oluştu.")
                    
                else:
                    st.warning("⚠️ Hisse verileri alınamadı. Lütfen sembol adlarını kontrol edin.")
            else:
                st.info("💡 Yukarıdan karşılaştırmak istediğiniz hisse sembollerini ekleyin (Enter ile)")

    elif st.session_state.menu_secim == "HABERLER":
        tab_news, = st.tabs(["📰 SON HABERLER"])
        with tab_news:
            st.markdown("## 📰 SON HABERLER")
            
            # Cache'li haber verisi al
            with st.spinner('Haberler yükleniyor...'):
                news_list = fetch_news_data(symbol)
            
            if news_list and len(news_list) > 0:
                st.success(f"✅ {len(news_list)} haber bulundu")
                
                # Haberleri göster
                for idx, news_item in enumerate(news_list):
                    title = news_item['title']
                    link = news_item['link']
                    source = news_item['source']
                    time_str = news_item['time']
                    
                    # Basit sentiment (başlığa göre)
                    positive_words = ['surge', 'gain', 'profit', 'growth', 'up', 'high', 'beat', 'strong', 'buy', 'bullish', 'rise', 'soar', 'rallies', 'jumps', 'climbs']
                    negative_words = ['fall', 'drop', 'loss', 'down', 'low', 'miss', 'weak', 'sell', 'bearish', 'crash', 'decline', 'plunge', 'tumbles', 'sinks', 'slumps']
                    
                    title_lower = title.lower()
                    pos_count = sum(1 for word in positive_words if word in title_lower)
                    neg_count = sum(1 for word in negative_words if word in title_lower)
                    
                    if pos_count > neg_count:
                        sentiment = "POZİTİF"
                        sentiment_color = "#238636"
                        sentiment_icon = "📈"
                    elif neg_count > pos_count:
                        sentiment = "NEGATİF"
                        sentiment_color = "#da3633"
                        sentiment_icon = "📉"
                    else:
                        sentiment = "NÖTR"
                        sentiment_color = "#8b949e"
                        sentiment_icon = "📊"
                    
                    with st.expander(f"{sentiment_icon} {title[:80]}...", expanded=(idx==0)):
                        st.markdown(f"""
                            <div style='margin-bottom:10px;'>
                                <span style='color:{sentiment_color}; font-weight:700;'>{sentiment}</span>
                                <span style='color:#6e7681; margin-left:10px;'>• {source}</span>
                                <span style='color:#6e7681; margin-left:10px;'>• {time_str}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"[📖 Haberi Oku]({link})")
            else:
                st.info(f"📭 {symbol} için son haberler bulunamadı.")
                st.markdown("""
                    <div style='background:#0d1117; border-left:4px solid #f69e5d; padding:15px; border-radius:6px; margin-top:15px;'>
                        <b style='color:#f69e5d;'>💡 Bilgi</b><br>
                        <span style='color:#8b949e; font-size:0.9rem; margin-top:5px; display:block;'>
                            Bu hisse için Google News'te son haberler bulunamadı. 
                            Daha popüler hisseler (örn: AAPL, TSLA, MSFT) için daha fazla haber bulunur.
                        </span>
                    </div>
                """, unsafe_allow_html=True)

    elif st.session_state.menu_secim == "PROFİL":
        tab_profile, = st.tabs(["🏢 ŞİRKET PROFİLİ"])
        with tab_profile:
            
            # Sektör ve industry bilgilerini al
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            # === ŞİRKET ÖZETİ ===
            st.markdown("### 🏢 Şirket Hakkında")
            
            company_name = info.get('longName', info.get('shortName', symbol))
            description = info.get('longBusinessSummary', 'Şirket açıklaması mevcut değil.')
            
            col_info1, col_info2 = st.columns([2, 1])
            
            with col_info1:
                st.markdown(f"**{company_name}**")
                st.markdown(f"> {description}")
            
            with col_info2:
                st.markdown("**📊 Temel Bilgiler**")
                
                website = info.get('website', 'N/A')
                founded = info.get('founded', 'N/A')
                employees = info.get('fullTimeEmployees', 'N/A')
                
                if website != 'N/A':
                    st.markdown(f"**🌐 Web Sitesi:** [{website}]({website})")
                else:
                    st.markdown(f"**🌐 Web Sitesi:** N/A")
                    
                st.markdown(f"**📅 Kuruluş:** {founded}")
                
                if isinstance(employees, int):
                    st.markdown(f"**👥 Çalışan:** {employees:,}")
                else:
                    st.markdown(f"**👥 Çalışan:** {employees}")
                
                st.markdown(f"**🏭 Sektör:** {sector}")
                st.markdown(f"**🏷️ Endüstri:** {industry}")
            
            st.markdown("---")
            
            # === İLETİŞİM BİLGİLERİ ===
            st.markdown("### 📞 İletişim Bilgileri")
            
            contact_cols = st.columns(3)
            
            address = info.get('address1', 'N/A')
            city = info.get('city', '')
            state = info.get('state', '')
            zip_code = info.get('zip', '')
            country = info.get('country', '')
            phone = info.get('phone', 'N/A')
            
            full_address = f"{address}"
            if city:
                full_address += f", {city}"
            if state:
                full_address += f", {state}"
            if zip_code:
                full_address += f" {zip_code}"
            if country:
                full_address += f", {country}"
            
            contact_cols[0].markdown(f"""
                <div class='tech-card'>
                    <small>ADRES</small><br>
                    <span style='font-size:0.9rem;'>{full_address if address != 'N/A' else 'N/A'}</span>
                </div>
            """, unsafe_allow_html=True)
            
            contact_cols[1].markdown(f"""
                <div class='tech-card'>
                    <small>TELEFON</small><br>
                    <span style='font-size:0.9rem;'>{phone}</span>
                </div>
            """, unsafe_allow_html=True)
            
            exchange = info.get('exchange', 'N/A')
            currency = info.get('currency', 'USD')
            
            contact_cols[2].markdown(f"""
                <div class='tech-card'>
                    <small>BORSA</small><br>
                    <span style='font-size:1.2rem; font-weight:700;'>{exchange}</span><br>
                    <small style='color:#8b949e;'>Para Birimi: {currency}</small>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # === YÖNETİM KADROSU ===
            st.markdown("### 👔 Yönetim Kadrosu")
            
            officers = info.get('companyOfficers', [])
            if officers and len(officers) > 0:
                officer_data = []
                for officer in officers[:8]:  # İlk 8 yönetici
                    name = officer.get('name', 'N/A')
                    title = officer.get('title', 'N/A')
                    age = officer.get('age', 'N/A')
                    pay = officer.get('totalPay', 0)
                    
                    officer_data.append({
                        'İsim': name,
                        'Pozisyon': title,
                        'Yaş': age if age != 'N/A' else '-',
                        'Ücret': f"${pay/1e6:.2f}M" if pay and pay > 0 else 'N/A'
                    })
                
                if officer_data:
                    import pandas as pd
                    df_officers = pd.DataFrame(officer_data)
                    st.dataframe(df_officers, use_container_width=True, hide_index=True)
            else:
                st.info("Yönetim kadrosu bilgisi mevcut değil.")
            
            st.markdown("---")
            
            # === HİSSE BİLGİLERİ ===
            st.markdown("### 📈 Hisse Bilgileri")
            
            share_cols = st.columns(4)
            
            shares_outstanding = info.get('sharesOutstanding', 0)
            float_shares = info.get('floatShares', 0)
            shares_short = info.get('sharesShort', 0)
            short_ratio = info.get('shortRatio', 0)
            
            share_cols[0].markdown(f"""
                <div class='tech-card'>
                    <small>TOPLAM HİSSE</small><br>
                    <b style='font-size:1.3rem;'>{shares_outstanding/1e9:.2f}B</b>
                </div>
            """, unsafe_allow_html=True)
            
            share_cols[1].markdown(f"""
                <div class='tech-card'>
                    <small>FLOAT</small><br>
                    <b style='font-size:1.3rem;'>{float_shares/1e9:.2f}B</b>
                </div>
            """, unsafe_allow_html=True)
            
            share_cols[2].markdown(f"""
                <div class='tech-card'>
                    <small>SHORT SHARES</small><br>
                    <b style='font-size:1.3rem;'>{shares_short/1e6:.1f}M</b>
                </div>
            """, unsafe_allow_html=True)
            
            share_cols[3].markdown(f"""
                <div class='tech-card'>
                    <small>SHORT RATIO</small><br>
                    <b style='font-size:1.3rem;'>{short_ratio:.2f}</b>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # === SAHİPLİK YAPISI ===
            st.markdown("### 🏛️ Sahiplik Yapısı")
            
            ownership_cols = st.columns(3)
            
            held_percent_insiders = info.get('heldPercentInsiders', 0) * 100
            held_percent_institutions = info.get('heldPercentInstitutions', 0) * 100
            
            ownership_cols[0].markdown(f"""
                <div class='tech-card'>
                    <small>INSIDER SAHİPLİĞİ</small><br>
                    <b style='font-size:1.5rem; color:#818cf8;'>{held_percent_insiders:.1f}%</b>
                </div>
            """, unsafe_allow_html=True)
            
            ownership_cols[1].markdown(f"""
                <div class='tech-card'>
                    <small>KURUMSAL SAHİPLİK</small><br>
                    <b style='font-size:1.5rem; color:#10b981;'>{held_percent_institutions:.1f}%</b>
                </div>
            """, unsafe_allow_html=True)
            
            public_ownership = 100 - held_percent_insiders - held_percent_institutions
            ownership_cols[2].markdown(f"""
                <div class='tech-card'>
                    <small>HALKA AÇIK</small><br>
                    <b style='font-size:1.5rem; color:#8b949e;'>{public_ownership:.1f}%</b>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # === TEMETTÜ BİLGİLERİ ===
            st.markdown("### 💰 Temettü Bilgileri")
            
            dividend_rate = info.get('dividendRate', 0)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            payout_ratio = info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0
            ex_dividend_date = info.get('exDividendDate', None)
            
            if dividend_rate > 0:
                div_cols = st.columns(4)
                
                div_cols[0].markdown(f"""
                    <div class='tech-card'>
                        <small>YILLIK TEMETTÜ</small><br>
                        <b style='font-size:1.3rem; color:#10b981;'>${dividend_rate:.2f}</b>
                    </div>
                """, unsafe_allow_html=True)
                
                div_cols[1].markdown(f"""
                    <div class='tech-card'>
                        <small>TEMETTÜ VERİMİ</small><br>
                        <b style='font-size:1.3rem; color:#10b981;'>{dividend_yield:.2f}%</b>
                    </div>
                """, unsafe_allow_html=True)
                
                div_cols[2].markdown(f"""
                    <div class='tech-card'>
                        <small>PAYOUT RATIO</small><br>
                        <b style='font-size:1.3rem;'>{payout_ratio:.1f}%</b>
                    </div>
                """, unsafe_allow_html=True)
                
                if ex_dividend_date:
                    from datetime import datetime
                    ex_date = datetime.fromtimestamp(ex_dividend_date).strftime('%Y-%m-%d')
                else:
                    ex_date = 'N/A'
                
                div_cols[3].markdown(f"""
                    <div class='tech-card'>
                        <small>SON TEMETTÜ TARİHİ</small><br>
                        <b style='font-size:1rem;'>{ex_date}</b>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Bu şirket temettü ödemiyor.")
            
            st.markdown("---")
            
            # === ANALİST TAVSİYELERİ ===
            st.markdown("### 📊 Analist Tavsiyeleri")
            
            recommendation = info.get('recommendationKey', 'N/A')
            target_mean = info.get('targetMeanPrice', 0)
            target_high = info.get('targetHighPrice', 0)
            target_low = info.get('targetLowPrice', 0)
            num_analysts = info.get('numberOfAnalystOpinions', 0)
            
            if num_analysts > 0:
                analyst_cols = st.columns(4)
                
                rec_color = {
                    'strong_buy': '#10b981',
                    'buy': '#10b981',
                    'hold': '#f59e0b',
                    'sell': '#ef4444',
                    'strong_sell': '#ef4444'
                }.get(recommendation.lower() if isinstance(recommendation, str) else 'hold', '#8b949e')
                
                rec_text = {
                    'strong_buy': 'GÜÇLÜ ALIM',
                    'buy': 'ALIM',
                    'hold': 'TUT',
                    'sell': 'SAT',
                    'strong_sell': 'GÜÇLÜ SATIM'
                }.get(recommendation.lower() if isinstance(recommendation, str) else 'hold', recommendation)
                
                analyst_cols[0].markdown(f"""
                    <div class='tech-card'>
                        <small>TAVSİYE</small><br>
                        <b style='font-size:1.2rem; color:{rec_color};'>{rec_text}</b>
                    </div>
                """, unsafe_allow_html=True)
                
                analyst_cols[1].markdown(f"""
                    <div class='tech-card'>
                        <small>HEDEF FİYAT (ORT.)</small><br>
                        <b style='font-size:1.3rem;'>${target_mean:.2f}</b>
                    </div>
                """, unsafe_allow_html=True)
                
                analyst_cols[2].markdown(f"""
                    <div class='tech-card'>
                        <small>HEDEF (YÜKSEK)</small><br>
                        <b style='font-size:1.3rem; color:#10b981;'>${target_high:.2f}</b>
                    </div>
                """, unsafe_allow_html=True)
                
                analyst_cols[3].markdown(f"""
                    <div class='tech-card'>
                        <small>HEDEF (DÜŞÜK)</small><br>
                        <b style='font-size:1.3rem; color:#ef4444;'>${target_low:.2f}</b>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='background:#0d1117; border-left:4px solid {rec_color}; padding:15px; border-radius:6px; margin-top:15px;'>
                        <b style='color:{rec_color};'>{num_analysts} Analist Görüşü</b><br>
                        <span style='color:#8b949e; margin-top:5px; display:block;'>
                            Ortalama hedef fiyat güncel fiyatın <b>%{((target_mean/curr_price - 1) * 100):+.1f}</b> {'üstünde' if target_mean > curr_price else 'altında'}.
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Analist tavsiyesi mevcut değil.")
            
            st.markdown("---")
            
            # === MALİ TAKVİM ===
            st.markdown("### 📅 Mali Takvim")
            
            earnings_date = info.get('earningsTimestamp', None)
            ex_dividend_date_ts = info.get('exDividendDate', None)
            
            calendar_cols = st.columns(2)
            
            if earnings_date:
                from datetime import datetime
                earnings_datetime = datetime.fromtimestamp(earnings_date)
                earnings_str = earnings_datetime.strftime('%Y-%m-%d %H:%M')
            else:
                earnings_str = 'Bilgi yok'
            
            calendar_cols[0].markdown(f"""
                <div class='tech-card'>
                    <small>SONRAKİ KAZANÇ AÇIKLAMASI</small><br>
                    <b style='font-size:1.1rem;'>{earnings_str}</b>
                </div>
            """, unsafe_allow_html=True)
            
            if ex_dividend_date_ts:
                from datetime import datetime
                ex_div_datetime = datetime.fromtimestamp(ex_dividend_date_ts)
                ex_div_str = ex_div_datetime.strftime('%Y-%m-%d')
            else:
                ex_div_str = 'Temettü yok'
            
            calendar_cols[1].markdown(f"""
                <div class='tech-card'>
                    <small>SON TEMETTÜ TARİHİ</small><br>
                    <b style='font-size:1.1rem;'>{ex_div_str}</b>
                </div>
            """, unsafe_allow_html=True)
            
            # ============================================
            # 📥 EXPORT ÖZELLİKLERİ
            # ============================================
            st.markdown("---")
            st.markdown("### 📥 Rapor İndir")
            
            st.markdown("""
                <div style='background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:15px; margin-bottom:15px;'>
                    <p style='color:#8b949e; margin:0;'>
                        📊 Analiz raporunuzu PDF, Excel veya CSV formatında indirebilirsiniz.
                        Teknik göstergeler ve finansal metrikler dahildir.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            export_cols = st.columns(3)
            
            # Teknik göstergeleri al (cache'den)
            try:
                indicators = calculate_technical_indicators_optimized(symbol, market_type)
            except:
                indicators = None
            
            # PDF Export
            with export_cols[0]:
                try:
                    pdf_buffer = generate_pdf_report(symbol, info, curr_price, df_long, indicators)
                    st.download_button(
                        label="📄 PDF İndir",
                        data=pdf_buffer,
                        file_name=f"{symbol}_analiz_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        help="Detaylı analiz raporu PDF formatında"
                    )
                except Exception as e:
                    st.error(f"PDF oluşturulamadı: {str(e)}")
            
            # Excel Export
            with export_cols[1]:
                try:
                    excel_buffer = generate_excel_report(symbol, info, df_long, indicators)
                    st.download_button(
                        label="📊 Excel İndir",
                        data=excel_buffer,
                        file_name=f"{symbol}_analiz_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Tüm veriler Excel formatında (çoklu sheet)"
                    )
                except Exception as e:
                    st.error(f"Excel oluşturulamadı: {str(e)}")
            
            # CSV Export
            with export_cols[2]:
                try:
                    # Son 500 günlük fiyat verisi
                    csv_data = df_long[['Open', 'High', 'Low', 'Close', 'Volume']].tail(500).to_csv()
                    st.download_button(
                        label="📋 CSV İndir",
                        data=csv_data,
                        file_name=f"{symbol}_fiyat_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Fiyat geçmişi CSV formatında (son 500 gün)"
                    )
                except Exception as e:
                    st.error(f"CSV oluşturulamadı: {str(e)}")
            
            # Export bilgilendirmesi
            st.markdown("""
                <div style='margin-top:15px; padding:10px; background:#1c2128; border-radius:6px;'>
                    <small style='color:#8b949e;'>
                        💡 <b>İpucu:</b> Excel dosyası 4 farklı sheet içerir: 
                        Genel Bilgiler, Fiyat Geçmişi, Teknik Göstergeler, Finansal Metrikler
                    </small>
                </div>
            """, unsafe_allow_html=True)



except (ConnectionError, TimeoutError) as e:
    st.error(f"🌐 Bağlantı Hatası: Sunucuya bağlanırken bir sorun oluştu. Lütfen internet bağlantınızı kontrol edin.")
    st.info(f"Hata detayı: {e}")
except (KeyError, IndexError, AttributeError) as e:
    st.error(f"📊 Veri Erişim Hatası: Bazı veriler eksik veya yanlış formatta. Lütfen farklı bir hisse deneyin.")
    st.info(f"Hata detayı: {e}")
except ValueError as e:
    st.error(f"⚠️ Değer Hatası: Veri işlenirken geçersiz bir değerle karşılaşıldı.")
    st.info(f"Hata detayı: {e}")
except pd.errors.EmptyDataError as e:
    st.error(f"📉 Veri Bulunamadı: Seçilen hisse için veri bulunamadı. Sembolü kontrol edin.")
except Exception as e:
    st.error(f"❌ Beklenmeyen Hata: {type(e).__name__}")
    st.info(f"Hata detayı: {e}")
    st.warning("Bu hatayı sistem yöneticisine bildirin.")
