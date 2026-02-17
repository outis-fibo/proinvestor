import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# -------------------------------------------------
# CUSTOM DARK THEME + ULTRA MODERN PRO SIDEBAR
# -------------------------------------------------
custom_css = """
<style>

/* Genel arka plan */
body, .stApp {
    background-color: #0d0f12 !important;
    color: #e6e6e6 !important;
    font-family: 'Inter', sans-serif;
}

/* Başlık */
h1, h2, h3, h4 {
    color: #f2f2f2 !important;
    font-weight: 600;
}

/* --- ULTRA MODERN PRO SIDEBAR (GLASS + NEON) --- */
section[data-testid="stSidebar"] {
    background: rgba(15, 18, 24, 0.55) !important;
    backdrop-filter: blur(14px) saturate(180%);
    -webkit-backdrop-filter: blur(14px) saturate(180%);
    border-right: 1px solid rgba(0, 238, 255, 0.18);
    padding: 0 !important;
}

section[data-testid="stSidebar"] > div {
    padding: 25px !important;
}

section[data-testid="stSidebar"] h2 {
    color: #00eaff !important;
    text-align: center;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Input alanları */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background-color: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(0, 238, 255, 0.25) !important;
    border-radius: 8px !important;
    color: #e6e6e6 !important;
}

section[data-testid="stSidebar"] input:hover,
section[data-testid="stSidebar"] select:hover {
    border-color: #00eaff !important;
}

/* Butonlar */
section[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #00eaff 0%, #007bff 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: black !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    transition: 0.2s ease-in-out !important;
}

section[data-testid="stSidebar"] button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 12px #00eaff !important;
}

/* Metric kartları */
[data-testid="stMetricValue"] {
    color: #00eaff !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 16px !important;
}

/* Plotly grafikleri */
.js-plotly-plot .plotly .main-svg {
    background-color: #0d0f12 !important;
}

/* Bölücü çizgiler */
hr {
    border: 1px solid #1f2228 !important;
}

/* Kart görünümü */
.block-container {
    padding-top: 2rem;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------------------------
# SAYFA AYARLARI
# -------------------------------------------------
st.set_page_config(
    page_title="ProInvestor Lite",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#00eaff;'>PROINVESTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9aa0a6;'>Next‑Gen Stock Analysis Terminal</p>", unsafe_allow_html=True)

# -------------------------------------------------
# TICKER FORMAT FONKSİYONU
# -------------------------------------------------
def build_ticker_symbol(symbol: str, market_type: str) -> str:
    symbol = symbol.upper().strip()

    if market_type == "Türkiye (BIST)":
        return symbol + ".IS"

    elif market_type == "İngiltere (LSE)":
        return symbol + ".L"

    elif market_type == "Avrupa (EUROPE)":
        if "." in symbol:
            return symbol
        return symbol + ".DE"  # Varsayılan: XETRA

    return symbol  # ABD

# -------------------------------------------------
# VERİ ÇEKME
# -------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="6mo")
    info = ticker.info
    return hist, info

# -------------------------------------------------
# GRAFİK
# -------------------------------------------------
def plot_price_chart(hist: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name=symbol
    ))
    fig.update_layout(
        title=f"{symbol} Fiyat Grafiği (Son 6 Ay)",
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# METRİKLER
# -------------------------------------------------
def show_basic_metrics(info: dict):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Son Fiyat", f"{info.get('regularMarketPrice', 'N/A')}")
    with col2:
        st.metric("Günlük Değişim", f"{info.get('regularMarketChange', 'N/A')}")
    with col3:
        st.metric("Günlük Değişim (%)", f"{info.get('regularMarketChangePercent', 'N/A')}")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#00eaff; text-align:center;'>⚙️ Settings</h2>", unsafe_allow_html=True)

    market_type = st.selectbox(
        "Market",
        [
            "ABD (Global)",
            "Türkiye (BIST)",
            "İngiltere (LSE)",
            "Avrupa (EUROPE)"
        ]
    )

    symbol = st.text_input(
        "Symbol",
        value="AAPL",
        placeholder="Örn: AAPL, TSLA, MSFT, THYAO, BMW, AIR, SAN"
    )

# -------------------------------------------------
# ANA İÇERİK
# -------------------------------------------------
if not symbol:
    st.info("Başlamak için sol taraftan bir sembol gir.")
else:
    ticker_symbol = build_ticker_symbol(symbol, market_type)

    try:
        with st.spinner(f"{ticker_symbol} verileri çekiliyor..."):
            hist, info = fetch_stock_data(ticker_symbol)

        if hist.empty:
            st.error(f"{ticker_symbol} için veri bulunamadı.")
        else:
            st.subheader(f"{symbol} ({ticker_symbol})")

            show_basic_metrics(info)

            st.markdown("---")

            plot_price_chart(hist, symbol)

            st.markdown("### Şirket Özeti")
            st.write(info.get("longName", "İsim bilgisi yok"))
            st.write(info.get("longBusinessSummary", "Özet bilgi bulunamadı."))

    except Exception as e:
        st.error("Veri çekilirken bir hata oluştu.")
        st.caption(str(e))
