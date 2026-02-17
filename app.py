import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------
# SAYFA AYARLARI
# -----------------------------
st.set_page_config(
    page_title="ProInvestor Lite",
    layout="wide"
)

st.title("📈 ProInvestor Lite – Hisse Analiz Terminali")

# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------
def build_ticker_symbol(symbol: str, market_type: str) -> str:
    symbol = symbol.upper().strip()
    if market_type == "Türkiye (BIST)":
        return symbol + ".IS"
    elif market_type == "İngiltere (LSE)":
        return symbol + ".L"
    return symbol  # ABD

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="6mo")
    info = ticker.info
    return hist, info

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

def show_basic_metrics(info: dict):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Son Fiyat", f"{info.get('regularMarketPrice', 'N/A')}")
    with col2:
        st.metric("Günlük Değişim", f"{info.get('regularMarketChange', 'N/A')}")
    with col3:
        st.metric("Günlük Değişim (%)", f"{info.get('regularMarketChangePercent', 'N/A')}")

# -----------------------------
# SIDEBAR – KULLANICI GİRİŞİ
# -----------------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")

    market_type = st.selectbox(
        "Borsa Bölgesi",
        ["ABD (Global)", "Türkiye (BIST)", "İngiltere (LSE)"]
    )

    symbol = st.text_input(
        "Sembol",
        value="AAPL",
        placeholder="Örn: AAPL, TSLA, MSFT, THYAO"
    )

    st.markdown("---")
    st.caption("Sembolü yazıp Enter'a basın.")

# -----------------------------
# ANA İÇERİK
# -----------------------------
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
            # Üst bilgi
            st.subheader(f"{symbol} ({ticker_symbol})")

            # Temel metrikler
            show_basic_metrics(info)

            st.markdown("---")

            # Grafik
            plot_price_chart(hist, symbol)

            # Ek bilgi
            st.markdown("### Şirket Özeti")
            st.write(info.get("longName", "İsim bilgisi yok"))
            st.write(info.get("longBusinessSummary", "Özet bilgi bulunamadı."))

    except Exception as e:
        st.error("Veri çekilirken bir hata oluştu.")
        st.caption(str(e))
