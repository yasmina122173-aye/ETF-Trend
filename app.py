from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (
    fetch_etf_history, fetch_etf_spot, merge_intraday_quote,
    add_indicators, score_etf, latest_snapshot,
)

BASE = Path(__file__).parent
DATA = BASE / "data"

st.set_page_config(page_title="ETF TrendOS Live", page_icon="📈", layout="wide")
st.title("📈 ETF TrendOS Live")
st.caption("盘中实时行情 · 临时日K评分 · 5日线趋势体系 · 手机端自适应")

@st.cache_data(ttl=900, show_spinner=False)
def load_hist(code: str):
    return fetch_etf_history(code)

@st.cache_data(ttl=15, show_spinner=False)
def load_spot():
    return fetch_etf_spot()


def load_csv(name, columns):
    p = DATA / name
    if not p.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(p, dtype={"code": str})

watch = load_csv("watchlist.csv", ["name", "code", "enabled", "category"])
holdings = load_csv("holdings.csv", ["name", "code", "shares", "cost_price"])

with st.sidebar:
    st.header("系统设置")
    total_capital = st.number_input("总资金（元）", min_value=1000.0, value=80000.0, step=1000.0)
    cash_reserve = st.slider("最低现金比例", 0, 50, 20, 5) / 100
    max_positions = st.slider("最多持有板块", 1, 6, 3)
    benchmark_code = st.text_input("市场基准ETF代码", value="510300", max_chars=6)
    refresh_seconds = st.select_slider("盘中刷新频率", options=[15, 30, 60, 120], value=30, format_func=lambda x: f"{x}秒")
    st.divider()
    st.info("盘中评分会把最新价、当日成交量等合并成一根‘临时日K’，收盘后才成为最终日线信号。")

valid = watch[(watch.get("enabled", 1) == 1) & watch["code"].fillna("").astype(str).str.fullmatch(r"\d{6}")]
if valid.empty:
    st.warning("观察池暂未配置有效ETF代码。请在 data/watchlist.csv 填写6位场内ETF代码。")
    st.stop()


def market_status() -> tuple[str, bool]:
    now = datetime.now()
    hm = now.hour * 60 + now.minute
    weekday = now.weekday() < 5
    open_now = weekday and ((570 <= hm <= 690) or (780 <= hm <= 900))
    if open_now:
        return "🟢 交易时段", True
    if weekday and hm < 570:
        return "⚪ 尚未开盘", False
    return "🔴 已收盘/休市", False

status_text, is_open = market_status()
st.caption(f"{status_text} · 页面打开时每 {refresh_seconds} 秒更新一次 · 最后刷新：{datetime.now():%H:%M:%S}")

# st.fragment only auto-reruns while the page/session is open.
@st.fragment(run_every=f"{refresh_seconds}s")
def live_dashboard():
    try:
        spot = load_spot()
        spot_map = spot.set_index("code")
    except Exception as e:
        st.error(f"实时行情暂时不可用：{e}")
        spot_map = pd.DataFrame()

    try:
        bh = load_hist(benchmark_code) if len(benchmark_code) == 6 else None
        if bh is not None and not spot_map.empty and benchmark_code in spot_map.index:
            bh = merge_intraday_quote(bh, spot_map.loc[benchmark_code])
    except Exception:
        bh = None

    rows, histories = [], {}
    for _, r in valid.iterrows():
        code = str(r.code)
        try:
            hist = load_hist(code)
            if not spot_map.empty and code in spot_map.index:
                hist = merge_intraday_quote(hist, spot_map.loc[code])
            histories[code] = hist
            s = score_etf(hist, bh)
            x = latest_snapshot(hist)
            rows.append({
                "板块": r["name"], "代码": code, "分类": r.get("category", ""),
                "最新价": x["close"], "涨跌幅%": x["pct_change"],
                "MA5": x["ma5"], "MA10": x["ma10"], "MA20": x["ma20"],
                "趋势": s.trend, "量能": s.volume, "强度": s.strength,
                "风险": s.risk, "市场": s.market, "总分": s.total,
                "信号": s.signal, "盘中判断": s.reason,
            })
        except Exception as e:
            rows.append({"板块": r["name"], "代码": code, "总分": 0, "信号": "数据失败", "盘中判断": str(e)})

    rank = pd.DataFrame(rows).sort_values("总分", ascending=False).reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("盘中最高分", int(rank.iloc[0]["总分"]) if not rank.empty else 0)
    c2.metric("达到试仓线", int((rank["总分"] >= 70).sum()))
    c3.metric("刷新时间", datetime.now().strftime("%H:%M:%S"))

    st.subheader("15:00前操作面板")
    top = rank[rank["总分"] >= 70].head(max_positions).copy()
    if top.empty:
        st.warning("当前没有板块达到试仓标准，保持现金。")
    else:
        investable = total_capital * (1 - cash_reserve)
        weights = top["总分"] / top["总分"].sum()
        top["建议金额"] = (weights * investable).round(-2)
        top["建议仓位%"] = (top["建议金额"] / total_capital * 100).round(1)
        st.dataframe(top[["板块", "代码", "最新价", "涨跌幅%", "总分", "信号", "建议金额", "建议仓位%", "盘中判断"]], use_container_width=True, hide_index=True)

    st.subheader("实时排行榜")
    cols = ["板块", "代码", "最新价", "涨跌幅%", "MA5", "MA10", "MA20", "趋势", "量能", "强度", "风险", "市场", "总分", "信号", "盘中判断"]
    st.dataframe(rank[[c for c in cols if c in rank.columns]], use_container_width=True, hide_index=True)

    st.subheader("趋势图")
    selected = st.selectbox("选择板块", rank["板块"].tolist(), key="live_select")
    code = rank.loc[rank["板块"] == selected, "代码"].iloc[0]
    if code in histories:
        d = add_indicators(histories[code]).tail(90)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=d.date, open=d.open, high=d.high, low=d.low, close=d.close, name="K线"))
        for n in (5, 10, 20):
            fig.add_trace(go.Scatter(x=d.date, y=d[f"ma{n}"], mode="lines", name=f"MA{n}"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=15, b=5))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("持仓实时风控")
    if holdings.empty:
        st.info("在 data/holdings.csv 填入持仓后，系统会盘中计算盈亏与减仓信号。")
    else:
        hrows = []
        for _, h in holdings.iterrows():
            try:
                code = str(h.code)
                hist = load_hist(code)
                if not spot_map.empty and code in spot_map.index:
                    hist = merge_intraday_quote(hist, spot_map.loc[code])
                x = latest_snapshot(hist)
                market_value = float(h.shares) * x["close"]
                cost = float(h.shares) * float(h.cost_price)
                pnl = market_value - cost
                pnl_pct = pnl / cost * 100 if cost else 0
                if x["close"] < x["ma20"]:
                    action = "退出/清仓信号"
                elif x["close"] < x["ma10"]:
                    action = "减仓40%信号"
                elif x["close"] < x["ma5"]:
                    action = "减仓20%信号"
                else:
                    action = "继续持有"
                hrows.append({"名称": h["name"], "代码": code, "最新价": x["close"], "市值": market_value, "盈亏": pnl, "收益率%": pnl_pct, "盘中风控": action})
            except Exception as e:
                hrows.append({"名称": h.get("name", ""), "代码": h.get("code", ""), "盘中风控": f"数据失败：{e}"})
        st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)

live_dashboard()

st.warning("盘中信号不是收盘确认：14:45后参考价值更高；临近15:00仍可能因最后几分钟价格和成交量变化而反转。系统只辅助执行纪律，不保证收益，也不会自动下单。")
