from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (
    fetch_etf_history,
    fetch_etf_spot,
    merge_intraday_quote,
    add_indicators,
    score_etf,
    latest_snapshot,
)

BASE = Path(__file__).parent
DATA = BASE / "data"

st.set_page_config(
    page_title="TrendOS V2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-first visual layer
st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 1rem; padding-bottom: 4rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    [data-testid="stMetric"] {background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.16); padding: 14px; border-radius: 18px;}
    .hero {padding: 22px; border-radius: 24px; background: linear-gradient(135deg, rgba(45,125,255,.18), rgba(122,87,255,.12)); border:1px solid rgba(127,127,127,.18); margin-bottom: 14px;}
    .hero-kicker {font-size: 13px; opacity: .70; margin-bottom: 8px;}
    .hero-title {font-size: clamp(30px, 6vw, 54px); font-weight: 800; line-height: 1.08; margin-bottom: 8px;}
    .hero-sub {font-size: 15px; opacity: .78;}
    .card {padding: 18px; border-radius: 20px; background: rgba(127,127,127,.07); border:1px solid rgba(127,127,127,.16); margin-bottom: 12px;}
    .score {font-size: 38px; font-weight: 800; line-height: 1;}
    .muted {opacity: .68; font-size: 13px;}
    .pill {display:inline-block; padding:5px 10px; border-radius:999px; font-size:12px; font-weight:700; background:rgba(45,125,255,.15); margin-right:6px;}
    .danger {background:rgba(255,70,70,.13);}
    .good {background:rgba(30,190,110,.14);}
    .warn {background:rgba(255,180,40,.15);}
    div[data-testid="stDataFrame"] {border-radius: 18px; overflow: hidden;}
    @media (max-width: 720px) {
      .block-container {padding-left: .8rem; padding-right: .8rem; padding-top: .5rem;}
      [data-testid="stMetric"] {padding: 10px; border-radius: 15px;}
      .hero {padding: 18px; border-radius: 20px;}
      .card {padding: 15px; border-radius: 18px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_KEYWORDS = [
    "恒生科技", "食品饮料", "绿色电力", "煤炭", "有色金属",
    "半导体", "电网设备", "锂电池", "固态电池",
]

@st.cache_data(ttl=900, show_spinner=False)
def load_history(code: str) -> pd.DataFrame:
    return fetch_etf_history(code)

@st.cache_data(ttl=15, show_spinner=False)
def load_spot() -> pd.DataFrame:
    return fetch_etf_spot()


def market_status() -> tuple[str, bool, int | None]:
    now = datetime.now()
    hm = now.hour * 60 + now.minute
    weekday = now.weekday() < 5
    open_now = weekday and ((570 <= hm <= 690) or (780 <= hm <= 900))
    mins_left = max(0, 900 - hm) if weekday and 780 <= hm <= 900 else None
    if open_now:
        return "交易中", True, mins_left
    if weekday and hm < 570:
        return "尚未开盘", False, None
    return "已收盘 / 休市", False, None


def resolve_keywords(spot: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """For each sector keyword, select the matching ETF with the highest current turnover."""
    records = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        matches = spot[spot["name"].astype(str).str.contains(kw, case=False, na=False)].copy()
        if matches.empty:
            records.append({"板块": kw, "代码": "", "ETF": "未匹配", "匹配状态": "需手动指定"})
            continue
        amount_col = "amount" if "amount" in matches.columns else None
        if amount_col:
            matches = matches.sort_values(amount_col, ascending=False, na_position="last")
        best = matches.iloc[0]
        records.append({
            "板块": kw,
            "代码": str(best["code"]).zfill(6),
            "ETF": str(best.get("name", kw)),
            "匹配状态": "按成交额自动选择",
        })
    return pd.DataFrame(records)


def load_holdings() -> pd.DataFrame:
    p = DATA / "holdings.csv"
    if not p.exists():
        return pd.DataFrame(columns=["name", "code", "shares", "cost_price"])
    df = pd.read_csv(p, dtype={"code": str})
    return df


def recommendation_text(rank: pd.DataFrame, held_codes: set[str]) -> tuple[str, str, str]:
    if rank.empty:
        return "今天先不操作", "行情数据不足", "neutral"
    top = rank.iloc[0]
    strong = rank[rank["总分"] >= 80]
    if not strong.empty:
        held_strong = strong[strong["代码"].isin(held_codes)]
        if not held_strong.empty:
            return "继续持有强势方向", f"{held_strong.iloc[0]['板块']}仍在强势区，先让趋势继续运行。", "good"
        return "出现可关注方向", f"{top['板块']}盘中评分最高，但更适合等回踩或收盘确认，不追瞬时拉升。", "good"
    weak_count = int((rank["总分"] < 60).sum())
    if weak_count >= max(1, math.ceil(len(rank) * 0.6)):
        return "今天以防守为主", "多数观察方向未达到试仓线，现金也是仓位。", "danger"
    return "今天先观察", "板块处在轮动或均线纠缠阶段，等待更清晰的确认。", "warn"


def action_label(row: pd.Series, held_codes: set[str]) -> str:
    held = row["代码"] in held_codes
    score = row["总分"]
    close, ma5, ma10, ma20 = row["最新价"], row["MA5"], row["MA10"], row["MA20"]
    if held:
        if close < ma20:
            return "退出信号"
        if close < ma10:
            return "减仓观察"
        if close < ma5:
            return "轻减 / 等收盘"
        return "继续持有"
    if score >= 90:
        return "重仓候选（分批）"
    if score >= 80:
        return "半仓候选（分批）"
    if score >= 70:
        return "试仓候选"
    if score >= 60:
        return "观察"
    return "回避"


def status_class(action: str) -> str:
    if any(x in action for x in ["持有", "候选"]):
        return "good"
    if any(x in action for x in ["退出", "回避"]):
        return "danger"
    return "warn"


# Header
status_text, is_open, mins_left = market_status()
now = datetime.now()
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">TrendOS V2 · {now:%Y-%m-%d %H:%M:%S}</div>
      <div class="hero-title">15:00 前，只看今天该怎么做</div>
      <div class="hero-sub">{status_text}{f' · 距离收盘约 {mins_left} 分钟' if mins_left is not None else ''} · 盘中信号为临时日K，收盘后才最终确认</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Settings kept inside expander for mobile simplicity
with st.expander("⚙️ 系统设置与观察池", expanded=False):
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        total_capital = st.number_input("总资金（元）", min_value=1000.0, value=80000.0, step=1000.0)
    with s2:
        cash_reserve = st.slider("最低现金比例", 0, 60, 20, 5) / 100
    with s3:
        max_positions = st.slider("最多持有方向", 1, 6, 3)
    with s4:
        refresh_seconds = st.select_slider("刷新频率", options=[15, 30, 60, 120], value=30, format_func=lambda x: f"{x}秒")
    benchmark_code = st.text_input("市场基准ETF代码", value="510300", max_chars=6)
    keyword_text = st.text_area(
        "观察板块关键词（每行一个，系统会自动匹配成交额最高的同名ETF）",
        value="\n".join(DEFAULT_KEYWORDS),
        height=190,
    )
    st.caption("自动匹配只是为了减少手动填代码。正式交易前，请在券商端核对ETF名称、跟踪指数、规模和流动性。")

keywords = [x.strip() for x in keyword_text.splitlines() if x.strip()]
holdings = load_holdings()
held_codes = set(holdings.get("code", pd.Series(dtype=str)).fillna("").astype(str).str.zfill(6))

@st.fragment(run_every="30s")
def dashboard():
    try:
        spot = load_spot()
    except Exception as exc:
        st.error(f"实时行情暂时不可用：{exc}")
        st.stop()

    resolved = resolve_keywords(spot, keywords)
    valid = resolved[resolved["代码"].astype(str).str.fullmatch(r"\d{6}")].copy()
    if valid.empty:
        st.warning("当前关键词没有匹配到ETF。展开“系统设置与观察池”修改关键词。")
        st.dataframe(resolved, use_container_width=True, hide_index=True)
        st.stop()

    spot_map = spot.set_index("code")
    try:
        benchmark = load_history(benchmark_code)
        if benchmark_code in spot_map.index:
            benchmark = merge_intraday_quote(benchmark, spot_map.loc[benchmark_code])
    except Exception:
        benchmark = None

    rows, histories = [], {}
    for _, item in valid.iterrows():
        code = item["代码"]
        try:
            hist = load_history(code)
            if code in spot_map.index:
                hist = merge_intraday_quote(hist, spot_map.loc[code])
            histories[code] = hist
            score = score_etf(hist, benchmark)
            snap = latest_snapshot(hist)
            rows.append({
                "板块": item["板块"], "ETF": item["ETF"], "代码": code,
                "最新价": snap["close"], "涨跌幅%": snap["pct_change"],
                "MA5": snap["ma5"], "MA10": snap["ma10"], "MA20": snap["ma20"],
                "量比20日": snap["volume_ratio"],
                "趋势": score.trend, "量能": score.volume, "强度": score.strength,
                "风险": score.risk, "市场": score.market, "总分": score.total,
                "系统判断": score.reason,
            })
        except Exception as exc:
            rows.append({
                "板块": item["板块"], "ETF": item["ETF"], "代码": code,
                "最新价": float("nan"), "涨跌幅%": float("nan"),
                "MA5": float("nan"), "MA10": float("nan"), "MA20": float("nan"),
                "量比20日": float("nan"), "趋势": 0, "量能": 0, "强度": 0,
                "风险": 0, "市场": 0, "总分": 0, "系统判断": f"数据失败：{exc}",
            })

    rank = pd.DataFrame(rows).sort_values("总分", ascending=False).reset_index(drop=True)
    rank["操作"] = rank.apply(lambda r: action_label(r, held_codes), axis=1)

    headline, explanation, tone = recommendation_text(rank, held_codes)
    top_name = rank.iloc[0]["板块"] if not rank.empty else "—"
    top_score = int(rank.iloc[0]["总分"]) if not rank.empty else 0
    eligible = int((rank["总分"] >= 70).sum())
    avg_score = round(float(rank["总分"].mean()), 1) if not rank.empty else 0

    st.markdown(
        f"""
        <div class="card">
          <div class="muted">今日总判断</div>
          <div style="font-size:30px;font-weight:800;margin:6px 0 8px;">{headline}</div>
          <span class="pill {tone}">{status_text}</span>
          <div style="margin-top:12px;opacity:.82;">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("市场温度", f"{avg_score}°")
    m2.metric("最强方向", top_name)
    m3.metric("最高评分", top_score)
    m4.metric("达到试仓线", eligible)

    tabs = st.tabs(["今日操作", "实时排行", "我的持仓", "趋势详情", "规则说明"])

    with tabs[0]:
        st.subheader("今日操作卡片")
        top_candidates = rank.head(max_positions)
        investable = total_capital * (1 - cash_reserve)
        positive = top_candidates[top_candidates["总分"] >= 70].copy()
        if not positive.empty:
            positive["建议金额"] = (positive["总分"] / positive["总分"].sum() * investable).round(-2)
        for idx, row in top_candidates.iterrows():
            action = row["操作"]
            cls = status_class(action)
            amount_text = ""
            if not positive.empty and row["代码"] in set(positive["代码"]):
                amount = float(positive.loc[positive["代码"] == row["代码"], "建议金额"].iloc[0])
                amount_text = f" · 上限参考 ¥{amount:,.0f}"
            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
                    <div>
                      <div class="muted">#{idx+1} · {row['ETF']} · {row['代码']}</div>
                      <div style="font-size:24px;font-weight:800;margin-top:5px;">{row['板块']}</div>
                      <div style="margin-top:8px;"><span class="pill {cls}">{action}</span></div>
                    </div>
                    <div style="text-align:right;">
                      <div class="score">{int(row['总分'])}</div>
                      <div class="muted">综合评分</div>
                    </div>
                  </div>
                  <div style="margin-top:13px;opacity:.80;">{row['系统判断']}{amount_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if positive.empty:
            st.info("暂无方向达到70分试仓线，建议保留现金。")
        else:
            st.caption(f"资金上限按总资金 ¥{total_capital:,.0f}、最低现金 {cash_reserve:.0%}、最多 {max_positions} 个方向估算；不是一次性买满建议。")

        avoid = rank[rank["总分"] < 60].head(4)
        if not avoid.empty:
            st.subheader("今天不要碰")
            st.dataframe(avoid[["板块", "ETF", "代码", "总分", "操作", "系统判断"]], use_container_width=True, hide_index=True)

    with tabs[1]:
        show = rank[["板块", "ETF", "代码", "最新价", "涨跌幅%", "MA5", "MA10", "MA20", "总分", "操作", "系统判断"]].copy()
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "最新价": st.column_config.NumberColumn(format="%.3f"),
                "涨跌幅%": st.column_config.NumberColumn(format="%.2f%%"),
                "MA5": st.column_config.NumberColumn(format="%.3f"),
                "MA10": st.column_config.NumberColumn(format="%.3f"),
                "MA20": st.column_config.NumberColumn(format="%.3f"),
                "总分": st.column_config.ProgressColumn(min_value=0, max_value=100),
            },
        )
        with st.expander("查看ETF自动匹配结果"):
            st.dataframe(resolved, use_container_width=True, hide_index=True)

    with tabs[2]:
        if holdings.empty:
            st.info("当前没有持仓。需要启用持仓风控时，在 GitHub 的 data/holdings.csv 填写：name,code,shares,cost_price。")
        else:
            hrows = []
            for _, h in holdings.iterrows():
                code = str(h.get("code", "")).zfill(6)
                try:
                    hist = load_history(code)
                    if code in spot_map.index:
                        hist = merge_intraday_quote(hist, spot_map.loc[code])
                    snap = latest_snapshot(hist)
                    shares = float(h.get("shares", 0))
                    cost_price = float(h.get("cost_price", 0))
                    market_value = shares * snap["close"]
                    cost_value = shares * cost_price
                    pnl = market_value - cost_value
                    pnl_pct = pnl / cost_value * 100 if cost_value else 0
                    if snap["close"] < snap["ma20"]:
                        action = "退出信号"
                    elif snap["close"] < snap["ma10"]:
                        action = "减仓40%观察"
                    elif snap["close"] < snap["ma5"]:
                        action = "轻减20% / 等收盘"
                    else:
                        action = "继续持有"
                    hrows.append({
                        "名称": h.get("name", code), "代码": code, "最新价": snap["close"],
                        "市值": market_value, "盈亏": pnl, "收益率%": pnl_pct, "风控": action,
                    })
                except Exception as exc:
                    hrows.append({"名称": h.get("name", code), "代码": code, "风控": f"数据失败：{exc}"})
            st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)

    with tabs[3]:
        available = rank[rank["代码"].isin(histories.keys())]
        if available.empty:
            st.warning("当前没有成功加载的K线数据。系统不会再报错；请稍后刷新或等待备用数据源恢复。")
        else:
            selected = st.selectbox("选择板块", available["板块"].tolist())
            row = available[available["板块"] == selected].iloc[0]
            code = row["代码"]
            d = add_indicators(histories[code]).tail(100)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=d.date, open=d.open, high=d.high, low=d.low, close=d.close, name="K线"))
            for n in (5, 10, 20):
                fig.add_trace(go.Scatter(x=d.date, y=d[f"ma{n}"], mode="lines", name=f"MA{n}"))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=10, b=5), legend_orientation="h")
            st.plotly_chart(fig, use_container_width=True)
            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric("趋势", int(row["趋势"]))
            q2.metric("量能", int(row["量能"]))
            q3.metric("强度", int(row["强度"]))
            q4.metric("风险安全", int(row["风险"]))
            q5.metric("市场", int(row["市场"]))
            st.info(row["系统判断"])

    with tabs[4]:
        st.markdown(
            """
            **核心纪律**

            1. 价格在5日、10日、20日线上方且均线多头，优先级最高。
            2. 跌破5日线先观察或轻减；跌破10日线明显降仓；跌破20日线退出。
            3. 放量突破优于无量突破；放量下跌明显扣分。
            4. 先试仓、再确认、再加仓；单日评分不等于一次性满仓。
            5. 14:45以后盘中信号更接近收盘，但最后几分钟仍可能反转。
            """
        )

    st.caption(f"最后刷新：{datetime.now():%H:%M:%S} · 免费公开行情可能延迟或限流 · 本系统不自动下单，也不保证收益")

# run_every is set statically for Streamlit compatibility; manual refresh remains available.
dashboard()
