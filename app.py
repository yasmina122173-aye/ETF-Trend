from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (
    fetch_etf_history, fetch_etf_spot, merge_intraday_quote,
    add_indicators, score_etf, latest_snapshot,
)

BASE = Path(__file__).parent
DATA = BASE / "data"

st.set_page_config(
    page_title="TrendOS V2.3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container{max-width:1120px;padding-top:1rem;padding-bottom:4rem}
h1,h2,h3{letter-spacing:-.02em}
[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.16);padding:14px;border-radius:18px}
.hero{padding:22px;border-radius:24px;background:linear-gradient(135deg,rgba(45,125,255,.18),rgba(122,87,255,.12));border:1px solid rgba(127,127,127,.18);margin-bottom:14px}
.hero-kicker{font-size:13px;opacity:.7;margin-bottom:8px}
.hero-title{font-size:clamp(30px,6vw,54px);font-weight:800;line-height:1.08;margin-bottom:8px}
.hero-sub{font-size:15px;opacity:.78}
.card{padding:18px;border-radius:20px;background:rgba(127,127,127,.07);border:1px solid rgba(127,127,127,.16);margin-bottom:12px}
.score{font-size:38px;font-weight:800;line-height:1}
.muted{opacity:.68;font-size:13px}
.pill{display:inline-block;padding:5px 10px;border-radius:999px;font-size:12px;font-weight:700;background:rgba(45,125,255,.15);margin-right:6px}
.good{background:rgba(30,190,110,.14)} .warn{background:rgba(255,180,40,.15)} .danger{background:rgba(255,70,70,.13)}
.total-score{padding:18px 20px;border-radius:20px;background:linear-gradient(135deg,rgba(45,125,255,.22),rgba(122,87,255,.14));border:1px solid rgba(127,127,127,.18);display:flex;justify-content:space-between;align-items:center;gap:16px;margin:16px 0 12px}
.total-score-number{font-size:52px;font-weight:900;line-height:1}
.total-score-label{font-size:14px;opacity:.72}
.score-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0}
.score-box{padding:14px;border-radius:16px;border:1px solid rgba(127,127,127,.16);background:rgba(127,127,127,.06)}
.score-name{font-size:13px;opacity:.72}.score-value{font-size:30px;font-weight:800;margin:3px 0}.score-note{font-size:12px;opacity:.72;line-height:1.4}
@media(max-width:720px){
 .block-container{padding-left:.8rem;padding-right:.8rem;padding-top:.5rem}
 .hero{padding:18px;border-radius:20px}.card{padding:15px;border-radius:18px}
 .total-score{padding:16px}.total-score-number{font-size:44px}
 .score-grid{grid-template-columns:1fr 1fr}.score-box:last-child{grid-column:1/-1}
}
</style>
""", unsafe_allow_html=True)

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

def market_status():
    now = datetime.now()
    hm = now.hour * 60 + now.minute
    weekday = now.weekday() < 5
    open_now = weekday and ((570 <= hm <= 690) or (780 <= hm <= 900))
    mins_left = max(0, 900 - hm) if weekday and 780 <= hm <= 900 else None
    if open_now:
        return "交易中", mins_left
    if weekday and hm < 570:
        return "尚未开盘", None
    return "已收盘 / 休市", None

def resolve_keywords(spot: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    records = []
    for kw in keywords:
        matches = spot[spot["name"].astype(str).str.contains(kw, case=False, na=False)].copy()
        if matches.empty:
            records.append({"板块":kw,"代码":"","ETF":"未匹配","匹配状态":"需手动指定"})
            continue
        if "amount" in matches.columns and matches["amount"].notna().any():
            matches = matches.sort_values("amount", ascending=False, na_position="last")
        best = matches.iloc[0]
        records.append({"板块":kw,"代码":str(best["code"]).zfill(6),"ETF":str(best.get("name",kw)),"匹配状态":"自动选择"})
    return pd.DataFrame(records)

def load_holdings():
    p = DATA / "holdings.csv"
    if not p.exists():
        return pd.DataFrame(columns=["name","code","shares","cost_price"])
    return pd.read_csv(p, dtype={"code":str})

def action_label(row, held_codes):
    held = row["代码"] in held_codes
    score = row["总分"]
    if held:
        if row["最新价"] < row["MA20"]: return "退出信号"
        if row["最新价"] < row["MA10"]: return "减仓观察"
        if row["最新价"] < row["MA5"]: return "轻减 / 等收盘"
        return "继续持有"
    if score >= 90: return "重仓候选（分批）"
    if score >= 80: return "半仓候选（分批）"
    if score >= 70: return "试仓候选"
    if score >= 60: return "观察"
    return "回避"

def status_class(action):
    if any(x in action for x in ["持有","候选"]): return "good"
    if any(x in action for x in ["退出","回避"]): return "danger"
    return "warn"

def level(score, maximum):
    ratio = score / maximum if maximum else 0
    if ratio >= .85: return "优秀", "good"
    if ratio >= .65: return "良好", "good"
    if ratio >= .45: return "一般", "warn"
    return "偏弱", "danger"

def score_explanations(row, hist, benchmark):
    d = add_indicators(hist).dropna(subset=["ma20","vol_ma20"])
    x = d.iloc[-1]
    close, ma5, ma10, ma20 = [float(x[k]) for k in ["close","ma5","ma10","ma20"]]
    above5 = int((d.tail(5)["close"] >= d.tail(5)["ma5"]).sum())
    vr = float(x["volume_ratio"]) if pd.notna(x["volume_ratio"]) else 1.0
    r5 = float(x["ret5"]) * 100 if pd.notna(x["ret5"]) else 0
    r20 = float(x["ret20"]) * 100 if pd.notna(x["ret20"]) else 0
    dist = (close / ma20 - 1) * 100
    breakout = close >= float(x["high20"]) * .995

    if close > ma5 > ma10 > ma20:
        trend_note = f"价格与均线多头排列，近5天有{above5}天站在5日线上。"
    elif close > ma5 > ma10:
        trend_note = f"价格站上5日、10日线，但20日线结构尚未完全走顺；近5天{above5}天站上5日线。"
    elif close > ma10:
        trend_note = f"价格站上10日线，但短中期均线仍纠缠；近5天{above5}天站上5日线。"
    else:
        trend_note = "价格仍在关键均线下方，趋势尚未确认。"

    if breakout and vr >= 1.25:
        volume_note = f"放量突破，当前成交量约为20日均量的{vr:.2f}倍。"
    elif vr < .75:
        volume_note = f"量能偏弱，当前约为20日均量的{vr:.2f}倍。"
    elif vr > 1.35:
        volume_note = f"量能明显放大至20日均量的{vr:.2f}倍，需结合涨跌判断。"
    else:
        volume_note = f"量能正常，当前约为20日均量的{vr:.2f}倍。"

    strength_note = f"近5日涨跌{r5:+.1f}%，近20日{r20:+.1f}%" + ("，并接近/突破20日新高。" if breakout else "。")
    risk_note = f"当前距离20日线{dist:+.1f}%。" + ("偏离较大，追高风险上升。" if dist > 12 else "与中期均线距离仍在可控范围。")
    market_note = f"市场环境得分{int(row['市场'])}/10。" + ("基准处于多头环境。" if row["市场"] >= 8 else "大盘环境一般，建议压低总体仓位。" if row["市场"] <= 4 else "大盘环境中性。")
    return trend_note, volume_note, strength_note, risk_note, market_note

status_text, mins_left = market_status()
now = datetime.now()
st.markdown(f"""
<div class="hero">
 <div class="hero-kicker">TrendOS V2.3 · {now:%Y-%m-%d %H:%M:%S}</div>
 <div class="hero-title">15:00 前，只看今天该怎么做</div>
 <div class="hero-sub">{status_text}{f" · 距离收盘约 {mins_left} 分钟" if mins_left is not None else ""} · 盘中信号为临时日K</div>
</div>
""", unsafe_allow_html=True)

with st.expander("⚙️ 系统设置与观察池"):
    c1,c2,c3,c4 = st.columns(4)
    total_capital = c1.number_input("总资金（元）", min_value=1000.0, value=80000.0, step=1000.0)
    cash_reserve = c2.slider("最低现金比例",0,60,20,5)/100
    max_positions = c3.slider("最多持有方向",1,6,3)
    c4.select_slider("刷新频率",options=[15,30,60,120],value=30,format_func=lambda x:f"{x}秒")
    benchmark_code = st.text_input("市场基准ETF代码",value="510300",max_chars=6)
    keyword_text = st.text_area("观察板块关键词（每行一个）",value="\n".join(DEFAULT_KEYWORDS),height=190)

keywords = [x.strip() for x in keyword_text.splitlines() if x.strip()]
holdings = load_holdings()
held_codes = set(holdings.get("code",pd.Series(dtype=str)).fillna("").astype(str).str.zfill(6))

@st.fragment(run_every="30s")
def dashboard():
    try:
        spot = load_spot()
    except Exception as exc:
        st.error(f"实时行情暂时不可用：{exc}")
        return

    resolved = resolve_keywords(spot, keywords)
    valid = resolved[resolved["代码"].astype(str).str.fullmatch(r"\d{6}")].copy()
    if valid.empty:
        st.warning("没有匹配到有效ETF。")
        return

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
                "板块":item["板块"],"ETF":item["ETF"],"代码":code,
                "最新价":snap["close"],"涨跌幅%":snap["pct_change"],
                "MA5":snap["ma5"],"MA10":snap["ma10"],"MA20":snap["ma20"],
                "趋势":score.trend,"量能":score.volume,"强度":score.strength,
                "风险":score.risk,"市场":score.market,"总分":score.total,
                "系统判断":score.reason,
            })
        except Exception as exc:
            rows.append({"板块":item["板块"],"ETF":item["ETF"],"代码":code,
                         "最新价":float("nan"),"涨跌幅%":float("nan"),"MA5":float("nan"),
                         "MA10":float("nan"),"MA20":float("nan"),"趋势":0,"量能":0,
                         "强度":0,"风险":0,"市场":0,"总分":0,"系统判断":f"数据失败：{exc}"})

    rank = pd.DataFrame(rows).sort_values("总分",ascending=False).reset_index(drop=True)
    rank["操作"] = rank.apply(lambda r:action_label(r,held_codes),axis=1)
    avg_score = round(float(rank["总分"].mean()),1)
    top = rank.iloc[0]
    strong = rank[rank["总分"]>=80]
    headline = "出现可关注方向" if not strong.empty else "今天以观察为主"
    explanation = f"{top['板块']}评分最高，但仍建议结合回踩或收盘确认。"

    st.markdown(f"""<div class="card"><div class="muted">今日总评</div>
    <div style="font-size:30px;font-weight:800;margin:6px 0">{headline}</div>
    <span class="pill good">{status_text}</span><div style="margin-top:12px">{explanation}</div></div>""",unsafe_allow_html=True)

    m1,m2,m3,m4=st.columns(4)
    m1.metric("市场温度",f"{avg_score}°");m2.metric("最强方向",top["板块"])
    m3.metric("最高评分",int(top["总分"]));m4.metric("达到试仓线",int((rank["总分"]>=70).sum()))

    tabs=st.tabs(["今日操作","实时排行","我的持仓","趋势详情","规则说明"])
    with tabs[0]:
        st.subheader("今日操作配置")
        candidates=rank.head(max_positions)
        positive=candidates[candidates["总分"]>=70].copy()
        if not positive.empty:
            positive["建议金额"]=(positive["总分"]/positive["总分"].sum()*total_capital*(1-cash_reserve)).round(-2)
        for idx,row in candidates.iterrows():
            amount=""
            if not positive.empty and row["代码"] in set(positive["代码"]):
                v=float(positive.loc[positive["代码"]==row["代码"],"建议金额"].iloc[0])
                amount=f" · 上限参考 ¥{v:,.0f}"
            st.markdown(f"""<div class="card"><div style="display:flex;justify-content:space-between">
            <div><div class="muted">#{idx+1} · {row['ETF']} · {row['代码']}</div>
            <div style="font-size:24px;font-weight:800">{row['板块']}</div>
            <span class="pill {status_class(row['操作'])}">{row['操作']}</span></div>
            <div><div class="score">{int(row['总分'])}</div><div class="muted">综合评分</div></div></div>
            <div style="margin-top:12px">{row['系统判断']}{amount}</div></div>""",unsafe_allow_html=True)

    with tabs[1]:
        st.dataframe(rank[["板块","ETF","代码","最新价","涨跌幅%","MA5","MA10","MA20","总分","操作","系统判断"]],
                     use_container_width=True,hide_index=True)

    with tabs[2]:
        st.info("持仓页沿用 data/holdings.csv；当前版本重点更新了评分解释。") if holdings.empty else st.dataframe(holdings,use_container_width=True,hide_index=True)

    with tabs[3]:
        available=rank[rank["代码"].isin(histories.keys())]
        if available.empty:
            st.warning("当前没有成功加载的K线数据。")
        else:
            selected=st.selectbox("选择板块",available["板块"].tolist())
            row=available[available["板块"]==selected].iloc[0]
            code=row["代码"]; hist=histories[code]
            d=add_indicators(hist).tail(100)
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=d["date"],open=d["open"],high=d["high"],low=d["low"],close=d["close"],name="K线"))
            for n in (5,10,20):
                fig.add_trace(go.Scatter(x=d["date"],y=d[f"ma{n}"],mode="lines",name=f"MA{n}"))
            fig.update_layout(height=520,xaxis_rangeslider_visible=False,margin=dict(l=5,r=5,t=10,b=5),legend_orientation="h")
            st.plotly_chart(fig,use_container_width=True)

            notes=score_explanations(row,hist,benchmark)
            total_score=int(row["总分"])
            total_tag,total_cls=level(total_score,100)
            st.markdown(f"""<div class="total-score">
            <div><div class="total-score-label">综合总分</div><div style="font-size:20px;font-weight:800;margin-top:5px">{row['板块']}</div>
            <div style="margin-top:9px"><span class="pill {total_cls}">{total_tag}</span></div></div>
            <div style="text-align:right"><div class="total-score-number">{total_score}</div><div class="total-score-label">满分 100</div></div>
            </div>""",unsafe_allow_html=True)

            labels=[("趋势",int(row["趋势"]),40,notes[0]),("量能",int(row["量能"]),20,notes[1]),
                    ("强度",int(row["强度"]),20,notes[2]),("风险安全",int(row["风险"]),10,notes[3]),
                    ("市场",int(row["市场"]),10,notes[4])]
            boxes=""
            for name,value,maximum,note in labels:
                tag,cls=level(value,maximum)
                boxes+=f"""<div class="score-box"><div class="score-name">{name} · {value}/{maximum}</div>
                <div class="score-value">{value}</div><span class="pill {cls}">{tag}</span>
                <div class="score-note" style="margin-top:9px">{note}</div></div>"""
            st.markdown(f'<div class="score-grid">{boxes}</div>',unsafe_allow_html=True)

            with st.expander("查看五项评分的详细计算逻辑"):
                st.markdown(f"""
**趋势 {int(row["趋势"])}/40**  
{notes[0]}

**量能 {int(row["量能"])}/20**  
{notes[1]}

**强度 {int(row["强度"])}/20**  
{notes[2]}

**风险安全 {int(row["风险"])}/10**  
{notes[3]}

**市场 {int(row["市场"])}/10**  
{notes[4]}
""")
            st.info(row["系统判断"])

    with tabs[4]:
        st.markdown("""
**五项评分**
- 趋势 40分：价格与MA5、MA10、MA20的结构，以及近5天站稳5日线的天数。
- 量能 20分：成交量相对20日均量，以及是否属于放量突破或放量下跌。
- 强度 20分：近5日、20日表现，以及是否接近20日新高。
- 风险安全 10分：距离20日线是否过远、是否出现异常放量等。
- 市场 10分：沪深300ETF等基准的均线环境。
""")
    st.caption(f"最后刷新：{datetime.now():%H:%M:%S} · 本系统不保证收益，交易前请用券商行情核对。")

dashboard()
