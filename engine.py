from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import akshare as ak
except Exception:
    ak = None
try:
    import yfinance as yf
except Exception:
    yf = None


@dataclass
class ScoreResult:
    trend: int
    volume: int
    strength: int
    risk: int
    market: int
    total: int
    signal: str
    reason: str


def _yf_symbol(code: str) -> str:
    code = str(code).strip().zfill(6)
    return f"{code}.SS" if code.startswith(("5", "6", "9")) else f"{code}.SZ"


def _normalise_history(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("未获取到历史行情")
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    rename = {
        "Date": "date", "Datetime": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change",
    }
    df = df.rename(columns=rename)
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError("历史行情字段异常")
    if "amount" not in df.columns:
        df["amount"] = np.nan
    if "pct_change" not in df.columns:
        df["pct_change"] = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100
    keep = ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").tail(days).reset_index(drop=True)


def fetch_etf_history(code: str, days: int = 180) -> pd.DataFrame:
    """历史行情：优先 AkShare，失败时自动切换 Yahoo Finance。"""
    code = str(code).strip().zfill(6)
    errors = []
    if ak is not None:
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date="20200101", end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
            return _normalise_history(df, days)
        except Exception as exc:
            errors.append(f"AkShare: {exc}")
    if yf is not None:
        try:
            df = yf.download(_yf_symbol(code), period="1y", interval="1d", auto_adjust=False, progress=False, threads=False, timeout=12)
            return _normalise_history(df, days)
        except Exception as exc:
            errors.append(f"Yahoo: {exc}")
    raise RuntimeError("；".join(errors) or "没有可用行情源")

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in (5, 10, 20, 60):
        out[f"ma{n}"] = out["close"].rolling(n).mean()
    out["vol_ma5"] = out["volume"].rolling(5).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / out["vol_ma20"].replace(0, np.nan)
    out["ret5"] = out["close"].pct_change(5)
    out["ret20"] = out["close"].pct_change(20)
    out["high20"] = out["high"].rolling(20).max()
    out["distance_ma20"] = out["close"] / out["ma20"] - 1
    return out


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def score_etf(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> ScoreResult:
    d = add_indicators(df).dropna(subset=["ma20", "vol_ma20"]).copy()
    if len(d) < 3:
        raise ValueError("历史数据不足，至少需要20个交易日")
    x = d.iloc[-1]
    prev = d.iloc[-2]

    # 1) 趋势 40分
    trend = 0
    if x.close > x.ma5 > x.ma10 > x.ma20:
        trend += 24
    elif x.close > x.ma5 > x.ma10:
        trend += 19
    elif x.close > x.ma10:
        trend += 13
    elif x.close > x.ma20:
        trend += 8
    else:
        trend += 2

    last5 = d.tail(5)
    above5 = int((last5.close >= last5.ma5).sum())
    trend += {5: 16, 4: 13, 3: 10, 2: 6, 1: 3, 0: 0}[above5]
    trend = _clamp(trend, 0, 40)

    # 2) 量能 20分
    breakout = bool(x.close >= x.high20 * 0.995)
    up_day = x.close > prev.close
    vr = float(x.volume_ratio) if pd.notna(x.volume_ratio) else 1.0
    if breakout and up_day and vr >= 1.25:
        volume = 20
    elif up_day and 0.75 <= vr <= 1.25:
        volume = 16
    elif up_day and vr < 0.75:
        volume = 13
    elif (not up_day) and vr < 0.85:
        volume = 12
    elif (not up_day) and vr >= 1.35:
        volume = 3
    else:
        volume = 9

    # 3) 强度 20分
    strength = 0
    r5 = float(x.ret5) if pd.notna(x.ret5) else 0
    r20 = float(x.ret20) if pd.notna(x.ret20) else 0
    strength += 8 if r5 > 0.04 else 6 if r5 > 0.015 else 4 if r5 > 0 else 1
    strength += 8 if r20 > 0.10 else 6 if r20 > 0.04 else 4 if r20 > 0 else 1
    strength += 4 if breakout else 2 if x.close >= d.tail(20).close.quantile(0.75) else 0
    strength = _clamp(strength, 0, 20)

    # 4) 风险 10分，分数越高越安全
    risk = 10
    dist = float(x.distance_ma20) if pd.notna(x.distance_ma20) else 0
    if dist > 0.18:
        risk -= 6
    elif dist > 0.12:
        risk -= 4
    elif dist > 0.08:
        risk -= 2
    if vr > 2.0 and not breakout:
        risk -= 3
    if x.close < x.ma10:
        risk -= 3
    risk = _clamp(risk, 0, 10)

    # 5) 市场环境 10分
    market = 5
    if benchmark_df is not None and not benchmark_df.empty:
        b = add_indicators(benchmark_df).dropna(subset=["ma20"])
        if not b.empty:
            bx = b.iloc[-1]
            if bx.close > bx.ma5 > bx.ma10 > bx.ma20:
                market = 10
            elif bx.close > bx.ma10 > bx.ma20:
                market = 8
            elif bx.close > bx.ma20:
                market = 6
            elif bx.close > bx.ma10:
                market = 4
            else:
                market = 2

    total = trend + volume + strength + risk + market
    if total >= 90:
        signal = "重仓候选"
    elif total >= 80:
        signal = "半仓候选"
    elif total >= 70:
        signal = "试仓候选"
    elif total >= 60:
        signal = "观察"
    else:
        signal = "回避"

    reasons = []
    reasons.append("多头排列" if x.close > x.ma5 > x.ma10 > x.ma20 else "均线尚未完全多头")
    reasons.append("放量突破" if breakout and vr >= 1.25 else "量能正常" if 0.75 <= vr <= 1.25 else "量能偏弱" if vr < 0.75 else "量能偏大")
    if x.close < x.ma5:
        reasons.append("跌破5日线")
    if x.close < x.ma10:
        reasons.append("跌破10日线")
    if dist > 0.12:
        reasons.append("偏离20日线较远")

    return ScoreResult(trend, volume, strength, risk, market, total, signal, "；".join(reasons))


def latest_snapshot(df: pd.DataFrame) -> dict:
    d = add_indicators(df).dropna(subset=["ma20", "vol_ma20"])
    x = d.iloc[-1]
    return {
        "date": x.date,
        "close": float(x.close),
        "pct_change": float(x.pct_change) if "pct_change" in d.columns and pd.notna(x.pct_change) else np.nan,
        "ma5": float(x.ma5), "ma10": float(x.ma10), "ma20": float(x.ma20),
        "volume_ratio": float(x.volume_ratio),
        "ret20": float(x.ret20) if pd.notna(x.ret20) else np.nan,
    }


ETF_UNIVERSE = {
    "510300": "沪深300ETF", "513180": "恒生科技ETF华夏", "515170": "食品饮料ETF华夏",
    "159625": "绿色电力ETF嘉实", "515220": "煤炭ETF国泰", "512400": "有色金属ETF",
    "512480": "半导体ETF", "561700": "电网设备ETF", "561160": "锂电池ETF",
    "159930": "能源ETF",
}


def fetch_etf_spot() -> pd.DataFrame:
    """实时行情：优先 AkShare；失败时用 Yahoo 5分钟行情覆盖默认观察池。"""
    if ak is not None:
        try:
            df = ak.fund_etf_spot_em()
            if df is not None and not df.empty:
                rename = {
                    "代码": "code", "名称": "name", "最新价": "close", "涨跌幅": "pct_change",
                    "开盘价": "open", "最高价": "high", "最低价": "low", "昨收": "prev_close",
                    "成交量": "volume", "成交额": "amount", "更新时间": "quote_time",
                }
                df = df.rename(columns=rename).copy()
                df["code"] = df["code"].astype(str).str.zfill(6)
                for c in ["close", "pct_change", "open", "high", "low", "prev_close", "volume", "amount"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception:
            pass
    if yf is None:
        raise RuntimeError("AkShare不可用，且未安装 yfinance 备用源")
    rows = []
    for code, name in ETF_UNIVERSE.items():
        try:
            t = yf.Ticker(_yf_symbol(code))
            d = t.history(period="5d", interval="5m", auto_adjust=False)
            if d is None or d.empty:
                continue
            d = d.dropna(subset=["Close"])
            last = d.iloc[-1]
            day = d.index[-1].date()
            today = d[d.index.date == day]
            prev_close = float(d["Close"].iloc[-2]) if len(d) > 1 else float(last["Close"])
            close = float(last["Close"])
            rows.append({
                "code": code, "name": name, "close": close,
                "pct_change": (close / prev_close - 1) * 100 if prev_close else np.nan,
                "open": float(today["Open"].iloc[0]), "high": float(today["High"].max()),
                "low": float(today["Low"].min()), "prev_close": prev_close,
                "volume": float(today["Volume"].sum()), "amount": np.nan,
                "quote_time": str(d.index[-1]),
            })
        except Exception:
            continue
    if not rows:
        raise RuntimeError("主数据源与备用数据源均不可用")
    return pd.DataFrame(rows)

def merge_intraday_quote(history: pd.DataFrame, quote: pd.Series | dict) -> pd.DataFrame:
    """Replace/add today's daily bar with the latest intraday quote for provisional scoring."""
    out = history.copy()
    q = dict(quote)
    today = pd.Timestamp.now().normalize()
    close = pd.to_numeric(q.get("close"), errors="coerce")
    if pd.isna(close):
        return out

    row = {
        "date": today,
        "open": pd.to_numeric(q.get("open", close), errors="coerce"),
        "high": pd.to_numeric(q.get("high", close), errors="coerce"),
        "low": pd.to_numeric(q.get("low", close), errors="coerce"),
        "close": close,
        "volume": pd.to_numeric(q.get("volume", 0), errors="coerce"),
        "amount": pd.to_numeric(q.get("amount", 0), errors="coerce"),
        "pct_change": pd.to_numeric(q.get("pct_change", np.nan), errors="coerce"),
    }
    mask = pd.to_datetime(out["date"]).dt.normalize() == today
    if mask.any():
        for k, v in row.items():
            out.loc[mask, k] = v
    else:
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)
