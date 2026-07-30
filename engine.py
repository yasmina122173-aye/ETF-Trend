from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf


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


ETF_UNIVERSE = {
    "510300": "沪深300ETF",
    "513180": "恒生科技ETF华夏",
    "515170": "食品饮料ETF华夏",
    "159625": "绿色电力ETF嘉实",
    "515220": "煤炭ETF国泰",
    "512400": "有色金属ETF",
    "512480": "半导体ETF",
    "561700": "电网设备ETF",
    "561160": "锂电池ETF",
    "159930": "能源ETF",
}


def _yf_symbol(code: str) -> str:
    code = str(code).strip().zfill(6)
    return f"{code}.SS" if code.startswith(("5", "6", "9")) else f"{code}.SZ"


def _normalise_history(df: pd.DataFrame, days: int = 180) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("未获取到历史行情")

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    rename = {
        "Date": "date", "Datetime": "date",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }
    df = df.rename(columns=rename)

    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError("历史行情字段异常")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["amount"] = np.nan
    df["pct_change"] = df["close"].pct_change() * 100

    cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
    return (
        df[cols]
        .dropna(subset=["date", "close"])
        .sort_values("date")
        .tail(days)
        .reset_index(drop=True)
    )


def fetch_etf_history(code: str, days: int = 180) -> pd.DataFrame:
    code = str(code).strip().zfill(6)
    if len(code) != 6 or not code.isdigit():
        raise ValueError("ETF代码必须是6位数字")

    df = yf.download(
        _yf_symbol(code),
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    return _normalise_history(df, days)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = ["close", "high", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"行情缺少字段：{','.join(missing)}")

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


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def score_etf(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> ScoreResult:
    d = add_indicators(df).dropna(subset=["ma20", "vol_ma20"]).copy()
    if len(d) < 3:
        raise ValueError("历史数据不足，至少需要20个交易日")

    x = d.iloc[-1]
    prev = d.iloc[-2]

    close = float(x["close"])
    ma5 = float(x["ma5"])
    ma10 = float(x["ma10"])
    ma20 = float(x["ma20"])
    high20 = float(x["high20"])
    prev_close = float(prev["close"])

    trend = 0
    if close > ma5 > ma10 > ma20:
        trend += 24
    elif close > ma5 > ma10:
        trend += 19
    elif close > ma10:
        trend += 13
    elif close > ma20:
        trend += 8
    else:
        trend += 2

    last5 = d.tail(5)
    above5 = int((last5["close"] >= last5["ma5"]).sum())
    trend += {5: 16, 4: 13, 3: 10, 2: 6, 1: 3, 0: 0}[above5]
    trend = _clamp(trend, 0, 40)

    breakout = bool(close >= high20 * 0.995)
    up_day = close > prev_close
    vr_raw = x["volume_ratio"]
    vr = float(vr_raw) if pd.notna(vr_raw) else 1.0

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

    r5_raw = x["ret5"]
    r20_raw = x["ret20"]
    r5 = float(r5_raw) if pd.notna(r5_raw) else 0.0
    r20 = float(r20_raw) if pd.notna(r20_raw) else 0.0
    strength = 0
    strength += 8 if r5 > 0.04 else 6 if r5 > 0.015 else 4 if r5 > 0 else 1
    strength += 8 if r20 > 0.10 else 6 if r20 > 0.04 else 4 if r20 > 0 else 1
    strength += 4 if breakout else 2 if close >= float(d.tail(20)["close"].quantile(0.75)) else 0
    strength = _clamp(strength, 0, 20)

    dist_raw = x["distance_ma20"]
    dist = float(dist_raw) if pd.notna(dist_raw) else 0.0
    risk = 10
    if dist > 0.18:
        risk -= 6
    elif dist > 0.12:
        risk -= 4
    elif dist > 0.08:
        risk -= 2
    if vr > 2.0 and not breakout:
        risk -= 3
    if close < ma10:
        risk -= 3
    risk = _clamp(risk, 0, 10)

    market = 5
    if benchmark_df is not None and not benchmark_df.empty:
        b = add_indicators(benchmark_df).dropna(subset=["ma20"])
        if not b.empty:
            bx = b.iloc[-1]
            bc = float(bx["close"])
            b5 = float(bx["ma5"])
            b10 = float(bx["ma10"])
            b20 = float(bx["ma20"])
            if bc > b5 > b10 > b20:
                market = 10
            elif bc > b10 > b20:
                market = 8
            elif bc > b20:
                market = 6
            elif bc > b10:
                market = 4
            else:
                market = 2

    total = trend + volume + strength + risk + market
    signal = (
        "重仓候选" if total >= 90 else
        "半仓候选" if total >= 80 else
        "试仓候选" if total >= 70 else
        "观察" if total >= 60 else
        "回避"
    )

    reasons = [
        "多头排列" if close > ma5 > ma10 > ma20 else "均线尚未完全多头",
        "放量突破" if breakout and vr >= 1.25 else
        "量能正常" if 0.75 <= vr <= 1.25 else
        "量能偏弱" if vr < 0.75 else "量能偏大",
    ]
    if close < ma5:
        reasons.append("跌破5日线")
    if close < ma10:
        reasons.append("跌破10日线")
    if dist > 0.12:
        reasons.append("偏离20日线较远")

    return ScoreResult(trend, volume, strength, risk, market, total, signal, "；".join(reasons))


def latest_snapshot(df: pd.DataFrame) -> dict:
    d = add_indicators(df).dropna(subset=["ma20", "vol_ma20"])
    if d.empty:
        raise ValueError("历史数据不足，无法计算均线")

    x = d.iloc[-1]
    pct = x["pct_change"] if "pct_change" in d.columns else np.nan
    ret20 = x["ret20"]

    return {
        "date": x["date"],
        "close": float(x["close"]),
        "pct_change": float(pct) if pd.notna(pct) else np.nan,
        "ma5": float(x["ma5"]),
        "ma10": float(x["ma10"]),
        "ma20": float(x["ma20"]),
        "volume_ratio": float(x["volume_ratio"]) if pd.notna(x["volume_ratio"]) else 1.0,
        "ret20": float(ret20) if pd.notna(ret20) else np.nan,
    }


def fetch_etf_spot() -> pd.DataFrame:
    rows = []

    for code, name in ETF_UNIVERSE.items():
        try:
            symbol = _yf_symbol(code)
            data = yf.download(
                symbol,
                period="5d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=15,
            )
            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.dropna(subset=["Close"])
            if data.empty:
                continue

            last = data.iloc[-1]
            current_day = data.index[-1].date()
            today = data[data.index.date == current_day]
            if today.empty:
                today = data.tail(1)

            close = float(last["Close"])
            previous_daily = yf.download(
                symbol,
                period="10d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=15,
            )
            if isinstance(previous_daily.columns, pd.MultiIndex):
                previous_daily.columns = previous_daily.columns.get_level_values(0)

            prev_close = close
            if previous_daily is not None and len(previous_daily) >= 2:
                prev_close = float(previous_daily["Close"].dropna().iloc[-2])

            rows.append({
                "code": code,
                "name": name,
                "close": close,
                "pct_change": (close / prev_close - 1) * 100 if prev_close else np.nan,
                "open": float(today["Open"].iloc[0]),
                "high": float(today["High"].max()),
                "low": float(today["Low"].min()),
                "prev_close": prev_close,
                "volume": float(today["Volume"].sum()),
                "amount": np.nan,
                "quote_time": str(data.index[-1]),
            })
        except Exception:
            continue

    if not rows:
        raise RuntimeError("备用行情源暂时没有返回数据，请稍后刷新")

    return pd.DataFrame(rows)


def merge_intraday_quote(history: pd.DataFrame, quote: pd.Series | dict) -> pd.DataFrame:
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
        "amount": pd.to_numeric(q.get("amount", np.nan), errors="coerce"),
        "pct_change": pd.to_numeric(q.get("pct_change", np.nan), errors="coerce"),
    }

    mask = pd.to_datetime(out["date"]).dt.normalize() == today
    if mask.any():
        for key, value in row.items():
            out.loc[mask, key] = value
    else:
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    return out.sort_values("date").reset_index(drop=True)
