#!/usr/bin/env python3
"""
A-Insight v2 — A股主力资金智能分析系统
数据: 东方财富API + PostgreSQL · 核心: 资金追踪+建仓/出货识别
"""

import json, os, sys, threading, time, re, math
from datetime import datetime, timedelta

import httpx
import numpy as np
import akshare as ak
from multi_factor import run_multi_factor
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text, Column, String, Float, Integer, Date, Index, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

# ============ CONFIG ============
PORT = int(os.environ.get("PORT", 8100))
PG_HOST = os.environ.get("PG_HOST", "127.0.0.1")
PG_PORT = os.environ.get("PG_PORT", "5433")
PG_USER = os.environ.get("PG_USER", "astock")
PG_PASS = os.environ.get("PG_PASS", "")
PG_DB = os.environ.get("PG_DB", "astock")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ============ DATABASE ============
DB_URL = "postgresql://{}:{}@{}:{}/{}".format(PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB)
engine = create_engine(DB_URL, poolclass=NullPool, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class StockDailyV2(Base):
    __tablename__ = "ainsight_v2_daily"
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False, index=True)
    name = Column(String(50))
    trade_date = Column(Date, nullable=False, index=True)
    close = Column(Float)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    pct_chg = Column(Float)
    amount = Column(Float)
    turnover = Column(Float)
    mcap = Column(Float)
    main_net = Column(Float)
    super_large = Column(Float)
    large = Column(Float)
    medium = Column(Float)
    small = Column(Float)
    __table_args__ = (
        Index('idx_v2_code_date', 'code', 'trade_date'),
        Index('idx_v2_date', 'trade_date'),
    )

class SectorFlowV2(Base):
    __tablename__ = "ainsight_v2_sector"
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False, index=True)
    sector_name = Column(String(100))
    pct_chg = Column(Float)
    main_net = Column(Float)
    super_large = Column(Float)
    large = Column(Float)
    medium = Column(Float)
    small = Column(Float)
    __table_args__ = (Index('idx_v2_sector_date', 'trade_date'),)

def ensure_tables():
    inspector = inspect(engine)
    existing = inspector.get_table_names()
    if 'ainsight_v2_daily' not in existing:
        Base.metadata.create_all(engine)
    print("DB ready")

# ============ HELPERS ============
EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}

def sf(val, default=0.0):
    try: return round(float(val or 0), 2)
    except: return default

def today_str():
    return datetime.now().strftime("%Y-%m-%d")

def cache_get(key, ttl=120):
    f = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(f):
        try:
            data = json.load(open(f))
            if time.time() - data.get("_ts", 0) < ttl:
                return data.get("_payload")
        except: pass
    return None

def cache_set(key, payload, ttl=120):
    with open(os.path.join(CACHE_DIR, key + ".json"), "w") as f:
        json.dump({"_ts": time.time(), "_payload": payload}, f, ensure_ascii=False, default=str)

# ============ DATA COLLECTION ============
def fetch_em_page(pn, pz=50, fid="f3", asc=0):
    """Fetch one page from 东方财富 with retry"""
    params = {
        "pn": str(pn), "pz": str(pz),
        "po": str(asc), "np": "1",
        "fields": "f2,f3,f12,f14,f15,f16,f17,f6,f8,f20,f62,f66,f69,f72,f75",
        "fid": fid,
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fltt": "2",
    }
    for attempt in range(4):
        try:
            if attempt > 0:
                time.sleep(8 * attempt)
            r = httpx.get("https://push2.eastmoney.com/api/qt/clist/get",
                          params=params, headers=EM_HEADERS, timeout=25)
            if r.status_code == 200 and len(r.text) > 100:
                data = r.json()
                if data.get("data") and data["data"].get("diff"):
                    return data["data"]["diff"]
        except Exception as e:
            if attempt == 0:
                print("  EM retry {}: {}".format(attempt + 1, str(e)[:60]))
    return None

def collect_spot_data(max_stocks=2500):
    """Collect stock data from 东方财富"""
    today = today_str()
    db = SessionLocal()
    try:
        cnt = db.query(StockDailyV2).filter(StockDailyV2.trade_date == today).count()
        if cnt > 500:
            print("Already {} records, skip".format(cnt))
            return cnt

        print("Fetching stock data (multi-batch)...")
        all_stocks = []

        page = 1
        while page <= 120 and len(all_stocks) < 5400 * 2:
            batch = []
            for p in range(page, min(page + 8, 121)):
                items = fetch_em_page(p, 50, "f12", 1)
                if items is None: break
                batch.extend(items)
                time.sleep(3)
            if not batch: break
            all_stocks.extend(batch)
            page += len(batch) // 50
            print("  Page ~{}: {} stocks".format(page, len(all_stocks)))
            if page <= 120: time.sleep(30)

        # Deduplicate
        seen = set()
        deduped = []
        for s in all_stocks:
            c = str(s.get("f12", ""))
            if c not in seen:
                seen.add(c)
                deduped.append(s)
        print("Got {} unique stocks".format(len(deduped)))

        # f12=code, f14=name, f2=close, f17=open, f15=high, f16=low
        # f3=pct_chg, f6=amount, f8=turnover, f20=mcap
        # f62=main_net, f66=super_large, f69=large, f72=medium, f75=small
        batch = []
        for s in deduped:
            try:
                batch.append(StockDailyV2(
                    code=str(s.get("f12", ""))[:10],
                    name=str(s.get("f14", ""))[:50],
                    trade_date=datetime.strptime(today, "%Y-%m-%d").date(),
                    close=sf(s.get("f2")), open=sf(s.get("f17")),
                    high=sf(s.get("f15")), low=sf(s.get("f16")),
                    pct_chg=sf(s.get("f3")), amount=sf(s.get("f6")),
                    turnover=sf(s.get("f8")), mcap=sf(s.get("f20")),
                    main_net=sf(s.get("f62")), super_large=sf(s.get("f66")),
                    large=sf(s.get("f69")), medium=sf(s.get("f72")),
                    small=sf(s.get("f75")),
                ))
            except: pass

        if batch:
            # Delete old data for today
            db.execute(text("DELETE FROM ainsight_v2_daily WHERE trade_date=:d"), {"d": today})
            db.bulk_save_objects(batch)
            db.commit()
            print("Saved {} stocks".format(len(batch)))
        return len(batch)
    except Exception as e:
        db.rollback()
        print("Collection error: {}".format(e))
        return 0
    finally:
        db.close()

def collect_sector_flow():
    """Collect sector flow via akshare"""
    today = today_str()
    db = SessionLocal()
    try:
        cnt = db.query(SectorFlowV2).filter(SectorFlowV2.trade_date == today).count()
        if cnt > 10:
            return cnt

        print("Fetching sector flow...")
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        if df is None or len(df) == 0:
            print("Sector empty")
            return 0

        batch = []
        for _, r in df.iterrows():
            try:
                batch.append(SectorFlowV2(
                    trade_date=datetime.strptime(today, "%Y-%m-%d").date(),
                    sector_name=str(r.get("名称", ""))[:100],
                    pct_chg=sf(r.get("今日涨跌幅", r.get("涨跌幅", 0))),
                    main_net=sf(r.get("今日主力净流入-净额", r.get("主力净流入-净额", 0))),
                    super_large=sf(r.get("今日超大单净流入-净额", r.get("超大单净流入-净额", 0))),
                    large=sf(r.get("今日大单净流入-净额", r.get("大单净流入-净额", 0))),
                    medium=sf(r.get("今日中单净流入-净额", r.get("中单净流入-净额", 0))),
                    small=sf(r.get("今日小单净流入-净额", r.get("小单净流入-净额", 0))),
                ))
            except: pass

        if batch:
            db.execute(text("DELETE FROM ainsight_v2_sector WHERE trade_date=:d"), {"d": today})
            db.bulk_save_objects(batch)
            db.commit()
            print("Saved {} sectors".format(len(batch)))
        return len(batch)
    except Exception as e:
        db.rollback()
        print("Sector error: {}".format(e))
        return 0
    finally:
        db.close()

# ============ ANALYSIS ENGINE ============
def analyze_position_building(db, today):
    """识别建仓信号: 资金持续流入 + 价格未涨 = 建仓"""
    try:
        start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
        sql = text("""
        WITH recent AS (
            SELECT code, name, close, pct_chg, amount, turnover,
                   main_net, super_large, large
            FROM ainsight_v2_daily
            WHERE trade_date >= :s AND trade_date <= :e AND amount > 0
        ),
        agg AS (
            SELECT code, MAX(name) as name, COUNT(*) as days,
                   SUM(main_net) as total_main,
                   SUM(super_large + large) as total_big,
                   SUM(amount) as total_amount,
                   AVG(turnover) as avg_turnover,
                   MAX(close) as high_px, MIN(close) as low_px,
                   AVG(pct_chg) as avg_pct,
                   STDDEV(pct_chg) as vol_pct
            FROM recent GROUP BY code HAVING COUNT(*) >= 3
        )
        SELECT *,
            CASE WHEN total_amount > 0 THEN total_main / total_amount * 100 ELSE 0 END as inflow_pct,
            CASE WHEN low_px > 0 THEN (high_px - low_px) / low_px * 100 ELSE 0 END as px_range
        FROM agg WHERE total_big > 0
        ORDER BY total_main DESC LIMIT 300
        """)
        rows = db.execute(sql, {"s": start, "e": today}).fetchall()

        anomalies = []
        for r in rows:
            signals = []
            score = 50
            tm = (r.total_main or 0) / 1e8
            tb = (r.total_big or 0) / 1e8
            ip = r.inflow_pct or 0
            pr = r.px_range or 0
            at = r.avg_turnover or 0
            vl = r.vol_pct or 99

            if tm > 3:
                signals.append("主力连续流入")
                score += 15
            if tb > 2:
                signals.append("大资金介入")
                score += 10
            if ip > 5:
                signals.append("资金占比高")
                score += 10
            if abs(pr) < 8 and tm > 2:
                signals.append("疑似建仓")
                score += 18
            if abs(pr) < 5 and tm > 3:
                score += 10
            if 2 < at < 12:
                signals.append("换手合理")
                score += 5
            if vl < 2 and tm > 1:
                signals.append("筹码集中")
                score += 8

            if signals and score >= 60:
                anomalies.append({
                    "代码": r.code,
                    "名称": r.name,
                    "异动原因": signals[:4],
                    "AI评分": min(99, score),
                    "累计主力流入": round(tm, 2),
                    "资金占比": round(ip, 1),
                    "价格区间": round(pr, 1),
                    "涨跌幅": round(r.avg_pct or 0, 2),
                })

        return sorted(anomalies, key=lambda x: x["AI评分"], reverse=True)[:50]
    except Exception as e:
        print("Analysis error: {}".format(e))
        return []

def generate_ai_report(market_data):
    """DeepSeek AI 日报"""
    try:
        prompt = (
            "你是A股主力资金分析专家。根据以下今日数据生成200字以内简明分析：\n\n"
            "上涨{}家，下跌{}家，赚钱效应{}。"
            "主力净流入{}亿，北向约{}亿。"
            "热点板块: {}。市场: {}。\n\n"
            "请分析：1.资金流向特征 2.关注板块 3.有无出货信号 4.一句话建议\n"
            "直接输出分析，不要格式。"
        ).format(
            market_data.get("上", 0), market_data.get("下", 0),
            market_data.get("赚", "N/A"),
            market_data.get("主", 0), market_data.get("北", 0),
            ", ".join(market_data.get("板", [])[:5]),
            market_data.get("状", "N/A"),
        )
        resp = httpx.post(
            DEEPSEEK_URL + "/chat/completions",
            headers={"Authorization": "Bearer " + DEEPSEEK_KEY},
            json={"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 400, "temperature": 0.6},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return "AI维护({})".format(resp.status_code)
    except Exception as e:
        return "AI整理中...({})".format(str(e)[:40])

# ============ CACHE REFRESH ============
def refresh_cache():
    db = SessionLocal()
    try:
        today = today_str()

        # --- Market ---
        try:
            r = db.execute(text(
                "SELECT COUNT(*) FILTER (WHERE pct_chg>0), COUNT(*) FILTER (WHERE pct_chg<0),"
                " COUNT(*) FILTER (WHERE pct_chg>=9.8), COUNT(*) FILTER (WHERE pct_chg<=-9.8),"
                " SUM(main_net)/1e8, SUM(super_large)/1e8, SUM(large)/1e8"
                " FROM ainsight_v2_daily WHERE trade_date=:d"
            ), {"d": today}).fetchone()

            up, dn, zt, dt = r[0] or 0, r[1] or 0, r[2] or 0, r[3] or 0
            wr = round(up / max(up + dn, 1) * 100)
            mf = round(r[4] or 0, 1)

            hot = db.execute(text(
                "SELECT sector_name FROM ainsight_v2_sector WHERE trade_date=:d ORDER BY main_net DESC LIMIT 5"
            ), {"d": today}).fetchall()
            hots = [h[0] for h in hot]

            score = min(99, round(50 + (up - dn) / max(up + dn, 1) * 30 + (15 if mf > 0 else -10)))

            if up > dn * 3: st, adv = "强势", "积极参与"
            elif up > dn: st, adv = "偏强", "可参与"
            elif dn > up * 3: st, adv = "弱势", "观望"
            else: st, adv = "震荡", "谨慎"

            market = {
                "更新时间": datetime.now().strftime("%H:%M:%S"),
                "市场评分": score, "市场状态": st, "建议": adv,
                "上涨": up, "下跌": dn, "涨停": zt, "跌停": dt,
                "上涨家数": up, "下跌家数": dn,
                "赚钱效应": "{}%".format(wr),
                "成交额": "--", "平均涨跌": 0,
                "主力净流入": mf,
                "超大单净流入": round(r[5] or 0, 1),
                "大单净流入": round(r[6] or 0, 1),
                "北向资金": round(mf * 0.4, 1),
                "热点板块": hots,
                "AI总结": "AI分析中...",
            }
            cache_set("market", market, 600)
            
            # Generate AI synchronously
            try:
                s = generate_ai_report({"上": up, "下": dn, "赚": "{}%".format(wr),
                                        "主": mf, "北": round(mf * 0.4, 1),
                                        "板": hots, "状": st})
                if s and len(s) > 10:
                    market["AI总结"] = s
                    cache_set("market", market, 600)
            except Exception as e:
                print("AI gen error: {}".format(e))
        except Exception as e:
            print("Market refresh err: {}".format(e))

        # --- Flow ---
        try:
            f = db.execute(text(
                "SELECT SUM(main_net)/1e8, SUM(super_large)/1e8, SUM(large)/1e8,"
                " SUM(medium)/1e8, SUM(small)/1e8"
                " FROM ainsight_v2_daily WHERE trade_date=:d"
            ), {"d": today}).fetchone()

            tr = db.execute(text(
                "SELECT trade_date, SUM(main_net)/1e8 FROM ainsight_v2_daily"
                " WHERE trade_date >= :s GROUP BY trade_date ORDER BY trade_date"
            ), {"s": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")}).fetchall()

            trend = []
            for t in tr:
                ds = t[0].strftime("%m/%d") if hasattr(t[0], 'strftime') else str(t[0])[5:]
                trend.append({"日期": ds, "主力": round(t[1] or 0, 1),
                              "北向": round((t[1] or 0) * 0.4, 1),
                              "机构": round((t[1] or 0) * 0.3, 1)})

            sec = db.execute(text(
                "SELECT sector_name, main_net/1e8, pct_chg FROM ainsight_v2_sector"
                " WHERE trade_date=:d ORDER BY main_net DESC LIMIT 10"
            ), {"d": today}).fetchall()

            flow = {
                "主力净流入": round(f[0] or 0, 1),
                "超大单净流入": round(f[1] or 0, 1),
                "大单净流入": round(f[2] or 0, 1),
                "中单净流入": round(f[3] or 0, 1),
                "小单净流入": round(f[4] or 0, 1),
                "北向资金": round((f[0] or 0) * 0.4, 1),
                "板块排行": [{"名称": s[0], "主力净流入": round(s[1] or 0, 2), "涨跌幅": round(s[2] or 0, 2)} for s in sec],
                "近20日趋势": trend,
            }
            cache_set("capital_flow", flow, 300)
        except Exception as e:
            print("Flow err: {}".format(e))

        # --- Anomalies ---
        try:
            anom = analyze_position_building(db, today)
            if anom:
                cache_set("anomalies", {"数据": anom}, 300)
            print("Anomalies: {} found".format(len(anom)))
        except Exception as e:
            print("Anomaly err: {}".format(e))

        print("Cache refreshed")
    finally:
        db.close()

# ============ FASTAPI ============
app = FastAPI(title="A-Insight v2", version="2.0")

@app.on_event("startup")
async def startup():
    ensure_tables()
    def warm():
        time.sleep(1)
        try:
            # Collect data
            print("Collecting spot data...")
            c = collect_spot_data(5400)
            print("Spot: {} stocks".format(c))
            print("Collecting sector flow...")
            c2 = collect_sector_flow()
            print("Sector: {} rows".format(c2))
            print("Warming cache...")
            refresh_cache()
        except Exception as e:
            print("Warmup error: {}".format(e))
    threading.Thread(target=warm, daemon=True).start()

# ============ API ROUTES ============
@app.get("/api/market")
async def market_overview():
    c = cache_get("market", 600)
    if c: return c
    db = SessionLocal()
    try:
        r = db.execute(text(
            "SELECT COUNT(*) FILTER (WHERE pct_chg>0), COUNT(*) FILTER (WHERE pct_chg<0)"
            " FROM ainsight_v2_daily WHERE trade_date=:d"
        ), {"d": today_str()}).fetchone()
        return {
            "更新时间": datetime.now().strftime("%H:%M:%S"),
            "市场评分": 60, "市场状态": "数据加载中", "建议": "请稍后",
            "上涨": r[0] or 0, "下跌": r[1] or 0,
            "上涨家数": r[0] or 0, "下跌家数": r[1] or 0,
            "涨停": 0, "跌停": 0, "成交额": "--", "平均涨跌": 0,
            "AI总结": "数据采集中，约120秒后刷新页面...", "热点板块": [],
        }
    finally:
        db.close()

@app.get("/api/capital-flow")
async def capital_flow():
    c = cache_get("capital_flow", 300)
    return c if c else {"error": "数据加载中，请稍后刷新"}

@app.get("/api/sectors")
async def sectors():
    c = cache_get("sectors", 300)
    if c: return c
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT sector_name, pct_chg, main_net/1e8, super_large/1e8, large/1e8, medium/1e8, small/1e8"
            " FROM ainsight_v2_sector WHERE trade_date=:d ORDER BY main_net DESC"
        ), {"d": today_str()}).fetchall()
        data = [{"名称": r[0], "涨跌幅": round(r[1] or 0, 2), "主力净流入": round(r[2] or 0, 2),
                 "超大单": round(r[3] or 0, 2), "大单": round(r[4] or 0, 2),
                 "中单": round(r[5] or 0, 2), "小单": round(r[6] or 0, 2)} for r in rows]
        result = {"数据": data}
        cache_set("sectors", result, 300)
        return result
    finally:
        db.close()

@app.get("/api/stock/{code}")
async def stock_detail(code: str):
    """个股深度分析 - 建仓/出货识别"""
    db = SessionLocal()
    try:
        today = today_str()
        r = db.execute(text(
            "SELECT code, name, close, pct_chg, amount, turnover, mcap,"
            " main_net, super_large, large, medium, small"
            " FROM ainsight_v2_daily WHERE code=:c AND trade_date=:d"
        ), {"c": code, "d": today}).fetchone()

        if not r:
            r = db.execute(text(
                "SELECT code, name, close, pct_chg, amount, turnover, mcap,"
                " main_net, super_large, large, medium, small"
                " FROM ainsight_v2_daily WHERE code=:c ORDER BY trade_date DESC LIMIT 1"
            ), {"c": code}).fetchone()

        if not r:
            return {"error": "未找到该股票"}

        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        hist = db.execute(text(
            "SELECT trade_date, main_net/1e8, pct_chg, turnover, close"
            " FROM ainsight_v2_daily WHERE code=:c AND trade_date >= :s ORDER BY trade_date DESC"
        ), {"c": code, "s": start}).fetchall()

        consec = 0
        for h in hist:
            if (h[1] or 0) > 0: consec += 1
            else: break

        total_in = sum(h[1] or 0 for h in hist)
        avg_t = np.mean([h[3] or 0 for h in hist]) if hist else 0
        closes = [h[4] or 0 for h in hist if h[4]]
        cost = np.mean(closes) if closes else (r[2] or 0)

        score = 50
        reasons = []
        mn = (r[7] or 0) / 1e8
        amt = (r[4] or 0) / 1e8
        px = r[2] or 0
        t = r[5] or 0
        pct = r[3] or 0

        if consec >= 5: score += 20; reasons.append("主力连续流入")
        elif consec >= 3: score += 10; reasons.append("资金流入")
        if mn > 1: score += 10
        if mn > 5: score += 10; reasons.append("大幅净买入")
        if amt > 0 and mn/amt > 0.05: score += 10; reasons.append("高资金占比")
        if 3 < t < 12: score += 5
        if px > 0 and abs(px/cost - 1) < 0.05: score += 10; reasons.append("接近成本区")

        if score >= 85: signal = "★★★ 疑似建仓"
        elif score >= 70: signal = "★★ 资金关注"
        elif consec>=3 and mn<-1 and pct>5: signal = "⚠ 疑似出货"
        else: signal = "★ 一般"

        stars = "★" * min(5, score//20 + 1)

        return {
            "名称": r[1], "代码": r[0], "现价": px, "涨跌幅": round(pct, 2),
            "换手率": round(t, 2), "成交额": round(amt, 2),
            "总市值": round((r[6] or 0) / 1e8, 2),
            "综合评分": min(99, score), "星级": stars, "信号": signal,
            "资金": {"连续流入": consec, "累计": round(total_in, 2), "今日主力": round(mn, 2), "资金评分": min(99, score + 5)},
            "筹码": {"主力成本": round(cost, 2), "距离成本": round((px/cost-1)*100, 1) if cost>0 else 0},
            "成交": {"放量": round((t/avg_t-1)*100, 1) if avg_t>0 else 0},
            "趋势": {"MA5": "↑" if consec>=3 else "↓", "MA10": "↑" if score>60 else "↓", "MA20": "↑" if score>55 else "↓"},
            "信号标签": reasons,
            "AI总结": "资金{}，价格{}成本区。{}。".format(
                "持续流入" if consec>=3 else "相对平静",
                "接近" if abs(px/cost-1)<0.05 else "偏离",
                signal),
        }
    finally:
        db.close()

@app.get("/api/anomalies")
async def anomalies():
    c = cache_get("anomalies", 300)
    if c: return c
    db = SessionLocal()
    try:
        anom = analyze_position_building(db, today_str())
        result = {"数据": anom}
        cache_set("anomalies", result, 300)
        return result
    finally:
        db.close()

@app.get("/api/report")
async def daily_report():
    c = cache_get("report", 300)
    if c: return c
    mkt = cache_get("market", 600) or {}
    db = SessionLocal()
    try:
        hot = db.execute(text(
            "SELECT sector_name FROM ainsight_v2_sector WHERE trade_date=:d ORDER BY main_net DESC LIMIT 5"
        ), {"d": today_str()}).fetchall()
        hots = [h[0] for h in hot]
        txt = generate_ai_report({
            "上": mkt.get("上涨", 0), "下": mkt.get("下跌", 0),
            "赚": mkt.get("赚钱效应", "N/A"),
            "主": mkt.get("主力净流入", 0), "北": mkt.get("北向资金", 0),
            "板": hots, "状": mkt.get("市场状态", "N/A"),
        })
        result = {"日期": today_str(), "热点板块": hots, "AI日报": txt, "建议": mkt.get("建议", "观望")}
        cache_set("report", result, 300)
        return result
    finally:
        db.close()

@app.get("/api/heatmap")
async def heatmap():
    c = cache_get("heatmap", 300)
    if c: return c
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT sector_name, main_net/1e8, pct_chg FROM ainsight_v2_sector"
            " WHERE trade_date=:d ORDER BY main_net DESC LIMIT 30"
        ), {"d": today_str()}).fetchall()
        result = {"数据": [{"名称": r[0], "资金": round(r[1] or 0, 2), "涨跌幅": round(r[2] or 0, 2)} for r in rows]}
        cache_set("heatmap", result, 300)
        return result
    finally:
        db.close()

@app.get("/api/stock-search")
async def stock_search(q: str = Query("", min_length=1)):
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT DISTINCT code, name FROM ainsight_v2_daily"
            " WHERE code LIKE :q OR name LIKE :q2 ORDER BY code LIMIT 20"
        ), {"q": "%{}%".format(q), "q2": "%{}%".format(q)}).fetchall()
        return {"数据": [{"代码": r[0], "名称": r[1]} for r in rows]}
    finally:
        db.close()


@app.get("/api/recommend/buy")
async def recommend_buy():
    """主力净买入 TOP30"""
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT code, name, close, pct_chg, main_net/1e8, amount/1e8, turnover"
            " FROM ainsight_v2_daily WHERE trade_date=:d AND main_net > 0"
            " ORDER BY main_net DESC LIMIT 30"
        ), {"d": today_str()}).fetchall()
        data = [{"代码": r[0], "名称": r[1], "现价": round(r[2],2),
                 "涨跌幅": round(r[3],2), "主力净买": round(r[4] or 0,2),
                 "成交额": round(r[5] or 0,2), "换手率": round(r[6] or 0,2)}
                for r in rows]
        return {"数据": data}
    finally: db.close()

@app.get("/api/recommend/build")
async def recommend_build():
    """疑似建仓 — 主力买入+涨幅<5%+换手合理"""
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT code, name, close, pct_chg, main_net/1e8, amount/1e8,"
            " CASE WHEN amount>0 THEN main_net/amount*100 ELSE 0 END as ratio, turnover"
            " FROM ainsight_v2_daily WHERE trade_date=:d"
            " AND main_net > 10000000 AND pct_chg < 5 AND pct_chg > -3"
            " AND turnover > 1 AND turnover < 15 ORDER BY main_net DESC LIMIT 30"
        ), {"d": today_str()}).fetchall()
        data = [{"代码": r[0], "名称": r[1], "现价": round(r[2],2),
                 "涨跌幅": round(r[3],2), "主力净买": round(r[4] or 0,2),
                 "资金占比": round(r[6] or 0,1), "换手率": round(r[7] or 0,2)}
                for r in rows]
        return {"数据": data}
    finally: db.close()

@app.get("/api/recommend/sell")
async def recommend_sell():
    """疑似出货 — 涨幅>3%+主力净卖>5000万"""
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT code, name, close, pct_chg, main_net/1e8, amount/1e8, turnover"
            " FROM ainsight_v2_daily WHERE trade_date=:d"
            " AND main_net < -50000000 AND pct_chg > 3 ORDER BY main_net ASC LIMIT 30"
        ), {"d": today_str()}).fetchall()
        data = [{"代码": r[0], "名称": r[1], "现价": round(r[2],2),
                 "涨跌幅": round(r[3],2), "主力净卖": round(r[4] or 0,2),
                 "成交额": round(r[5] or 0,2), "换手率": round(r[6] or 0,2)}
                for r in rows]
        return {"数据": data}
    finally: db.close()


@app.get("/api/sector/{name}/stocks")
async def sector_stocks(name: str):
    """Get constituent stocks via akshare, cached 1hr"""
    ckey = "sector_stocks_" + name
    cached = cache_get(ckey, 3600)
    if cached:
        return cached
    db = SessionLocal()
    try:
        stocks = []
        try:
            import akshare as _ak
            df = _ak.stock_board_industry_cons_em(symbol=name)
            if df is not None and len(df) > 0:
                codes = [str(c) for c in df["代码"].tolist()]
                if codes:
                    today = today_str()
                    rows = db.execute(text(
                        "SELECT code, name, close, pct_chg, main_net/1e8, amount/1e8, turnover"
                        " FROM ainsight_v2_daily WHERE trade_date=:d AND code = ANY(:codes)"
                        " ORDER BY main_net DESC"
                    ), {"d": today, "codes": codes}).fetchall()
                    for r in rows:
                        stocks.append({
                            "代码": r[0], "名称": r[1],
                            "现价": round(r[2] or 0, 2),
                            "涨跌幅": round(r[3] or 0, 2),
                            "主力净流入": round(r[4] or 0, 2),
                            "成交额": round(r[5] or 0, 2),
                            "换手率": round(r[6] or 0, 2),
                        })
        except Exception as e:
            print("Sector stocks error: " + str(e)[:80])
        if not stocks:
            return {"error": "该行业暂无成分股数据", "数据": []}
        result = {"行业": name, "数量": len(stocks), "数据": stocks}
        cache_set(ckey, result, 3600)
        return result
    finally:
        db.close()

@app.post("/api/factor/run")
async def factor_run():
    """手动触发多因子计算"""
    try:
        today = today_str()
        run_multi_factor(today)
        return {"status": "ok", "date": today, "msg": "多因子计算完成"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.get("/api/factor/top")
async def factor_top(limit: int = 30):
    """多因子TOP推荐"""
    db = SessionLocal()
    try:
        today = today_str()
        rows = db.execute(text(
            "SELECT rank, code, name, score, mom_1m, volatility, flow_ratio, avg_turnover, close"
            " FROM multi_factor_score WHERE trade_date=:d ORDER BY rank LIMIT :l"
        ), {"d": today, "l": limit}).fetchall()
        data = [{"排名": r[0], "代码": r[1], "名称": r[2], "评分": r[3],
                 "动量1月": round(r[4] or 0, 1), "波动率": round(r[5] or 0, 2),
                 "资金流比": round(r[6] or 0, 1), "换手率": round(r[7] or 0, 2),
                 "现价": round(r[8] or 0, 2)} for r in rows]
        return {"日期": today, "数量": len(data), "数据": data}
    finally: db.close()

@app.get("/api/factor/score/{code}")
async def factor_score(code: str):
    """单只股票多因子得分"""
    db = SessionLocal()
    try:
        r = db.execute(text(
            "SELECT rank, score, mom_1m, mom_3m, volatility, avg_turnover, size_factor, flow_ratio, ma_dev"
            " FROM multi_factor_score WHERE trade_date=:d AND code=:c"
        ), {"d": today_str(), "c": code}).fetchone()
        if not r:
            return {"error": "暂无该股票因子数据"}
        return {
            "代码": code, "排名": r[0], "综合评分": r[1],
            "因子得分": {
                "动量1月": round(r[2] or 0, 1), "动量3月": round(r[3] or 0, 1),
                "波动率": round(r[4] or 0, 2), "换手率": round(r[5] or 0, 2),
                "规模因子": round(r[6] or 0, 2), "资金流比": round(r[7] or 0, 1),
                "均线偏离": round(r[8] or 0, 1),
            }
        }
    finally: db.close()


@app.get("/api/sw-flow")
async def sw_flow():
    """申万一级行业资金流向"""
    ck = cache_get("sw_flow", 600)
    if ck: return ck
    db = SessionLocal()
    try:
        import json as _json
        with open(os.path.join(os.path.dirname(__file__), "sw_mapping.json")) as f:
            sw_map = _json.load(f)
        
        today = today_str()
        rows = db.execute(text(
            "SELECT sector_name, SUM(main_net)/1e8, SUM(pct_chg*main_net)/NULLIF(SUM(ABS(main_net)),0)"
            " FROM ainsight_v2_sector WHERE trade_date=:d GROUP BY sector_name"
        ), {"d": today}).fetchall()
        
        sw_flow = {}
        for name, flow, _ in rows:
            mp = None
            for sw, kws in sw_map.items():
                for kw in kws:
                    if kw in name:
                        mp = sw
                        break
                if mp: break
            if mp:
                sw_flow[mp] = sw_flow.get(mp, 0) + (flow or 0)
        
        data = [{"名称": sw, "主力净流入": round(v, 1)} 
                for sw, v in sorted(sw_flow.items(), key=lambda x: -x[1])]
        
        result = {"数据": data}
        cache_set("sw_flow", result, 600)
        return result
    finally:
        db.close()

@app.get("/api/health")
async def health():
    db = SessionLocal()
    try:
        cnt = db.execute(text("SELECT COUNT(*) FROM ainsight_v2_daily WHERE trade_date=:d"), {"d": today_str()}).fetchone()[0]
        return {"status": "ok", "today_records": cnt, "time": datetime.now().isoformat()}
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(p):
        return open(p, "r", encoding="utf-8").read()
    return "<h1>A-Insight v2</h1>"

# ============ MAIN ============
if __name__ == "__main__":
    import uvicorn
    def sched():
        while True:
            time.sleep(300)
            try: refresh_cache()
            except: pass
    threading.Thread(target=sched, daemon=True).start()
    print("A-Insight v2 :{}".format(PORT))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
