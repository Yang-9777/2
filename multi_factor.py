"""
华泰多因子选股引擎 v1
因子: 动量 | 波动 | 换手 | 规模 | 资金流 | 均线偏离
每日9:00自动计算 → 综合打分 → TOP推荐
"""

import psycopg2
import numpy as np
from datetime import datetime, timedelta

import os
DB = f"host={os.environ.get("PG_HOST","127.0.0.1")} port={os.environ.get("PG_PORT","5433")} user={os.environ.get("PG_USER","astock")} password={os.environ.get("PG_PASS","")} dbname={os.environ.get("PG_DB","astock")}"  # set via env: PG_HOST PG_PORT PG_USER PG_PASS PG_DB

def get_db():
    return psycopg2.connect(DB)

def run_multi_factor(trade_date=None):
    """执行多因子打分，返回推荐列表"""
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    conn = get_db()
    cur = conn.cursor()
    
    # 1. 获取最近20天和60天的数据
    d20 = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    d60 = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    
    print(f"[多因子] 计算日期: {trade_date}  回溯: {d20} ~ {trade_date}")
    
    # 用 ainsight_v2_daily 获取近期数据（会持续积累）
    cur.execute("""
        SELECT d.code, COALESCE(s.name, d.code) as name, 
               d.trade_date, d.close, d.change_pct, d.turnover,
               COALESCE(s.market_cap, 5000000000) as mcap,
               d.amount, 0 as main_net
        FROM daily_bars d
        LEFT JOIN stocks s ON d.code = s.code
        WHERE d.trade_date >= %s AND d.trade_date <= %s AND d.close > 0
        ORDER BY d.code, d.trade_date
    """, (d60, trade_date))
    
    rows = cur.fetchall()
    print(f"  获取 {len(rows)} 条日线数据")
    
    # 2. 按股票分组计算因子
    from collections import defaultdict
    stock_data = defaultdict(list)
    for r in rows:
        stock_data[r[0]].append({
            'date': r[2], 'close': r[3], 'pct_chg': r[4] or 0,
            'turnover': r[5] or 0, 'mcap': r[6] or 1, 'amount': r[7] or 0, 'main_net': r[8] or 0
        })
    
    print(f"  覆盖 {len(stock_data)} 只股票")
    
    # 3. 逐只计算因子
    factors = []
    for code, bars in stock_data.items():
        if len(bars) < 5:  # 至少5天数据
            continue
        
        # Get name from first bar
        name = bars[0].get('name', code)
        bars.sort(key=lambda x: x['date'])
        closes = np.array([b['close'] for b in bars])
        turnovers = np.array([b['turnover'] for b in bars])
        main_nets = np.array([b['main_net'] for b in bars])
        amounts = np.array([b['amount'] for b in bars])
        
        latest = bars[-1]
        name = ""
        for r in rows:
            if r[0] == code:
                name = r[1]
                break
        
        # === 因子计算 ===
        
        # 1. 动量_1M: 近20日收益率
        if len(closes) >= 20:
            mom_1m = (closes[-1] / closes[-20] - 1) * 100
        elif len(closes) >= 5:
            mom_1m = (closes[-1] / closes[0] - 1) * 100
        else:
            mom_1m = 0
        
        # 2. 动量_3M: 近60日收益率（如果有）
        if len(closes) >= 60:
            mom_3m = (closes[-1] / closes[-60] - 1) * 100
        elif len(closes) >= 20:
            mom_3m = (closes[-1] / closes[-20] - 1) * 100 * 3  # 近似
        else:
            mom_3m = mom_1m * 3
        
        # 3. 波动率: 近20日收益率标准差
        if len(closes) >= 20:
            returns = np.diff(closes[-21:]) / closes[-21:-1] * 100 if len(closes) >= 21 else np.diff(closes) / closes[:-1] * 100
            volatility = np.std(returns) if len(returns) > 0 else 0
        else:
            volatility = np.std(np.diff(closes) / closes[:-1] * 100) if len(closes) > 1 else 0
        
        # 4. 换手率: 近5日平均
        avg_turnover = np.mean(turnovers[-5:]) if len(turnovers) >= 5 else turnovers[-1]
        
        # 5. 规模因子: log(市值) — 负向（小盘溢价）
        size_factor = -np.log(latest['mcap'] / 1e8 + 1)  # 市值单位: 亿
        
        # 6. 资金流因子: 近5日主力净流入/成交额
        recent_amounts = amounts[-5:] if len(amounts) >= 5 else amounts
        recent_mains = main_nets[-5:] if len(main_nets) >= 5 else main_nets
        total_amount = np.sum(recent_amounts)
        if total_amount > 0:
            flow_ratio = np.sum(recent_mains) / total_amount * 100
        else:
            flow_ratio = 0
        
        # 7. 均线偏离: (close - MA20) / MA20
        if len(closes) >= 20:
            ma20 = np.mean(closes[-20:])
            ma_dev = (closes[-1] - ma20) / ma20 * 100 if ma20 > 0 else 0
        else:
            ma_dev = 0
        
        factors.append({
            'code': code, 'name': name,
            'close': latest['close'], 'mcap': latest['mcap'],
            'mom_1m': mom_1m, 'mom_3m': mom_3m,
            'volatility': volatility, 'avg_turnover': avg_turnover,
            'size_factor': size_factor, 'flow_ratio': flow_ratio,
            'ma_dev': ma_dev,
        })
    
    if len(factors) < 50:
        print("  ⚠ 股票太少，无法计算")
        return []
    
    # 4. 标准化 (Z-score, 截面去极值)
    print(f"  标准化 + 打分 ({len(factors)} 只)...")
    
    factor_names = ['mom_1m', 'mom_3m', 'volatility', 'avg_turnover', 'size_factor', 'flow_ratio', 'ma_dev']
    # 因子方向: 1=越大越好(正向), -1=越小越好(负向)
    directions = {'mom_1m': -1, 'mom_3m': -1, 'volatility': -1, 'avg_turnover': 1,
                  'size_factor': 1, 'flow_ratio': 1, 'ma_dev': -1}  # 回测校准: A股反转效应
    
    # 权�(华泰研报典型配置)
    weights = {'mom_1m': 0.23, 'mom_3m': 0.24, 'volatility': 0.14, 'avg_turnover': 0.06,
               'size_factor': 0.16, 'flow_ratio': 0.00, 'ma_dev': 0.17}
    
    # 去极值: 1%和99%分位数截断
    for fn in factor_names:
        vals = np.array([f[fn] for f in factors])
        p1, p99 = np.percentile(vals, [1, 99])
        vals_clipped = np.clip(vals, p1, p99)
        
        # Z-score
        mean, std = np.mean(vals_clipped), np.std(vals_clipped)
        if std > 0:
            z_scores = (vals_clipped - mean) / std
        else:
            z_scores = np.zeros_like(vals)
        
        for i, f in enumerate(factors):
            f[fn + '_z'] = z_scores[i] * directions[fn]
    
    # 5. 加权合成
    for f in factors:
        score = 0
        for fn in factor_names:
            score += f.get(fn + '_z', 0) * weights[fn]
        f['total_score'] = score
    
    # 6. 归一化到0-100
    scores = np.array([f['total_score'] for f in factors])
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        for f in factors:
            f['score_100'] = round((f['total_score'] - s_min) / (s_max - s_min) * 100)
    else:
        for f in factors:
            f['score_100'] = 50
    
    # 7. 排序，返回 TOP
    factors.sort(key=lambda x: x['score_100'], reverse=True)
    
    print(f"  评分完成! TOP5:")
    for f in factors[:5]:
        print(f"    {f['code']} {f['name']}  {f['score_100']}分  "
              f"mom={f['mom_1m']:.1f}%  vol={f['volatility']:.2f}  flow={f['flow_ratio']:.1f}%")
    
    # 8. 存入数据库
    cur.execute("""
        CREATE TABLE IF NOT EXISTS multi_factor_score (
            id SERIAL PRIMARY KEY,
            trade_date DATE NOT NULL,
            code VARCHAR(10) NOT NULL,
            name VARCHAR(50),
            score INTEGER,
            mom_1m FLOAT, mom_3m FLOAT, volatility FLOAT,
            avg_turnover FLOAT, size_factor FLOAT,
            flow_ratio FLOAT, ma_dev FLOAT,
            close FLOAT, mcap FLOAT,
            rank INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mf_date ON multi_factor_score(trade_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mf_code ON multi_factor_score(code)")
    
    # 删除今日旧数据
    cur.execute("DELETE FROM multi_factor_score WHERE trade_date = %s", (trade_date,))
    
    # 插入新数据
    for rank, f in enumerate(factors, 1):
        cur.execute("""
            INSERT INTO multi_factor_score (trade_date, code, name, score, 
                mom_1m, mom_3m, volatility, avg_turnover, size_factor, flow_ratio, ma_dev, close, mcap, rank)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (trade_date, f['code'], f['name'], int(f['score_100']),
              float(f['mom_1m']), float(f['mom_3m']), float(f['volatility']), float(f['avg_turnover']),
              float(f['size_factor']), float(f['flow_ratio']), float(f['ma_dev']),
              float(f['close']), float(f['mcap']), rank))
    
    conn.commit()
    conn.close()
    
    print(f"  已存入 multi_factor_score 表 ({len(factors)} 条)")
    
    # 返回 TOP50 供 API 使用
    return [
        {
            "排名": i+1,
            "代码": f['code'], "名称": f['name'],
            "评分": f['score_100'], "现价": round(f['close'], 2),
            "动量1月": round(f['mom_1m'], 1),
            "波动率": round(f['volatility'], 2),
            "资金流比": round(f['flow_ratio'], 1),
            "换手率": round(f['avg_turnover'], 2),
        }
        for i, f in enumerate(factors[:50])
    ]

if __name__ == "__main__":
    # 测试运行
    today = datetime.now().strftime("%Y-%m-%d")
    result = run_multi_factor(today)
    print(f"\n最终推荐 {len(result)} 只")
