"""
因子权重回测引擎
- IC分析: 每期因子值与下期收益的相关系数
- IC_IR: IC均值/IC标准差 → 因子稳定性
- 权重优化: IC_IR加权
- 分层回测: 按得分5分组，看各组表现
"""

import psycopg2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

import os
DB = f"host={os.environ.get("PG_HOST","127.0.0.1")} port={os.environ.get("PG_PORT","5433")} user={os.environ.get("PG_USER","astock")} password={os.environ.get("PG_PASS","")} dbname={os.environ.get("PG_DB","astock")}"  # set via env: PG_HOST PG_PORT PG_USER PG_PASS PG_DB

def get_conn():
    return psycopg2.connect(DB)

def backtest_5years():
    conn = get_conn()
    cur = conn.cursor()
    
    # 获取全部历史数据
    cur.execute("""
        SELECT code, trade_date, close, change_pct, turnover, volume, amount
        FROM daily_bars 
        WHERE close > 0
        ORDER BY code, trade_date
    """)
    rows = cur.fetchall()
    print(f"历史数据: {len(rows)} 条")
    
    # 按股票组织
    stock_data = defaultdict(list)
    for r in rows:
        stock_data[r[0]].append({
            'date': r[1], 'close': r[2], 'chg': r[3] or 0,
            'turnover': r[4] or 0, 'volume': r[5] or 0, 'amount': r[6] or 0
        })
    
    # 获取所有交易日期
    all_dates = sorted(set(r[1] for r in rows))
    print(f"交易日: {len(all_dates)} ({all_dates[0]} ~ {all_dates[-1]})")
    
    # 逐期计算因子值和下期收益
    all_ics = defaultdict(list)  # factor -> [IC values]
    portfolio_returns = []       # (date, group_returns)
    
    factor_names = ['mom_1m', 'mom_3m', 'volatility', 'avg_turnover', 'size', 'ma_dev']
    # 至少需要60天前导数据 + 20天未来收益
    for i in range(60, len(all_dates) - 21):
        t_date = all_dates[i]
        fwd_date = all_dates[i + 20]  # ~1个月后
        
        factor_scores = []  # [{code, factors..., fwd_return}]
        
        for code, bars in stock_data.items():
            # 找 t_date 和 fwd_date 的数据
            bars_t = [b for b in bars if b['date'] <= t_date]
            bars_fwd = [b for b in bars if b['date'] == fwd_date]
            
            if len(bars_t) < 60 or not bars_fwd:
                continue
            
            closes = np.array([b['close'] for b in bars_t[-90:]])
            turnovers = np.array([b['turnover'] for b in bars_t[-90:]])
            
            # === 因子计算 ===
            current_close = closes[-1]
            
            # 动量1M
            mom_1m = (closes[-1] / closes[-20] - 1) * 100 if len(closes) >= 20 else 0
            # 动量3M
            mom_3m = (closes[-1] / closes[-60] - 1) * 100 if len(closes) >= 60 else 0
            # 波动率(20d)
            rets = np.diff(closes[-21:]) / closes[-21:-1] * 100 if len(closes) >= 21 else np.zeros(1)
            volatility = np.std(rets)
            # 换手率(5d avg)
            avg_turnover = np.mean(turnovers[-5:]) if len(turnovers) >= 5 else 0
            # 规模因子
            size = -np.log(current_close * 1e8 + 1)
            # 均线偏离
            ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            ma_dev = (closes[-1] - ma20) / ma20 * 100
            
            # 未来收益
            fwd_return = (bars_fwd[0]['close'] / closes[-1] - 1) * 100
            
            factor_scores.append({
                'code': code,
                'mom_1m': mom_1m, 'mom_3m': mom_3m,
                'volatility': volatility, 'avg_turnover': avg_turnover,
                'size': size, 'ma_dev': ma_dev,
                'fwd_return': fwd_return
            })
        
        if len(factor_scores) < 100:
            continue
        
        # 计算IC
        pass
        factor_values = {fn: np.array([f[fn] for f in factor_scores]) for fn in factor_names}
        fwd_returns = np.array([f['fwd_return'] for f in factor_scores])
        
        for fn in factor_names:
            # Rank IC
            vals = factor_values[fn]
            if np.std(vals) > 0:
                ic = np.corrcoef(vals, fwd_returns)[0, 1]
                if not np.isnan(ic):
                    all_ics[fn].append(ic)
        
        # 每20期打印一次进度
        if i % 100 == 0:
            print(f"  回测进度: {t_date} ({i}/{len(all_dates)}) 样本={len(factor_scores)}")
    
    print(f"\n{'='*60}")
    print(f"  IC 分析结果 ({len(all_dates)}个交易日)")
    print(f"{'='*60}")
    print(f"  {'因子':<15} {'IC均值':>8} {'IC标准差':>8} {'IC_IR':>8} {'胜率':>8} {'建议权重':>8}")
    print(f"  {'-'*55}")
    
    ic_results = {}
    for fn in factor_names:
        ics = all_ics.get(fn, [])
        if len(ics) < 10:
            continue
        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        win_rate = sum(1 for ic in ics if ic > 0) / len(ics) * 100
        
        ic_results[fn] = {
            'ic_mean': ic_mean, 'ic_std': ic_std,
            'ic_ir': ic_ir, 'win_rate': win_rate, 'count': len(ics)
        }
        
        print(f"  {fn:<15} {ic_mean:>8.4f} {ic_std:>8.4f} {ic_ir:>8.3f} {win_rate:>7.1f}%", end="")
    
    # 计算优化权重
    ic_irs = {fn: abs(d['ic_ir']) for fn, d in ic_results.items()}
    total_ir = sum(ic_irs.values())
    
    print(f"\n\n{'='*60}")
    print(f"  优化权重 (IC_IR加权)")
    print(f"{'='*60}")
    
    opt_weights = {}
    for fn, ir in ic_irs.items():
        w = ir / total_ir * 100 if total_ir > 0 else 0
        opt_weights[fn] = round(w, 1)
        d = ic_results[fn]
        sign = "+" if d['ic_mean'] > 0 else "-"
        print(f"  {fn:<15} 权重={w:>5.1f}%  IC={d['ic_mean']:+.4f}  IR={ir:.3f}  Win={d['win_rate']:.1f}%")
    
        print(f"\n  {'='*55}")
    print(f"  因子方向: + = 正向   - = 反向(反转)")
    for fn, d in ic_results.items():
        direction = "正向" if d['ic_mean'] > 0 else "反转"
        print(f"    {fn}: {direction} (IC={d['ic_mean']:+.4f}, 胜率={d['win_rate']:.1f}%)")
    
    print(f"\n  {'='*55}")