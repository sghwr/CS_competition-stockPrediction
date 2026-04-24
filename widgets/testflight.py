import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ====================== 读取你的特殊格式数据 ======================
file_path = r"E:\USELESS\数据分析学习\数分竞赛学习\CS-competition\ensemble\output\eval\ensemble_results.csv"

# 读取原始文本
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 解析数据行
data_lines = [line.strip() for line in lines[1:] if line.strip() and not line.startswith("=====")]
rows = []
for line in data_lines:
    items = line.split(',')
    date = items[0]
    ret = float(items[-1])
    rows.append({"date": date, "return": ret})

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

returns = df['return'].values
dates = df['date'].values

# ====================== 计算真实净值（5天非重叠） ======================
def compute_real_nav(returns, hold_days=5):
    nav = [1.0]
    current = 0
    cnt = 0
    for r in returns:
        current += r
        cnt += 1
        if cnt == hold_days:
            nav.append(nav[-1] * (1 + current / hold_days))
            current = 0
            cnt = 0
    if cnt > 0:
        nav.append(nav[-1] * (1 + current / cnt))
    return np.array(nav)

real_nav = compute_real_nav(returns, 5)
real_dates = pd.date_range(start=dates[0], periods=len(real_nav), freq='5D')

# 回撤
def max_dd(nav):
    peak = np.maximum.accumulate(nav)
    return (nav - peak) / peak
dd = max_dd(real_nav)

# ====================== 统计指标 ======================
mean_ret = np.mean(returns)
std_ret = np.std(returns)
sharpe = mean_ret / std_ret * np.sqrt(252)
win_rate = np.mean(returns > 0)
profit = returns[returns>0].mean() if (returns>0).any() else 0
loss = -returns[returns<0].mean() if (returns<0).any() else 1
pl_ratio = profit / loss

print("="*50)
print("        集成模型回测最终统计（真实无虚高）")
print("="*50)
print(f"总交易日：{len(returns)}")
print(f"平均日收益：{mean_ret*100:.2f}%")
print(f"年化夏普：{sharpe:.4f}")
print(f"胜率：{win_rate*100:.2f}%")
print(f"盈亏比：{pl_ratio:.4f}")
print(f"真实累计收益：{(real_nav[-1]-1)*100:.2f}%")
print(f"最大回撤：{dd.min()*100:.2f}%")
print("="*50)

# ====================== ✅ 修复绘图：4张图全部正常输出 ======================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# 1. 真实净值曲线
ax1.plot(real_dates, real_nav, linewidth=3, color='#1f77b4')
ax1.set_title("真实净值曲线（5天非重叠 · 无虚高）", fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# 2. 每日加权收益率（已修复，100%显示）
colors = ['#d62728' if x < 0 else '#2ca02c' for x in returns]
ax2.bar(dates, returns, color=colors, alpha=0.7, width=1.0)
ax2.axhline(0, color='black', linewidth=1.2)
ax2.set_title("每日加权收益率", fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45) # 日期旋转防止重叠

# 3. 回撤曲线
ax3.fill_between(real_dates, dd, 0, color='#ff7f0e', alpha=0.6)
ax3.set_title("回撤曲线", fontsize=14, fontweight='bold')

# 4. 收益率分布
ax4.hist(returns, bins=30, color='#9467bd', alpha=0.7, edgecolor='black')
ax4.axvline(mean_ret, color='red', linestyle='--', label=f'均值 {mean_ret*100:.2f}%')
ax4.legend()
ax4.set_title("收益率分布", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("ensemble_final_report.png", dpi=300, bbox_inches='tight')
plt.show()