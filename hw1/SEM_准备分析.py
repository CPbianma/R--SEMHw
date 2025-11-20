#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构方程模型 (SEM) 数据准备和初步分析 - Python版本
作者: Claude
日期: 2025-11-17
用途: 手机依赖与心理健康的SEM分析

需要安装的包:
pip install pandas numpy scipy scikit-learn semopy matplotlib seaborn openpyxl factor_analyzer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 数据读取和准备
# ============================================================================

print("=" * 80)
print("结构方程模型 (SEM) 数据准备和分析")
print("=" * 80)

# 读取数据
data_raw = pd.read_excel('问卷数据-已编码.xlsx')
print(f"\n数据读取成功！样本量: {data_raw.shape[0]}, 变量数: {data_raw.shape[1]}")

# 创建工作数据集
data = data_raw.copy()

# ============================================================================
# 2. 变量重命名
# ============================================================================

# 定义新的列名
new_columns = ['ID', 'Gender', 'Age', 'OnlyChild', 'ParentEdu', 'InLove',
               'PhoneYears', 'PhoneCost',
               'PF1', 'PF2', 'PF3', 'PF4', 'PF5',
               'PM1', 'PM2', 'PM3', 'PM4', 'PM5',
               'DailyUse',
               'WS1', 'WS2', 'WS3', 'WS4', 'WS5', 'WS6', 'WS_Total',
               'CR1', 'CR2', 'CR3', 'CR_Total',
               'PMI1', 'PMI2', 'PMI3', 'PMI4', 'PMI_Total',
               'MPD_Total',
               'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S_Total',
               'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A_Total',
               'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D_Total']

data.columns = new_columns

# ============================================================================
# 3. 描述性统计
# ============================================================================

print("\n" + "=" * 80)
print("描述性统计")
print("=" * 80)

# 样本特征
print(f"\n样本量: {len(data)}")
print("\n性别分布:")
print(data['Gender'].value_counts())
print(f"\n年龄描述: M={data['Age'].mean():.2f}, SD={data['Age'].std():.2f}")

# 主要变量
main_vars = ['MPD_Total', 'WS_Total', 'CR_Total', 'PMI_Total',
             'S_Total', 'A_Total', 'D_Total']

print("\n主要变量描述统计:")
desc_stats = data[main_vars].describe().round(2)
print(desc_stats)

# 保存描述性统计
desc_stats.to_csv('描述性统计.csv', encoding='utf-8-sig')
print("\n描述性统计已保存至: 描述性统计.csv")

# ============================================================================
# 4. 信度分析 (Cronbach's Alpha)
# ============================================================================

def cronbach_alpha(df):
    """计算Cronbach's Alpha"""
    df_corr = df.corr()
    n = df.shape[1]
    avg_corr = df_corr.values[np.triu_indices_from(df_corr.values, 1)].mean()
    alpha = (n * avg_corr) / (1 + (n - 1) * avg_corr)
    return alpha

print("\n" + "=" * 80)
print("信度分析 (Cronbach's Alpha)")
print("=" * 80)

# 戒断症状维度
ws_items = ['WS1', 'WS2', 'WS3', 'WS4', 'WS5', 'WS6']
alpha_ws = cronbach_alpha(data[ws_items])
print(f"\n戒断症状维度 α = {alpha_ws:.3f}")

# 渴求性维度
cr_items = ['CR1', 'CR2', 'CR3']
alpha_cr = cronbach_alpha(data[cr_items])
print(f"渴求性维度 α = {alpha_cr:.3f}")

# 身心影响维度
pmi_items = ['PMI1', 'PMI2', 'PMI3', 'PMI4']
alpha_pmi = cronbach_alpha(data[pmi_items])
print(f"身心影响维度 α = {alpha_pmi:.3f}")

# 手机依赖总量表
mpd_items = ws_items + cr_items + pmi_items
alpha_mpd = cronbach_alpha(data[mpd_items])
print(f"手机依赖总量表 α = {alpha_mpd:.3f}")

# DASS-21各维度
s_items = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
alpha_s = cronbach_alpha(data[s_items])
print(f"压力维度 α = {alpha_s:.3f}")

a_items = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
alpha_a = cronbach_alpha(data[a_items])
print(f"焦虑维度 α = {alpha_a:.3f}")

d_items = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
alpha_d = cronbach_alpha(data[d_items])
print(f"抑郁维度 α = {alpha_d:.3f}")

# 保存信度结果
reliability_results = pd.DataFrame({
    '维度': ['戒断症状', '渴求性', '身心影响', '手机依赖总表',
            '压力', '焦虑', '抑郁'],
    '题目数': [len(ws_items), len(cr_items), len(pmi_items), len(mpd_items),
              len(s_items), len(a_items), len(d_items)],
    "Cronbach's Alpha": [alpha_ws, alpha_cr, alpha_pmi, alpha_mpd,
                         alpha_s, alpha_a, alpha_d]
})
reliability_results.to_csv('信度分析结果.csv', index=False, encoding='utf-8-sig')
print("\n信度分析结果已保存至: 信度分析结果.csv")

# ============================================================================
# 5. 相关性分析
# ============================================================================

print("\n" + "=" * 80)
print("相关性分析")
print("=" * 80)

# 计算相关矩阵
corr_matrix = data[main_vars].corr()
print("\n主要变量相关矩阵:")
print(corr_matrix.round(3))

# 保存相关矩阵
corr_matrix.to_csv('相关性矩阵.csv', encoding='utf-8-sig')

# 可视化相关矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Main Variables Correlation Matrix', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('相关性矩阵热图.png', dpi=300, bbox_inches='tight')
print("\n相关性矩阵热图已保存至: 相关性矩阵热图.png")
plt.close()

# ============================================================================
# 6. 正态性检验
# ============================================================================

print("\n" + "=" * 80)
print("正态性检验 (Shapiro-Wilk Test)")
print("=" * 80)

normality_results = []
for var in main_vars:
    statistic, pvalue = stats.shapiro(data[var])
    normality_results.append({
        '变量': var,
        'W统计量': statistic,
        'p值': pvalue,
        '是否正态': 'Yes' if pvalue > 0.05 else 'No'
    })

normality_df = pd.DataFrame(normality_results)
print(normality_df.to_string(index=False))
normality_df.to_csv('正态性检验结果.csv', index=False, encoding='utf-8-sig')

# ============================================================================
# 7. 数据分布可视化
# ============================================================================

print("\n生成数据分布图...")

# 主要变量分布图
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, var in enumerate(main_vars):
    axes[idx].hist(data[var], bins=15, edgecolor='black', alpha=0.7)
    axes[idx].set_title(var, fontsize=12)
    axes[idx].set_xlabel('Score')
    axes[idx].set_ylabel('Frequency')
    axes[idx].axvline(data[var].mean(), color='red', linestyle='--',
                      label=f'Mean={data[var].mean():.2f}')
    axes[idx].legend()

# 删除多余的子图
for idx in range(len(main_vars), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('变量分布图.png', dpi=300, bbox_inches='tight')
print("变量分布图已保存至: 变量分布图.png")
plt.close()

# ============================================================================
# 8. 结构方程模型分析 (使用semopy)
# ============================================================================

print("\n" + "=" * 80)
print("结构方程模型分析 (SEM)")
print("=" * 80)

try:
    from semopy import Model

    # 定义测量模型（CFA）
    measurement_model = """
    # 潜变量定义
    MPD =~ WS_Total + CR_Total + PMI_Total
    NegativeAffect =~ S_Total + A_Total + D_Total
    """

    # 拟合测量模型
    print("\n拟合测量模型（CFA）...")
    model_cfa = Model(measurement_model)
    results_cfa = model_cfa.fit(data)

    print("\n测量模型拟合结果:")
    print(results_cfa)

    # 定义结构模型
    structural_model = """
    # 测量模型
    MPD =~ WS_Total + CR_Total + PMI_Total
    NegativeAffect =~ S_Total + A_Total + D_Total

    # 结构模型
    NegativeAffect ~ MPD
    """

    # 拟合结构模型
    print("\n拟合结构模型...")
    model_sem = Model(structural_model)
    results_sem = model_sem.fit(data)

    print("\n结构模型拟合结果:")
    print(results_sem)

    # 获取拟合指数
    inspect_sem = model_sem.inspect()
    print("\n拟合指数:")
    print(inspect_sem)

    # 保存结果
    with open('SEM分析结果_Python.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("结构方程模型分析结果\n")
        f.write("=" * 80 + "\n\n")
        f.write("测量模型结果:\n")
        f.write(str(results_cfa) + "\n\n")
        f.write("结构模型结果:\n")
        f.write(str(results_sem) + "\n\n")
        f.write("拟合指数:\n")
        f.write(str(inspect_sem) + "\n")

    print("\nSEM分析结果已保存至: SEM分析结果_Python.txt")

except ImportError:
    print("\n注意: semopy包未安装，跳过SEM分析")
    print("可以使用以下命令安装: pip install semopy")

# ============================================================================
# 9. 回归分析（备选方法）
# ============================================================================

print("\n" + "=" * 80)
print("回归分析（备选方法）")
print("=" * 80)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 准备数据
X = data[['MPD_Total']].values
y = data[['S_Total', 'A_Total', 'D_Total']].values

# 创建总负性情绪分数
data['NegativeAffect_Total'] = data['S_Total'] + data['A_Total'] + data['D_Total']

# 简单回归: MPD -> NegativeAffect
X_simple = data[['MPD_Total']].values
y_simple = data['NegativeAffect_Total'].values

model_reg = LinearRegression()
model_reg.fit(X_simple, y_simple)
y_pred = model_reg.predict(X_simple)

r2 = r2_score(y_simple, y_pred)
beta = model_reg.coef_[0]
intercept = model_reg.intercept_

print(f"\n手机依赖 → 负性情绪总分")
print(f"回归系数 β = {beta:.4f}")
print(f"截距 = {intercept:.4f}")
print(f"R² = {r2:.4f}")

# 绘制回归图
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y_simple, alpha=0.5, label='Observed')
plt.plot(X_simple, y_pred, 'r-', linewidth=2, label='Regression line')
plt.xlabel('Mobile Phone Dependence', fontsize=12)
plt.ylabel('Negative Affect Total', fontsize=12)
plt.title(f'MPD → Negative Affect (R² = {r2:.3f})', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('回归分析图.png', dpi=300, bbox_inches='tight')
print("\n回归分析图已保存至: 回归分析图.png")
plt.close()

# ============================================================================
# 10. 保存清理后的数据
# ============================================================================

data.to_csv('数据_重命名版_Python.csv', index=False, encoding='utf-8-sig')
print("\n清理后的数据已保存至: 数据_重命名版_Python.csv")

# ============================================================================
# 11. 生成分析报告摘要
# ============================================================================

report = f"""
{'=' * 80}
手机依赖与心理健康SEM分析报告摘要
{'=' * 80}

分析日期: 2025-11-17
样本量: {len(data)}

一、样本特征
-----------
性别分布: 男性 {(data['Gender']==1).sum()}人 ({(data['Gender']==1).sum()/len(data)*100:.1f}%),
         女性 {(data['Gender']==2).sum()}人 ({(data['Gender']==2).sum()/len(data)*100:.1f}%)
年龄: M = {data['Age'].mean():.2f}, SD = {data['Age'].std():.2f}

二、信度分析
-----------
戒断症状维度: α = {alpha_ws:.3f}
渴求性维度: α = {alpha_cr:.3f}
身心影响维度: α = {alpha_pmi:.3f}
手机依赖总表: α = {alpha_mpd:.3f}
压力维度: α = {alpha_s:.3f}
焦虑维度: α = {alpha_a:.3f}
抑郁维度: α = {alpha_d:.3f}

三、描述性统计
-------------
手机依赖总分: M = {data['MPD_Total'].mean():.2f}, SD = {data['MPD_Total'].std():.2f}
压力总分: M = {data['S_Total'].mean():.2f}, SD = {data['S_Total'].std():.2f}
焦虑总分: M = {data['A_Total'].mean():.2f}, SD = {data['A_Total'].std():.2f}
抑郁总分: M = {data['D_Total'].mean():.2f}, SD = {data['D_Total'].std():.2f}

四、相关性分析
-------------
手机依赖 - 压力: r = {corr_matrix.loc['MPD_Total', 'S_Total']:.3f}
手机依赖 - 焦虑: r = {corr_matrix.loc['MPD_Total', 'A_Total']:.3f}
手机依赖 - 抑郁: r = {corr_matrix.loc['MPD_Total', 'D_Total']:.3f}

五、回归分析
-----------
手机依赖 → 负性情绪总分
β = {beta:.4f}, R² = {r2:.4f}

六、生成文件清单
--------------
1. 描述性统计.csv
2. 信度分析结果.csv
3. 相关性矩阵.csv
4. 相关性矩阵热图.png
5. 正态性检验结果.csv
6. 变量分布图.png
7. 回归分析图.png
8. 数据_重命名版_Python.csv
9. 分析报告摘要.txt

{'=' * 80}
分析完成！
{'=' * 80}
"""

print(report)

# 保存报告
with open('分析报告摘要.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n分析报告摘要已保存至: 分析报告摘要.txt")
print("\n所有分析完成！请查看生成的文件。")
