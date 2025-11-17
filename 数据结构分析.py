#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据结构分析脚本
用于整理问卷数据的结构信息，为结构方程模型分析做准备
"""

import pandas as pd
import json

# 读取数据
df = pd.read_excel('问卷数据-已编码.xlsx')

# 定义变量结构
data_structure = {
    "样本信息": {
        "样本量": df.shape[0],
        "变量数": df.shape[1],
        "缺失值": df.isnull().sum().sum()
    },

    "变量分组": {
        "1. 人口统计学变量": {
            "变量列表": list(df.columns[0:8]),
            "变量数": 8,
            "说明": "包括序号、性别、年龄、是否独生子女、父母文化水平、是否恋爱、手机使用年限、手机消费"
        },

        "2. 手机功能使用": {
            "变量列表": list(df.columns[8:13]),
            "变量数": 5,
            "说明": "测量手机不同功能的使用频率"
        },

        "3. 手机使用动机": {
            "变量列表": list(df.columns[13:18]),
            "变量数": 5,
            "说明": "测量使用手机的不同动机"
        },

        "4. 手机使用时间": {
            "变量列表": [df.columns[18]],
            "变量数": 1,
            "说明": "每日手机使用时间"
        },

        "5. 手机依赖量表": {
            "变量数": 17,
            "子维度": {
                "5.1 戒断症状维度": {
                    "题目": list(df.columns[19:25]),
                    "题目数": 6,
                    "总分变量": df.columns[25]
                },
                "5.2 渴求性维度": {
                    "题目": list(df.columns[26:29]),
                    "题目数": 3,
                    "总分变量": df.columns[29]
                },
                "5.3 身心影响维度": {
                    "题目": list(df.columns[30:34]),
                    "题目数": 4,
                    "总分变量": df.columns[34]
                },
                "总分": df.columns[35]
            },
            "说明": "测量手机依赖程度的量表，包含3个子维度，共13个题目"
        },

        "6. DASS-21心理健康量表": {
            "变量数": 24,
            "子维度": {
                "6.1 压力维度": {
                    "题目": list(df.columns[36:43]),
                    "题目数": 7,
                    "总分变量": df.columns[43]
                },
                "6.2 焦虑维度": {
                    "题目": list(df.columns[44:51]),
                    "题目数": 7,
                    "总分变量": df.columns[51]
                },
                "6.3 抑郁维度": {
                    "题目": list(df.columns[52:59]),
                    "题目数": 7,
                    "总分变量": df.columns[59]
                }
            },
            "说明": "DASS-21量表，测量压力、焦虑、抑郁三个维度，每个维度7个题目"
        }
    },

    "主要构念（潜变量）": {
        "手机依赖": {
            "观测变量": ["戒断症状", "渴求性", "身心影响"],
            "测量题目数": 13
        },
        "心理健康": {
            "观测变量": ["压力", "焦虑", "抑郁"],
            "测量题目数": 21
        }
    },

    "描述性统计": {
        "年龄": {
            "均值": round(df['年龄'].mean(), 2),
            "标准差": round(df['年龄'].std(), 2),
            "最小值": int(df['年龄'].min()),
            "最大值": int(df['年龄'].max())
        },
        "性别分布": {
            "男性": int((df['性别'] == 1).sum()),
            "女性": int((df['性别'] == 2).sum())
        },
        "手机依赖总分": {
            "均值": round(df['手机依赖-总分'].mean(), 2),
            "标准差": round(df['手机依赖-总分'].std(), 2),
            "最小值": int(df['手机依赖-总分'].min()),
            "最大值": int(df['手机依赖-总分'].max())
        },
        "压力总分": {
            "均值": round(df['压力维度-总分'].mean(), 2),
            "标准差": round(df['压力维度-总分'].std(), 2)
        },
        "焦虑总分": {
            "均值": round(df['焦虑维度-总分'].mean(), 2),
            "标准差": round(df['焦虑维度-总分'].std(), 2)
        },
        "抑郁总分": {
            "均值": round(df['抑郁维度-总分'].mean(), 2),
            "标准差": round(df['抑郁维度-总分'].std(), 2)
        }
    }
}

# 打印数据结构
print("=" * 100)
print("问卷数据结构分析报告")
print("=" * 100)
print(f"\n【基本信息】")
print(f"样本量: {data_structure['样本信息']['样本量']}")
print(f"变量数: {data_structure['样本信息']['变量数']}")
print(f"缺失值: {data_structure['样本信息']['缺失值']}")

print(f"\n【变量分组详情】")
for key, value in data_structure['变量分组'].items():
    print(f"\n{key}")
    if '子维度' in value:
        print(f"  总变量数: {value['变量数']}")
        for sub_key, sub_value in value['子维度'].items():
            print(f"  {sub_key}")
            if isinstance(sub_value, dict) and '题目' in sub_value:
                print(f"    题目数: {sub_value['题目数']}")
                print(f"    总分变量: {sub_value['总分变量']}")
    else:
        print(f"  变量数: {value['变量数']}")
        print(f"  说明: {value['说明']}")

print(f"\n【潜变量结构】")
for construct, details in data_structure['主要构念（潜变量）'].items():
    print(f"\n{construct}:")
    print(f"  观测变量: {', '.join(details['观测变量'])}")
    print(f"  测量题目数: {details['测量题目数']}")

print(f"\n【描述性统计】")
for var_name, stats in data_structure['描述性统计'].items():
    print(f"\n{var_name}:")
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name}: {stat_value}")

# 保存为JSON格式
with open('数据结构.json', 'w', encoding='utf-8') as f:
    json.dump(data_structure, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 100)
print("数据结构已保存至: 数据结构.json")
print("=" * 100)

# 创建变量映射表
print("\n\n【完整变量列表】")
print("=" * 100)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# 计算相关性矩阵（主要变量）
print("\n\n【主要变量相关性】")
print("=" * 100)
key_vars = ['手机依赖-总分', '压力维度-总分', '焦虑维度-总分', '抑郁维度-总分',
            '戒断症状-总分', '渴求性-总分', '身心影响-总分']
corr_matrix = df[key_vars].corr()
print(corr_matrix.round(3))

# 保存相关性矩阵
corr_matrix.to_csv('主要变量相关性矩阵.csv', encoding='utf-8-sig')
print("\n相关性矩阵已保存至: 主要变量相关性矩阵.csv")
