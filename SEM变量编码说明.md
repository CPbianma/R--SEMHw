# SEM分析变量编码说明

## 变量命名规范

为了便于进行结构方程模型分析，建议使用以下变量简称：

### 1. 人口统计学变量
- `ID` - 序号
- `Gender` - 性别 (1=男, 2=女)
- `Age` - 年龄
- `OnlyChild` - 是否独生子女
- `ParentEdu` - 父母文化水平
- `InLove` - 是否恋爱
- `PhoneYears` - 手机使用年限
- `PhoneCost` - 月均手机消费

### 2. 手机使用相关变量
**手机功能使用 (Phone Functions)**
- `PF1` - 打电话
- `PF2` - 发短信
- `PF3` - 拍照摄影
- `PF4` - 看电视/听音乐
- `PF5` - 上网

**手机使用动机 (Phone Motives)**
- `PM1` - 人际交往
- `PM2` - 打发时间
- `PM3` - 体现个性
- `PM4` - 娱乐消遣
- `PM5` - 学习/工作需要

**手机使用时间**
- `DailyUse` - 每日手机使用时间

### 3. 手机依赖量表 (Mobile Phone Dependence, MPD)

**戒断症状维度 (Withdrawal Symptoms, WS)**
- `WS1` - 企图减少或停止使用手机时沮丧易怒
- `WS2` - 必须关机时感到烦躁
- `WS3` - 脑海浮现手机有未接来电
- `WS4` - 幻听手机铃声或震动
- `WS5` - 不查看手机感到焦虑
- `WS6` - 没有手机感到不知所措
- `WS_Total` - 戒断症状总分

**渴求性维度 (Craving, CR)**
- `CR1` - 觉得使用手机时间不够
- `CR2` - 需要更多时间才能满足
- `CR3` - 做有关手机的梦
- `CR_Total` - 渴求性总分

**身心影响维度 (Physical and Mental Impact, PMI)**
- `PMI1` - 玩手机导致睡眠不足
- `PMI2` - 宁愿玩手机不处理紧迫事情
- `PMI3` - 休闲活动时间减少
- `PMI4` - 影响学习或工作效率
- `PMI_Total` - 身心影响总分

- `MPD_Total` - 手机依赖总分

### 4. DASS-21量表

**压力维度 (Stress, S)**
- `S1` - 很难让自己安静下来
- `S2` - 对环境容易反应过度
- `S3` - 消耗很多精神
- `S4` - 感到忐忑不安
- `S5` - 很难放松自己
- `S6` - 无法容忍阻碍
- `S7` - 很容易被触怒
- `S_Total` - 压力总分

**焦虑维度 (Anxiety, A)**
- `A1` - 口干舌燥
- `A2` - 呼吸困难
- `A3` - 感到颤抖
- `A4` - 担心恐慌或出丑
- `A5` - 快要恐慌
- `A6` - 心律不正常
- `A7` - 无缘无故害怕
- `A_Total` - 焦虑总分

**抑郁维度 (Depression, D)**
- `D1` - 不再有愉快舒畅的感觉
- `D2` - 很难主动开始工作
- `D3` - 对将来没什么可盼望
- `D4` - 感到忧郁沮丧
- `D5` - 对任何事不热衷
- `D6` - 觉得自己不配做人
- `D7` - 感到生命毫无意义
- `D_Total` - 抑郁总分

## SEM模型中的潜变量定义

### 二阶因子模型结构

```
手机依赖 (MPD - 二阶潜变量)
├── 戒断症状 (WS - 一阶潜变量)
│   ├── WS1, WS2, WS3, WS4, WS5, WS6
├── 渴求性 (CR - 一阶潜变量)
│   ├── CR1, CR2, CR3
└── 身心影响 (PMI - 一阶潜变量)
    ├── PMI1, PMI2, PMI3, PMI4

负性情绪/心理健康问题 (Negative Affect - 二阶潜变量)
├── 压力 (Stress - 一阶潜变量)
│   ├── S1, S2, S3, S4, S5, S6, S7
├── 焦虑 (Anxiety - 一阶潜变量)
│   ├── A1, A2, A3, A4, A5, A6, A7
└── 抑郁 (Depression - 一阶潜变量)
    ├── D1, D2, D3, D4, D5, D6, D7
```

### 一阶因子模型结构（简化版）

如果样本量较小，可以使用各维度总分作为观测变量：

```
手机依赖 (MPD)
├── WS_Total
├── CR_Total
└── PMI_Total

负性情绪 (Negative Affect)
├── S_Total
├── A_Total
└── D_Total
```

## R语言lavaan包示例代码

### 测量模型（二阶因子）

```r
# 定义测量模型
measurement_model <- '
  # 一阶因子
  WS =~ WS1 + WS2 + WS3 + WS4 + WS5 + WS6
  CR =~ CR1 + CR2 + CR3
  PMI =~ PMI1 + PMI2 + PMI3 + PMI4

  Stress =~ S1 + S2 + S3 + S4 + S5 + S6 + S7
  Anxiety =~ A1 + A2 + A3 + A4 + A5 + A6 + A7
  Depression =~ D1 + D2 + D3 + D4 + D5 + D6 + D7

  # 二阶因子
  MPD =~ WS + CR + PMI
  NegativeAffect =~ Stress + Anxiety + Depression
'
```

### 简化测量模型（使用总分）

```r
# 定义简化测量模型
simple_model <- '
  # 潜变量定义
  MPD =~ WS_Total + CR_Total + PMI_Total
  NegativeAffect =~ S_Total + A_Total + D_Total
'
```

### 结构模型示例

```r
# 结构模型：手机依赖影响负性情绪
structural_model <- '
  # 测量模型
  MPD =~ WS_Total + CR_Total + PMI_Total
  NegativeAffect =~ S_Total + A_Total + D_Total

  # 结构模型
  NegativeAffect ~ MPD

  # 控制变量
  NegativeAffect ~ Gender + Age + DailyUse
  MPD ~ Gender + Age
'
```

## Python SEM包示例 (semopy)

```python
import pandas as pd
from semopy import Model

# 定义模型
model_desc = '''
# 测量模型
MPD =~ WS_Total + CR_Total + PMI_Total
NegativeAffect =~ S_Total + A_Total + D_Total

# 结构模型
NegativeAffect ~ MPD

# 控制变量
NegativeAffect ~ Gender + Age + DailyUse
MPD ~ Gender + Age
'''

# 读取数据
data = pd.read_excel('问卷数据-已编码.xlsx')

# 拟合模型
model = Model(model_desc)
results = model.fit(data)

# 查看结果
print(results)
```

## 注意事项

1. **样本量**: 当前样本量为59，对于复杂的二阶因子模型可能不够充分
   - 建议: 使用简化模型或题目包裹策略
   - 推荐: 使用稳健估计方法（MLR）

2. **模型识别**: 确保模型可识别
   - 二阶因子模型需要至少3个一阶因子
   - 每个因子至少需要3个观测变量

3. **拟合指数阈值**:
   - CFI/TLI ≥ 0.90 (较好 ≥ 0.95)
   - RMSEA ≤ 0.08 (较好 ≤ 0.06)
   - SRMR ≤ 0.08

4. **缺失值处理**: 当前数据无缺失值，无需处理

5. **编码检查**:
   - 确认反向计分题已正确编码
   - 检查异常值和极端值
