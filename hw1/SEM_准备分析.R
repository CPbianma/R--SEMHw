# ============================================================================
# 结构方程模型 (SEM) 数据准备和初步分析
# 作者: Claude
# 日期: 2025-11-17
# 用途: 手机依赖与心理健康的SEM分析
# ============================================================================

# 安装和加载必要的包
# install.packages(c("lavaan", "semPlot", "readxl", "psych", "ggplot2", "corrplot"))

library(readxl)      # 读取Excel文件
library(lavaan)      # 结构方程模型
library(semPlot)     # SEM图形化
library(psych)       # 心理测量分析
library(corrplot)    # 相关性可视化
library(ggplot2)     # 数据可视化

# ============================================================================
# 1. 数据读取和准备
# ============================================================================

# 读取数据
data_raw <- read_excel("问卷数据-已编码.xlsx")

# 查看数据结构
str(data_raw)
summary(data_raw)

# 检查缺失值
sum(is.na(data_raw))

# ============================================================================
# 2. 变量重命名（可选，便于SEM分析）
# ============================================================================

# 创建工作数据集
data <- data_raw

# 人口统计学变量
names(data)[1:8] <- c("ID", "Gender", "Age", "OnlyChild",
                       "ParentEdu", "InLove", "PhoneYears", "PhoneCost")

# 手机功能（9-13）
names(data)[9:13] <- paste0("PF", 1:5)

# 手机动机（14-18）
names(data)[14:18] <- paste0("PM", 1:5)

# 每日使用时间
names(data)[19] <- "DailyUse"

# 戒断症状（20-25）+ 总分（26）
names(data)[20:26] <- c(paste0("WS", 1:6), "WS_Total")

# 渴求性（27-29）+ 总分（30）
names(data)[27:30] <- c(paste0("CR", 1:3), "CR_Total")

# 身心影响（31-34）+ 总分（35）
names(data)[31:35] <- c(paste0("PMI", 1:4), "PMI_Total")

# 手机依赖总分（36）
names(data)[36] <- "MPD_Total"

# 压力维度（37-43）+ 总分（44）
names(data)[37:44] <- c(paste0("S", 1:7), "S_Total")

# 焦虑维度（45-51）+ 总分（52）
names(data)[45:52] <- c(paste0("A", 1:7), "A_Total")

# 抑郁维度（53-59）+ 总分（60）
names(data)[53:60] <- c(paste0("D", 1:7), "D_Total")

# ============================================================================
# 3. 描述性统计
# ============================================================================

cat("\n========== 描述性统计 ==========\n")

# 样本特征
cat("\n样本量:", nrow(data), "\n")
cat("性别分布:\n")
table(data$Gender)
cat("\n年龄描述:\n")
summary(data$Age)

# 主要变量描述统计
main_vars <- c("MPD_Total", "WS_Total", "CR_Total", "PMI_Total",
               "S_Total", "A_Total", "D_Total")

cat("\n主要变量描述统计:\n")
describe(data[, main_vars])

# ============================================================================
# 4. 信度分析
# ============================================================================

cat("\n========== 信度分析 ==========\n")

# 戒断症状维度
cat("\n戒断症状维度 Cronbach's α:\n")
ws_items <- paste0("WS", 1:6)
alpha_ws <- alpha(data[, ws_items])
print(alpha_ws$total$raw_alpha)

# 渴求性维度
cat("\n渴求性维度 Cronbach's α:\n")
cr_items <- paste0("CR", 1:3)
alpha_cr <- alpha(data[, cr_items])
print(alpha_cr$total$raw_alpha)

# 身心影响维度
cat("\n身心影响维度 Cronbach's α:\n")
pmi_items <- paste0("PMI", 1:4)
alpha_pmi <- alpha(data[, pmi_items])
print(alpha_pmi$total$raw_alpha)

# 手机依赖总量表
cat("\n手机依赖总量表 Cronbach's α:\n")
mpd_items <- c(ws_items, cr_items, pmi_items)
alpha_mpd <- alpha(data[, mpd_items])
print(alpha_mpd$total$raw_alpha)

# DASS-21 各维度
cat("\n压力维度 Cronbach's α:\n")
s_items <- paste0("S", 1:7)
alpha_s <- alpha(data[, s_items])
print(alpha_s$total$raw_alpha)

cat("\n焦虑维度 Cronbach's α:\n")
a_items <- paste0("A", 1:7)
alpha_a <- alpha(data[, a_items])
print(alpha_a$total$raw_alpha)

cat("\n抑郁维度 Cronbach's α:\n")
d_items <- paste0("D", 1:7)
alpha_d <- alpha(data[, d_items])
print(alpha_d$total$raw_alpha)

# ============================================================================
# 5. 相关性分析
# ============================================================================

cat("\n========== 相关性分析 ==========\n")

# 主要变量相关矩阵
cor_matrix <- cor(data[, main_vars])
print(round(cor_matrix, 3))

# 相关性可视化
pdf("相关性矩阵图.pdf", width = 10, height = 10)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.7,
         tl.col = "black", tl.srt = 45,
         title = "主要变量相关性矩阵")
dev.off()

cat("\n相关性矩阵图已保存至: 相关性矩阵图.pdf\n")

# ============================================================================
# 6. 测量模型 - 验证性因子分析 (CFA)
# ============================================================================

cat("\n========== 验证性因子分析 (CFA) ==========\n")

# 模型1: 使用维度总分的简化模型
model1 <- '
  # 潜变量定义
  MPD =~ WS_Total + CR_Total + PMI_Total
  NegativeAffect =~ S_Total + A_Total + D_Total
'

fit1 <- cfa(model1, data = data, estimator = "MLR")
summary(fit1, fit.measures = TRUE, standardized = TRUE)

cat("\n模型1拟合指数:\n")
fit_measures <- fitMeasures(fit1, c("chisq", "df", "pvalue", "cfi", "tli",
                                     "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                                     "srmr"))
print(round(fit_measures, 3))

# 绘制路径图
pdf("CFA模型1路径图.pdf", width = 10, height = 8)
semPaths(fit1, what = "std", layout = "tree",
         edge.label.cex = 1.2, curvePivot = TRUE,
         title = "测量模型1: 简化CFA模型")
dev.off()

cat("\nCFA模型1路径图已保存至: CFA模型1路径图.pdf\n")

# ============================================================================
# 7. 结构模型 - 手机依赖对心理健康的影响
# ============================================================================

cat("\n========== 结构方程模型 (SEM) ==========\n")

# 基本结构模型
structural_model <- '
  # 测量模型
  MPD =~ WS_Total + CR_Total + PMI_Total
  NegativeAffect =~ S_Total + A_Total + D_Total

  # 结构模型
  NegativeAffect ~ MPD
'

fit_sem <- sem(structural_model, data = data, estimator = "MLR")
summary(fit_sem, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)

cat("\n结构模型拟合指数:\n")
fit_measures_sem <- fitMeasures(fit_sem, c("chisq", "df", "pvalue", "cfi", "tli",
                                            "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                                            "srmr"))
print(round(fit_measures_sem, 3))

# 提取路径系数
cat("\n路径系数:\n")
params <- parameterEstimates(fit_sem, standardized = TRUE)
print(params[params$op == "~", c("lhs", "op", "rhs", "est", "se", "pvalue", "std.all")])

# 绘制SEM路径图
pdf("SEM结构模型路径图.pdf", width = 10, height = 8)
semPaths(fit_sem, what = "std", layout = "tree",
         edge.label.cex = 1.2, curvePivot = TRUE,
         title = "结构方程模型: 手机依赖 → 负性情绪")
dev.off()

cat("\nSEM结构模型路径图已保存至: SEM结构模型路径图.pdf\n")

# ============================================================================
# 8. 控制变量的结构模型
# ============================================================================

cat("\n========== 包含控制变量的SEM模型 ==========\n")

structural_model_control <- '
  # 测量模型
  MPD =~ WS_Total + CR_Total + PMI_Total
  NegativeAffect =~ S_Total + A_Total + D_Total

  # 结构模型（含控制变量）
  NegativeAffect ~ MPD + Gender + Age + DailyUse
  MPD ~ Gender + Age + DailyUse
'

fit_sem_control <- sem(structural_model_control, data = data, estimator = "MLR")
summary(fit_sem_control, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)

# 提取路径系数
cat("\n包含控制变量的路径系数:\n")
params_control <- parameterEstimates(fit_sem_control, standardized = TRUE)
print(params_control[params_control$op == "~", c("lhs", "op", "rhs", "est", "se", "pvalue", "std.all")])

# ============================================================================
# 9. 保存结果
# ============================================================================

# 保存清理后的数据
write.csv(data, "数据_重命名版.csv", row.names = FALSE, fileEncoding = "UTF-8")

# 保存模型结果摘要
sink("SEM分析结果摘要.txt")
cat("========== 手机依赖与心理健康SEM分析结果 ==========\n")
cat("\n日期:", Sys.Date(), "\n")
cat("样本量:", nrow(data), "\n\n")

cat("\n========== 测量模型拟合指数 ==========\n")
print(round(fit_measures, 3))

cat("\n\n========== 结构模型拟合指数 ==========\n")
print(round(fit_measures_sem, 3))

cat("\n\n========== 路径系数（标准化） ==========\n")
print(params[params$op == "~", c("lhs", "op", "rhs", "est", "pvalue", "std.all")])

sink()

cat("\n所有分析结果已保存！\n")
cat("- 数据_重命名版.csv\n")
cat("- 相关性矩阵图.pdf\n")
cat("- CFA模型1路径图.pdf\n")
cat("- SEM结构模型路径图.pdf\n")
cat("- SEM分析结果摘要.txt\n")

cat("\n========== 分析完成 ==========\n")
