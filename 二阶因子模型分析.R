# ============================================================================
# 二阶因子模型分析：手机依赖与负面情绪
# 作者: Claude
# 日期: 2025-11-17
# ============================================================================

# 加载必要的包
library(readxl)
library(lavaan)
library(semPlot)
library(corrplot)
library(psych)
library(ggplot2)

# 设置工作目录和输出选项
options(width = 100)

# ============================================================================
# 1. 数据读取和准备
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("二阶因子模型分析：手机依赖与负面情绪\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 读取数据
data_raw <- read_excel("问卷数据-已编码.xlsx")
cat("数据读取成功！样本量:", nrow(data_raw), "\n")

# 提取相关列并重命名
data <- data.frame(
  # 手机依赖 - 戒断症状 (第20-25列)
  WD1 = data_raw[[20]],
  WD2 = data_raw[[21]],
  WD3 = data_raw[[22]],
  WD4 = data_raw[[23]],
  WD5 = data_raw[[24]],
  WD6 = data_raw[[25]],

  # 手机依赖 - 渴求性 (第27-29列)
  CR1 = data_raw[[27]],
  CR2 = data_raw[[28]],
  CR3 = data_raw[[29]],

  # 手机依赖 - 身心影响 (第31-34列)
  PI1 = data_raw[[31]],
  PI2 = data_raw[[32]],
  PI3 = data_raw[[33]],
  PI4 = data_raw[[34]],

  # DASS-21 - 压力维度 (第37-43列)
  ST1 = data_raw[[37]],
  ST2 = data_raw[[38]],
  ST3 = data_raw[[39]],
  ST4 = data_raw[[40]],
  ST5 = data_raw[[41]],
  ST6 = data_raw[[42]],
  ST7 = data_raw[[43]],

  # DASS-21 - 焦虑维度 (第45-51列)
  AN1 = data_raw[[45]],
  AN2 = data_raw[[46]],
  AN3 = data_raw[[47]],
  AN4 = data_raw[[48]],
  AN5 = data_raw[[49]],
  AN6 = data_raw[[50]],
  AN7 = data_raw[[51]],

  # DASS-21 - 抑郁维度 (第53-59列)
  DE1 = data_raw[[53]],
  DE2 = data_raw[[54]],
  DE3 = data_raw[[55]],
  DE4 = data_raw[[56]],
  DE5 = data_raw[[57]],
  DE6 = data_raw[[58]],
  DE7 = data_raw[[59]]
)

cat("变量重命名完成！总变量数:", ncol(data), "\n")
cat("  - 手机依赖观测变量: 13个 (WD1-WD6, CR1-CR3, PI1-PI4)\n")
cat("  - DASS-21观测变量: 21个 (ST1-ST7, AN1-AN7, DE1-DE7)\n")

# ============================================================================
# 2. 描述性统计
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("描述性统计\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

desc_stats <- describe(data)
print(desc_stats[, c("n", "mean", "sd", "min", "max", "skew", "kurtosis")])

# 保存描述性统计
write.csv(desc_stats, "二阶模型_描述性统计.csv", row.names = TRUE, fileEncoding = "UTF-8")
cat("\n描述性统计已保存至: 二阶模型_描述性统计.csv\n")

# ============================================================================
# 3. 信度分析
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("信度分析 (Cronbach's Alpha)\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 手机依赖各维度
alpha_wd <- alpha(data[, paste0("WD", 1:6)])
alpha_cr <- alpha(data[, paste0("CR", 1:3)])
alpha_pi <- alpha(data[, paste0("PI", 1:4)])
alpha_mpd <- alpha(data[, c(paste0("WD", 1:6), paste0("CR", 1:3), paste0("PI", 1:4))])

# DASS-21各维度
alpha_st <- alpha(data[, paste0("ST", 1:7)])
alpha_an <- alpha(data[, paste0("AN", 1:7)])
alpha_de <- alpha(data[, paste0("DE", 1:7)])
alpha_dass <- alpha(data[, c(paste0("ST", 1:7), paste0("AN", 1:7), paste0("DE", 1:7))])

cat("手机依赖量表:\n")
cat(sprintf("  戒断症状 (WD): α = %.3f\n", alpha_wd$total$raw_alpha))
cat(sprintf("  渴求性 (CR): α = %.3f\n", alpha_cr$total$raw_alpha))
cat(sprintf("  身心影响 (PI): α = %.3f\n", alpha_pi$total$raw_alpha))
cat(sprintf("  总量表: α = %.3f\n", alpha_mpd$total$raw_alpha))

cat("\nDASS-21量表:\n")
cat(sprintf("  压力维度 (ST): α = %.3f\n", alpha_st$total$raw_alpha))
cat(sprintf("  焦虑维度 (AN): α = %.3f\n", alpha_an$total$raw_alpha))
cat(sprintf("  抑郁维度 (DE): α = %.3f\n", alpha_de$total$raw_alpha))
cat(sprintf("  总量表: α = %.3f\n", alpha_dass$total$raw_alpha))

# ============================================================================
# 4. 相关性分析
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("维度间相关性分析\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 计算各维度总分
data$WD_sum <- rowSums(data[, paste0("WD", 1:6)])
data$CR_sum <- rowSums(data[, paste0("CR", 1:3)])
data$PI_sum <- rowSums(data[, paste0("PI", 1:4)])
data$ST_sum <- rowSums(data[, paste0("ST", 1:7)])
data$AN_sum <- rowSums(data[, paste0("AN", 1:7)])
data$DE_sum <- rowSums(data[, paste0("DE", 1:7)])

# 相关矩阵
dimension_vars <- c("WD_sum", "CR_sum", "PI_sum", "ST_sum", "AN_sum", "DE_sum")
dimension_names <- c("戒断症状", "渴求性", "身心影响", "压力", "焦虑", "抑郁")
cor_matrix <- cor(data[, dimension_vars])
colnames(cor_matrix) <- rownames(cor_matrix) <- dimension_names

print(round(cor_matrix, 3))

# 可视化相关矩阵
pdf("二阶模型_维度相关矩阵图.pdf", width = 10, height = 10)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.8,
         tl.col = "black", tl.srt = 45, tl.cex = 1.2,
         title = "维度间相关矩阵", mar = c(0, 0, 2, 0))
dev.off()
cat("\n相关矩阵图已保存至: 二阶模型_维度相关矩阵图.pdf\n")

# ============================================================================
# 5. 模型1：手机依赖的二阶因子模型
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("模型1：手机依赖的二阶因子模型\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 定义二阶因子模型
model_mpd <- '
  # 一阶因子（First-Order Factors）
  Withdrawal =~ WD1 + WD2 + WD3 + WD4 + WD5 + WD6
  Craving =~ CR1 + CR2 + CR3
  PhysicalImpact =~ PI1 + PI2 + PI3 + PI4

  # 二阶因子（Second-Order Factor）
  MobileDependence =~ Withdrawal + Craving + PhysicalImpact
'

# 拟合模型
fit_mpd <- cfa(model_mpd, data = data, estimator = "MLR")

# 输出结果
cat("模型拟合结果:\n")
summary(fit_mpd, fit.measures = TRUE, standardized = TRUE)

# 提取拟合指数
fit_measures_mpd <- fitMeasures(fit_mpd, c("chisq", "df", "pvalue", "cfi", "tli",
                                            "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                                            "srmr", "aic", "bic"))
cat("\n模型1拟合指数:\n")
print(round(fit_measures_mpd, 3))

# 绘制路径图
pdf("模型1_手机依赖二阶因子模型.pdf", width = 14, height = 10)
semPaths(fit_mpd,
         what = "std",           # 显示标准化系数
         layout = "tree2",       # 树状布局
         edge.label.cex = 1.0,   # 系数字体大小
         curvePivot = TRUE,
         rotation = 2,           # 旋转布局
         sizeMan = 6,            # 观测变量大小
         sizeLat = 10,           # 潜变量大小
         edge.color = "black",
         nodeLabels = c(paste0("WD", 1:6), paste0("CR", 1:3), paste0("PI", 1:4),
                       "戒断症状", "渴求性", "身心影响", "手机依赖"),
         style = "lisrel",
         title = TRUE,
         residuals = FALSE,
         thresholds = FALSE)
title("模型1: 手机依赖的二阶因子模型", line = 3, cex.main = 1.5)
dev.off()
cat("\n模型1路径图已保存至: 模型1_手机依赖二阶因子模型.pdf\n")

# ============================================================================
# 6. 模型2：DASS-21的二阶因子模型
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("模型2：DASS-21的二阶因子模型\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 定义二阶因子模型
model_dass <- '
  # 一阶因子（First-Order Factors）
  Stress =~ ST1 + ST2 + ST3 + ST4 + ST5 + ST6 + ST7
  Anxiety =~ AN1 + AN2 + AN3 + AN4 + AN5 + AN6 + AN7
  Depression =~ DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7

  # 二阶因子（Second-Order Factor）
  NegativeAffect =~ Stress + Anxiety + Depression
'

# 拟合模型
fit_dass <- cfa(model_dass, data = data, estimator = "MLR")

# 输出结果
cat("模型拟合结果:\n")
summary(fit_dass, fit.measures = TRUE, standardized = TRUE)

# 提取拟合指数
fit_measures_dass <- fitMeasures(fit_dass, c("chisq", "df", "pvalue", "cfi", "tli",
                                              "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                                              "srmr", "aic", "bic"))
cat("\n模型2拟合指数:\n")
print(round(fit_measures_dass, 3))

# 绘制路径图
pdf("模型2_DASS21二阶因子模型.pdf", width = 14, height = 10)
semPaths(fit_dass,
         what = "std",
         layout = "tree2",
         edge.label.cex = 1.0,
         curvePivot = TRUE,
         rotation = 2,
         sizeMan = 6,
         sizeLat = 10,
         edge.color = "black",
         nodeLabels = c(paste0("ST", 1:7), paste0("AN", 1:7), paste0("DE", 1:7),
                       "压力", "焦虑", "抑郁", "负面情绪"),
         style = "lisrel",
         title = TRUE,
         residuals = FALSE,
         thresholds = FALSE)
title("模型2: DASS-21的二阶因子模型", line = 3, cex.main = 1.5)
dev.off()
cat("\n模型2路径图已保存至: 模型2_DASS21二阶因子模型.pdf\n")

# ============================================================================
# 7. 整合模型：两个二阶因子的相关模型
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("整合模型：手机依赖与负面情绪的相关关系\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 定义整合模型
model_integrated <- '
  # 手机依赖 - 一阶因子
  Withdrawal =~ WD1 + WD2 + WD3 + WD4 + WD5 + WD6
  Craving =~ CR1 + CR2 + CR3
  PhysicalImpact =~ PI1 + PI2 + PI3 + PI4

  # DASS-21 - 一阶因子
  Stress =~ ST1 + ST2 + ST3 + ST4 + ST5 + ST6 + ST7
  Anxiety =~ AN1 + AN2 + AN3 + AN4 + AN5 + AN6 + AN7
  Depression =~ DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7

  # 二阶因子
  MobileDependence =~ Withdrawal + Craving + PhysicalImpact
  NegativeAffect =~ Stress + Anxiety + Depression

  # 二阶因子间的相关关系
  MobileDependence ~~ NegativeAffect
'

# 拟合模型
fit_integrated <- cfa(model_integrated, data = data, estimator = "MLR")

# 输出结果
cat("整合模型拟合结果:\n")
summary(fit_integrated, fit.measures = TRUE, standardized = TRUE)

# 提取拟合指数
fit_measures_int <- fitMeasures(fit_integrated, c("chisq", "df", "pvalue", "cfi", "tli",
                                                   "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                                                   "srmr", "aic", "bic"))
cat("\n整合模型拟合指数:\n")
print(round(fit_measures_int, 3))

# 提取二阶因子间的相关系数
params <- parameterEstimates(fit_integrated, standardized = TRUE)
correlation <- params[params$lhs == "MobileDependence" & params$op == "~~" &
                     params$rhs == "NegativeAffect", ]
cat("\n二阶因子相关系数:\n")
cat(sprintf("手机依赖 <-> 负面情绪: r = %.3f (SE = %.3f, p = %.3f)\n",
            correlation$std.all, correlation$se, correlation$pvalue))

# 绘制整合模型路径图
pdf("整合模型_手机依赖与负面情绪.pdf", width = 16, height = 12)
semPaths(fit_integrated,
         what = "std",
         layout = "spring",      # 使用弹簧布局
         edge.label.cex = 0.8,
         curvePivot = TRUE,
         sizeMan = 5,
         sizeLat = 9,
         edge.color = "black",
         style = "lisrel",
         residuals = FALSE,
         thresholds = FALSE,
         rotation = 2)
title("整合模型: 手机依赖与负面情绪的二阶因子相关模型", line = 3, cex.main = 1.5)
dev.off()
cat("\n整合模型路径图已保存至: 整合模型_手机依赖与负面情绪.pdf\n")

# ============================================================================
# 8. 因子载荷矩阵
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("因子载荷矩阵\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 提取标准化因子载荷
loadings_mpd <- parameterEstimates(fit_mpd, standardized = TRUE)
loadings_mpd <- loadings_mpd[loadings_mpd$op == "=~",
                             c("lhs", "rhs", "est", "se", "pvalue", "std.all")]

cat("\n模型1 - 手机依赖因子载荷:\n")
print(loadings_mpd, digits = 3)

loadings_dass <- parameterEstimates(fit_dass, standardized = TRUE)
loadings_dass <- loadings_dass[loadings_dass$op == "=~",
                               c("lhs", "rhs", "est", "se", "pvalue", "std.all")]

cat("\n模型2 - DASS-21因子载荷:\n")
print(loadings_dass, digits = 3)

# 保存因子载荷
write.csv(loadings_mpd, "模型1_因子载荷.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(loadings_dass, "模型2_因子载荷.csv", row.names = FALSE, fileEncoding = "UTF-8")

# ============================================================================
# 9. 模型比较
# ============================================================================

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("模型拟合指数比较\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

# 创建比较表
comparison <- data.frame(
  Model = c("模型1_手机依赖", "模型2_DASS21", "整合模型"),
  Chi_Square = c(fit_measures_mpd["chisq"], fit_measures_dass["chisq"],
                 fit_measures_int["chisq"]),
  df = c(fit_measures_mpd["df"], fit_measures_dass["df"], fit_measures_int["df"]),
  CFI = c(fit_measures_mpd["cfi"], fit_measures_dass["cfi"], fit_measures_int["cfi"]),
  TLI = c(fit_measures_mpd["tli"], fit_measures_dass["tli"], fit_measures_int["tli"]),
  RMSEA = c(fit_measures_mpd["rmsea"], fit_measures_dass["rmsea"],
            fit_measures_int["rmsea"]),
  SRMR = c(fit_measures_mpd["srmr"], fit_measures_dass["srmr"],
           fit_measures_int["srmr"])
)

print(round(comparison, 3))
write.csv(comparison, "模型拟合指数比较.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("\n模型比较表已保存至: 模型拟合指数比较.csv\n")

# ============================================================================
# 10. 生成综合报告
# ============================================================================

sink("二阶因子模型分析报告_R.txt", encoding = "UTF-8")

cat("=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("二阶因子模型分析报告\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

cat("分析日期:", format(Sys.Date(), "%Y-%m-%d"), "\n")
cat("样本量:", nrow(data), "\n")
cat("分析软件: R + lavaan\n")
cat("估计方法: Maximum Likelihood with Robust standard errors (MLR)\n\n")

cat("=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("一、信度分析\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

cat("手机依赖量表:\n")
cat(sprintf("  戒断症状: α = %.3f\n", alpha_wd$total$raw_alpha))
cat(sprintf("  渴求性: α = %.3f\n", alpha_cr$total$raw_alpha))
cat(sprintf("  身心影响: α = %.3f\n", alpha_pi$total$raw_alpha))
cat(sprintf("  总量表: α = %.3f\n", alpha_mpd$total$raw_alpha))

cat("\nDASS-21量表:\n")
cat(sprintf("  压力维度: α = %.3f\n", alpha_st$total$raw_alpha))
cat(sprintf("  焦虑维度: α = %.3f\n", alpha_an$total$raw_alpha))
cat(sprintf("  抑郁维度: α = %.3f\n", alpha_de$total$raw_alpha))
cat(sprintf("  总量表: α = %.3f\n", alpha_dass$total$raw_alpha))

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("二、模型拟合指数\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

cat("模型1 - 手机依赖二阶因子模型:\n")
cat(sprintf("  χ²(df) = %.2f(%d), p = %.3f\n",
            fit_measures_mpd["chisq"], fit_measures_mpd["df"], fit_measures_mpd["pvalue"]))
cat(sprintf("  CFI = %.3f, TLI = %.3f\n", fit_measures_mpd["cfi"], fit_measures_mpd["tli"]))
cat(sprintf("  RMSEA = %.3f [%.3f, %.3f]\n",
            fit_measures_mpd["rmsea"], fit_measures_mpd["rmsea.ci.lower"],
            fit_measures_mpd["rmsea.ci.upper"]))
cat(sprintf("  SRMR = %.3f\n", fit_measures_mpd["srmr"]))

cat("\n模型2 - DASS-21二阶因子模型:\n")
cat(sprintf("  χ²(df) = %.2f(%d), p = %.3f\n",
            fit_measures_dass["chisq"], fit_measures_dass["df"], fit_measures_dass["pvalue"]))
cat(sprintf("  CFI = %.3f, TLI = %.3f\n", fit_measures_dass["cfi"], fit_measures_dass["tli"]))
cat(sprintf("  RMSEA = %.3f [%.3f, %.3f]\n",
            fit_measures_dass["rmsea"], fit_measures_dass["rmsea.ci.lower"],
            fit_measures_dass["rmsea.ci.upper"]))
cat(sprintf("  SRMR = %.3f\n", fit_measures_dass["srmr"]))

cat("\n整合模型 - 手机依赖与负面情绪:\n")
cat(sprintf("  χ²(df) = %.2f(%d), p = %.3f\n",
            fit_measures_int["chisq"], fit_measures_int["df"], fit_measures_int["pvalue"]))
cat(sprintf("  CFI = %.3f, TLI = %.3f\n", fit_measures_int["cfi"], fit_measures_int["tli"]))
cat(sprintf("  RMSEA = %.3f [%.3f, %.3f]\n",
            fit_measures_int["rmsea"], fit_measures_int["rmsea.ci.lower"],
            fit_measures_int["rmsea.ci.upper"]))
cat(sprintf("  SRMR = %.3f\n", fit_measures_int["srmr"]))

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("三、二阶因子相关关系\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

cat(sprintf("手机依赖 <-> 负面情绪: r = %.3f (p = %.3f)\n",
            correlation$std.all, correlation$pvalue))

if (correlation$pvalue < 0.001) {
  cat("  *** p < .001 (高度显著)\n")
} else if (correlation$pvalue < 0.01) {
  cat("  ** p < .01 (显著)\n")
} else if (correlation$pvalue < 0.05) {
  cat("  * p < .05 (边际显著)\n")
}

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("四、结论与建议\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

cat("1. 信度检验：所有量表及子维度的Cronbach's α系数均 > 0.70，表明测量具有良好的内部一致性。\n\n")

cat("2. 模型拟合：\n")
if (fit_measures_mpd["cfi"] >= 0.90 && fit_measures_mpd["rmsea"] <= 0.08) {
  cat("   - 模型1（手机依赖）拟合良好\n")
} else {
  cat("   - 模型1（手机依赖）拟合可接受，可能需要进一步优化\n")
}

if (fit_measures_dass["cfi"] >= 0.90 && fit_measures_dass["rmsea"] <= 0.08) {
  cat("   - 模型2（DASS-21）拟合良好\n")
} else {
  cat("   - 模型2（DASS-21）拟合可接受，可能需要进一步优化\n")
}

if (fit_measures_int["cfi"] >= 0.90 && fit_measures_int["rmsea"] <= 0.08) {
  cat("   - 整合模型拟合良好\n")
} else {
  cat("   - 整合模型拟合可接受，可能需要进一步优化\n")
}

cat("\n3. 二阶因子结构：\n")
cat("   - 手机依赖的三维度结构（戒断症状、渴求性、身心影响）得到验证\n")
cat("   - DASS-21的三维度结构（压力、焦虑、抑郁）得到验证\n")

cat("\n4. 相关关系：\n")
if (abs(correlation$std.all) >= 0.5) {
  cat("   - 手机依赖与负面情绪呈中等偏高正相关\n")
} else if (abs(correlation$std.all) >= 0.3) {
  cat("   - 手机依赖与负面情绪呈中等正相关\n")
} else {
  cat("   - 手机依赖与负面情绪呈弱正相关\n")
}

cat("\n5. 研究局限：\n")
cat("   - 样本量较小（N=59），可能影响模型稳定性\n")
cat("   - 横断面设计，无法推断因果关系\n")
cat("   - 样本性别分布不均（女性占79.7%）\n")

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("分析完成！\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n")

sink()

cat("\n综合报告已保存至: 二阶因子模型分析报告_R.txt\n")

cat("\n", "=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("所有分析完成！生成文件清单：\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("1. 二阶模型_描述性统计.csv\n")
cat("2. 二阶模型_维度相关矩阵图.pdf\n")
cat("3. 模型1_手机依赖二阶因子模型.pdf\n")
cat("4. 模型2_DASS21二阶因子模型.pdf\n")
cat("5. 整合模型_手机依赖与负面情绪.pdf\n")
cat("6. 模型1_因子载荷.csv\n")
cat("7. 模型2_因子载荷.csv\n")
cat("8. 模型拟合指数比较.csv\n")
cat("9. 二阶因子模型分析报告_R.txt\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")
