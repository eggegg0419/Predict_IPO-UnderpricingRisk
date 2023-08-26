# logistic

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from collections import Counter

# 加载数据集和预处理
data = pd.read_csv("D.csv")
X = data.drop(["y"], axis=1)  # 特征
y = data["y"]  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 过采样
oversampler = RandomOverSampler(sampling_strategy=1)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_scaled, y_train)

# 构建逻辑回归模型
logistic_model = LogisticRegression(random_state=42)

# 训练模型
logistic_model.fit(X_train_resampled, y_train_resampled)

# 预测测试集
y_pred = logistic_model.predict(X_test_scaled)

# 模型准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)

# F1 分数
f1 = f1_score(y_test, y_pred)
print("模型F1分数:", f1)

# AUC
y_pred_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]  # 获取正类的预测概率
auc = roc_auc_score(y_test, y_pred_prob)
print("模型AUC:", auc)

# 使用statsmodels获取统计信息
X_train_resampled = sm.add_constant(X_train_resampled)  # 加入常数列
logit_model = sm.Logit(y_train_resampled, X_train_resampled)
result = logit_model.fit()

# 创建系数表格
coef_summary = result.summary2().tables[1]

print(coef_summary)
