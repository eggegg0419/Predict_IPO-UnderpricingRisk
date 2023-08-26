# Random forest

import numpy as np
import pandas as pd
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_graphviz

# 预处理
data = pd.read_csv("D.csv")
X = data.drop(["y","是否为科创版"], axis=1)
y = data["y"] 

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 显示原始类别分布
print("原始类别分布:", Counter(y_train))

# 过采样
oversampler = RandomOverSampler(sampling_strategy=1)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
print("过采样后的类别分布:", Counter(y_train_resampled))

# 随机森林模型
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=50)

# 训练模型
random_forest_model.fit(X_train_resampled, y_train_resampled)

# 预测测试集
y_pred = random_forest_model.predict(X_test)

# 准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)

# 测试集中 y=1 的预测准确率
y_test_positive = y_test[y_test == 1]
y_pred_positive = y_pred[y_test == 1]
accuracy_positive = accuracy_score(y_test_positive, y_pred_positive)
print("测试集中 y=1 的预测准确率:", accuracy_positive)

y_test_negative = y_test[y_test == 0]
y_pred_negative = y_pred[y_test == 0]
error_rate_negative = 1 - accuracy_score(y_test_negative, y_pred_negative)
print("测试集中 y=0 的预测错误率:", error_rate_negative)

# F1 分数
f1 = f1_score(y_test, y_pred)
print("模型F1分数:", f1)

# 重要性
feature_importance = random_forest_model.feature_importances_

# 创建特征重要性的DataFrame
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# AUC
y_pred_prob = random_forest_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
auc = roc_auc_score(y_test, y_pred_prob)
print("模型AUC:", auc)

# 获取随机森林中的第一棵树
first_tree = random_forest_model.estimators_[0]

# 导出树的可视化表示
dot_data = export_graphviz(first_tree, out_file=None, 
                           feature_names=X.columns, 
                           class_names=["0", "1"],  # 根据你的类别标签调整
                           filled=True, rounded=True, special_characters=True)

# 使用Graphviz绘制树状图
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree")  # 可选：将树状图保存为文件
graph.view()  # 在默认浏览器中显示树状图





# 模型提升曲线(未完成，直接比较模型优劣)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 预处理
data = pd.read_csv("D.csv") 
X = data.drop("y", axis=1) 
y = data["y"] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=50)

# 随机森林模型
n_estimators_values = [45,90,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
accuracy_scores = []

for n_estimators in n_estimators_values:
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# 绘制提升曲线
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, accuracy_scores, marker='o')
plt.title("Random Forest Model Boosting Curve")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
