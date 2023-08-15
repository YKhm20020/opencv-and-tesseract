import pandas as pd
from sklearn.model_selection import train_test_split

features = pd.read_csv("bert/data/features.csv")
labels = pd.read_csv("bert/data/labels.csv")

# データの分割（テストデータを30%に設定）
x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2)

# 分割したデータの保存
x_train.to_csv("bert/data/trains/features_train.csv", index=False)
x_test.to_csv("bert/data/tests/features_test.csv", index=False)
y_train.to_csv("bert/data/trains/labels_train.csv", index=False)
y_test.to_csv("bert/data/tests/labels_test.csv", index=False)