import pandas as pd

# 路径
train_path = r"E:\USELESS\数据分析学习\数分竞赛学习\CS-competition\data\train.csv"
test_path  = r"E:\USELESS\数据分析学习\数分竞赛学习\CS-competition\data\test.csv"

# 删训练集
df_train = pd.read_csv(train_path, encoding="utf-8-sig")
df_train = df_train[df_train["股票代码"] != 600930].copy()
df_train.to_csv(train_path, index=False, encoding="utf-8-sig")

# 删测试集
df_test = pd.read_csv(test_path, encoding="utf-8-sig")
df_test = df_test[df_test["股票代码"] != 600930].copy()
df_test.to_csv(test_path, index=False, encoding="utf-8-sig")

print("✅ 已彻底删除 华电新能 (600930)")
print("训练集股票数：", df_train["股票代码"].nunique())
print("测试集股票数：", df_test["股票代码"].nunique())