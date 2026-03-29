# #bài 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Đọc dữ liệu
df = pd.read_csv(r"C:\Users\FPT\OneDrive\Desktop\ITA105\lab3\ITA105_Lab_3_Sports.csv")
df = df.select_dtypes(include='number')

# 2. Kiểm tra dữ liệu
print("Missing:\n", df.isnull().sum())
print("\nDescribe:\n", df.describe())

# 3. Histogram + Boxplot (tất cả cột cùng lúc)
df.hist(figsize=(10,6))
plt.suptitle("Histogram - Original")
plt.show()

df.plot(kind='box', subplots=True, layout=(2,3), figsize=(10,6))
plt.suptitle("Boxplot - Original")
plt.show()

# 4. Chuẩn hóa
minmax = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
zscore = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

# 5. Histogram sau chuẩn hóa
minmax.hist(figsize=(10,6))
plt.suptitle("Histogram - MinMax")
plt.show()

zscore.hist(figsize=(10,6))
plt.suptitle("Histogram - Z-score")
plt.show()

# 6. So sánh (gộp tất cả cột)
plt.figure(figsize=(8,5))
plt.hist(df.values.flatten(), alpha=0.5, label="Original")
plt.hist(minmax.values.flatten(), alpha=0.5, label="MinMax")
plt.hist(zscore.values.flatten(), alpha=0.5, label="Z-score")
plt.legend()
plt.title("Overall Comparison")
plt.show()















#bài 2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Đọc dữ liệu
df = pd.read_csv(r"C:\Users\FPT\OneDrive\Desktop\ITA105\lab3\ITA105_Lab_3_Health.csv")
df = df.select_dtypes(include='number')
# Thống kê
print(df.isnull().sum(), "\n", df.describe())

# Histogram cho tất cả biến
df.hist(figsize=(8,6))
plt.show()

# Boxplot cho tất cả biến
df.plot(kind='box')
plt.show()

# Phát hiện outliers (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).sum()
print("\nOutliers:\n", outliers)

# Chuẩn hóa
mm = MinMaxScaler().fit_transform(df)
zs = StandardScaler().fit_transform(df)

# So sánh (vẽ tất cả cùng lúc)
pd.DataFrame(mm, columns=df.columns).hist(figsize=(8,6))
plt.suptitle("Min-Max")
plt.show()

pd.DataFrame(zs, columns=df.columns).hist(figsize=(8,6))
plt.suptitle("Z-score")
plt.show()








#bài 3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Đọc dữ liệu 
df = pd.read_csv(r"C:\Users\FPT\OneDrive\Desktop\ITA105\lab3\ITA105_Lab_3_Finance.csv")
df = df.select_dtypes(include='number')
# 2. Thống kê + boxplot
print(df.describe())
df.plot(kind='box', title="Boxplot - Company Data")
plt.show()


x = df["doanh_thu_musd"]
y = df["loi_nhuan_musd"]

# scatter gốc
plt.scatter(x, y)
plt.title("Original")
plt.xlabel("Revenue"); plt.ylabel("Profit")
plt.show()

# chuẩn hóa
mm = MinMaxScaler().fit_transform(df[["doanh_thu_musd","loi_nhuan_musd"]])
zs = StandardScaler().fit_transform(df[["doanh_thu_musd","loi_nhuan_musd"]])

# scatter Min-Max
plt.scatter(mm[:,0], mm[:,1])
plt.title("Min-Max")
plt.show()

# scatter Z-score
plt.scatter(zs[:,0], zs[:,1])
plt.title("Z-score")
plt.show()






# bài 4
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Đọc dữ liệu
df = pd.read_csv(r"C:\Users\FPT\OneDrive\Desktop\ITA105\lab3\ITA105_Lab_3_Gaming.csv")
df = df.select_dtypes(include='number')
# 2. Kiểm tra dữ liệu
print("Missing:\n", df.isnull().sum())
print("\nDescribe:\n", df.describe())

# 3. Histogram dữ liệu gốc
df.hist(figsize=(8,6))
plt.suptitle("Original Data")
plt.show()

# 4. Chuẩn hóa
mm = MinMaxScaler().fit_transform(df)
zs = StandardScaler().fit_transform(df)

# 5. Histogram sau chuẩn hóa
pd.DataFrame(mm, columns=df.columns).hist(figsize=(8,6))
plt.suptitle("Min-Max")
plt.show()

pd.DataFrame(zs, columns=df.columns).hist(figsize=(8,6))
plt.suptitle("Z-score")
plt.show()

