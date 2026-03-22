# bài 1
# 1,
import pandas as pd
import matplotlib.pyplot as plt

# đọc file
df = pd.read_csv(r"C:\Users\FPT\OneDrive\Desktop\ITA105\lab2\ITA105_Lab_2_Housing.csv")

# shape
print("Kích thước:", df.shape)

# # kiểm tra thiếu dữ liệu
print("\nMissing values:")
print(df.isnull().sum())
# #2,
print(df.describe())

# # median
print("\nMedian:")
print(df.median(numeric_only=True))
#3,
df.plot(kind='box', subplots=True, layout=(3,3), figsize=(10,8))
plt.show()
#4,

plt.scatter(df['dien_tich'], df['gia'])
plt.xlabel("Diện tích")
plt.ylabel("Giá")
plt.title("Diện tích vs Giá")
plt.show()
#5,
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outlier_iqr = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
print("Số outlier (IQR):")
print(outlier_iqr.sum())
#6,
import numpy as np

# chọn cột số
df_num = df.select_dtypes(include='number')

# tính Z-score thủ công
z_scores = np.abs((df_num - df_num.mean()) / df_num.std())

# xác định outlier
outliers_z = z_scores > 3

# đếm số lượng
print("Số lượng outlier theo Z-score:")
print(outliers_z.sum())
#8
# Do nhập sai (ví dụ giá quá lớn)
# Do thực tế (nhà biệt thự, vị trí đẹp)
#9
df_clean = df.copy()

for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df_clean[col] = df[col].clip(lower, upper)
#10
df_clean.plot(kind='box', subplots=True, layout=(3,3), figsize=(10,8))
plt.show()    









import numpy as np

df_sub = df[['area', 'price']]  # ví dụ Housing

z_scores = np.abs((df_sub - df_sub.mean()) / df_sub.std())

outliers_z = (z_scores > 3).any(axis=1)