# ==============================
# 1. Load dữ liệu & kiểm tra
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load dữ liệu
df = pd.read_csv(r"C:\Users\FPT\OneDrive\Desktop\ITA105\lab2\\ITA105_Lab_2_Ecommerce.csv")

# Xem dữ liệu
print("5 dòng đầu:")
print(df.head())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Thống kê mô tả
print("\nDescribe:")
print(df.describe())


# ==============================
# 2. Boxplot ban đầu
# ==============================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.boxplot(y=df['price'])
plt.title("Price")

plt.subplot(1,3,2)
sns.boxplot(y=df['quantity'])
plt.title("Quantity")

plt.subplot(1,3,3)
sns.boxplot(y=df['rating'])
plt.title("Rating")

plt.show()


# ==============================
# 3. IQR & Z-score
# ==============================

# Hàm IQR
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return data[(data[col] < lower) | (data[col] > upper)]

# Outliers theo IQR
out_price = detect_outliers_iqr(df, 'price')
out_quantity = detect_outliers_iqr(df, 'quantity')
out_rating = detect_outliers_iqr(df, 'rating')

print("\nSố outliers (IQR):")
print("price:", len(out_price))
print("quantity:", len(out_quantity))
print("rating:", len(out_rating))

# Z-score
z_scores = np.abs(zscore(df[['price','quantity','rating']]))
outliers_z = (z_scores > 3)

df_outliers_z = df[(outliers_z).any(axis=1)]
print("\nOutliers theo Z-score:")
print(df_outliers_z)


# ==============================
# 4. Scatterplot + đánh dấu outliers
# ==============================

# Tạo cột đánh dấu outlier price
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['outlier'] = (df['price'] < lower) | (df['price'] > upper)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='price', y='quantity', hue='outlier')
plt.title("Price vs Quantity (Outliers)")
plt.show()


# ==============================
# 5. Kiểm tra nguyên nhân
# ==============================
print("\nPrice = 0:")
print(df[df['price'] == 0])

print("\nRating > 5:")
print(df[df['rating'] > 5])

print("\nQuantity lớn:")
print(df[df['quantity'] > 50])

print("\nCategory distribution:")
print(df['category'].value_counts())


# ==============================
# 6. Xử lý ngoại lệ
# ==============================

df_clean = df.copy()

# Loại lỗi nhập liệu
df_clean = df_clean[df_clean['price'] > 0]
df_clean = df_clean[df_clean['rating'] <= 5]

# Clip giá trị lớn
df_clean['price'] = df_clean['price'].clip(
    lower=df_clean['price'].quantile(0.01),
    upper=df_clean['price'].quantile(0.99)
)

df_clean['quantity'] = df_clean['quantity'].clip(
    lower=df_clean['quantity'].quantile(0.01),
    upper=df_clean['quantity'].quantile(0.99)
)

# Log transform (optional)
df_clean['price_log'] = np.log1p(df_clean['price'])


# ==============================
# 7. Vẽ lại sau xử lý
# ==============================

# Boxplot sau xử lý
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.boxplot(y=df_clean['price'])
plt.title("Price (cleaned)")

plt.subplot(1,3,2)
sns.boxplot(y=df_clean['quantity'])
plt.title("Quantity (cleaned)")

plt.subplot(1,3,3)
sns.boxplot(y=df_clean['rating'])
plt.title("Rating (cleaned)")

plt.show()


# Scatterplot sau xử lý
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_clean, x='price', y='quantity')
plt.title("After Cleaning")
plt.show()


# ==============================
# Thống kê sau xử lý
# ==============================
print("\nDescribe sau xử lý:")
print(df_clean.describe())