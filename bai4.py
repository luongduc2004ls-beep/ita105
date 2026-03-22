import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =======================
# 1. Load dữ liệu
# =======================
housing = pd.read_csv(r'C:\Users\FPT\OneDrive\Desktop\ITA105\lab2\ITA105_Lab_2_Housing.csv')
iot = pd.read_csv(r'C:\Users\FPT\OneDrive\Desktop\ITA105\lab2\ITA105_Lab_2_Iot.csv')
ecom = pd.read_csv(r'C:\Users\FPT\OneDrive\Desktop\ITA105\lab2\\ITA105_Lab_2_Ecommerce.csv')

# =======================
# 2. Hàm IQR
# =======================
def detect_outliers_iqr(df, cols):
    outlier_mask = pd.Series([False]*len(df))
    
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
        outlier_mask = outlier_mask | mask
    
    return outlier_mask

# =======================
# 3. Hàm Z-score (optional)
# =======================
def detect_outliers_zscore(df, cols, threshold=3):
    outlier_mask = pd.Series([False]*len(df))
    
    for col in cols:
        z = (df[col] - df[col].mean()) / df[col].std()
        mask = abs(z) > threshold
        outlier_mask = outlier_mask | mask
    
    return outlier_mask

# =======================
# 4. Detect outliers
# =======================

# Housing
housing_iqr = detect_outliers_iqr(housing, ['dien_tich', 'gia'])
housing_z = detect_outliers_zscore(housing, ['dien_tich', 'gia'])

# IoT
iot_iqr = detect_outliers_iqr(iot, ['temperature', 'pressure'])
iot_z = detect_outliers_zscore(iot, ['temperature', 'pressure'])

# Ecommerce
ecom_iqr = detect_outliers_iqr(ecom, ['price', 'quantity', 'rating'])
ecom_z = detect_outliers_zscore(ecom, ['price', 'quantity', 'rating'])

# =======================
# 5. Vẽ biểu đồ
# =======================

def plot_outliers(df, x, y, mask, title):
    plt.figure()
    plt.scatter(df[x], df[y], label='Normal')
    plt.scatter(df[mask][x], df[mask][y], label='Outliers')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.show()

# Housing
plot_outliers(housing, 'dien_tich', 'gia', housing_iqr, 'Housing (IQR)')

# IoT
plot_outliers(iot, 'temperature', 'pressure', iot_iqr, 'IoT (IQR)')

# Ecommerce (2D projection)
plot_outliers(ecom, 'price', 'quantity', ecom_iqr, 'E-commerce (IQR)')

# =======================
# 6. Kết quả
# =======================

print("=== SỐ OUTLIERS ===")
print("Housing (IQR):", housing_iqr.sum())
print("Housing (Z-score):", housing_z.sum())

print("IoT (IQR):", iot_iqr.sum())
print("IoT (Z-score):", iot_z.sum())

print("E-commerce (IQR):", ecom_iqr.sum())
print("E-commerce (Z-score):", ecom_z.sum())

# IQR thường phát hiện nhiều outlier hơn khi dữ liệu lệch (skewed)
# Z-score phù hợp khi dữ liệu gần phân phối chuẩn
# Multivariate giúp phát hiện outlier thực tế hơn so với univariate