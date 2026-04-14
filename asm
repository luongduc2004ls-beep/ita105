# =====================================================
# ===== 0. IMPORT =====================================
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import plotly.express as px
import joblib


# =====================================================
# ===== 1. LOAD DATA ==================================
# =====================================================
df = pd.read_csv("data.csv")


# =====================================================
# ===== 2. DATA UNDERSTANDING =========================
# =====================================================
print("=== BASIC STATISTICS ===")
print(df.describe())
print("\nMissing:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())


# =====================================================
# ===== 3. DATA CLEANING ==============================
# =====================================================
df['gia'] = df['gia'].fillna(df['gia'].median())
df['quan'] = df['quan'].fillna(df['quan'].mode()[0])

df = df[df['gia'] > 0]
df = df[df['so_phong'] > 0]

df['quan'] = df['quan'].str.lower().str.strip()
df = df.drop_duplicates()


# =====================================================
# ===== 4. VISUALIZATION ==============================
# =====================================================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
df['gia'].hist()
plt.title("Raw Price")

plt.subplot(1,3,2)
sns.boxplot(x=df['gia'])

plt.subplot(1,3,3)
sns.violinplot(x=df['so_phong'], y=df['gia'])

plt.show()


# =====================================================
# ===== 5. OUTLIER ====================================
# =====================================================
Q1 = df['gia'].quantile(0.25)
Q3 = df['gia'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['gia'] >= Q1 - 1.5*IQR) & (df['gia'] <= Q3 + 1.5*IQR)]

z = (df['gia'] - df['gia'].mean()) / df['gia'].std()
df = df[np.abs(z) < 3]


# =====================================================
# ===== 6. FEATURE ENGINEERING ========================
# =====================================================

# TEXT
df['mo_ta'] = df['mo_ta'].astype(str)

df['text_len'] = df['mo_ta'].apply(lambda x: len(x.split()))
df['sentiment'] = df['mo_ta'].apply(lambda x: TextBlob(x).sentiment.polarity)

for kw in ['luxury', 'cozy', 'modern']:
    df[f'kw_{kw}'] = df['mo_ta'].str.contains(kw, case=False).astype(int)

# TIME
if 'sale_date' in df.columns:
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['month'] = df['sale_date'].dt.month
    df['quarter'] = df['sale_date'].dt.quarter

# INTERACTION
df['area_room_interaction'] = df['dien_tich'] * df['so_phong']

# KPI
df['price_per_m2'] = df['gia'] / (df['dien_tich'] + 1)


# =====================================================
# ===== 7. SKEW (KHÔNG ĐỤNG TARGET) ==================
# =====================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if 'gia' in numeric_cols:
    numeric_cols.remove('gia')

skewed_cols = [col for col in numeric_cols if abs(skew(df[col])) > 0.75]

if len(skewed_cols) > 0:
    pt = PowerTransformer(method='yeo-johnson')
    df[skewed_cols] = pt.fit_transform(df[skewed_cols])


# =====================================================
# ===== 8. TF-IDF =====================================
# =====================================================
tfidf = TfidfVectorizer(max_features=20)
tfidf_matrix = tfidf.fit_transform(df['mo_ta']).toarray()

tfidf_df = pd.DataFrame(tfidf_matrix, index=df.index)

# detect duplicate text
cos_sim = cosine_similarity(tfidf_matrix)
duplicates = [(i,j) for i in range(len(cos_sim)) for j in range(i+1,len(cos_sim)) if cos_sim[i][j]>0.8]
print("Text duplicates:", duplicates)

df = pd.concat([df, tfidf_df], axis=1)

#  REMOVE TEXT
df = df.drop(columns=['mo_ta'])


# =====================================================
# ===== 9. ENCODING ===================================
# =====================================================
df = pd.get_dummies(df, columns=['quan'])

# FIX BOOL → INT
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)


# =====================================================
# ===== 10. DROP DATETIME =============================
# =====================================================
if 'sale_date' in df.columns:
    df = df.drop(columns=['sale_date'])


# =====================================================
# ===== 11. FIX COLUMN NAME ===========================
# =====================================================
df.columns = df.columns.astype(str)


# =====================================================
# ===== 12. SPLIT =====================================
# =====================================================
X = df.drop('gia', axis=1)
y = np.log1p(df['gia'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# =====================================================
# ===== 13. MODEL COMPARISON ==========================
# =====================================================
models = {
    "Linear": LinearRegression(),
    "RF": RandomForestRegressor(),
    "GB": GradientBoostingRegressor(),
    "XGB": XGBRegressor()
}

print("\n=== MODEL RESULTS ===")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")


# =====================================================
# ===== 14. PIPELINE ROBUST ===========================
# =====================================================
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

final_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBRegressor())
])

final_pipeline.fit(X_train, y_train)

joblib.dump(final_pipeline, "house_price_pipeline.pkl")


# =====================================================
# ===== 15. MINI DASHBOARD ============================
# =====================================================
px.histogram(df, x="gia", title="Price Distribution").show()
px.histogram(df, x=np.log1p(df["gia"]), title="Log Price").show()

plt.figure()
sns.boxplot(x=df['gia'])
plt.title("Outliers")
plt.show()


# =====================================================
# ===== DONE ==========================================
# =====================================================
print("\n DONE - FULL PIPELINE RUN SUCCESSFULLY")
print("Final shape:", df.shape)
