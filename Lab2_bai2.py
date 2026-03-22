import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dữ liệu và set index [cite: 49]
df_iot = pd.read_csv(r"C:\Users\FPT\OneDrive\Desktop\ITA105\lab2\ITA105_Lab_2_Iot.csv")
df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'])
df_iot.set_index('timestamp', inplace=True)

# 2. Vẽ line plot temperature cho từng sensor [cite: 50]
for sensor in df_iot['sensor_id'].unique():
    df_iot[df_iot['sensor_id'] == sensor]['temperature'].plot(label=sensor)
plt.legend()
plt.show()

# 3. Rolling mean ± 3 x std (window = 10) [cite: 51]
window = 10
rolling_mean = df_iot['temperature'].rolling(window=window).mean()
rolling_std = df_iot['temperature'].rolling(window=window).std()
upper_bound = rolling_mean + (3 * rolling_std)
lower_bound = rolling_mean - (3 * rolling_std)

outliers_rolling = df_iot[(df_iot['temperature'] > upper_bound) | (df_iot['temperature'] < lower_bound)]

# 7. Xử lý bằng interpolation 
df_iot['temperature_fixed'] = df_iot['temperature'].where(~df_iot.index.isin(outliers_rolling.index))
df_iot['temperature_fixed'] = df_iot['temperature_fixed'].interpolate(method='linear')