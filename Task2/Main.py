import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
from tqdm import tqdm

data = pd.read_csv('sample_data.txt', sep=';', low_memory=False)
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
data.drop(['Date', 'Time'], axis=1, inplace=True)

for col in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna().reset_index(drop=True)

X = data.drop(['Global_active_power', 'Datetime'], axis=1)
y = data['Global_active_power']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

sample_size = int(0.1 * X_train.shape[0])  

if sample_size > 0:
    X_train = X_train[:sample_size]  
    y_train = y_train.iloc[:sample_size]
else:
    print("Error: size")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Neural Network Regressor": MLPRegressor(random_state=42, max_iter=100)
}

for model_name, model in models.items():
    print('')
    with tqdm(total=1, desc=f"Training {model_name}", position=0, leave=True) as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name}:")
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}')

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Cross-validated MAE: {np.mean(np.abs(cv_scores))}\n')

print("success!")
