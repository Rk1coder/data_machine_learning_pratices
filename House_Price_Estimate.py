import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

veri_kumesi = pd.read_csv('ev_fiyat_veri.csv')


X = veri_kumesi.drop('fiyat', axis=1)
y = veri_kumesi['fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
print('Eğitim veri seti RMSE:', train_rmse)

test_predictions = model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
print('Test veri seti RMSE:', test_rmse)