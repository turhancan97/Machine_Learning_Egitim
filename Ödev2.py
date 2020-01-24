#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
tahmin = lin_reg.predict(x)

import statsmodels.api as sm
r_ols = sm.OLS(tahmin, x)
r = r_ols.fit()
print(r.summary())




plt.scatter(x,y,color='red')
plt.plot(x,tahmin, color = 'blue')
plt.show()



print("Linear R2 degeri:")
print(r2_score(y, tahmin))