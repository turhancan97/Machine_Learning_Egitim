
"""
@author: turhancankargin
"""

#Kütüphane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yukleme
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

#veri on isleme
boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

#eksik veriler (eksik verinin olduğu yere sütünün ortalamasını koyma)
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

    
    
    
    
    
    
    
    
    
    
    
    

