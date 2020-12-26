import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

#<---- INICIO DO PREPROCESSAMENTO -------->

base = pd.read_csv("autos.csv", encoding = "ISO-8859-1" )
base = base.drop ("dateCrawled", axis = 1)
base = base.drop ("dateCreated", axis = 1)
base = base.drop ("nrOfPictures", axis = 1)
base = base.drop ("postalCode", axis = 1)
base = base.drop ("lastSeen", axis = 1)

#contador de itens e variaveis na base da
base["name"].value_counts()
base = base.drop ("name", axis = 1)

base = base.drop ("seller", axis = 1)
base = base.drop ("offerType", axis = 1)

#localização de dados inconsistentes i1 
i1 = base.loc[base.price <= 10]
base.price.mean()
#retirada de valores inconsistentes 
base = base[base.price > 10]
##localização de dados inconsistentes i2
i2 = base.loc[base.price > 350000]
#retirada de valores inconsistentes 
base = base [base.price < 350000 ]

#dados inexistentes e suas respectivas maiorias
base.loc[pd.isnull(base["vehicleType"])]
base["vehicleType"].value_counts() # limousine
base.loc[pd.isnull(base["gearbox"])]
base["gearbox"].value_counts() # manuell
base.loc[pd.isnull(base["model"])]
base["model"].value_counts() # golf
base.loc[pd.isnull(base["fuelType"])]
base["fuelType"].value_counts() # benzin
base.loc[pd.isnull(base["notRepairedDamage"])]
base["notRepairedDamage"].value_counts() # nein

#dicionario dos tipos e suas substituicoes ou maiorias
valores = {"vehicleType" :"limousine" , "gearbox" :
           "manuell","model" : "golf", "fuelType" : "benzin" ,
           "notRepairedDamage" : "nein"}
#substituicao da base de dados usando o dicionario
base = base.fillna(value = valores )

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:,10])

ct = ColumnTransformer([(['abtest','vehicleType','gearbox','model',
                          'fuelType','brand','notRepairedDamage'],
                         OneHotEncoder(),
                         [0,1,3,5,8,9,10])],
                         remainder='passthrough')
previsores = ct.fit_transform(previsores.tolist())

#<---- FIM DO PREPROCESSAMENTO -------->

regressor = Sequential()

regressor.add (Dense(units = 159, activation="relu",
                     input_dim = 316))
regressor.add (Dense(units = 159, activation="relu"))
regressor.add (Dense(units = 1 , activation="linear"))
regressor.compile (loss = "mean_absolute_error", optimizer ="adam",
                  metrics = ["mean_absolute_error"])

regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)



        




























