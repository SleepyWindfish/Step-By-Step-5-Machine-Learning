import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

y=y.reshape(len(y),1)
print(y)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)


from sklearn.svm import SVR
regression=SVR(kernel='rbf')
regression.fit(X,y)



plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color="red")
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regression.predict(X).reshape(-1,1)),color="blue")
plt.title("Truth or Bluff(SVR)")
plt.xlabel("Positional Level")
plt.ylabel("Salery")
plt.show()


