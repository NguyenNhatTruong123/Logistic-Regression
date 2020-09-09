import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

data=pd.read_csv("english_score.csv")
X=np.array(data["TOEFL Score"].values)
Y=np.array(data["GRE Score"].values)

one=np.ones((X.shape[0],1))
x=np.concatenate((one,X.reshape(X.shape[0],1)),axis=1)
w=np.dot(np.linalg.pinv(np.dot(x.T,x)),np.dot(x.T,Y.reshape(Y.shape[0],1)))

y_hat=w[0]+w[1]*X

plt.scatter(X,Y)
plt.plot(X,y_hat)
plt.show()

reg=LinearRegression(fit_intercept=False)
reg.fit(x,Y)
slope,intercept,r,p,std_err=stats.linregress(X,Y)

print(reg.coef_)
print(w)

SSE=0
SST=0
SSR=0
for i in range(Y.shape[0]):
    y_pred=intercept+slope*X[i]
    SSE=SSE+math.pow(Y[i]-y_pred,2)
    SST=SST+math.pow(Y[i]-np.mean(Y),2)
    SSR=SSR+math.pow(y_pred-np.mean(Y),2)
print(1-SSE/SST)
print(SSR/SST)


