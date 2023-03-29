import pandas as pd
import plotly.express as px

df = pd.read_csv("data2.csv")

velocity=df["Velocity"].tolist()
escaped=df["Escaped"].tolist()

fig=px.scatter(x=velocity, y=escaped)
fig.show()

import numpy as np
velocity_array=np.array(velocity)
escaped_array=np.array(escaped)

m, c= np.polyfit(velocity_array, escaped_array, 1)

y=[]

for x in velocity_array:
  y_value=m*x+c
  y.append(y_value)

fig=px.scatter(x=velocity_array, y=escaped_array)
fig.update_layout(shapes=[dict(
    type='line',
    y0=min(y), y1=max(y),
    x0=min(velocity_array), x1=max(velocity_array)
)])
fig.show()


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.reshape(velocity, (len(velocity), 1))
Y = np.reshape(escaped, (len(escaped), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

#Using the line formula 
X_test = np.linspace(0, 100, 200)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

#do hit and trial by changing the vlaue of X_test here.
plt.axvline(x=X_test[23], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(0, 30)
plt.show()
