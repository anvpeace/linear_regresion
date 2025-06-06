import pandas as pd

data = pd.read_csv("celsius_.csv")

data.info()

data.head()

import seaborn as sb 

sb.scatterplot(x="celsius", y="fahrenheit", data=data, hue="fahrenheit", palette="coolwarm")


#characteristics (x), etiquite (y)
X = data[["celsius"]]
y = data[["fahrenheit"]]

y

X_processed = X.values.reshape(-1,1)
y_processed = y.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

model = LinearRegression();
model.fit(X_processed, y_processed)

celsius = 7900
prediction = model.predict([[celsius]])
print(f"{celsius} degrees celsius are {prediction} degrees fehrenheit ")

model.score(X_processed, y_processed)
