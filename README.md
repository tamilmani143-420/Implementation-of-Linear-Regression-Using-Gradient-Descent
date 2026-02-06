# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and prepare the dataset (select Population and Profit, scale values).

2.Initialize parameters (X, y, and theta).

3.Apply gradient descent to update theta and minimize cost.

4.Predict results and visualize the regression line with cost reduction.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("50_Startups.csv")
data = data.iloc[:, [0, 4]]
data.columns = ["Population", "Profit"]

data["Population"] = (data["Population"] - data["Population"].mean()) / data["Population"].std()

plt.scatter(data["Population"], data["Profit"])
plt.xlabel("Scaled Population")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()

def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    return (1/(2*m)) * np.sum((h - y)**2)

m = len(data)
X_raw = data["Population"].values.reshape(m, 1)
X = np.append(np.ones((m, 1)), X_raw, axis=1)
y = data["Profit"].values.reshape(m, 1)
theta = np.zeros((2, 1))

print("Initial Cost:", computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.T, (predictions - y))
        theta -= alpha * (1/m) * error
        j_history.append(computeCost(X, y, theta))
    return theta, j_history

theta, j_history = gradientDescent(X, y, theta, 0.01, 1500)

print(f"Model: h(x) = {round(theta[0,0],2)} + {round(theta[1,0],2)}x")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\\Theta)$")
plt.title("Cost Function Reduction")
plt.show()

plt.scatter(data["Population"], data["Profit"])
x_line = np.linspace(data["Population"].min(), data["Population"].max(), 100)
y_line = theta[0,0] + theta[1,0] * x_line
plt.plot(x_line, y_line, color="r")
plt.xlabel("Scaled Population")
plt.ylabel("Profit ($10,000)")
plt.title("Linear Regression Fit")
plt.show()
```

## Output:
<img width="749" height="613" alt="Screenshot 2026-02-06 150339" src="https://github.com/user-attachments/assets/1c648da2-d3a4-48c5-8818-1d12ff90c0f9" />
<img width="701" height="570" alt="Screenshot 2026-02-06 150355" src="https://github.com/user-attachments/assets/3f25ed2b-fc4b-493d-8f1c-20dee05d2c74" />
<img width="747" height="568" alt="Screenshot 2026-02-06 150433" src="https://github.com/user-attachments/assets/68ad1d31-291a-4dcf-97ee-7e84b9530e99" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
