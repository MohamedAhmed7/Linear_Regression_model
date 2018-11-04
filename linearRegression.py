import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv('Boston.csv')

#print(data.head())
# just pick 2 parameters as x and y for linear regression
# crim zn indus chas nox rm  age dis rad tax ptratio b lstat   medv

x = np.array(data['medv'])
y = np.array(data['rm'])

n = np.size(x)
# 80% are training set
x_training = np.array(x[0: int(0.8 * n) + 1])
y_training = np.array(y[0: int(0.8 * n) + 1])
# 20% for testing the model
x_test = np.array(x[int(-0.2 * n):])
y_test = np.array(y[int(-0.2 * n):])

#our linear regression model
#finding the coffecients
#y = m x + c
def find_coef(x, y):
    n = np.size(x)
    m = (np.mean(x) * np.mean(y) - np.mean(x * y)) / (np.mean(x) * np.mean(x)  - np.mean(x ** 2))
    c = np.mean(y) - m * np.mean(x)

    return (m, c)


m,c = find_coef(x_training, y_training)

regression_line = [(m * i) + c for i in x_training]

y_predict = np.array([(m * i) + c for i in x_test])

#claculate errors
mean_error = np.mean(sum((y_predict - y_test) ** 2))

#drawig results

#drawing the training set and the model
plt.plot(x, y, 'g.', label = 'data') # draw data as points
plt.plot(x_training, regression_line, 'r-', label = 'Linear Regression') #draw our line
plt.plot(x_test, y_test, 'bx' , label='testing 20%')

#drwing features
plt.title('Simple Linear Regression')
plt.xlabel('parameter 1')
plt.ylabel('parameter 2')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()

#Statsmodel
X = sm.add_constant(x)
Y = y
mod = sm.OLS(Y, X)
res = mod.fit()

#print results
print("Model results : \n",find_coef(x_training, y_training))
#print("Mean Error : ", mean_error)

#statsmodel
print("Statsmodel results : \n",res.params[::-1])

#print(res.summary())



