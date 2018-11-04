# Linear_Regression_model
# Implementing a simple Linear Regression Model in python Using Boston Dataset
Using least square error method to estimate the coefficients (slope, intercept)
and Comparing the results with the output from Statsmodel
<br>
slope = (np.mean(x) * np.mean(y) - np.mean(x * y)) / (np.mean(x) * np.mean(x)  - np.mean(x ** 2))<br>
intercept = np.mean(y) - slope * np.mean(x)
# Results 
![image](https://user-images.githubusercontent.com/19196061/47970259-ddbc9600-e08b-11e8-9b0e-3d33be494f23.png)

