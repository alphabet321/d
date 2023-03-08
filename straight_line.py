import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Reading Data
data = pd.read_csv('linear-reg.csv')
print(data.shape) # print number of rows and column
print(data.head()) # to print remaining data
#Computing data
X = data['SAT'].values[:43] # training set 43 not included
Y = data['GPA'].values[:43]
X_test = data['SAT'].values[43:] # test set
Y_test = data['GPA'].values[43:]
mean_x = np.mean(X)  
mean_y = np.mean(Y)

n = len(X)
print(data)
#finding m and c
numer = 0
denom = 0
for i in range(n):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = numer/denom
c = mean_y - (m*mean_x)

print("m = ", m)
print("c = ", c)
# Plotting Values and Regression Line
 
max_x = np.max(X) + 100
min_x = np.min(X) - 100
print(max_x,min_x)

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x

# Ploting Line
plt.plot(x, y, color='#00FF00', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#FF0000', label='Scatter Plot')
 
plt.xlabel('SAT scores')
plt.ylabel('GPA in college')
plt.legend()
plt.show()
#plotting with test data
max_x = np.max(X) + 100
min_x = np.min(X) - 100
 
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x
 
# Ploting Line
plt.plot(x, y, color='#00FF00', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X_test, Y_test, c='#FF0000', label='Scatter Plot')
 
plt.xlabel('SAT scores')
plt.ylabel('GPA in college')
plt.legend()
plt.show()
#Calculating error

error = 0
n = len(X_test)
for i in range(n):
  pred = m * X_test[i] + c
  y = Y_test[i]
  error+= (pred - y)**2

print("Sum of squares error on test data is: ", error)