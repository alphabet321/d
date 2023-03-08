import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('linear-reg.csv')
X = data['SAT']
Y = data['GPA']
data = np.array(data)
np.random.shuffle(data)
trainingSplit = 0.8
trainingSize = int(trainingSplit * len(data))
XTrain = data[:trainingSize,0]
YTrain = data[:trainingSize,1]
XTest = data[trainingSize:,0]
YTest = data[trainingSize:,1]
plt.scatter(X,Y)
plt.show()

#The 2 in the polyfit function is the degree of the polynomial - use 1 for straight line, 2 for quadratic, 3 for cubic, etc.
model = np.poly1d(np.polyfit(X,Y,2))
max_x = np.max(X) + 100
min_x = np.min(X) - 100
line = np.linspace(min_x, max_x, 1000) # Why 1000????????
plt.scatter(X,Y)
plt.plot(line, model(line), color='red')
plt.show()
#Sum of Squares Error calculation
error = 0
for i in range(len(XTrain)):
    error += (YTrain[i] - model(XTrain[i]))**2

print(error)