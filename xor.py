from keras.models import Sequential
from keras.layers import Dense
X_train = [[0,0], [0,1], [1,0], [1,1]]
y_train = [0, 1, 1, 0]
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, batch_size=2)
X_test = [[0,0], [0,1], [1,0], [1,1]]
y_test = [0, 1, 1, 0]
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy*100))

# Use the trained model to make predictions
X_new = [[1,1]]
y_new = model.predict(X_new)

print("Input: {}".format(X_new))
print("Predicted output: {}".format(y_new))
