from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.model_selection import StratifiedKFold

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
print ("Loading dataset")
dataset = numpy.loadtxt("Dataset\pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvScores = []
print (X)
print (Y)
"""
# create model
print ("Creating model")
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
print ("Compiling model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
print ("Training model")
model.fit(X, Y, epochs=150, batch_size=10, verbose=2)

# evaluate the model
print ("Evaluating model")
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
"""