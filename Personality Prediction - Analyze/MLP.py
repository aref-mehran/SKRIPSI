import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy

# Generate dummy data
# x_train = np.random.random((1000, 20))
# y_train = np.random.randint(2, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))

INPUT_LAYER = 125
HIDDEN_LAYER = 25
OUTPUT_LAYER = 5
BATCH_SIZE = 50
EPOCH = 1000

numpy.random.seed(7)

#dataset = np.loadtxt("data_splice_features_openness.csv", delimiter=",")
#dataset = np.loadtxt("dataset_splice_conscientiousness.csv", delimiter=",")
dataset = np.loadtxt("dataset_splice_openness.csv", delimiter=",")
x_train = dataset[:,0:74]
y_train = dataset[:,74]
  
x_test = dataset[:,0:74]
y_test = dataset[:,74]
 
model = Sequential()
model.add(Dense(INPUT_LAYER,init='uniform', input_dim=74, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(HIDDEN_LAYER,init='uniform', activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(OUTPUT_LAYER,init='uniform', activation='relu'))

model.add(Dense(1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
model.fit(x_train, y_train,
          epochs=EPOCH,
          batch_size=BATCH_SIZE)
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print("\n\n%s: %f%%" % (model.metrics_names[0], score[0])) 
print("%s: %f%%" % (model.metrics_names[1], score[1]*100))

# calculate predictions
predictions = model.predict(x_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)