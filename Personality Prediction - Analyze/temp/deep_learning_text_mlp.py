from __future__ import print_function

import numpy
import json
import os
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, Flatten
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import confusion_matrix

seed = 3
OOV_CHAR = 2
max_features = 2800
maxlen = 500
batch_size = 25
epochs = 20
validation_split=0.2
classWeight = {0:2.3, 1:1}

numpy.random.seed(seed)

print("Loading dataset")
filename = r'Dataset\Personality Prediction\Keras\status_fb.json'
jsonFile = open(filename).read()
f = json.loads(jsonFile)
data = (f['X_train'])
labels = (f['Y_train']['conscientiousness'])
for i in range(0, len(data)):
    for j in range(0, len(data[i])):
        if (data[i][j] >= max_features):
            data[i][j] = OOV_CHAR


# print("Cross Validation")
# kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# cvValues = []
# confusionMatrix = numpy.array([[0, 0], [0, 0]])
# kFoldIdx = 1
# for train, test in kFold.split(data, labels):
#     print ("K-Fold: " + str(kFoldIdx))
#     kFoldIdx += 1
#     print("Padding sequences")
data = sequence.pad_sequences(data, maxlen=maxlen)
labels = to_categorical(labels)
print("Splitting train/test")
indices = numpy.arange(250)
numpy.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
val_sample = int(validation_split * len(data))
 
x_train = data[:-val_sample]
y_train = labels[:-val_sample]
x_test = data[-val_sample:]
y_test = labels[-val_sample:]

#     print("Creating model")
# Simple MLP
model = Sequential()
model.add(Embedding(max_features, 50, input_length=maxlen))
model.add(Flatten())
model.add(Dense(50, init='glorot_uniform', activation='tanh'))
model.add(Dense(2, activation='softmax'))

#     print("Compiling model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print (model.summary())

#     print("Training model")
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# predictions = model.predict(x_test)
# predicted = []
# yTest = []
# for x in predictions:
#     if(x[0] > x[1]):
#         predicted.append(0)
#     else:
#         predicted.append(1)
# for y in y_test:
#     if(y[0] > y[1]):
#         yTest.append(0)
#     else:
#         yTest.append(1)
# confusionMatrix += confusion_matrix(yTest, predicted)
# print("\nEvaluating model")
# scores = model.evaluate(x_train, y_train, verbose=0)
# print("%s: %f%%" % (model.metrics_names[0], scores[0]))
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# predictions = model.predict(x_test)
# predicted = []
# for x in predictions:
#     if(x[0] > x[1]):
#         predicted.append(0)
#     else:
#         predicted.append(1)
# print(predicted)

# cvLoss = []
# cvAcc = []
# for value in cvValues:
#     cvLoss.append(value.history['val_loss'][epochs-1])
#     cvAcc.append(value.history['val_acc'][epochs-1])
# print("Validation Average Loss: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvLoss), numpy.std(cvLoss)))
# print("Validation Average Accuracy: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvAcc), numpy.std(cvAcc)))
# print("Confusion Matrix: " + str(confusionMatrix))


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
predictions = loaded_model.predict(x_test)
rounded = [round(x[1]) for x in predictions]
print(rounded)