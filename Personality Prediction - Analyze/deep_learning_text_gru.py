from __future__ import print_function

import numpy
import json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold

seed = 3
numpy.random.seed(seed)
OOV_CHAR = 2
max_features = 2800
maxlen = 500
batch_size = 25
epochs = 10
validation_split=0.2

print("Loading dataset")
filename = r'Dataset\Personality Prediction\Keras\status_fb.json'
jsonFile = open(filename).read()
f = json.loads(jsonFile)
data = (f['X_train'])
labels = (f['Y_train']['openness'])
for i in range(0, len(data)):
    for j in range(0, len(data[i])):
        if (data[i][j] >= max_features):
            data[i][j] = OOV_CHAR

print("Cross Validation")
kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvValues = []
kFoldIdx = 1
for train, test in kFold.split(data, labels):
    print ("K-Fold: " + str(kFoldIdx))
    kFoldIdx += 1
#     print("Padding sequences")
    data = sequence.pad_sequences(data, maxlen=maxlen)
    labels = to_categorical(labels)
    
#     print("Splitting train/test")
#     indices = numpy.arange(250)
#     numpy.random.shuffle(indices)
#     data = data[indices]
#     labels = labels[indices]
#     val_sample = int(validation_split * len(data))
    
    x_train = data[train]
    y_train = labels[train]
    x_test = data[test]
    y_test = labels[test]
    
#     print("Creating model")
    # GRU
    model = Sequential()
    
#     print("Compiling model")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print (model.summary())
    
#     print("Training model")
    cvValues.append(model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2))
    
#     print("\nEvaluating model")
#     scores = model.evaluate(x_train, y_train, verbose=0)
#     print("%s: %f%%" % (model.metrics_names[0], scores[0]))
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     predictions = model.predict(x_test)
#     predicted = []
#     for x in predictions:
#         if(x[0] > x[1]): predicted.append(0)
#         else: predicted.append(1)
#     print(predicted)

cvLoss = []
cvAcc = []
for value in cvValues:
    cvLoss.append(value.history['val_loss'][epochs-1])
    cvAcc.append(value.history['val_acc'][epochs-1])
print("Validation Average Loss: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvLoss), numpy.std(cvLoss)))
print("Validation Average Accuracy: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvAcc), numpy.std(cvAcc)))