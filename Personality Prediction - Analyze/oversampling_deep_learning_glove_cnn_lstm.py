from __future__ import print_function

import os
# import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, LSTM
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'Dataset/Personality Prediction/Keras/GloVe/6B/'
TEXT_DATA_DIR = BASE_DIR + 'Dataset/20 Newsgroup/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 2800 #20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

np.random.seed(8)

# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

def train(datasetFilename, classFilename):
    # second, prepare text samples and their labels
    print('Processing text dataset')
    texts = []  # list of text samples
    labels = []  # list of label ids
    dataset = open(datasetFilename).read() #Read file
    texts = dataset.split('#SEPARATOR#') #Split Status Dataset by each user
    labelClass = open(classFilename).read()
    labels = labelClass.split('\n')
    print('Found %s texts.' % len(texts))
    
    tempTexts = []
    tempLabels = []
    indices = np.arange(250)
    np.random.shuffle(indices)
    for i in range(0, 250):
        tempTexts.append(texts[indices[i]])
        tempLabels.append(labels[indices[i]])
    texts = tempTexts
    labels = tempLabels
        
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = []
    x_test = data[-num_validation_samples:]
    y_test = []
    startIdx = 0; finishIdx = 0.8 * data.shape[0]
    for i in range(int(startIdx), int(finishIdx)):
        y_train.append(labels[i])
    startIdx = data.shape[0] - 0.2 * data.shape[0]; finishIdx = data.shape[0]
    for i in range(int(startIdx), int(finishIdx)):
        y_test.append(labels[i])
     
    # Start of Oversampling #######################################################################################################
    y_train = np.reshape(y_train, (len(y_train)))
    y_test = np.reshape(y_test, (len(y_test)))
     
    # kind = ['regular', 'borderline1', 'borderline2', 'svm']
    kind = ['borderline1']
    sm = [SMOTE(kind=k) for k in kind]
    for method in sm:
        x_train_resampled, y_train_resampled = method.fit_sample(x_train, y_train)
        x_test_resampled, y_test_resampled = method.fit_sample(x_test, y_test)
         
    print ("x_train= " + str(x_train.shape) + " x_test= " + str(x_test.shape))
    print ("y_train= " + str(y_train.shape) + " y_test= " + str(y_test.shape))
    print ("x_train_resampled= " + str(x_train_resampled.shape) + " x_test_resampled= " + str(x_test_resampled.shape))
    print ("y_train_resampled= " + str(y_train_resampled.shape) + " y_test_resampled= " + str(x_test_resampled.shape))
    x_train = x_train_resampled
    x_test = x_test_resampled
    y_train = to_categorical(np.asarray(y_train_resampled.tolist()))
    y_test = to_categorical(np.asarray(y_test_resampled.tolist()))
    # End of Oversampling #########################################################################################################
    
    print('Shape of data tensor:', x_train.shape+x_test.shape)
    print('Shape of label tensor:', y_train.shape+y_test.shape)
    
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    print('Training model.')
    # CNN LSTM
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(Conv1D(50, 3, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(70))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), verbose=2)
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    # Evaluate Model
    confusion = np.array([[0, 0], [0, 0]])
    
    predictions = model.predict(x_test)
    predictedRound = [round(x[1]) for x in predictions]
    predicted = [x[1] for x in predictions]
    tested = [round(x[1]) for x in y_test]
    
    confusion += confusion_matrix(tested, predictedRound)
    precisionScore = precision_score(tested, predictedRound, pos_label=1.) #pos_label= (1.=yes, 0.=no)
    recallScore = recall_score(tested, predictedRound, pos_label=1.)
    accuracyScore = accuracy_score(tested, predictedRound)
    f1Score = f1_score(tested, predictedRound, pos_label=1.)
    rocAucScore = roc_auc_score(tested, predictedRound)
    maeScore = mean_absolute_error(tested, predictedRound)
    mseScore = mean_squared_error(tested, predictedRound)
    r2Score = r2_score(tested, predictedRound)
    
    trainResult = ''
    #     trainResult += 'Filename: ' + dsetFilename + '\n'
    #     trainResult += 'Classifier: ' + str(classifier) + '\n'
    trainResult += str(confusion[0]) + str(confusion[1]) + ','
    trainResult += str(precisionScore) + ','
    trainResult += str(recallScore) + ','
    trainResult += str(f1Score) + ','
    trainResult += str(accuracyScore) + ','
    trainResult += str(rocAucScore) + ','
    trainResult += str(maeScore) + ','
    trainResult += str(mseScore) + ','
    trainResult += str(r2Score) + '\n'
    print ("Trained successfully\n")
    
    return trainResult
    
    # Draw ROC Plot
    # fpr, tpr, thresholds = roc_curve(tested, predicted)
    # rocAuc = auc(fpr, tpr)
    # plt.title('ROC ' + classPersonality)
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% rocAuc)
    # plt.legend(loc='lower right')
    # plt.plot([0,1],[0,1],'r--')
    # plt.xlim([-0.1,1.2])
    # plt.ylim([-0.1,1.2])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
    
datasetName = 'mypersonality'
traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
# traits = ['openness']
datasetFilename = "Dataset\Personality Prediction\Dataset\\" + datasetName + "\\" + datasetName + "_status_dataset.txt"
for trait in traits:
    classFilename = r'Dataset\Personality Prediction\Dataset\\' + datasetName + '\Deep Learning\Class\\' + datasetName + '_' + trait + '_class.txt'
    trainResult = ""
    trainResult += train(datasetFilename, classFilename)
    
    resultFileName = open("Dataset/Personality Prediction/Dataset/" + datasetName + "/Deep Learning/Training Result/tr_dl_text_nopreprop_nores_"+trait[0]+".txt", "a")
    resultFileName.write(trainResult)
    resultFileName.close
print("All training result has been saved.")
# print(trainResult)