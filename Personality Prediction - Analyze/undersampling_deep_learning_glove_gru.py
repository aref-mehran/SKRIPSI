from __future__ import print_function

import os
# import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense
from keras.layers import GRU
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.under_sampling import ClusterCentroids

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
    
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Start of Undersampling #####################################################################################################
    labels = np.reshape(labels, (len(labels)))
    
    cc = ClusterCentroids()
    data_resampled, labels_resampled = cc.fit_sample(data, labels)
            
    print ("data= " + str(data.shape))
    print ("labels= " + str(labels.shape))
    print ("data_resampled= " + str(data_resampled.shape))
    print ("labels_resampled= " + str(labels_resampled.shape))
    data = data_resampled
    labels = to_categorical(np.asarray(labels_resampled.tolist()))
    # End of Undersampling #######################################################################################################
    
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_test = data[-num_validation_samples:]
    y_test = labels[-num_validation_samples:]
    
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
    # GRU
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'mse', 'mae'])
    
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), verbose=2)
    
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