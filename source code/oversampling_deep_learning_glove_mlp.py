import os
import numpy as np
from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Flatten
from keras.layers import LSTM, GRU, Conv1D, MaxPooling1D
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'Dataset/Personality Prediction/Keras/GloVe/6B/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 2800
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

np.random.seed(8)

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
    print('Processing text dataset')
    texts = []
    labels = []
    dataset = open(datasetFilename).read()
    texts = dataset.split('#SEPARATOR#')
    labelClass = open(classFilename).read()
    labels = labelClass.split('\n')
    
    tempTexts = []
    tempLabels = []
    indices = np.arange(400)
    np.random.shuffle(indices)
    for i in range(0, 400):
        tempTexts.append(texts[indices[i]])
        tempLabels.append(labels[indices[i]])
    texts = tempTexts
    labels = tempLabels

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
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

    kind = ['borderline1']
    sm = [SMOTE(kind=k) for k in kind]
    for method in sm:
        print(str(x_train.shape) + " " + str(x_test.shape))
        x_train_resampled, y_train_resampled = method.fit_sample(x_train, y_train)
        x_test_resampled, y_test_resampled = method.fit_sample(x_test, y_test)
        
    x_train = x_train_resampled
    x_test = x_test_resampled
    y_train = to_categorical(np.asarray(y_train_resampled.tolist()))
    y_test = to_categorical(np.asarray(y_test_resampled.tolist()))
    # End of Oversampling #########################################################################################################
    
    print('Preparing embedding matrix.')
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    print('Training model.')
    # MLP
    model = Sequential()
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(50, init='glorot_uniform', activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # LSTM
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    
    # GRU
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'mse', 'mae'])
    
    # CNN1D GLOBALMAXPOOLING
#     sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences = embedding_layer(sequence_input)
#     x = Conv1D(32, 5, activation='tanh')(embedded_sequences)
#     x = MaxPooling1D(5)(x)
#     x = Conv1D(32, 5, activation='tanh')(x)
#     x = MaxPooling1D(5)(x)
#     x = Conv1D(32, 5, activation='tanh')(x)
#     x = MaxPooling1D(35)(x)
#     x = Flatten()(x)
#     x = Dense(32, activation='tanh')(x)
#     preds = Dense(2, activation='softmax')(x)
#     model = Model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    
    # CNN LSTM
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(Dropout(0.2))
#     model.add(Conv1D(50, 3, padding='valid', activation='relu', strides=1))
#     model.add(MaxPooling1D(pool_size=4))
#     model.add(LSTM(70))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), verbose=2)
    
    predictions = model.predict(x_test)
    predictedRound = [round(x[1]) for x in predictions]
    predicted = [x[1] for x in predictions]
    tested = [round(x[1]) for x in y_test]
    accuracyScore = accuracy_score(tested, predictedRound)
    trainResult = ''
    trainResult += str(accuracyScore) + ','
    print ("Trained successfully\n")
    
    return trainResult
    
datasetName = 'mix'
traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
datasetFilename = "Dataset\Personality Prediction\Dataset\\" + datasetName + "\\" + datasetName + "_preprop_status_dataset.txt"
for trait in traits:
    classFilename = r'Dataset\Personality Prediction\Dataset\\' + datasetName + '\Deep Learning\Class\\' + datasetName + '_' + trait + '_class.txt'
    trainResult = ""
    trainResult += train(datasetFilename, classFilename)
    
    resultFileName = open("Dataset/Personality Prediction/Dataset/" + datasetName + "/Deep Learning/Training Result/tr_dl_text_preprop_over_"+trait[0]+".txt", "a")
    resultFileName.write(trainResult)
    resultFileName.close
print("All training result has been saved.")