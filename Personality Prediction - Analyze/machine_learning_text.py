import os
from pandas import DataFrame
import numpy
import uuid
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfTransformer

TOTAL_DATASET = 250
CLASS_WEIGHT = {0:2.3, 1:1}

def build_data_frame(data, labels):
    rows = []
    index = []
    rows.append({'data': data, 'labels': labels})
    index.append(str(uuid.uuid1()))

    data_frame = DataFrame(rows, index=index)
    return data_frame

#BUILD DATASETS
#Read Data
dataFilename = r'Dataset\Personality Prediction\myPersonality Status Dataset.txt'
data = unicode(open(dataFilename).read(), errors='ignore')
data = data.split('#SEPARATOR#')
#Read Labels
labelFilename = r'Dataset\Personality Prediction\class\class_openness.txt'
labels = open(labelFilename).read()
labels = labels.split('\n')

labelEncoder = preprocessing.LabelEncoder()
encodedLabels = labelEncoder.fit(['yes', 'no'])
transformedEncodeLabels = labelEncoder.transform(labels)
datasets = DataFrame({'data': [], 'labels': []})
for i in range(0, TOTAL_DATASET):
    datasets = datasets.append(build_data_frame(data[i], transformedEncodeLabels[i])) 
numpy.random.seed(3)
datasets = datasets.reindex(numpy.random.permutation(datasets.index))

# for path, classification in SOURCES:
#     print(path)
#     data = data.append(build_data_frame(path, classification))
# data = data.reindex(numpy.random.permutation(data.index))

#WORD COUNTS AS FEATURE
#count_vectorizer = CountVectorizer()
#counts = count_vectorizer.fit_transform(data['text'].values)

#TRAINING
#classifier = MultinomialNB()
#targets = data['class'].values
#classifier.fit(counts, targets)
#print ("Training finished")
#print(len(counts[0]))
#print(data['text'].values[0])

#TEST PREDICT
#examples = ['Free Viagra call today!', "I'm going to attend the Linux users group tomorrow."]
#example_counts = count_vectorizer.transform(examples)
#predictions = classifier.predict(example_counts)
#predictions # [1, 0]

#TRAINING USING PIPELINE
# pipeline = Pipeline([
#     ('vectorizer',  CountVectorizer()),
#     ('classifier',  MultinomialNB())
# ])
# pipeline = Pipeline([
#     ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
#     ('classifier',       MultinomialNB())
# ])
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',       MultinomialNB())
])
#pipeline = Pipeline([
#    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
#    ('classifier',       BernoulliNB(binarize=0.0))
#])

#CROSS-VALIDATION
k_fold = KFold(n=len(datasets), n_folds=10)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
precision_scores = []
recall_scores = []
accuracy_scores = []
f1_scores = []
roc_auc_scores = []
mae_scores = []
mse_scores = []
r2_scores = []
foldIndex = 1;
for train_indices, test_indices in k_fold:
    print ("kFold-" + str(foldIndex))
    foldIndex = foldIndex + 1
    
    x_train = datasets.iloc[train_indices]['data'].values
    y_train = datasets.iloc[train_indices]['labels'].values
    x_test = datasets.iloc[test_indices]['data'].values
    y_test = datasets.iloc[test_indices]['labels'].values

    pipeline.fit(x_train, y_train)
    
    predictions = pipeline.predict(x_test)
    confusion += confusion_matrix(y_test, predictions)
    precisionScore = precision_score(y_test, predictions, pos_label=1) #pos_label= (1=yes, 0=no)
    recallScore = recall_score(y_test, predictions, pos_label=1)
    accuracyScore = accuracy_score(y_test, predictions)
    f1Score = f1_score(y_test, predictions, pos_label=1)
    rocAucScore = roc_auc_score(y_test, predictions)
    maeScore = mean_absolute_error(y_test, predictions)
    mseScore = mean_squared_error(y_test, predictions)
    r2Score = r2_score(y_test, predictions)
    
    precision_scores.append(precisionScore)
    recall_scores.append(recallScore)
    accuracy_scores.append(accuracyScore)
    f1_scores.append(f1Score)
    roc_auc_scores.append(rocAucScore)
    mae_scores.append(maeScore)
    mse_scores.append(mseScore)
    r2_scores.append(r2Score)
    
trainResult = ''
#     trainResult += 'Filename:' + dsetFilename + '\n'
#     trainResult += 'Classifier:' + str(classifier) + '\n'
trainResult += 'Confusion Matrix:\n' + str(confusion) + '\n'
trainResult += 'Precision: ' + str(sum(precision_scores)/len(precision_scores)) + '\n'
trainResult += 'Recall: ' + str(sum(recall_scores)/len(recall_scores)) + '\n'
trainResult += 'Accuracy: ' + str(sum(accuracy_scores)/len(accuracy_scores)) + '\n'
trainResult += 'F1 Measure: ' + str(sum(f1_scores)/len(f1_scores)) + '\n'
trainResult += 'ROC AUC: ' + str(sum(roc_auc_scores)/len(roc_auc_scores)) + '\n'
trainResult += 'MAE: ' + str(sum(mae_scores)/len(mae_scores)) + '\n'
trainResult += 'MSE: ' + str(sum(mse_scores)/len(mse_scores)) + '\n'
trainResult += 'R2: ' + str(sum(r2_scores)/len(r2_scores)) + '\n\n'
print ("Trained successfully\n")
print (trainResult)