import os
from pandas import DataFrame
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import arff
import uuid

TOTAL_DATASET = 250
INDEX_CLASS = 170

class LengthStatusTransformer(TransformerMixin):
    def transform(self, status):
        rows = []
        for i in range(0, TOTAL_DATASET):
            rows.append(len(status[i]))
        return DataFrame(rows)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
def build_data_frame(liwcStatusDataset, classification):
    rows = []
    index = []
    rows.append({'text': liwcStatusDataset, 'class': classification})
    index.append(str(uuid.uuid1()))

    data_frame = DataFrame(rows, index=index)
    return data_frame

#BUILD DATASETS
statusFileName = r'LIWC\LIWC Status Dataset Edited.txt' #Dataset filename
liwcStatusDset = unicode(open(statusFileName).read(), errors='ignore') #Read file
liwcStatusDataset = liwcStatusDset.split('#SEPARATOR#') #Split LIWC Status Dataset by each user

dset = arff.load(open('Dataset\Personality Prediction\dataset_agreeableness.arff', 'rb'))
classification = []
for index in (range(0, TOTAL_DATASET)):
    classification.append(dset['data'][index][INDEX_CLASS])
    
data = DataFrame({'text': [], 'class': []})
for i in range(0, TOTAL_DATASET):
    data = data.append(build_data_frame(liwcStatusDataset[i], classification[i]))
data = data.reindex(numpy.random.permutation(data.index))

#lengthStatus = LengthStatusTransformer()
#print lengthStatus.transform(data['text'])

#TRAINING USING PIPELINE
#pipeline = Pipeline([
#    ('union', Pipeline([
#        ('vectorizer',  CountVectorizer()),
#        ('length_status', LengthStatusTransformer())
#    ])),
#    ('classifier',  MultinomialNB())
#])
#pipeline = Pipeline([
#    ('count_vectorizer', CountVectorizer()),
#    ('classifier',       MultinomialNB())
#])
#pipeline = Pipeline([
#    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
#    ('classifier',       MultinomialNB())
#])
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',       MultinomialNB())
])
#pipeline = Pipeline([
#    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
#    ('classifier',       BernoulliNB(binarize=0.0))
#])
pipeline.fit(data['text'].values, data['class'].values)
print ("Finished training")

#CROSS-VALIDATING
k_fold = KFold(n=len(data), n_folds=10)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
foldIndex = 1;
for train_indices, test_indices in k_fold:
    print ("kFold-" + str(foldIndex))
    foldIndex = foldIndex + 1
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label="yes")
    scores.append(score)

print('Total statuses classified:', len(data['text']))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)