import os
import arff
import uuid
from pandas import DataFrame
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

TOTAL_FEATURES = 170
TOTAL_DATA = 250
CLASS_INDEX = 170
DATASET_FILENAME = [
    'Dataset/Personality Prediction/dataset_extraversion.arff'
    #'Dataset/Personality Prediction/dataset_conscientiousness.arff',
    #'Dataset/Personality Prediction/dataset_extraversion.arff',
    #'Dataset/Personality Prediction/dataset_neuroticism.arff',
    #'Dataset/Personality Prediction/dataset_openness.arff',
]

class GetFeatures(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        dset = arff.load(open('Dataset\Personality Prediction\dataset_agreeableness.arff', 'rb'))
        featuresKey = []
        features = []
        
        for i in (range(0, TOTAL_FEATURES)):
            featuresKey.append(dset['attributes'][i][0])
        
        for i in (range(0, TOTAL_DATA)):
            featuresKeyValue = dict()
            for j in (range(0, 1)):
                featuresKeyValue[featuresKey[0]] = int(dset['data'][i][j])
            features.append(featuresKeyValue)
                
        return features

    def fit(self, X, y=None, **fit_params):
        return self

def build_data_frame(path):
    rows = []
    index = []
    dset = arff.load(open('Dataset\Personality Prediction\dataset_agreeableness.arff', 'rb'))
    
    for features_value in dset['data']:
        classification = features_value[CLASS_INDEX];
        del features_value[CLASS_INDEX]
        rows.append({'features_value': features_value, 'class': classification})
        index.append(path + "/" + str(uuid.uuid1()))

    data_frame = DataFrame(rows, index=index)
    return data_frame

#BUILD DATASETS
data = DataFrame({'features_value': [], 'class': []})
for path in DATASET_FILENAME:
    print("Build" + path + "DataFrame")
    data = data.append(build_data_frame(path))
data = data.reindex(numpy.random.permutation(data.index))
print("Finish loading all dataset")

features = GetFeatures();
featuresResult = features.fit_transform(data['features_value'], data['class'])
print featuresResult

#TRAINING USING PIPELINE
#pipeline = Pipeline([
#    ('vectorizer',  CountVectorizer()),
#    ('classifier',  MultinomialNB()) ])
pipeline = Pipeline([
    ('fUnion', FeatureUnion([
        ('features', GetFeatures())
    ])),
    ('classifier',       MultinomialNB())
])
#pipeline = Pipeline([
#    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
#    ('tfidf_transformer',  TfidfTransformer()),
#    ('classifier',       MultinomialNB())
#])
#pipeline = Pipeline([
#    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
#    ('classifier',       BernoulliNB(binarize=0.0))
#])
pipeline.fit(data['features_value'], data['class'])
print ("Finished training")

#CROSS-VALIDATING
#k_fold = KFold(n=len(data), n_folds=10)
#scores = []
#confusion = numpy.array([[0, 0], [0, 0]])
#foldIndex = 1;
#for train_indices, test_indices in k_fold:
#    print ("kFold-" + str(foldIndex))
#    foldIndex = foldIndex + 1
#    train_text = data.iloc[train_indices]['text'].values
#    train_y = data.iloc[train_indices]['class'].values

#    test_text = data.iloc[test_indices]['text'].values
#    test_y = data.iloc[test_indices]['class'].values

#    pipeline.fit(train_text, train_y)
#    predictions = pipeline.predict(test_text)

#    confusion += confusion_matrix(test_y, predictions)
#    score = f1_score(test_y, predictions, pos_label=SPAM)
#    scores.append(score)

#print('Total emails classified:', len(data))
#print('Score:', sum(scores)/len(scores))
#print('Confusion matrix:')
#print(confusion)