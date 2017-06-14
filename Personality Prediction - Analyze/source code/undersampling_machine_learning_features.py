import arff
import uuid
import numpy
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.under_sampling import ClusterCentroids

TOTAL_DATASET = 400
total_features = 60
index_class = 60
# FB_Features=7; LIWC=85; Splice=74
# fs_FB_Features=5; fs_LIWC=7; fs_Splice=60

class GetFeatures(TransformerMixin):
    def transform(self, features):
        featuresResult = []
        for i in range(0, features.shape[0]):
            rows = []
            for j in range(0, total_features):
                rows.append(features[i][j])
            featuresResult.append(rows)
        return numpy.asarray(featuresResult).reshape(features.shape[0], total_features)
    
    def fit(self, X, y=None):
        return self
    
def build_data_frame(features, classification):
    rows = []
    index = []
    rows.append({'features': features, 'class': classification})
    index.append(str(uuid.uuid1()))

    data_frame = DataFrame(rows, index=index)
    return data_frame

def train(dsetFilename, classifier):
    numpy.random.seed(8)
    
    dset = arff.load(open(dsetFilename, 'rb'))
    features = []
    classification = []
    labelEncoder = preprocessing.LabelEncoder()
    for index_dataset in (range(0, TOTAL_DATASET)):
        ftrs = []
        for index_features in (range(0, total_features)):
            ftrs.append(dset['data'][index_dataset][index_features])
        features.append(ftrs)
        classification.append(dset['data'][index_dataset][index_class])
    
    encodedClassification = labelEncoder.fit(['yes', 'no'])
    transformedEncodeClassification = labelEncoder.transform(classification)
    data = DataFrame({'features': [], 'class': []})
    for i in range(0, TOTAL_DATASET):
        data = data.append(build_data_frame(features[i], transformedEncodeClassification[i]))
    data = data.reindex(numpy.random.permutation(data.index))
    
    #TRAINING USING PIPELINE
    pipeline = Pipeline([
        ('features', GetFeatures()),
        ('classifier', classifier)
    ])
    
    #CROSS-VALIDATING
    print ("Start training " + dsetFilename + " with\n" + str(classifier))
    k_fold = KFold(n=len(data), n_folds=10)
    accuracy_scores = []
    foldIndex = 1;
    for train_indices, test_indices in k_fold:
        print ("kFold-" + str(foldIndex))
        foldIndex = foldIndex + 1
        x_train = data.iloc[train_indices]['features'].values
        y_train = data.iloc[train_indices]['class'].values
        x_test = data.iloc[test_indices]['features'].values
        y_test = data.iloc[test_indices]['class'].values
    
        # Start of UnderSampling #######################################################################################################
        element_x_train = []
        for i in range(0, len(x_train)):
            element_x_train.append(x_train[i])
        temp_x_train = numpy.array(element_x_train)
        temp_x_train = numpy.reshape(temp_x_train, (len(temp_x_train), total_features))
        
        cc = ClusterCentroids()
        x_train_resampled, y_train_resampled = cc.fit_sample(temp_x_train, y_train)
             
        x_train = x_train_resampled
        y_train = y_train_resampled
        # End of Undersampling #########################################################################################################
        
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracyScore = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracyScore)
    
    trainResult = ''
    trainResult += str(sum(accuracy_scores)/len(accuracy_scores)) + ','
    print ("Trained successfully\n")
    
    return trainResult

traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
classifiers = [GaussianNB(), LinearSVC(), LogisticRegression(), GradientBoostingClassifier(), LinearDiscriminantAnalysis()]
for trait in traits:
    dsetFilename = "Dataset\Personality Prediction\Dataset\mix\Machine Learning\Arff\Splice\mix_fs_"+trait+"_dataset.arff"
    trainResult = ""
    for classifier in classifiers:
         trainResult += train(dsetFilename, classifier)

    resultFileName = open("Dataset/Personality Prediction/Dataset/mix/Machine Learning/Training Result/tr_ml_splice_fs_under_"+trait[0]+".txt", "w")
    resultFileName.write(trainResult)
    resultFileName.close
print("All training result has been saved.")