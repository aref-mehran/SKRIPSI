import arff
import uuid
import numpy
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2

TOTAL_DATASET = 250
total_features = 170
index_class = 170

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
        train_text = data.iloc[train_indices]['features'].values
        train_y = data.iloc[train_indices]['class'].values
        
        test_text = data.iloc[test_indices]['features'].values
        test_y = data.iloc[test_indices]['class'].values
    
        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)
        print (predictions)
        print (test_y)
        confusion += confusion_matrix(test_y, predictions)
        precisionScore = precision_score(test_y, predictions, pos_label=1) #pos_label= (1=yes, 0=no)
        recallScore = recall_score(test_y, predictions, pos_label=1)
        accuracyScore = accuracy_score(test_y, predictions)
        f1Score = f1_score(test_y, predictions, pos_label=1)
        rocAucScore = roc_auc_score(test_y, predictions)
        maeScore = mean_absolute_error(test_y, predictions)
        mseScore = mean_squared_error(test_y, predictions)
        r2Score = r2_score(test_y, predictions)
        
        precision_scores.append(precisionScore)
        recall_scores.append(recallScore)
        accuracy_scores.append(accuracyScore)
        f1_scores.append(f1Score)
        roc_auc_scores.append(rocAucScore)
        mae_scores.append(maeScore)
        mse_scores.append(mseScore)
        r2_scores.append(r2Score)
    
    trainResult = ''
    trainResult += 'Filename:' + dsetFilename + '\n'
    trainResult += 'Classifier:' + str(classifier) + '\n'
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
    
    return trainResult

# total_features = 7
# index_class = 7
dsetFilename = "Dataset\Personality Prediction\Arff\All features\dataset_openness.arff"
classifier = [GaussianNB(), BernoulliNB(), LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), LinearDiscriminantAnalysis()]
trainResult = ""
for clsfier in classifier:
     trainResult += train(dsetFilename, clsfier)

# resultFileName = open("Dataset\Personality Prediction\Training Result.txt", "w")
# resultFileName.write(trainResult)
# resultFileName.close
print(trainResult)