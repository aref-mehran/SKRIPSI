import arff
import uuid
import numpy
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE, RandomOverSampler

TOTAL_DATASET = 389
total_features = 74
index_class = 74
# FB_Features=7; LIWC=85; Splice=74

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
        x_train = data.iloc[train_indices]['features'].values
        y_train = data.iloc[train_indices]['class'].values
        x_test = data.iloc[test_indices]['features'].values
        y_test = data.iloc[test_indices]['class'].values
        
        # Start of Oversampling #######################################################################################################
        element_x_train = []
        for i in range(0, len(x_train)):
            element_x_train.append(x_train[i])
        temp_x_train = numpy.array(element_x_train)
        temp_x_train = numpy.reshape(temp_x_train, (len(temp_x_train), total_features))

#         kind = ['regular', 'borderline1', 'borderline2', 'svm']
        kind = ['borderline1']
        sm = [SMOTE(kind=k) for k in kind]
        for method in sm:
            x_train_resampled, y_train_resampled = method.fit_sample(temp_x_train, y_train)
             
        x_train = x_train_resampled
        y_train = y_train_resampled
        # End of Oversampling #########################################################################################################
        
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
#     trainResult += 'Filename: ' + dsetFilename + '\n'
    trainResult += 'Classifier: ' + str(classifier) + '\n'
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

traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
# traits = ['openness']
classifiers = [GaussianNB(), BernoulliNB(), LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), LinearDiscriminantAnalysis()]
# classifiers = [GaussianNB()]
for trait in traits:
    dsetFilename = "Dataset\Personality Prediction\Dataset\mix\Machine Learning\Arff\Splice\mix_"+trait+"_dataset.arff"
    trainResult = ""
    for classifier in classifiers:
         trainResult += train(dsetFilename, classifier)

#     resultFileName = open("Dataset/Personality Prediction/Dataset/mix/Machine Learning/Arff/Splice/Training Result/Without Preprocessing/Oversampling/training_result_"+trait+".txt", "w")
#     resultFileName.write(trainResult)
#     resultFileName.close
print("All training result has been saved.")
print(trainResult)