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

NEWLINE = '\n'
SKIP_FILES = {'cmds'}
HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    ('Dataset/Spam or Ham/Spam/spam',        SPAM),
    ('Dataset/Spam or Ham/Ham/easy_ham',    HAM),
    ('Dataset/Spam or Ham/Ham/hard_ham',    HAM)
]

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path)
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content

def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': unicode(text, errors="ignore"), 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

#BUILD DATASETS
data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    print(path)
    data = data.append(build_data_frame(path, classification))
data = data.reindex(numpy.random.permutation(data.index))

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
#pipeline = Pipeline([
#    ('vectorizer',  CountVectorizer()),
#    ('classifier',  MultinomialNB()) ])
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
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
pipeline.fit(data['text'].values, data['class'].values)
print ("Finished training")
#print pipeline.predict(examples) # ['spam', 'ham']

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
    score = f1_score(test_y, predictions, pos_label=SPAM)
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)