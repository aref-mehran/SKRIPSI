import re
import nltk
from nltk.tag.stanford import StanfordNERTagger

filename = r'Dataset\Personality Prediction\Dataset\chamndod\testing.txt'
dset = open(filename).read()
dataset = dset.split('#SEPARATOR#')

st = StanfordNERTagger('Dataset/Personality Prediction/Dataset/Pre-Processing/Stanford NER Tagger/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz', 'Dataset/Personality Prediction/Dataset/Pre-Processing/Stanford NER Tagger/stanford-ner-2014-06-16/stanford-ner.jar')

print ("Start Pre-processing Dataset")
for i in range(0, len(dataset)):
    print ("Dataset-" + str(i+1))
    dataset[i] = dataset[i].replace("*PROPNAME*", "") # Remove *PROPNAME*
    dataset[i] = re.sub(r"http\S+", "", dataset[i]) # Remove URLs
    dataset[i] = re.sub("[^a-zA-Z ']", "", dataset[i]) # Remove All Symbols except (single quote), Numbers, Punctuations
    
    # Remove Person Name
    nerResult = st.tag(dataset[i].split())
    result = ""
    for res in nerResult:
        if (res[1] != 'PERSON'):
            result += res[0] + " "
    dataset[i] = result
    
    dataset[i] = dataset[i].strip() # Remove Space
    dataset[i] = re.sub("[ ]{2,}", " ", dataset[i]) # Remove Duplicate Space
    dataset[i] = dataset[i].lower(); # Lower case
    
    # Stemming
#     stemmer = nltk.stem.SnowballStemmer('english')
#     result = ""
#     dataSplit = dataset[i].split()
#     for data in dataSplit:
#         result += stemmer.stem(data) + " "
#     dataset[i] = result
print ("Finish Pre-processing Dataset")
    
resultFileName = open("Dataset\Personality Prediction\Dataset\chamndod\preprocessing_testing.txt", "w") #LIWC JSON result filename
resultPreprocessing = ""
for i in range(0, len(dataset)):
    resultPreprocessing += dataset[i] + "#SEPARATOR#"
resultFileName.write(resultPreprocessing)
resultFileName.close
print ("Pre-processing Dataset has been saved")