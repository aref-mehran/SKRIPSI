from keras.preprocessing.text import text_to_word_sequence
from collections import Counter
import json

def generateJsonString(countDatasetDict):
    totalUniqueWord = len(countDatasetDict)
    jsonString = "{"
    for i in range(0, totalUniqueWord):
        if (i < totalUniqueWord-1):
            jsonString += "\"" + countDatasetDict[i][0] + "\": " + str(i+1) + ", "
        else:
            jsonString += "\"" + countDatasetDict[i][0] + "\": " + str(i+1)
    jsonString += "}"
    
    return jsonString
        
def wordCount():
    filename = r'Dataset\Personality Prediction\Keras\status_dataset.txt' #Dataset filename
    dset = unicode(open(filename).read(), errors='ignore') #Read file
    dataset = dset.split('#SEPARATOR#') #Split Status Dataset by each user
    dataset = '\n'.join(dataset)
    
    listDataset = text_to_word_sequence(str(dataset))
    countDatasetDict = Counter(listDataset)

    #Manually generate json string to be saved as json file,
    #because countDatasetDict does not satisfy json's requirements
    jsonString = generateJsonString(countDatasetDict.most_common())
    
    resultFileName = open("Dataset\Personality Prediction\Keras\status_word_index.json", "w")
    resultFileName.write(jsonString)
    resultFileName.close
    print ("Word Count complete")

def encodeDataset():
    datasetFilename = r'Dataset\Personality Prediction\Keras\status_dataset.txt'
    dset = unicode(open(datasetFilename).read(), errors='ignore')
    dataset = dset.split('#SEPARATOR#')
    
    labelClass = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    binaryClass = []
    for lblClass in labelClass:
        classFilename = r'Dataset\Personality Prediction\Keras\Class\class_' + lblClass + '.txt'
        dset = open(classFilename).read()
        binaryClass.append(dset.split('\n'))
    classResult = {"Y_train": {"openness": binaryClass[0], "conscientiousness": binaryClass[1], "extraversion": binaryClass[2], "agreeableness": binaryClass[3], "neuroticism": binaryClass[4]}}
        
    wordIndexFilename= r'Dataset\Personality Prediction\Keras\status_word_index.json'
    wIndex = open(wordIndexFilename).read()
    wordIndex = json.loads(wIndex)
    
    for i in range(0, len(dataset)):
        dataset[i] = text_to_word_sequence(str(dataset[i]))

    encodedDataset = []
    for i in range(0, len(dataset)):
        encodedDset = []
        for j in range(0, len(dataset[i])):
            encodedDset.append(wordIndex[dataset[i][j]])
        encodedDataset.append(encodedDset)
    encodedResult = {"X_train": encodedDataset, "Y_train": {"openness": binaryClass[0], "conscientiousness": binaryClass[1], "extraversion": binaryClass[2], "agreeableness": binaryClass[3], "neuroticism": binaryClass[4]}}
    
    resultFileName = open("Dataset\Personality Prediction\Keras\status_fb.json", "w") #LIWC JSON result filename
    resultFileName.write(json.dumps(encodedResult))
    resultFileName.close
    print ("Encode Dataset complete")
    
# wordCount()
# encodeDataset()