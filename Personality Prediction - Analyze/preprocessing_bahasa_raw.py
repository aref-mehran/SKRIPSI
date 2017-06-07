import re
from nltk.tokenize import word_tokenize

datasetFilename = r'Dataset\Personality Prediction\Dataset\chamndod\Dataset sources\raw.txt'
dset = unicode(open(datasetFilename).read(), errors='ignore')
dataset = dset.split('#SEPARATOR#')

slangFilename = r'Dataset\Personality Prediction\Dataset\Preprocessing\Slang Words\slang.txt'
slangData = open(slangFilename).read()
slangWords = slangData.split('\n')

bakuFilename = r'Dataset\Personality Prediction\Dataset\Preprocessing\Slang Words\baku.txt'
bakuData = open(bakuFilename).read()
bakuWords = bakuData.split('\n')

print ("Start Pre-processing Dataset")
for i in range(0, len(dataset)):
    print ("Dataset-" + str(i+1) + " of " + str(len(dataset)))
    dataset[i] = dataset[i].replace("*PROPNAME*", "") # Remove *PROPNAME*
    dataset[i] = re.sub(r"http\S+", "", dataset[i]) # Remove URLs
    
    # Replace Slang Words
    words = word_tokenize(dataset[i])
    wordsFiltered = ""
    for word in words:
        if word in slangWords:
            wordIndex = 0
            for j in range(0, len(slangWords)):
                if word == slangWords[j]:
                    wordIndex = j
                    j = len(slangWords)
            wordsFiltered += bakuWords[wordIndex] + " "
        else:
            wordsFiltered += word + " "
    dataset[i] = wordsFiltered
    
    dataset[i] = dataset[i].strip() # Remove Space
    dataset[i] = re.sub("[ ]{2,}", " ", dataset[i]) # Remove Duplicate Space
print ("Finish Pre-processing Dataset")

resultFileName = open("Dataset/Personality Prediction/Dataset/chamndod/Dataset sources/raw.txt", "w")
resultPreprocessing = ""
for i in range(0, len(dataset)):
    if (i < len(dataset)-1):
        resultPreprocessing += dataset[i] + "#SEPARATOR#"
    else:
        resultPreprocessing += dataset[i]
resultFileName.write(resultPreprocessing)
resultFileName.close
print ("Preprocessing Dataset has been saved")