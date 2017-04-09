import json

import arff

import xml.etree.ElementTree as ET


totalDataset = 250
#Fb attribute = 7
#LIWC attribute = 89
#Splice attribute = 74
totalAttribute = 170

def writeToArff():
    #READ ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\attributes.txt'
    attributes = open(attributeFileName).read()
    attribute = attributes.split('\n')
    test = "test";
    
    #READ DATA_FB_FEATURES
    fbFeaturesFileName= r'Dataset\Personality Prediction\data_fb_features.txt'
    fbFeatures = open(fbFeaturesFileName).read()
    fbFeature = fbFeatures.split('\n')
    
    #READ DATA_LIWC_FEATURES
    liwcFeaturesFileName = r'Dataset\Personality Prediction\data_liwc_features.txt'
    liwcFeatures = open(liwcFeaturesFileName).read()
    liwcFeature = liwcFeatures.split('\n')
                
    #READ_DATA_SPLICE_FEATURES
    spliceFeaturesFileName = r'Dataset\Personality Prediction\data_splice_features.txt'
    spliceFeatures = open(spliceFeaturesFileName).read()
    spliceFeature = spliceFeatures.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\class_' + personalityClass[datasetCounter] + '.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attribute:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            arffContent = arffContent + fbFeature[index] + "," + liwcFeature[index] + "," + spliceFeature[index] + "," + personalityClassResult[index] + "\n"
            
        resultFileName = open("Dataset\Personality Prediction\\dataset_" + personalityClass[datasetCounter] + ".arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")
    
writeToArff() #Don't forget to change destination file, so it won't be replaced