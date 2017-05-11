import arff

totalDataset = 250
#Fb attribute = 7
#LIWC attribute = 89
#Splice attribute = 74
totalAttribute = 170

def writeAllToArff():
    #READ ALL ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\attributes\all_attributes.txt'
    attributes = open(attributeFileName).read()
    attribute = attributes.split('\n')
    
    #READ DATA_FB_FEATURES
    fbFeaturesFileName= r'Dataset\Personality Prediction\data\data_fb_features.txt'
    fbFeatures = open(fbFeaturesFileName).read()
    fbFeature = fbFeatures.split('\n')
    
    #READ DATA_LIWC_FEATURES
    liwcFeaturesFileName = r'Dataset\Personality Prediction\data\data_liwc_features.txt'
    liwcFeatures = open(liwcFeaturesFileName).read()
    liwcFeature = liwcFeatures.split('\n')
                
    #READ_DATA_SPLICE_FEATURES
    spliceFeaturesFileName = r'Dataset\Personality Prediction\data\data_splice_features.txt'
    spliceFeatures = open(spliceFeaturesFileName).read()
    spliceFeature = spliceFeatures.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\class\class_' + personalityClass[datasetCounter] + '.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attribute:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + fbFeature[index] + "," + liwcFeature[index] + "," + spliceFeature[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + fbFeature[index] + "," + liwcFeature[index] + "," + spliceFeature[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Arff\All features\dataset_" + personalityClass[datasetCounter] + ".arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")

def writeFbFeaturesToArff():
    #READ FB FEATURES ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\attributes\fb_features_attributes.txt'
    attributes = open(attributeFileName).read()
    attribute = attributes.split('\n')
    
    #READ FB FEATURES
    featuresFileName= r'Dataset\Personality Prediction\data\data_fb_features.txt'
    features = open(featuresFileName).read()
    feature = features.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\class\class_' + personalityClass[datasetCounter] + '.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attribute:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + feature[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + feature[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Arff\Fb features\dataset_fb_features_" + personalityClass[datasetCounter] + ".arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")
        
def writeLiwcToArff():
    #READ LIWC ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\attributes\liwc_attributes.txt'
    attributes = open(attributeFileName).read()
    attribute = attributes.split('\n')
    
    #READ LIWC FEATURES
    featuresFileName= r'Dataset\Personality Prediction\data\data_liwc_features.txt'
    features = open(featuresFileName).read()
    feature = features.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\class\class_' + personalityClass[datasetCounter] + '.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attribute:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + feature[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + feature[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Arff\LIWC features\dataset_liwc_features_" + personalityClass[datasetCounter] + ".arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")

def writeSpliceToArff():
    #READ SPLICE ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\attributes\splice_attributes.txt'
    attributes = open(attributeFileName).read()
    attribute = attributes.split('\n')
    
    #READ SPLICE FEATURES
    featuresFileName= r'Dataset\Personality Prediction\data\data_splice_features.txt'
    features = open(featuresFileName).read()
    feature = features.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\class\class_' + personalityClass[datasetCounter] + '.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attribute:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + feature[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + feature[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Arff\Splice features\dataset_splice_features_" + personalityClass[datasetCounter] + ".arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")

def writeLiwcSpliceToArff():
    #READ LIWC ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\attributes\liwc_attributes.txt'
    attributes = open(attributeFileName).read()
    attribute = attributes.split('\n')
    
    #READ SPLICE ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\attributes\splice_attributes.txt'
    attributes = open(attributeFileName).read()
    attribute = attributes.split('\n')
    
    #READ DATA_LIWC_FEATURES
    liwcFeaturesFileName = r'Dataset\Personality Prediction\data\data_liwc_features.txt'
    liwcFeatures = open(liwcFeaturesFileName).read()
    liwcFeature = liwcFeatures.split('\n')
                
    #READ_DATA_SPLICE_FEATURES
    spliceFeaturesFileName = r'Dataset\Personality Prediction\data\data_splice_features.txt'
    spliceFeatures = open(spliceFeaturesFileName).read()
    spliceFeature = spliceFeatures.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\class\class_' + personalityClass[datasetCounter] + '.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attribute:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + liwcFeature[index] + "," + spliceFeature[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + liwcFeature[index] + "," + spliceFeature[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Arff\dataset_" + personalityClass[datasetCounter] + ".arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")
    
# writeLiwcSpliceToArff() #Don't forget to change destination file, so it won't be replaced