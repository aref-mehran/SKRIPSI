import arff

totalDataset = 150
#Fb attribute = 7
#LIWC attribute = 85
#Splice attribute = 74
# totalAttribute = 166

def writeAllToArff():
    #READ ALL ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Attributes\fs_all_attributes.txt'
    attributes = open(attributeFileName).read()
    attributes = attributes.split('\n')
    
    #READ DATA_FB_FEATURES
    fbFeaturesFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Features\FB Features\mypersonality_fs_fb_features_data.txt'
    fbFeatures = open(fbFeaturesFileName).read()
    fbFeatures = fbFeatures.split('\n')
    
    #READ DATA_LIWC_FEATURES
    liwcFeaturesFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Features\LIWC\mypersonality_fs_liwc_data.txt'
    liwcFeatures = open(liwcFeaturesFileName).read()
    liwcFeatures = liwcFeatures.split('\n')
                
    #READ_DATA_SPLICE_FEATURES
    spliceFeaturesFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Features\Splice\mypersonality_fs_splice_data.txt'
    spliceFeatures = open(spliceFeaturesFileName).read()
    spliceFeatures = spliceFeatures.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Class\mypersonality_' + personalityClass[datasetCounter] + '_class.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + fbFeatures[index] + "," + liwcFeatures[index] + "," + spliceFeatures[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + fbFeatures[index] + "," + liwcFeatures[index] + "," + spliceFeatures[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Arff\Fb Features LIWC Splice\mypersonality_fs_" + personalityClass[datasetCounter] + "_dataset.arff", "w")
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")

def writeFbFeaturesToArff():
    #READ FB FEATURES ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Attributes\fs_fb_features_attributes.txt'
    attributes = open(attributeFileName).read()
    attributes = attributes.split('\n')
    
    #READ FB FEATURES
    featuresFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Features\FB Features\mypersonality_fs_fb_features_data.txt'
    features = open(featuresFileName).read()
    features = features.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Class\mypersonality_' + personalityClass[datasetCounter] + '_class.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + features[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + features[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Arff\FB Features\mypersonality_fs_" + personalityClass[datasetCounter] + "_dataset.arff", "w")
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")
        
def writeLiwcToArff():
    #READ LIWC ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Attributes\fs_liwc_attributes.txt'
    attributes = open(attributeFileName).read()
    attributes = attributes.split('\n')
    
    #READ LIWC FEATURES
    featuresFileName= r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Features\LIWC\chamndod_fs_liwc_data.txt'
    features = open(featuresFileName).read()
    features = features.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Class\chamndod_' + personalityClass[datasetCounter] + '_class.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + features[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + features[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Arff\LIWC\chamndod_fs_" + personalityClass[datasetCounter] + "_dataset.arff", "w")
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")

def writeSpliceToArff():
    #READ SPLICE ATTRIBUTES
    attributeFileName= r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Attributes\fs_splice_attributes.txt'
    attributes = open(attributeFileName).read()
    attributes = attributes.split('\n')
    
    #READ SPLICE FEATURES
    featuresFileName= r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Features\Splice\chamndod_fs_splice_data.txt'
    features = open(featuresFileName).read()
    features = features.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Class\chamndod_' + personalityClass[datasetCounter] + '_class.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in attributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + features[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + features[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Dataset\chamndod\Machine learning\Arff\Splice\chamndod_fs_" + personalityClass[datasetCounter] + "_dataset.arff", "w")
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")

def writeLiwcSpliceToArff():
    #READ LIWC ATTRIBUTES
    liwcAttributeFileName= r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Attributes\fs_liwc_attributes.txt'
    liwcAttributes = open(liwcAttributeFileName).read()
    liwcAttributes = liwcAttributes.split('\n')
    
    #READ SPLICE ATTRIBUTES
    spliceAttributeFileName= r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Attributes\fs_splice_attributes.txt'
    spliceAttributes = open(spliceAttributeFileName).read()
    spliceAttributes = spliceAttributes.split('\n')
    
    #READ DATA_LIWC_FEATURES
    liwcFeaturesFileName = r'Dataset\Personality Prediction\Dataset\chamndod\Machine learning\Features\LIWC\chamndod_fs_liwc_data.txt'
    liwcFeatures = open(liwcFeaturesFileName).read()
    liwcFeatures = liwcFeatures.split('\n')
                
    #READ_DATA_SPLICE_FEATURES
    spliceFeaturesFileName = r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Features\Splice\chamndod_fs_splice_data.txt'
    spliceFeatures = open(spliceFeaturesFileName).read()
    spliceFeatures = spliceFeatures.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Class\chamndod_' + personalityClass[datasetCounter] + '_class.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in liwcAttributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        for value in spliceAttributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + liwcFeatures[index] + "," + spliceFeatures[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + liwcFeatures[index] + "," + spliceFeatures[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Arff\LIWC Splice\chamndod_fs_" + personalityClass[datasetCounter] + "_dataset.arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")
    
def writeFbFeaturesLiwcToArff():
    #READ FB Features ATTRIBUTES
    fbFeaturesAttributeFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Attributes\fs_fb_features_attributes.txt'
    fbFeaturesAttributes = open(fbFeaturesAttributeFileName).read()
    fbFeaturesAttributes = fbFeaturesAttributes.split('\n')
    
    #READ LIWC ATTRIBUTES
    liwcAttributeFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Attributes\fs_liwc_attributes.txt'
    liwcAttributes = open(liwcAttributeFileName).read()
    liwcAttributes = liwcAttributes.split('\n')
    
    #READ DATA FB FEATURES
    fbFeaturesFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Features\FB Features\mypersonality_fs_fb_features_data.txt'
    fbFeatures = open(fbFeaturesFileName).read()
    fbFeatures = fbFeatures.split('\n')
    
    #READ DATA LIWC FEATURES
    liwcFeaturesFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine learning\Features\LIWC\mypersonality_fs_liwc_data.txt'
    liwcFeatures = open(liwcFeaturesFileName).read()
    liwcFeatures = liwcFeatures.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Class\mypersonality_' + personalityClass[datasetCounter] + '_class.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in fbFeaturesAttributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        for value in liwcAttributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + fbFeatures[index] + "," + liwcFeatures[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + fbFeatures[index] + "," + liwcFeatures[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Arff\FB Features LIWC\mypersonality_fs_" + personalityClass[datasetCounter] + "_dataset.arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")
    
def writeFbFeaturesSpliceToArff():
    #READ FB Features ATTRIBUTES
    fbFeaturesAttributeFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Attributes\fs_fb_features_attributes.txt'
    fbFeaturesAttributes = open(fbFeaturesAttributeFileName).read()
    fbFeaturesAttributes = fbFeaturesAttributes.split('\n')
    
    #READ Splice ATTRIBUTES
    spliceAttributeFileName= r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Attributes\fs_splice_attributes.txt'
    spliceAttributes = open(spliceAttributeFileName).read()
    spliceAttributes = spliceAttributes.split('\n')
    
    #READ DATA FB FEATURES
    fbFeaturesFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Features\FB Features\mypersonality_fs_fb_features_data.txt'
    fbFeatures = open(fbFeaturesFileName).read()
    fbFeatures = fbFeatures.split('\n')
    
    #READ DATA Splice FEATURES
    spliceFeaturesFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine learning\Features\Splice\mypersonality_fs_splice_data.txt'
    spliceFeatures = open(spliceFeaturesFileName).read()
    spliceFeatures = spliceFeatures.split('\n')

    for datasetCounter in (range(0, 5)):
        personalityClass = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        #READ_PERSONALITY_CLASS
        personalityClassFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Class\mypersonality_' + personalityClass[datasetCounter] + '_class.txt'
        personalityClassResults = open(personalityClassFileName).read()
        personalityClassResult = personalityClassResults.split('\n')
    
        arffContent = "@RELATION " + personalityClass[datasetCounter] + "_personality\n\n";
        for value in fbFeaturesAttributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        for value in spliceAttributes:
            arffContent = arffContent + "@ATTRIBUTE " + value + " numeric\n"
        
        arffContent = arffContent + "@ATTRIBUTE " + personalityClass[datasetCounter] + " {yes, no}\n"
        arffContent = arffContent + "\n" + "@DATA\n"
        for index in range(0, totalDataset):
            if index < totalDataset-1:
                arffContent = arffContent + fbFeatures[index] + "," + spliceFeatures[index] + "," + personalityClassResult[index] + "\n"
            else:
                arffContent = arffContent + fbFeatures[index] + "," + spliceFeatures[index] + "," + personalityClassResult[index]
        
        resultFileName = open("Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Arff\FB Features Splice\mypersonality_fs_" + personalityClass[datasetCounter] + "_dataset.arff", "w") #LIWC JSON result filename
        resultFileName.write(arffContent)
        resultFileName.close
    print ("All Personality Datasets have been written")

#Don't forget to change destination file, so it won't be replaced
# writeAllToArff()
# writeFbFeaturesToArff()
# writeLiwcToArff()
# writeSpliceToArff()
# writeLiwcSpliceToArff()
# writeFbFeaturesLiwcToArff()
# writeFbFeaturesSpliceToArff()