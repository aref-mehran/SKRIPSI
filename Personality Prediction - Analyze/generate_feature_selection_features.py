def saveFbFeaturesData():
    fbFeaturesResultFileName = r'Dataset\Personality Prediction\Dataset\mypersonality\Machine Learning\Features\FB Features\mypersonality_fb_features_data.txt' #Dataset filename
    fbFeaturesResult = open(fbFeaturesResultFileName).read() #Read file
    fbFeaturesResult = fbFeaturesResult.split('\n')
    
    index_feature_selection = [0, 1, 2, 3, 4]
    fsResult = ""
    for i in range(0, len(fbFeaturesResult)):
        fbFeaturesResultRow = fbFeaturesResult[i].split(',')
        for j in index_feature_selection:
            if (j < index_feature_selection[len(index_feature_selection)-1]):
                fsResult += fbFeaturesResultRow[j] + ","
            else:
                fsResult += fbFeaturesResultRow[j]
        if (i < len(fbFeaturesResult)-1):
            fsResult += '\n'   
    
    fbFeaturesDataFilename = open("Dataset/Personality Prediction/Dataset/mypersonality/Machine Learning/Features/FB Features/mypersonality_fs_fb_features_data.txt", "w")
    fbFeaturesDataFilename.write(fsResult)
    fbFeaturesDataFilename.close
    print("FB Features data has been saved.")
    
def saveLiwcData():
    liwcResultFileName = r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Features\LIWC\chamndod_liwc_data.txt' #Dataset filename
    liwcResult = open(liwcResultFileName).read() #Read file
    liwcResult = liwcResult.split('\n')
    
    index_feature_selection = [23, 37, 39, 44, 52, 64, 83]
    fsResult = ""
    for i in range(0, len(liwcResult)):
        liwcResultRow = liwcResult[i].split(',')
        for j in index_feature_selection:
            if (j < index_feature_selection[len(index_feature_selection)-1]):
                fsResult += liwcResultRow[j] + ","
            else:
                fsResult += liwcResultRow[j]
        if (i < len(liwcResult)-1):
            fsResult += '\n'   
    
    liwcDataFilename = open("Dataset/Personality Prediction/Dataset/chamndod/Machine Learning/Features/LIWC/chamndod_fs_liwc_data.txt", "w")
    liwcDataFilename.write(fsResult)
    liwcDataFilename.close
    print("LIWC data has been saved.")
    
def saveSpliceData():
    spliceResultFileName = r'Dataset\Personality Prediction\Dataset\chamndod\Machine Learning\Features\Splice\chamndod_splice_data.txt' #Dataset filename
    spliceResult = open(spliceResultFileName).read() #Read file
    spliceResult = spliceResult.split('\n')
    
    index_feature_selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 22, 23, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]
    fsResult = ""
    for i in range(0, len(spliceResult)):
        spliceResultRow = spliceResult[i].split(',')
        for j in index_feature_selection:
            if (j < index_feature_selection[len(index_feature_selection)-1]):
                fsResult += spliceResultRow[j] + ","
            else:
                fsResult += spliceResultRow[j]
        if (i < len(spliceResult)-1):
            fsResult += '\n'   
    
    spliceDataFilename = open("Dataset/Personality Prediction/Dataset/chamndod/Machine Learning/Features/Splice/chamndod_fs_splice_data.txt", "w")
    spliceDataFilename.write(fsResult)
    spliceDataFilename.close
    print("Splice data has been saved.")
    
# saveFbFeaturesData()
# saveLiwcData()
saveSpliceData()