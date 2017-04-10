import requests
import xml.etree.ElementTree as ET

statusFileName = r'Splice\Splice Status Dataset Edited.txt' #Dataset filename
spliceStatusDset = unicode(open(statusFileName).read(), errors='ignore') #Read file
spliceStatusDataset = spliceStatusDset.split('#SEPARATOR#') #Split Splice Status Dataset by each user
totalDataset = 250

def splicePost():
    spliceCues = "numChars numCharsMinusSpacesAndPunctuation numWords numSentences numPunctuation numNouns numVerbs \
                numAdjectives numAdverbs numPassiveVerbs firstPersonSingular firstPersonPlural secondPerson \
                thirdPersonSingular thirdPersonPlural iCanDoIt doKnow posSelfImage iCantDoIt dontKnow negSelfImage \
                numImperatives suggestionPhrases inflexibility contradict totalDominance numAgreement askPermission \
                seekGuidance totalSubmissiveness Imagery Pleasantness Activation avgWordLength avgSentenceLength \
                numSyllables avgSyllablesPerWord numWordsWith3OrMoreSyllables rateWordsWith3OrMoreSyllables \
                numWordsWith6OrMoreChars rateWordsWith6OrMoreChars numWordsWith7OrMoreChars rateWordsWith7OrMoreChars \
                LexicalDiversity complexityComposite hedgeVerb hedgeConj hedgeAdj hedgeModal hedgeAll numDisfluencies \
                numInterjections numSpeculate Expressivity numIgnorance Pausality questionCount hedgeUncertain pastTense \
                presentTense SWNpositivity SWNnegativity SWNobjectivity ARI FRE FKGL CLI LWRF FOG SMOG DALE LIX RIX FRY" # Put list of cues here in a string seperated by spaces
    
    url = 'http://splice.cmi.arizona.edu/SPLICE/post/postargs/' #Splice API
    
    result = []
    for index in range(0, totalDataset): #Looping All Splice Status Dataset by each user
        print("Start analyzing user-" + str(index+1) + "...")
        data = 'text=' + spliceStatusDataset[index] + '&cues=' + spliceCues #Splice API Parameters
        response = requests.post(url, data) #Call Splice API
        result.append(response.text) #Read response from Splice API
        print (result[index]) #Print response result from Splice API
        print("Complete analysis user-" + str(index+1) + "...")
    print ("Complete analysis all user")
    
    resultFileName = open("Splice Result.xml", "w") #LIWC JSON result filename
    resultFileName.write("<?xml version=\"1.0\"?>\n<Splice>")
    for index in range(0, totalDataset):
        resultFileName.write("\n\t<Data>\n\t\t")
        resultFileName.write(result[index].replace("<?xml version=\"1.0\"?>", "").replace("<Splice>", "").replace("</Splice>", ""))
        resultFileName.write("\n\t</Data>")
    resultFileName.write("\n</Splice>")
    resultFileName.close

def readSpliceResult():
    tree = ET.parse('Splice\Splice Result.xml')
    root = tree.getroot()
    print(root)
    index1=0; index2=0
    for child1 in root:
        print("Splice result user-" + str(index1+1) + ":")
        spliceResult = ""
        for child2 in child1:
            spliceResult = spliceResult + child2.tag + ": " + root[index1][index2].text + ", "
            index2 = index2 + 1
        print(spliceResult + "\n")
        index1 = index1 + 1
        index2 = 0
        
#splicePost() #Don't forget to change destination file, so it won't be replaced
readSpliceResult()