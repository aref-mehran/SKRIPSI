import requests
import uuid
import json

receptivitiApiSecretKey = "eFr0mf8T7wx4Ns98wd21s6zuHewp4WC33nwzHgmt3Jg"
receptivitiApiKey = "592d6f09bf893305d37bf93d"
receptivitiName = "hendro ndod" #Unique name for Receptiviti's user
receptivitiGender = 2 #0=Undefined, 1=Female, 2=Male
contentLanguage = "english"
contentSource = 4 #4=Media Social (Except Twitter)
totalDataset = 389

def liwcPersonPost(): #liwcPersonPost run once, the data will be saved in Receptiviti
    statusFileName = r'Dataset\Personality Prediction\Dataset\mix\mix_status_dataset.txt' #Dataset filename
    liwcStatusDset = unicode(open(statusFileName).read(), errors='ignore') #Read file
    liwcStatusDataset = liwcStatusDset.split('#SEPARATOR#') #Split LIWC Status Dataset by each user
    
    content_data = []
    person_data = []
    for index in range(0, totalDataset): #Looping All LIWC Status Dataset by each user
        content_data.append({"language": contentLanguage, "content_source": contentSource, "language_content": liwcStatusDataset[index]}) #Set content_data each user's status
        person_data.append({"name": receptivitiName, "gender": receptivitiGender, "person_handle": uuid.uuid4().hex}) #Set person_data each user AuthID
        person_data[index]["content"] = content_data[index] #Join content_data to person_data
        
    auth_headers = {"X-API-SECRET-KEY": receptivitiApiSecretKey, "X-API-KEY": receptivitiApiKey}
    
    url = 'https://app.receptiviti.com/v2/api/person' #LIWC Person API
    
    result = []
    for index in range(0, totalDataset): #Looping All LIWC Status Dataset by each user
        print("Start analyzing user-" + str(index+1) + "...")
        response = requests.post(url, json=person_data[index], headers=auth_headers);
        result.append(response.text)
        print(result[index])
        print("Complete analysis user-" + str(index+1) + "...")    
    print("Complete analysis all user")
    
    resultFileName = open("Dataset\Personality Prediction\Dataset\mix\Features\LIWC\mix_liwc.json", "w") #LIWC JSON result filename
    resultFileName.write(json.dumps(result).replace("\"", "").replace("\\", "\""))
    resultFileName.close

def liwcPersonGet():
    baseurl = 'https://app.receptiviti.com/v2/api/person'
    url = baseurl + "?name=" + receptivitiName
    
    auth_headers = {"X-API-SECRET-KEY": receptivitiApiSecretKey, "X-API-KEY": receptivitiApiKey}
     
    response = requests.get(url, headers=auth_headers)
    result = response.text
    print(result)

def saveLiwcData():
    liwcResultFileName = r'Dataset\Personality Prediction\Dataset\mix\Features\LIWC\mix_liwc.json' #Dataset filename
    liwcResultJson = open(liwcResultFileName).read() #Read file
    liwcResult = json.loads(liwcResultJson)
    
    liwcAttributes = []
    for key in liwcResult[0]["contents"][0]["liwc_scores"]["categories"]:
        liwcAttributes.append(key)
    
    liwcData = ""
    for i in range(0, totalDataset): #Looping All LIWC Status Dataset by each user
        print("LIWC-" + str(i+1) + " of " + str(totalDataset))
        for j in range(0, len(liwcAttributes)):
            if (j < len(liwcAttributes)-1):
                liwcData += str(liwcResult[i]["contents"][0]["liwc_scores"]["categories"][liwcAttributes[j]]) + ","
            else:
                liwcData += str(liwcResult[i]["contents"][0]["liwc_scores"]["categories"][liwcAttributes[j]])
        if (i < totalDataset-1):
            liwcData += "\n"
    
    liwcDataFilename = open("Dataset\Personality Prediction\Dataset\mix\Features\LIWC\mix_liwc_data.txt", "w")
    liwcDataFilename.write(liwcData)
    liwcDataFilename.close
    print("LIWC data has been saved.")
        
# liwcPersonPost() #Don't forget to change destination file, so it won't be replaced
# liwcPersonGet() #Not used
saveLiwcData()