import requests
import uuid
import json

receptivitiApiSecretKey = "1DRZXuYw5YRrc4kFSYi4BKrhy6e4ogiuU0DBbNlIYy0"
receptivitiApiKey = "58d6f24ae53b0b05af52cd7c"
receptivitiName = "hendro-tommy" #Unique name for Receptiviti's user
receptivitiGender = 2 #0=Undefined, 1=Female, 2=Male
contentLanguage = "english"
contentSource = 4 #4=Media Social (Except Twitter)
totalDataset = 250

def liwcPersonPost(): #liwcPersonPost run once, the data will be saved in Receptiviti
    statusFileName = r'LIWC\LIWC Status Dataset Edited.txt' #Dataset filename
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
    
    resultFileName = open("LIWC\LIWC Result.json", "w") #LIWC JSON result filename
    resultFileName.write(json.dumps(result).replace("\"", "").replace("\\", "\""))
    resultFileName.close

def liwcPersonGet():
    baseurl = 'https://app.receptiviti.com/v2/api/person'
    url = baseurl + "?name=" + receptivitiName
    
    auth_headers = {"X-API-SECRET-KEY": receptivitiApiSecretKey, "X-API-KEY": receptivitiApiKey}
     
    response = requests.get(url, headers=auth_headers)
    result = response.text
    print(result)

def readLiwcResult():
    liwcResultFileName= r'LIWC\LIWC Result.json' #Dataset filename
    liwcResultJson = open(liwcResultFileName).read() #Read file
    liwcResult = json.loads(liwcResultJson)
        
    for index in range(0, totalDataset): #Looping All LIWC Status Dataset by each user
        print("LIWC result user-" + str(index+1) + ":")
        print("sixLtr:" + str(liwcResult[index]["contents"][0]["liwc_scores"]["sixLtr"]) + "\n" + 
              "wc:" + str(liwcResult[index]["contents"][0]["liwc_scores"]["wc"]) + "\n" + 
              "wps:" + str(liwcResult[index]["contents"][0]["liwc_scores"]["wps"]) + "\n" + 
              "dic:" + str(liwcResult[index]["contents"][0]["liwc_scores"]["dic"]) + "\n" +
               str(liwcResult[index]["contents"][0]["liwc_scores"]["categories"]) + "\n")
    
#liwcPersonPost() #Don't forget to change destination file, so it won't be replaced
#liwcPersonGet() #Not used
readLiwcResult()