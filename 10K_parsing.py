# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:29:01 2016

@author: u472290
"""

path="C:/Users/u472290/Desktop/reference materials/cyt-20141231.xml"
path1="C:/Users/u472290/Desktop/reference materials/aapl-20150926.xml"
writeBasePath="C:\Users\u472290\workfolder\Project 10_k\\test"
RSSfeedPath="C:\Users\u472290\AppData\Local\Microsoft\Windows\Temporary Internet Files\Enclosure\{65762E5E-6643-45B9-AAB2-F9159EF37275}"


import os
import zipfile
from bs4 import BeautifulSoup
import pandas as pd
import shutil
import datetime
import math
#function for parsing the facts and other information from an XBRL xml file and return as a dataframe
def xbrlParser(path):
    processedXbrl = pd.DataFrame()
    soup=BeautifulSoup(open(path),'xml')
    allTags=soup.findAll()
    allTags=allTags[1:]
    counter=0
    for ind,currentTag in enumerate(allTags):
        attributes=currentTag.attrs
        if attributes.has_key("contextRef"):
            elementId="_".join((currentTag.prefix,currentTag.name)) 
            if attributes.has_key("contextRef"):
                contextId=attributes["contextRef"]
            else:
                contextId=""
            if attributes.has_key("unitRef"):
                unitId=attributes["unitRef"]
            else:
                unitId=""
            try:
                fact=currentTag.contents[0]
            except:
                fact=""
            if attributes.has_key("decimals"):
                decimals=attributes["decimals"]
            else:
                decimals=""        
            if attributes.has_key("scale"):
                scale=attributes["scale"]
            else:
                scale="" 
            if attributes.has_key("sign"):
                decimals=attributes["sign"]
            else:
                sign=""             
            if attributes.has_key("id"):
                factId=attributes["id"]
            else:
                factId="" 
            ns=currentTag.namespace    
            #adding all the data to the processedXbrl dataframe
            if counter==0:
                processedXbrl=pd.DataFrame({"elementId":pd.Series(elementId),"contextId":pd.Series(contextId),"unitId":pd.Series(unitId),"fact":pd.Series(fact),"decimals":pd.Series(decimals),"scale":pd.Series(scale),"sign":pd.Series(sign),"factId":pd.Series(factId),"ns":ns})
                counter=1
            else:
                processedXbrl.ix[ind,"elementId"] = elementId
                processedXbrl.ix[ind,"contextId"] = contextId
                processedXbrl.ix[ind,"unitId"] = unitId
                processedXbrl.ix[ind,"fact"] = fact
                processedXbrl.ix[ind,"decimals"] = decimals
                processedXbrl.ix[ind,"scale"] = scale
                processedXbrl.ix[ind,"sign"] = sign
                processedXbrl.ix[ind,"factId"] = factId
                processedXbrl.ix[ind,"ns"] = ns
       
    processedXbrl["fact"] = processedXbrl.fact.apply(lambda x:x.encode('ascii', 'ignore'))
    
    #adding Path, Company_Name, CIK, form_type, Date.Filed, qtr
    processedXbrl["Path"]  = "RSS-Feed"
    processedXbrl["Company_Name"]  = processedXbrl[processedXbrl.elementId == "dei_EntityRegistrantName"].fact.values[0]
    processedXbrl["CIK"]  = processedXbrl[processedXbrl.elementId == "dei_EntityCentralIndexKey"].fact.values[0]   
    processedXbrl["form_type"]  = processedXbrl[processedXbrl.elementId == "dei_DocumentType"].fact.values[0]   
    processedXbrl["Date.Filed"] = datetime.datetime.now().strftime("%Y-%m-%d")
    date1  = datetime.datetime.strptime(str(processedXbrl[processedXbrl.elementId == "dei_DocumentPeriodEndDate"].fact.values[0]),"%Y-%m-%d")
    year=date1.strftime("%y")
    qtr=int(math.ceil(int(date1.strftime("%m"))/3))
    date = "-".join((str(year),str(qtr)))
    processedXbrl["qtr"] = date
    processedXbrl = processedXbrl.reindex_axis(['elementId','contextId', 'unitId', 'fact', 'decimals','scale', 'sign', 'factId', 'ns','Path', 'Company_Name', 'CIK', 'form_type', 'Date.Filed', 'qtr'],axis=1)
    return processedXbrl       
    
    
    
#xbrlFile=xbrlParser(path)    
#xbrlFile.to_csv(os.path.join(writeBasePath,"parsed_CYT.csv"),sep="|")    


#reading the recent filings and parsing them as required
while True:
    listOfFiles=os.listdir(RSSfeedPath)
    pathOfZips=[os.path.join(RSSfeedPath,files) for files in listOfFiles if files.find(".zip") > 0]
    nameOfFiles=[files for files in listOfFiles if files.find(".zip") > 0]
    collatedFormsPath=os.path.join(RSSfeedPath,"collatedForms")   
    try:
        os.mkdir(collatedFormsPath) 
    except:
         exp="folder collatedForms already exists."    
   
    for ind,zips in enumerate(pathOfZips):
        print "processing file " + str(ind+1) + " of " + str(len(pathOfZips))
        fh = open(zips, 'rb')
        z = zipfile.ZipFile(fh)
        files=z.extract(z.namelist()[0], collatedFormsPath)
        z.close()
        fh.close()
        shutil.copyfile(zips,os.path.join(collatedFormsPath,nameOfFiles[ind]))
        parsedXBRL=xbrlParser(files)
        os.remove(files)
        os.chmod(zips,666)
        os.remove(zips)
        parsedXBRL.to_csv(os.path.join(writeBasePath,z.namelist()[0].replace(".xml",".csv")),sep="|",index=False)