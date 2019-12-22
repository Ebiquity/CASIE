import sys
import os
import codecs
import json

def cuthead(txtfile):
    txt=readFileEncode(txtfile,'utf-8')
    return txt.index("<text>")+7

def loadJsontoDict(jsonfile):
    content=readFileEncode(jsonfile,'utf-8')
    dictContent=json.loads(content)
    return dictContent

def readFileEncode(inputFile,encodings):
    try:
        f=codecs.open(inputFile, 'r', encodings)
        tmpstr=f.read()
    except FileNotFoundError:
        tmpstr=''
        print ('file ',inputFile, ' not found')
    return tmpstr

def readFile(inputFile):
    with open (inputFile,'r') as f:
        inputText=f.read()
    f.close()
    return inputText

def jsonToDict(inputText):
    outputList=[]
    inputList=inputText.strip().split('\n')    
    for x in inputList:        
        inputDict=json.loads(x)
        outputList.append(inputDict)
            
    return outputList

def writeJsonToFile(outputMatrix,filename):
    """ write jsons data to a file"""
    with open (filename, 'w') as f:
        for senDict in outputMatrix:
            for word in senDict:
                j=json.dumps(word)
                f.write(str(j))
                f.write('\n')            
    f.close()
    

def findTokenOffset(content, inputMatrix):
    """ find token offset in content"""
    start=0
    outputMatrix=[]
    for senDict in inputMatrix:
        outputSen=[]
        for word in senDict:
            if word['depfn']=='punct':
                word['offset']=-1
                pass
            else:
                try:
                    position=content.index(word['surface'],start,start+200)
                    start=position+len(word['surface'])-1
                except ValueError:
                    position = -1
                word['offset']=position
            outputSen.append(word)
        outputMatrix.append(outputSen)
    return outputMatrix
