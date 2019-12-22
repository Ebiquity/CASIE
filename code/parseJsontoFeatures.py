# 3 ner 04/05/2019
#input: CoreNLP json output
#output: surface lemma pos dep_parent parent_pos dep_child child_pos startOffset endOffset level chunk-type 
# 2 options; dict , file
# dict structure: sentences->tokens->each features above
# file structure: one line per token
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random as rn
rn.seed(1)
import numpy as np

import sys
sys.path.append('../')

import utils
import tree
import wd_search

from sklearn.preprocessing import OneHotEncoder	
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

import re
import json
import inflect
from datetime import date

encoded=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','y','z','aa','x']

def removeinnermost(parsetree):
    parsetree = parsetree.replace("\n", "")
    parsetree = parsetree.replace('(. .)', '')
    leaves = parsetree.split()

    right = 0
    for i in range(0, len(leaves) - 1):
        if leaves[i].startswith('(') and leaves[i + 1].endswith(')'):
            leaves[i] = ''
            right += 1
        elif leaves[i].count(')') == 1:
            leaves[i] = leaves[i].replace(')', '')
            right -= 1
        elif leaves[i].count(')') > 1:
            while right > 0:
                leaves[i] = leaves[i][:-1]
                right -= 1
        leaves[i] = leaves[i].strip()

    return " ".join(leaves)

def combineCoref(features,corefs):
    """input coref info to dict"""
    for x in corefs:
        ner=''
        for c in corefs[x]:
            if c["isRepresentativeMention"]==True:
                sentences=features[c["sentNum"]-1]
                if sentences[c["headIndex"]-1]["ner"]!='O':
                    ner=sentences[c["headIndex"]-1]["ner"][2:]
                

                if c["isRepresentativeMention"]==False:
                    sentences=features[c["sentNum"]-1] 
                    if sentences[c["headIndex"]-1]["ner"]=='O':
                        sentences[c["headIndex"]-1]["ner"]="B-"+ner

    return features

def readCoref(features,corefs):
    """read coref info from Stanford CoreNLP output"""
    corefinfo={}
    for x in corefs.keys():
        if len(corefs[x])>1:
            corefinfo[x]=[]

            for c in corefs[x]:
                csentid = c['sentNum'] - 1
                cstarttokenid = c['startIndex'] - 1
                cendtokenid = c['endIndex'] - 1
                for d in corefs[x]:
                    if c['id']!=d['id']:
                        sentid=d['sentNum']-1
                        starttokenid=d['startIndex']-1
                        endtokenid=d['endIndex']-1
                        startOffset=features[sentid][starttokenid]['characterOffsetBegin']
                        if endtokenid< len(features[sentid]):
                            endOffset = features[sentid][endtokenid]['characterOffsetBegin']-1
                        else:
                            endOffset = features[sentid][-1]['characterOffsetBegin'] - 1

                        for i in range(cstarttokenid,cendtokenid,1):
                            if 'coref' not in features[csentid][i]:
                                features[csentid][i]['coref']=[]
                            features[csentid][i]['coref'].append((startOffset,endOffset))


    return features

def combineDep(sentence):
    """add dependency relation info to dict"""
    enPPDep=sentence["enhancedPlusPlusDependencies"]
    tokens=sentence["tokens"]
    #add governor originalText, governor pos    
    for e in enPPDep:
        if e["governor"]==0:
            tokens[e["dependent"]-1]["gov_pos"]="ROOT"
        else:
            tokens[e["dependent"]-1]["gov_pos"]=tokens[e["governor"]-1]["pos"]

        if "dep_set" not in tokens[e["governor"]-1]:
            tokens[e["governor"]-1]["dep_set"]=[]    
            tokens[e["governor"]-1]["dep_words"]=[]

        if "gov_id" not in tokens[e["dependent"]-1]:
            tokens[e["dependent"]-1]["gov_id"]=[]
            tokens[e["dependent"]-1]["gov_words"]=[]
            tokens[e["dependent"] - 1]["gov_rel"] = []

        if e["dep"] not in tokens[e["governor"]-1]["dep_set"]:
            tokens[e["governor"]-1]["dep_set"].append(e["dep"])

        if e["dep"] not in tokens[e["dependent"] - 1]["gov_rel"]:
            tokens[e["dependent"] - 1]["gov_rel"].append(e["dep"])

        tokens[e["governor"]-1]["dep_words"].append(e["dependent"])
        tokens[e["dependent"]-1]["gov_id"].append(e["governor"])
        tokens[e["dependent"]-1]["gov_words"].append(e["governorGloss"])

    return tokens

def combineChnkLvl(tokens,np,chnklist,depthlist):
    """add chunk type and its level in dict"""
    for i in range(len(tokens)):
        if "chunk" not in tokens[i]:
            tokens[i]["depthchunk"]=[]
            tokens[i]["chunk"]=[]
        word=tokens[i]["originalText"]+'_@_'+str(tokens[i]["index"])
        for j in range(len(chnklist)):
            
            if word == chnklist[j][0]:
                if np:
                    tokens[i]["chunk"].append("B-NP")
                else:
                    tokens[i]["chunk"].append("B-VP")
                tokens[i]["depthchunk"].append(encoded[depthlist[j]])
            elif word in chnklist[j]:
                if np:
                    tokens[i]["chunk"].append("I-NP")
                else:
                    tokens[i]["chunk"].append("I-VP")         
                tokens[i]["depthchunk"].append(encoded[depthlist[j]])
                              
    return tokens      

selectedNER=["LOCATION","STATE_OR_PROVINCE","CITY","COUNTRY","DATE","SET","TIME","DURATION","NUMBER","URL","MONEY","ORGANIZATION","EMAIL","PERSON","CURRENCY",
"SOFTWARE","DEVICE","SYSTEM","CVE","File","GPE","Version","Tool","PII","MODIFIER","CONSEQUENCES","OPERATINGSYSTEM","NETWORK","ATTACK","MEANS","HARDWARE","OTHER","FILE"]


def combineNe_st(tokens):
    """add named entities from Stanford CoreNLP to dict"""
    for x in range(0,len(tokens)):

        if tokens[x]['ner']!='O' and tokens[x]['ner'] not in selectedNER :            
            tokens[x]['ner']='O'

        elif tokens[x]['ner']!='O' and tokens[x]['ner'] in selectedNER:
            if x>0:
                if tokens[x]['ner'].lower() == tokens[x-1]['ner'][2:].lower():                        
                    tokens[x]['ner']='I-'+tokens[x]['ner'][0]+tokens[x]['ner'][1:].lower()
                else: 
                    tokens[x]['ner']='B-'+tokens[x]['ner'][0]+tokens[x]['ner'][1:].lower()
            else: #x=0
                tokens[x]['ner']='B-'+tokens[x]['ner'][0]+tokens[x]['ner'][1:].lower()        
       

        ## find version number ###
        p1=re.compile("(v)*\d+\.\d+(\.\d+)*")
        version1=re.match(p1,tokens[x]['originalText'])
        if version1: # or version2:
            if tokens[x-1]['ner'][2:].lower()=='software' or tokens[x-1]['ner'][2:].lower()=='system':
                tokens[x]['ner']='B-Version'
            elif tokens[x-1]['originalText'].lower().startswith('ver'): #ver(sions) 10.1
                tokens[x-1]['ner']='B-Version'
                tokens[x]['ner']='I-Version'
            elif tokens[x-1]['ner']=='O' and tokens[x-1]['originalText'][0].isupper():
                tokens[x]['ner']='B-Version'
                tokens[x-1]['ner']='B-Software'

        if (tokens[x-1]['ner'][2:].lower()=='software' or tokens[x-1]['ner'][2:].lower()=='system') and tokens[x]['ner'][2:].lower()=='number':
            tokens[x]['ner']='B-Version'

        # for Windows 10 Mobile
        if tokens[x]['originalText'].isdigit():

            if x-1 >= 0:
                if tokens[x-1]['ner'][2:].lower()=='software' or tokens[x-1]['ner'][2:].lower()=='system':
                    name=tokens[x-1]['originalText']+' '+tokens[x]['originalText']
                    namelen=2
                    if x+1<len(tokens):
                        if tokens[x+1]['pos'].startswith('NNP'):
                            name=name+' '+tokens[x+1]['originalText']
                            namelen=3
                    result=wd_search.wd_search(name)
                    typelist=[]
                    if result:
                        for res in result:
                            for t in res['types']:
                                typelist.append(t[1])
                        if 'software version' in typelist:
                             tokens[x-1]['ner']='B-Version'
                             tokens[x]['ner']='I-Version'
                             if namelen==3:
                                 tokens[x+1]['ner']='I-Version'
        ## find CVE ###
        cve=''
        p2=re.compile("CVE-\d+-\d+")
        cve=re.match(p2,tokens[x]['originalText'])
        if cve:
            tokens[x]['ner']='B-CVE'

    return tokens

def combineExtNe(tokens, wklist1, wklist2, casielist, dbplist):
    """add named entities from external knowledge base into dict"""
    for x in range(0, len(tokens)):
        tokens[x]['wk_ner'] = 'O' #casie
        tokens[x]['finerner'] = 'O' #wk
        tokens[x]['db_ner'] = 'O'
        ## main source of NER; wikidata, dbpedi
        ## use this ner as the first choice
        for text, start, ne in wklist1:
            if tokens[x]['characterOffsetBegin'] == start and tokens[x]['originalText'] == text:
                tokens[x]['finerner'] = ne
                break
        for text, start, ne in wklist2:
            if tokens[x]['characterOffsetBegin'] == start and tokens[x]['originalText'] == text:
                tokens[x]['finerner2'] = ne
                break
        for text, start, ne in casielist:
            if tokens[x]['characterOffsetBegin'] == start and tokens[x]['originalText'] == text:
                tokens[x]['wk_ner'] = ne
                break
        for text, start, ne in dbplist:
            if tokens[x]['characterOffsetBegin'] == start and tokens[x]['originalText'] == text:
                tokens[x]['db_ner'] = ne
                break
    return tokens

NuggetList10=["I-Phishing","B-Phishing","I-DiscoverVulnerability","B-DiscoverVulnerability","B-Databreach","I-Databreach","B-PatchVulnerability","I-PatchVulnerability","B-Ransom","I-Ransom"]
ArgumentList=["B-Patch","I-Data","I-Money","I-Capabilities","B-Time","B-Organization","B-Device","I-GPE","B-File","B-Version","B-Person","B-Software","I-Organization","B-Capabilities","I-Software","B-GPE","I-Time","I-Person","B-Vulnerability","B-Data","I-Patch","I-Version","B-Purpose","B-Money","I-Purpose","I-PaymentMethod","B-PaymentMethod","B-CVE","I-System","I-Vulnerability","I-Device","B-System","I-File","I-Number","B-Number","B-PII","I-PII","B-Malware","I-Malware", "B-Website", "I-Website"]

def processLabel(lbcontent,content,labeltype):
    """
    surface offset  label       role      Realis   coref
    hacker  1263  B-Person  B-Attacker  O   O
    hacked  1275  B-Databreach    O   Actual  Same_0
    """
    labellist={}
    lines=lbcontent.split('\n')[:-1]
    sentenceid=0
    selectedlist={}
    position=0
    if labeltype=='nugget':
        selectedlist=NuggetList10
    elif labeltype=='argument':
        selectedlist=ArgumentList

    for line in lines:
        if len(line)>3:
            words=line.split('\t')
            if words[2] in selectedlist:
                tokens=content["sentences"][sentenceid]["tokens"]
                tokenid=''
                for x in tokens:
                    if int(words[1])==x["characterOffsetBegin"]:
                        tokenid=x["index"]-1
                        break
                if sentenceid not in labellist: 
                    labellist[sentenceid]=[]

                labellist[sentenceid].append((words[0],words[1],words[2],tokenid))
        else:
            sentenceid+=1
    
    return labellist

def findhead(focustrigger,tokens,deptree):
    """ find heads of triggers"""
    entlist=[]
    ent=[]
    for x in range(len(focustrigger)):
        text,position,eventType,tokenid=focustrigger[x]
        if eventType.startswith('B-'):
            if ent:
                entlist.append(ent)
                ent=[]
            ent.append(tokenid)
        elif eventType.startswith('I-'):
            ent.append(tokenid)
    entlist.append(ent)
    # find head of each trigger
    headlist=[]

    for i in range(len(entlist)):
        if len(entlist[i])>1:
            # multi-word entity
            head=set([])
            nodej = tokens[entlist[i][0]]['originalText'] + '_@_' + str(entlist[i][0] + 1)
            nodek = tokens[entlist[i][-1]]['originalText'] + '_@_' + str(entlist[i][-1] + 1)
            path=tree.find_path(deptree,nodej,nodek,[],{})
            respath = []
            for node1 in path:
                # select only root node
                if node1 in deptree.keys():
                    respath.append(node1)
            if len(respath)==1:
                if respath[0]==nodej or respath[0]==nodek:
                    head.add(respath[0])

            else:
                path=set([])
                for node1 in respath:
                    # find upper root
                    for node2 in respath:
                        if node1!= node2 and node2 in deptree[node1]:
                            path.discard(node2)
                            path.add(node1)
                            break
                        elif node2!= node1 and node1 in deptree[node2]:
                            path.discard(node1)
                            path.add(node2)
                            break

                for node in path:
                    if node==nodej or node==nodek:
                        head.add(node)
            if head:
                h=head.pop()
                headlist.append(int(h.split('_@_')[1])-1)
            else:
                if tokens[entlist[i][0]]['pos'].startswith('VB'):
                    headlist.append(entlist[i][0])
                else:
                    headlist.append(entlist[i][-1])
        elif len(entlist[i])==1:
        # one word trigger
            headlist.append(entlist[i][0])

    return headlist

def combineNearTrggr(tokens,trggrlist,sentenceid,deptree,enh,parsetree):
    """find nearest trigger in dependency path"""
    focustrigger=[]
    triggersentence=0
    if sentenceid in trggrlist:
        focustrigger=trggrlist[sentenceid]
        triggersentence=sentenceid
    elif sentenceid-1 in trggrlist:
        focustrigger=trggrlist[sentenceid-1]
        triggersentence=sentenceid-1    

    nearEvent,nearTrigger='O',''    

    # found trigger in the same sentence
    if len(focustrigger)==0:
        return tokens

    if focustrigger and triggersentence!=sentenceid:
        triggertokenid=0
        for i in range(len(focustrigger)):
            # find trigger info
            text, position, eventType, tokenid = focustrigger[i]
            if triggertokenid <= int(tokenid):
                triggertokenid=tokenid
                nearEvent=eventType[2:]
                nearTrigger=text
        for x in range(len(tokens)):
            tokens[x]['triggerPosition']='differsentence'
            tokens[x]['nearEvent']=nearEvent
            tokens[x]['nearTrigger']=nearTrigger
            tokens[x]['distFromTrigger']='aa'
            tokens[x]['triggerPosition']='differsentence'
        return tokens

    elif focustrigger and triggersentence==sentenceid:
        headlist = findhead(focustrigger, tokens, deptree)
        for y in focustrigger:
            #clear all trigger tokens
            text,position,eventType,tokenid=y
            tokens[tokenid]['distFromTrigger']='x'
            tokens[tokenid]['triggerPosition']='this'
            tokens[tokenid]['nearEvent']='-'
            tokens[tokenid]['nearTrigger']='-'

        if len(headlist) == 1:
            # one candidate, no choice
            for i in range(len(focustrigger)):
                # find trigger info
                text, position, eventType, tokenid = focustrigger[i]
                if headlist[0] == int(tokenid):
                    triggertokenid=int(tokenid)
                    nearEvent=eventType[2:]
                    triggerposition=int(position)
                    break

            source = tokens[triggertokenid]['originalText'] + '_@_' + str(triggertokenid + 1)
            for x in range(len(tokens)):
                #clear other tokens to nearest trigger
                if 'nearEvent' not in tokens[x]: #not a trigger
                    target = tokens[x]['originalText'] + '_@_' + str(x + 1)
                    path = tree.find_path(deptree, source, target, [], {})
                    if isinstance(path, list) and len(path) > 1:
                        deppath = tree.todeppath(path, enh)
                        dist = len(deppath)

                    else:
                        dist = 1000
                        deppath = ['-']
                    tokens[x]['nearTriggerTokenid']=triggertokenid
                    tokens[x]['nearEvent'] = nearEvent
                    tokens[x]['nearTrigger'] = text
                    tokens[x]['distFromTrigger'] = encoded[min(abs(triggertokenid - x - 1),26)]
                    tokens[x]['deppathtoTrigger'] = deppath
                    tokens[x]['deppathtoTriggerLength'] = encoded[min(dist,26)]

                    if triggerposition < int(tokens[x]['characterOffsetBegin']):  # trigger x
                        tokens[x]['triggerPosition'] = 'before'
                        for j in range(triggertokenid + 1, x):
                            if tokens[j]['originalText'] == ',':
                                tokens[x]['triggerPosition'] += 'separated'
                                break

                    elif triggerposition > int(tokens[x]['characterOffsetBegin']):  # x trigger
                        tokens[x]['triggerPosition'] = 'after'
                        for j in range(x + 1, triggertokenid):
                            if tokens[j]['originalText'] == ',':
                                tokens[x]['triggerPosition'] += 'separated'
                                break

            tokens=commonRootParse(tokens,parsetree)
            return tokens

        elif len(headlist)>1:
            # few choices for trigger, check distance every tokens
            for x in range(len(tokens)):
                if 'nearEvent' not in tokens[x]:
                    distance,deppath={},{}
                    for head in headlist:
                        # find nearest trigger for x
                        source=tokens[head]['originalText']+'_@_'+str(head+1)
                        target=tokens[x]['originalText'] + '_@_' + str(x + 1)
                        path = tree.find_path(deptree,source ,target ,[],{})
                        if isinstance(path,list) and len(path)>1:
                            deppath[head] = tree.todeppath(path,enh)
                            distance[head]=len(deppath)
                        else:
                            distance[head]=1000
                            deppath[head]=['-']
                    triggertokenid = min(distance.items(), key=lambda a: a[1])[0]

                    for i in range(len(focustrigger)):
                        #find trigger info
                        text, position, eventType, tokenid = focustrigger[i]
                        if triggertokenid==tokenid:
                            break

                    nearEvent=eventType[2:]
                    nearTrigger=text
                    triggerposition=int(position)
                    triggertokenid=int(triggertokenid)

                    tokens[x]['nearTriggerTokenid'] = triggertokenid
                    tokens[x]['nearEvent']=nearEvent
                    tokens[x]['nearTrigger']=nearTrigger
                    tokens[x]['distFromTrigger']=encoded[min(abs(triggertokenid-x-1),26)]
                    tokens[x]['deppathtoTrigger']=deppath[triggertokenid]
                    tokens[x]['deppathtoTriggerLength']=encoded[min(distance[triggertokenid],26)]

                    if triggerposition < int(tokens[x]['characterOffsetBegin']):  # trigger x
                        tokens[x]['triggerPosition']='before'
                        for j in range(triggertokenid+1,x):
                            if tokens[j]['originalText']==',':
                                tokens[x]['triggerPosition']+='separated'
                                break

                    elif triggerposition > int(tokens[x]['characterOffsetBegin']): # x trigger
                        tokens[x]['triggerPosition']='after'
                        for j in range(x+1,triggertokenid):
                            if tokens[j]['originalText']==',':
                                tokens[x]['triggerPosition']+='separated'
                                break

            tokens=commonRootParse(tokens,parsetree)
            return tokens

def commonRootParse(tokens,parsetree):
    """add common root parse between the target token and the nearest event nugget into dict"""
    for x in range(len(tokens)):
        if tokens[x]['triggerPosition']!='this':
            source=tokens[x]['originalText']+'_@_'+str(x+1)
            triggertokenid=tokens[x]['nearTriggerTokenid']
            target=tokens[triggertokenid]['originalText']+'_@_'+str(triggertokenid+1)
            rootparseidx=['ARQ',u'-RRB-']
            #find common root, and its depth in parsetree between this token and trigger -> argument
            path = tree.find_path(parsetree,source ,target , [], {})
            if len(path)>1:
                if '-RRB-' in path:
                    path.remove('-RRB-')
                if "'s" in path:
                    path.remove("'s")
                common = tree.commonroot(parsetree,path)
                root=common.split('_@_')[0]

                tokens[x]['commonRootwTriggerParse']=root

                pathtoroot=tree.find_path(parsetree,common,'ROOT_@_0',[],{})
                tokens[x]['depthOfCommonRootwTrigger']=encoded[min(len(pathtoroot),25)]

        if 'nearTriggerTokenid' in tokens[x]:
            triggertokenid=tokens[x]['nearTriggerTokenid']
        else:
            continue
        if tokens[x]['ner']!='O' and 'nearEntityType' in tokens[x]:
            if len(tokens[x]['nearEntityType'])>0:
                for i in range(len(tokens[x]['nearEntityType'])):
                    entityid=[]
                    if tokens[x]['nearEntityType'][i]==tokens[x]['ner'][2:]:
                        tokens[x]['isOnly1ItsType']='no'

                        for j in range(len(tokens)):
                            if j!=x:
                                if tokens[j]['originalText']==tokens[x]['nearEntity'][i]:
                                    entityid.append(j)

                        if entityid:
                            dist=[]
                            dist.append(abs(triggertokenid-x))
                            for j in range(len(entityid)):
                                dist.append(abs(triggertokenid-entityid[j]))
                            if dist.index(min(dist))==0:
                                tokens[x]['isNearestItsType']='yes'
                            else:
                                tokens[x]['isNearestItsType']='no'
                        else:
                            print ('cannot find another entity position')
                        break
                else:
                    tokens[x]['isOnly1ItsType']='yes'
                    tokens[x]['isNearestItsType']='yes'

    return tokens

def combineNearEntity(tokens,deptree,enh,parsetree):
    """add near entity type into dict"""
    # find entities in the sentence
    entlist=[]
    ent=[]
    for x in range(len(tokens)):
        if tokens[x]['ner'].startswith('B-'):
            if ent:
                entlist.append(ent)
                ent=[]
            ent.append(x)
        elif tokens[x]['ner'].startswith('I-'):
            ent.append(x)
    entlist.append(ent)
 
    # find head of each entity
    headlist=[]
    for i in range(len(entlist)):        
        if len(entlist[i])>1:
            head=[]
            for j in entlist[i]:
                nodej=tokens[j]['originalText']+'_@_'+str(j+1)
                for k in entlist[i]:
                    if j!=k:
                        nodek=tokens[k]['originalText']+'_@_'+str(k+1)                                          
                        if nodek in deptree.keys() and nodej in deptree[nodek]:
                            if nodek not in head:
                                head.append(nodek)
                            if nodej in head:
                                head.remove(nodej)
                        elif nodej in deptree.keys() and nodek in deptree[nodej]:
                            if nodej not in head:
                                head.append(nodej)
                            if nodek in head:
                                head.remove(nodek)
            if not head:
                for node in deptree.keys():
                    node0=tokens[entlist[i][0]]['originalText']+'_@_'+str(entlist[i][0]+1)
                    if node0 in deptree[node]:
                        target=int(node.split('_@_')[1])-1
                        tokens[target]['ner']=tokens[entlist[i][0]]['ner']
                        head.append(node)
                        entlist[i].append(target)
                        break  
            if head:
                headlist.append(int(head[0].split('_@_')[1])-1)
            else:
                headlist.append(entlist[i][-1])
        elif len(entlist[i])==1:
            headlist.append(entlist[i][0])
 
    #set near entity list for each token
    if not headlist:
        return tokens
    rootparseidx=['ARQ',u'-RRB-']

    for x in range(len(tokens)):
        tokens[x]['nearEntity'],tokens[x]['nearEntityType']=[],[]
        tokens[x]['deppathtoEntity'],tokens[x]['deppathtoEntityLength']=[],[]
        tokens[x]['commonRootParse'],tokens[x]['depthOfCommonRoot']=[],[]
        tokens[x]['shortdeppath']=[]
        source=tokens[x]['originalText']+'_@_'+str(x+1)
        for i in range(len(entlist)):
            if x not in entlist[i]:
                tokens[x]['nearEntity'].append(tokens[headlist[i]]['originalText'])
                tokens[x]['nearEntityType'].append(tokens[headlist[i]]['ner'][2:])
                target=tokens[headlist[i]]['originalText']+'_@_'+str(headlist[i]+1)  
                # find dependency path and its length from this token to each entity head
                path = tree.find_path(deptree,source ,target ,[],{})

                deppath = tree.todeppath(path,enh)
                
                rpath = [a.split(':')[0] for a in deppath]
                short=[]
                for r in rpath:
                    if r not in short:
                        short.append(r)
                if not rpath:
                    rpath=['']
                if not short:
                    short=['']
                tokens[x]['deppathtoEntity'].append(rpath)
                tokens[x]['deppathtoEntityLength'].append(encoded[min(len(deppath),25)])
                
                tokens[x]['shortdeppath'].append(short)
                # find common root in parse tree from this token to each entity head
                path = tree.find_path(parsetree,source ,target , [], {})
                if '-RRB-' in path:
                    path.remove('-RRB-')                   
                if "'s" in path:
                    path.remove("'s")
                if len(path)>1:
                    common = tree.commonroot(parsetree,path)
                    root=common.split('_@_')[0]

                    tokens[x]['commonRootParse'].append(root)
                    pathtoroot=tree.find_path(parsetree,common,'ROOT_@_0',[],{})
                    tokens[x]['depthOfCommonRoot'].append(encoded[min(len(pathtoroot),25)])
                else:
                    tokens[x]['depthOfCommonRoot'].append('')
                    tokens[x]['commonRootParse'].append('')

    return tokens

def combineNearArgument(tokens, deptree, enh, parsetree, arglist):
    """find near argument type info into dict"""
    # find entities in the sentence
    secondArg=['Capabilities','Purpose']
    headlist=findhead(arglist,tokens,deptree)
    entlist={}
    ent=[]
    oldarg=''
    entityid=0
    for x in range(len(arglist)):
        text,position,argType,tokenid=arglist[x]
        if argType.startswith('B-'):
            if ent:
                entlist[entityid]={'tokenlist':ent, 'argType':oldarg}
                entityid+=1
                ent=[]
            ent.append(tokenid)
            oldarg=argType[2:]
        elif argType.startswith('I-'):
            ent.append(tokenid)
    entlist[entityid]={'tokenlist':ent, 'argType':oldarg}

    # set near entity list for each token
    if not headlist:
        return tokens
    rootparseidx = ['ARQ', u'-RRB-']

    for x in range(len(tokens)):
        tokens[x]['nearArgument'], tokens[x]['nearArgumentType'] = [], []
        tokens[x]['deppathtoArgument'], tokens[x]['deppathtoArgumentLength'] = [], []
        tokens[x]['commonRootParseWArg'], tokens[x]['depthOfCommonRootWArg'] = [], []
        tokens[x]['shortdeppathArg'] = []
        tokens[x]['isArgument']=False
        source = tokens[x]['originalText'] + '_@_' + str(x + 1)
        for i in entlist.keys():
            if x not in entlist[i]['tokenlist']:
                tokens[x]['nearArgument'].append(tokens[headlist[i]]['originalText'])
                tokens[x]['nearArgumentType'].append(entlist[i]['argType'])
                target = tokens[headlist[i]]['originalText'] + '_@_' + str(headlist[i] + 1)
                # find dependency path and its length from this token to each entity head
                path = tree.find_path(deptree, source, target, [], {})

                deppath = tree.todeppath(path, enh)

                rpath = [a.split(':')[0] for a in deppath]

                short = []
                for r in rpath:
                    if r not in short:
                        short.append(r)
                if not rpath:
                    rpath = ['']
                if not short:
                    short = ['']
                tokens[x]['deppathtoArgument'].append(rpath)
                tokens[x]['deppathtoArgumentLength'].append(encoded[min(len(deppath), 25)])

                tokens[x]['shortdeppathArg'].append(short)
                # find common root in parse tree from this token to each entity head
                path = tree.find_path(parsetree, source, target, [], {})
                if '-RRB-' in path:
                    path.remove('-RRB-')
                if "'s" in path:
                    path.remove("'s")
                if len(path) > 1:
                    common = tree.commonroot(parsetree, path)
                    root = common.split('_@_')[0]

                    tokens[x]['commonRootParseWArg'].append(root)
                    pathtoroot = tree.find_path(parsetree, common, 'ROOT_@_0', [], {})
                    tokens[x]['depthOfCommonRootWArg'].append(encoded[min(len(pathtoroot), 25)])
                else:
                    tokens[x]['depthOfCommonRootWArg'].append('')
                    tokens[x]['commonRootParseWArg'].append('')
            else:
                if entlist[i]['argType'] not in secondArg:
                    tokens[x]['isArgument']=True

    return tokens

def combine4Role(tokens,arglist,deptree,enh):
    """find features for role identification and add into dict"""
    args=[]
    surfaces,labels=[],[]
    a,s=[],[]
    oldlabel,thislabel='',''
    hasVersion=[]
    for t, offset, label, tokenid in arglist:

        if label.startswith('B-') or (label.startswith('I-') and label[2:]!=oldlabel):
            if a:
                args.append(a)
                surfaces.append(s)
                s,a=[],[]
            a.append(tokenid)
            s.append(tokens[tokenid]['originalText'])
            labels.append(label[2:])
            if label[2:]=='Version':
                hasVersion.append(tokenid)
        elif label.startswith('I-') and label[2:] == oldlabel:
            a.append(tokenid)
            s.append(tokens[tokenid]['originalText'])
        oldlabel = label[2:]


    args.append(a)
    surfaces.append(s)

    # find head of each argument
    headlist = findhead(arglist, tokens, deptree)

    #find nearest verb
    #verblist tokenid
    verblist,passivelist=[],[]
    for i in range(0,len(tokens)-1):
        if tokens[i]['pos'].startswith('V') and not(tokens[i+1]['pos'].startswith('V')):
            verblist.append(i)
            if 'dep_set' in tokens[i]:
                for x in tokens[i]['dep_set']:
                    if 'pass' in x:
                        passivelist.append(i)   
                        break

    ## assign left and right argument to each argument
    if len(args)==1:
        for j in args[0]:
            tokens[j]['leftargument'] = None
            tokens[j]['rightargument'] = None
    elif len(args)==2:
        for j in args[0]:
            tokens[j]['rightargument']=labels[1]
            tokens[j]['leftargument'] = None
        for j in args[1]:
            tokens[j]['rightargument'] = None
            tokens[j]['leftargument'] = labels[0]
    elif len(args)>2:
        for j in args[0]:
            tokens[j]['rightargument']=labels[1]
            tokens[j]['leftargument'] = None
        for j in args[-1]:
            tokens[j]['rightargument'] = None
            tokens[j]['leftargument'] = labels[-2]
        for i in range(1,len(args)-1,1):
            for j in args[i]:
                tokens[j]['rightargument'] = labels[i+1]
                tokens[j]['leftargument'] = labels[i-1]

    if len(hasVersion)>0:
        for verpos in hasVersion:
            for i in range(len(args)):
                if verpos in args[i]:
                    versionstr=i
                    break
            distancetosys,deppathtosys={},{}
            distancetopatch, deppathtopatch = {}, {}
            source = tokens[verpos]['originalText'] + '_@_' + str(verpos + 1)
            for i in range(len(labels)):
                if labels[i]=='System' or labels[i]=='Software' or labels[i]=='Device' or labels[i]=='Website':
                    #find deppath from version to system
                    target = tokens[headlist[i]]['originalText'] + '_@_' + str(headlist[i] + 1)
                    path = tree.find_path(deptree, source, target, [], {})

                    if isinstance(path, list) and len(path) > 1:
                        deppathtosys[i] = tree.todeppath(path, enh)
                        distancetosys[i] = len(deppathtosys[i])
                    else:
                        distancetosys[i] = 1000
                        deppathtosys[i] = ['-']
                elif labels[i]=='Patch':
                    target = tokens[headlist[i]]['originalText'] + '_@_' + str(headlist[i] + 1)
                    path = tree.find_path(deptree, source, target, [], {})

                    if isinstance(path, list) and len(path) > 1:
                        deppathtopatch[i] = tree.todeppath(path, enh)
                        distancetopatch[i] = len(deppathtopatch[i])
                    else:
                        distancetopatch[i] = 1000
                        deppathtopatch[i] = ['-']
            if len(distancetopatch)>0:
                mindisttopatch = min(distancetopatch.items(), key=lambda a: a[1])[0]
                for j in args[versionstr]:
                    tokens[j]['distVer2NearestPatch'] = encoded[min(distancetopatch[mindisttopatch], 26)]
                    tokens[j]['deppathVer2NearestPatch'] = deppathtopatch[mindisttopatch]

            if len(distancetosys) > 0:
                mindisttosys = min(distancetosys.items(), key=lambda a: a[1])[0]
                for j in args[versionstr]:
                    tokens[j]['distVer2NearestSystem']=encoded[min(distancetosys[mindisttosys],26)]
                    tokens[j]['deppathVer2NearestSystem']=deppathtosys[mindisttosys]

    for i in range(len(headlist)):
        head=tokens[headlist[i]]['originalText']+'_@_'+str(headlist[i]+1)
        nearestverb,distance=tree.verbsamedept(head,verblist,deptree)
        if nearestverb!=-1:
            tokens[headlist[i]]['surfaceentity']=" ".join(surfaces[i])
            tokens[headlist[i]]['nearestverbpos']=tokens[nearestverb]['pos']
            tokens[headlist[i]]['nearestverb']=tokens[nearestverb]['originalText']
            tokens[headlist[i]]['dist2nearestverb']=encoded[min(distance,25)]
            source=tokens[headlist[i]]['originalText']+'_@_'+str(headlist[i]+1)  
            target=tokens[nearestverb]['originalText']+'_@_'+str(nearestverb+1)  
            path = tree.find_path(deptree,source ,target ,[],{})
            if isinstance(path, list) and len(path)>1:
                deppath = tree.todeppath(path,enh)
            else:                
                deppath =['-']
            
            tokens[headlist[i]]['deppathtonearestverb']=deppath
            if nearestverb in passivelist:
                tokens[headlist[i]]['passive']='yes'
            else:
                tokens[headlist[i]]['passive']='no'
    return tokens

devicelist=['wificam','hardware','appliance','kit','gateway','bear','toy','doll','thermostat','fridge','door','kettle','monitor', 'webcam','desktop','television','car','vehicle','transmitter','pacemaker','electronic','model','microphone','speaker', 'register','handset','stripe','card','oven','pump','harddrive','scanner','recorder','smartwatch','wearable','watch', 'TV','accelerometer','gadget','smartphone','sensor','workstation','gear','switch','router','CPU','defibrillator', 'pacemaker','chipset','equipment','adapter','printer','platform','product','family','fuse','console','chip','memory', 'robot','camera', 'machine','device','component','module','PC','laptop','computer','modem','processor']
gpelist=['country', 'internationally']
syssoftlist=['network', 'system', 'server', 'page', 'website', 'site', 'interface', 'service', 'portal', 'version', 'module', 'component', 'subsystem', 'library', 'framework', 'product', 'mode', 'installation', 'engine', 'management',  'code', 'software', 'codebase', 'fork', 'platform', 'browser', 'login', 'drive', 'core', 'filesystem', 'technology', 'repository', 'feature', 'database', 'distro', 'distribution', 'client', 'suite', 'tool', 'assessment', 'desktop', 'console', 'landscape','functionality', 'domain', 'rail', 'function','implementation', 'panel', 'security',  'process', 'intranet', 'server-side', 'redirection',  'channel', 'traffic', 'host', 'controller',  'handler', 'verification', 'infrastructure','product', 'app', 'application', 'software', 'installer', 'game', 'browser',  'project', 'program', 'tool', 'client', 'installation', 'production', 'blunder', 'image', 'solution', 'interface', 'firewall', 'service', 'services', 'variant', 'extension', 'script', 'update', 'anti-virus', 'module', 'driver']
orgperlist=['party','group','target']
perlist=['party','group','target','patient', 'staff', 'woman', 'person', 'customer', 'worker', 'employee', 'volunteer', 'student', 'family', 'member', 'passenger', 'hacker', 'parent', 'teacher', 'citizen', 'fraudster', 'complainant', 'attacker', 'resident', 'researcher', 'reader', 'conspirator', 'official', 'someone', 'victim', 'infiltrator', 'individual', 'user', 'scammer', 'consumer', 'actor', 'thief', 'expert', 'boss', 'artist', 'criminal', 'executive', 'friend', 'man',  'conman', 'guest', 'malefactor', 'cybercriminal', 'developer', 'guy', 'authority', 'anyone', 'assailant','perpetrator', 'cyber-attacker', 'operator', 'owner', 'crook', 'extortionist',  'culprit', 'recipient', 'colleague', 'journalist', 'girl', 'buyer', 'administrator',  'scamster', 'subscriber', 'visitor', 'phisher', 'inspector', 'taxpayer', 'sender', 'swindler',  'cyberattacker', 'client', 'shopper', 'author',  'motorist', 'leader', 'caller', 'investor', 'defendant',  'somebody', 'taxman', 'creator', 'engineer', 'imposter', 'partner', 'folk', 'officer',  'stealer', 'cyber-criminal', 'threat', 'player', 'advisory',  'gamer', 'athlete', 'other', 'director', 'seeker', 'voter',  'insider', 'miscreant', 'spokesperson', 'veteran',  'end-user', 'duo', 'politician', 'celebrity',  'crew', 'chairman', 'intruder', 'cryptologist', 'blackmailer', 'seller', 'chief', 'source', 'maker', 'moniker', 'employer', 'preparer', 'spear-phisher', 'brother', 'sister', 'activism', 'producer', 'speaker', 'physician', 'trickster', 'manager', 'influencer',  'child', 'demander', 'adult', 'kid', 'co-worker', 'stranger', 'serviceman', 'contractor', 'adversary', 'maintainer', 'distributor','team','diplomat','cyberoperative']
orglist=['party','group','target','town', 'hospital', 'village', 'district', 'firm', 'organization', 'organisation', 'association', 'supplier', 'company', 'third-party', 'group', 'vendor', 'department', 'business', 'finance', 'manufacturer', 'giant', 'party', 'hotel', 'bank', 'sector', 'institution', 'victim', 'government', 'county', 'city', 'national', 'state', 'provider', 'enforcement', 'industry', 'brand', 'maker', 'operation', 'enterprise',  'community', 'advisory', 'chipmaker', 'project', 'site', 'charity', 'municipality', 'corporation', 'logistic', 'program', 'pair', 'office', 'authority', 'port', 'school', 'channel', 'entity', 'source', 'chain', 'host', 'subcontractor', 'facility', 'healthcare', 'shipping', 'other', 'university', 'asset', 'restaurant', 'bar', 'branding', 'airline', 'copycat', 'studio', 'establishment', 'telecom','library', 'management', 'ministry', 'utility','college', 'partner','store', 'clinic','artillery','payroll']

nermap={'Person':['human','whistleblower','hacktivism','black hat','hacker group','security hacker'],
'Organization':['organization','company','government organization','computer security consultant','intelligence agency','software company','social media'],
'GPE':['geographic region','political territorial entity'],
'Software': ['software','process','instant messaging','authentication protocol','operating system','mobile app','Dropper','software release','instant messaging client'],
'Version':['software version'],
'Device': ['computer','mobile phone','computer hardware','computer memory'],
'System': ['network','service on internet','payment system'],
'Currency':['cryptocurrency','currency'],
'File':['computer file'],
'CVE':['Common Vulnerabilities and Exposures'],
'Tool': ['malware','botnet']}


def findNNPNN(tokens):
    """find named entities pronoun"""
    unknnplist=[]
    latestnnp=-2
    unkword=''
    nnlist=[]
    latestnn=-2
    nnword=''
    start,end=0,0
    s,e=0,0
    for i in range(len(tokens)):
        if tokens[i]['pos'].startswith('NNP') and tokens[i]['ner']=='O':
            if latestnnp==i-1:
                unkword+=' '+tokens[i]['originalText']
                e=i
            elif not unkword:
                unkword=tokens[i]['originalText']
                s=i
                e=i
            elif unkword:
                unk=(unkword.strip(), (s,e))
                unknnplist.append(unk)               
                unkword=tokens[i]['originalText']     
                s=i
                e=i
            latestnnp=i
        elif tokens[i]['pos'].startswith('NN') and tokens[i]['ner']=='O':
            if latestnn==i-1:
                nnword+=' '+tokens[i]['originalText']
                end=i
            elif not nnword:
                nnword=tokens[i]['originalText']
                start=i
                end=i
            elif nnword:      
                nn=(nnword.strip(),(start,end))
                nnlist.append(nn)
                
                nnword=tokens[i]['originalText'] 
                start=i
                end=i
            latestnn=i
    if unkword:
        unk=(unkword.strip(), (s,e))
        unknnplist.append(unk)
    if nnword:
        nn=(nnword.strip(),(start,end))
        nnlist.append(nn)

    return unknnplist,nnlist


def findnearbyner(nn,position,nertype,sentno,features):
    """find named entities"""
    nnpos=features[sentno][position[1]]['pos']  

    for i in range(sentno,-1,-1):
        for tokens in features[i]:
            if tokens['ner'][2:] in nertype:
                return tokens['ner']

def combineNNNer(nnlist,features,corefs):
    """add more named entities pronoun into dict"""
    for sentno in nnlist.keys():
        for nnstr, position in nnlist[sentno]:
            nn=nnstr.split()[-1]
            near=[]
            if features[sentno][position[1]]['pos']=='NNS':
                nn=infl.singular_noun(nn)            
            if nn in orglist:
                near.append('Organization')
            elif nn in perlist:
                near.append('Person')            
            elif nn in syssoftlist:
                near.append('System')
                near.append('Software')
            elif nn in devicelist:
                near.append('Device')
            elif nn in gpelist:
                near.append('GPE')
            
            ner=findnearbyner(nn,position,near,sentno,features)
            if ner:
                if features[sentno][position[1]]['originalText']==nn:
                    features[sentno][position[1]]['ner']=ner

    return features

def trggrtophrase(trggrs,headlist):
    """combine triggers to phrase and find header of trigger
        Input:  trggrs list of words which labeled as trigger using BIO
                headlist list of trigger head
        Output: trggrlist list of trigger in phrase
    """
    trggrlist=[]
    startToken=0
    phrase=''

    for tr in trggrs:
        text, position, eventType, tokenid = tr
        if eventType.startswith('B-'):
            if phrase:
                trggrlist.append({'startToken': startToken, 'text': phrase.strip(), 'endToken': endToken, 'eventType':oldeventType[2:]})
            startToken=tokenid
            phrase=''
        phrase+=' '+text
        endToken=tokenid
        oldeventType=eventType
    if phrase:
        trggrlist.append({'startToken': startToken, 'text': phrase.strip(), 'endToken': endToken, 'eventType':oldeventType[2:]})

    assert(len(headlist)==len(trggrlist))
    for i in range(len(trggrlist)):
        if headlist[i]>=trggrlist[i]['startToken'] and headlist[i]<=trggrlist[i]['endToken']:
            trggrlist[i]['head']=headlist[i]
        else:
            print ('head not found')

    return trggrlist


def combine4Realis_dep(tokens,trggrs,deptree,datestr):
    """ add features for realis classification
        Input:  tokens dict contained pre-computed features
                trggrs list of triggers in this sentence
                args list of arguments in this sentence
    """
    general=['every','always', 'constantly', 'daily', 'frequently', 'often', 'regularly', 'eventually',  'sometimes','annually', 'weekly', 'yearly', 'hourly', 'generally', 'occasionally', 'monthly', 'quarterly', 'normally', 'nightly']
    other=['infrequently', 'never', 'rarely', 'soon', 'yet', 'later', 'next', 'then']
    actual=['now', 'today', 'tonight', 'yesterday', 'ever']
    negative=['not','no']
    cond=['either', 'or', 'whether', 'if', 'unless']
    datetype=[ 'DATE','TIME','DURATION','SET'] # SET is a mention of frequency

    for tr in trggrs:
        text=tr['text']
        tokenid = tr['head']
        tokens[tokenid]['nuggetphrase']=text
        tokens[tokenid]['startToken']= tr['startToken']
        tokens[tokenid]['endToken'] = tr['endToken']
        eventType=tr['eventType']

        trggrNode = tokens[tokenid]['originalText'] + '_@_' + str(tokenid + 1)
        leave,sibling=[],[]

        # find suspect node from trigger's leave, trigger's siblings, and trigger's root
        if trggrNode in deptree.keys():
            leave=deptree[trggrNode]
        for node in deptree.keys():
            if trggrNode in deptree[node]:
                sibling=deptree[node]
                sibling.append(node)
                break

        suspect=set(leave+sibling)
        vblist=[]
        for node in suspect:
            nodeid=int(node.split('_@_')[1])-1

            if tokens[nodeid]['lemma'] in negative:
                # negative
                tokens[tokenid]['neg'] = True

            if tokens[nodeid]['pos'] == 'MD':
                # modal
                tokens[tokenid]['modal'] = True

            if tokens[nodeid]['originalText'] == 'going':
                # future
                if nodeid in deptree.keys():
                    for l in deptree[nodeid]:
                        lid=int(l.split('_@_')[1])-1

                        if tokens[lid]['lemma']==be:
                            tokens[tokenid]['future'] = True
                            break
            subSuspect=[]
            if tokens[nodeid]['pos'].startswith('NN') and 'neg' not in tokens[tokenid]:
                if node in deptree.keys():
                    subSuspect=deptree[node]
                for snode in subSuspect:
                    snodeid = int(snode.split('_@_')[1]) - 1
                    if tokens[snodeid]['lemma']=='no':
                        tokens[tokenid]['neg']=True
                        break

            if tokens[nodeid]['pos'].startswith('VB'):
                vblist.append(tokens[nodeid]['pos'])
        samevb=''
        for vb in vblist:
            if vb==tokens[tokenid]['pos']:
                samevb=vb
                break
        if samevb:
            vblist.remove(samevb)
        tokens[tokenid]['vblist']=vblist

        for i in range(0, len(tokens)):
            # conditional form
            if tokens[i]['lemma'] in cond:
                tokens[tokenid]['cond'] = True
                break

        for i in range(0, len(tokens)):
            # frequency
            if tokens[i]['originalText'] in actual:
                tokens[tokenid]['freq'] = 'a'
                break
            elif tokens[i]['originalText'] in general:
                tokens[tokenid]['freq'] = 'g'
                break
            elif tokens[i]['originalText'] in other:
                tokens[tokenid]['freq'] = 'o'
                break

        tidlist,timetype=[],[]
        yearr,monthr,dayr=datestr.split('_')
        dr = date(int(yearr), int(monthr), int(dayr))

        for i in range(0,len(tokens)):
            #datetime

            if tokens[i]['ner'][2:].upper() in datetype:
                # ner
                if 'timex' in tokens[i]:
                    timex=tokens[i]['timex']
                    tid=timex['tid']
                    if tid not in tidlist:
                        tidlist.append(tid)
                        timetype.append(tokens[i]['originalText'][2:])
                else:
                    timetype.append(tokens[i]['originalText'][2:])

        tokens[tokenid]['time']=timetype

        if 'neg' not in tokens[tokenid]:
            tokens[tokenid]['neg']=False
        if 'cond' not in tokens[tokenid]:
            tokens[tokenid]['cond']=False
        if 'modal' not in tokens[tokenid]:
            tokens[tokenid]['modal']=False
        if 'future' not in tokens[tokenid]:
            tokens[tokenid]['future']=False
        if 'freq' not in tokens[tokenid]:
            tokens[tokenid]['freq'] = None

    return tokens

def processNER(nerinfo):
    """add named entities from external knowledge base into dict"""
    wknelist1, wknelist2, casielist, dbpnelist =[],[],[],[]
    for ner in nerinfo['NERinfo']:
        text=ner['text'].split()
        start = int(ner['startOffset'])
        wikiner,casiener,dbpner=[],'',''
        if 'wikidata' in ner:
            if 'types' in ner['wikidata']:
                wikiner=ner['wikidata']['types']
                wknelist1.append((text[0], start, 'B-' + wikiner[0]))
                if len(wikiner) > 1:
                    wknelist2.append((text[0], start, 'B-' + wikiner[1]))
        if 'dbpedia' in ner:
            if 'type' in ner['dbpedia']:
                dbpner=ner['dbpedia']['type']
                dbpnelist.append((text[0], start, 'B-' + dbpner))
        if 'casietype' in ner:
            casiener=ner['casietype']
            casielist.append((text[0], start, 'B-' + casiener))

        for t in range(1,len(text),1):
            start+=len(text[t-1])+1
            if wikiner:
                wknelist1.append((text[t], start, 'I-' + wikiner[0]))
                if len(wikiner) > 1:
                    wknelist2.append((text[t], start, 'I-' + wikiner[1]))
            if casiener:
                casielist.append((text[t], start, 'I-' + casiener))
            if dbpner:
                dbpnelist.append((text[t], start, 'I-' + dbpner))
    return wknelist1, wknelist2, casielist, dbpnelist

def parse(jsonfile,labelfile,options):
    """entry point to add feature into dict"""
    parsefile=jsonfile.replace('ver3n/','ver3n/json4parsetree/')
    parsecontent = utils.loadJsontoDict(parsefile)
    content=utils.loadJsontoDict(jsonfile)
    nefile=jsonfile.replace('.content.json','.ner.json')
    nerinfo=utils.loadJsontoDict(nefile)
    wknelist1, wknelist2, casienelist, dbpnelist = processNER(nerinfo)

    trggrlist,arglist=[],[]
    lbcontent = utils.readFileEncode(labelfile, 'utf8')
    if options=='argument': # second pass, already have trigger labels
        trggrlist=processLabel(lbcontent,content,'nugget')
    elif options=='role' or options=='realis'  or options=='secondargument': # third pass, had trigger and arguments
        trggrlist =  processLabel(lbcontent,content,'nugget')
        arglist = processLabel(lbcontent,content,'argument')

    features,deptree,parsetree={},{},{}
    nnplist,nnlist={},{}

    for i in range(len(content["sentences"])):
        x=content["sentences"][i]
        deptree[x["index"]]=tree.build_deptree(x["enhancedPlusPlusDependencies"])
        y=parsecontent["sentences"][i]
        parsetree[y["index"]]=tree.build_parsetree(removeinnermost(y["parse"]),y["tokens"])

        nplist,vplist,depthNP,depthVP=tree.list_chunk(parsetree[y["index"]])

        features[x["index"]]=combineDep(x)
        features[x["index"]]=combineChnkLvl(features[x["index"]],True,nplist,depthNP)
        features[x["index"]]=combineChnkLvl(features[x["index"]],False,vplist,depthVP)
        features[x['index']] = combineExtNe(features[x["index"]], wknelist1, wknelist2, casienelist, dbpnelist)
        features[x["index"]] = combineNe_st(features[x["index"]])
        if options=='secondargument' and x['index'] in arglist.keys():
            features[x["index"]]= combineNearArgument(features[x["index"]],deptree[x['index']],x["enhancedPlusPlusDependencies"],parsetree[x['index']],arglist[x["index"]])

    for sentno in features.keys():
        nnplist[sentno],nnlist[sentno]=findNNPNN(features[sentno])
    features=combineNNNer(nnlist,features,content["corefs"])
    features = readCoref(features, content['corefs'])
    features=combineCoref(features,content["corefs"])

    if options=='argument' or options=='role':
        for x in range(len(features)):
            features[x]=combineNearEntity(features[x],deptree[x],content["sentences"][x]["enhancedPlusPlusDependencies"],parsetree[x])

            if options=='argument' or options=='secondargument': # have trigger use for find argument
                features[x]=combineNearTrggr(features[x],trggrlist, x,deptree[x],content["sentences"][x]["enhancedPlusPlusDependencies"],parsetree[x])

            if options=='role' and x in arglist.keys(): # have trigger and arguments use for find role
                features[x]=combineNearTrggr(features[x],trggrlist, x,deptree[x],content["sentences"][x]["enhancedPlusPlusDependencies"],parsetree[x])
                features[x]=combine4Role(features[x],arglist[x],deptree[x],content["sentences"][x]["enhancedPlusPlusDependencies"]) 

    if options=='realis':
        txtfile = jsonfile.replace('.content.json', '.txt')
        txtcontent=utils.readFileEncode(txtfile,'utf8')
        start=txtcontent.index("<date>")
        end=txtcontent.index("</date>")
        datestr=txtcontent[start+6:end].strip()

        for x in range(len(features)):
            # may set source of name entity
            features[x] = combineNearEntity(features[x], deptree[x],content["sentences"][x]["enhancedPlusPlusDependencies"], parsetree[x])
            if x>0:
                features[x]=combineNearTrggr(features[x],trggrlist, x,deptree[x],content["sentences"][x]["enhancedPlusPlusDependencies"],parsetree[x])
            else:
                features[x]=combineNearTrggr(features[x],trggrlist, x,deptree[x],content["sentences"][x]["enhancedPlusPlusDependencies"],parsetree[x])
            if x in trggrlist:
                headlist = findhead(trggrlist[x], features[x], deptree[x])
                tlist=trggrtophrase(trggrlist[x],headlist)
                features[x] = combine4Realis_dep(features[x],tlist,deptree[x],datestr)

    return features
    
infl=inflect.engine()


