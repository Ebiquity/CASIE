from __future__ import absolute_import
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import sys
sys.path.append('../')

import utils
import parseJsontoFeatures
import prepare
import x2index
import role_phrase
import realis_identify

import codecs
import copy
import argparse
import re
import gensim
import operator
import pickle
import datetime
from keras_contrib.utils import save_load_utils
import os
from nltk.corpus import stopwords
import string
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine,squareform

TriggerList=["Phishing","DiscoverVulnerability","Databreach","PatchVulnerability","Ransom"]
ArgumentList=["Patch","Data","Money","Time","Organization","GPE","File","Version","Person","Vulnerability",
"PaymentMethod","CVE","Device","Website","System","Number","PII","Malware","Capabilities","Purpose","Software"]
Pair={'Phishing':["Capabilities","File","GPE","Money","Number","Organization","Person","Website","System","PII","Malware","Purpose","Time","Data","Software"],
      'Ransom':["Capabilities","Money","Time","Organization","GPE","File","Person","PaymentMethod","Device","System","Number","Malware","Website","Software"],
      'Databreach':["Data","Money","Time","Organization","GPE","File","Person","Device","System","Number","Website","PII","Malware","Capabilities","Purpose","Software"],
      'DiscoverVulnerability':["Time","Organization","Version","Person","Vulnerability","CVE","Device","System","Capabilities","Website","Software"],
      'PatchVulnerability':["Time","Organization","Version","Person","Vulnerability","CVE","Device","System","Capabilities","Patch","Website","Software"]}
Wordset={'DiscoverVulnerability':['find','publish','identify','demonstrate','discover','investigate','observe','uncover','notice','notify','acknowledge','exist','report','say','exploit','reside','study','reveal','disclose','expose','declare','introduce','develop','exposure','explain','describe','vulnerable','appear','suggest','indicate'],
         'PatchVulnerability':['install','firmware','reportedly','note','deploy','reveal','develop','build','address','resolve','announce','release','update','bug','fix','issue','improve','distribute'],
         'Databreach':['eavesdrop','intercept','credential-harvesting','leakage','dump','sold','harvest','corrupted','copied','collecting','handed','published','grab','steal','privacy','obtain','pilfered','collected','sensitive','circulated','compromised','allegedly','unauthorized'],
         'Ransom':['asking','demanding','payment','ransom','bitcoin','provide','refusing','paying','effort','holding','payout','infects','disrupted','requesting','stating','instruct','extorting','wannacry'],
         'Phishing':['appear','defraud','trick','hit','phishing','lure','claim','pose','load','display','distribute','redirect','disguise','pretending','serve','impersonate','create','scam','dupe','sent','craft','misleading']}
GeneralN={'general':['attack','attacks','cyberattacks','cyber-attack','incident','incidents','threats'],
          'Databreach':['breach','breaches'],
          'PatchVulnerability':['patch','patches']}
DoNothing= {'Databreach':['Time','Purpose','Capabilities'],
            'Phishing':['Money','Time','Purpose','Capabilities'],
            'Ransom':['Time','Purpose','Capabilities'],
            'PatchVulnerability':['Time','Purpose','Capabilities'],
            'DiscoverVulnerability':['Person','Time','Purpose','Capabilities']}
Date=['Date','Time','Duration']
GeneralArg={'Vulnerability':['flaw', 'flaws', 'issues', 'issue','bug','bugs'],'Patch':['patch','release','update']}

RolePair={'Phishing':{"Capabilities":["Attack-Pattern"],"File":["Trusted-Entity","Tool"],"GPE":["Place"],\
                      "Money":["Damage-Amount"],"Number":["Number-of-Victim"], \
                      "Organization":["Attacker","Victim","Trusted-Entity"],"Person":["Attacker","Victim","Trusted-Entity"], \
                      "Website":["Trusted-Entity","Tool"],"System":["Trusted-Entity"],"PII":["Trusted-Entity"],\
                      "Malware":["Tool"],"Purpose":["Purpose"],"Time":["Time"],"Data":["Trusted-Entity"]},
          'Ransom':{"Capabilities":["Attack-Pattern"],"Money":["Ransom-Price"],"Time":["Time"], \
                "Organization":["Attacker","Victim"],"GPE":["Place"],"File":["Tool"],"Person":["Attacker","Victim"],\
                "PaymentMethod":["Payment-Method"],"Device":["Victim"],"System":["Victim"],"Number":["Number-of-Victim"],\
                "Malware":["Tool"],"Website":["Victim"]},
      'Databreach':{"Data":["Compromised-Data"],"Money":["Damage-Amount"],"Time":["Time"],"Organization":["Attacker","Victim"],\
                    "GPE":["Place"],"File":["Tool"],"Person":["Attacker","Victim"],"Device":["Victim"],"System":["Victim"], \
                    "Number":["Number-of-victim","Number-of-Compromised-Data"],"Website":["Victim"],"PII":["Compromised-Data"],\
                    "Malware":["Tool"],"Capabilities":["Attack-Pattern"],"Purpose":["Purpose"]} ,
      'DiscoverVulnerability':{"Time":["Time"],"Organization":["Attacker","Victim"],"Version":"Vulnerable-System-Version", \
                               "Person":["Discoverer","Vulnerable-System-Owner"],"Vulnerability":["Vulnerability"], \
                               "CVE":["CVE"],"Device":["Vulnerable-System","Supported-Platform"], \
                               "System":["Vulnerable-System","Supported-Platform"],"Capabilities":["Capabilities"],\
                               "Website":["Vulnerable-System","Supported-Platform"]} ,
      'PatchVulnerability':{"Time":["Time"],"Organization":["Patch-Releaser","Vulnerable-System-Owner"],\
                            "Version":["Patch-Number","Vulnerable-System-Version"],"Person":["Patch-Releaser","Vulnerable-System-Owner"],\
                            "Vulnerability":["Vulnerability"],"CVE":["CVE"],"Device":["Vulnerable-System","Supported-Platform"], \
                            "System":["Vulnerable-System","Supported-Platform"],"Capabilities":["Issues-Addressed"], \
                            "Patch":["Patch"],"Website":["Vulnerable-System","Supported-Platform"]}}
MaxLength={'Databreach':8,'Phishing':6, 'Ransom':7,'DiscoverVulnerability': 8,'PatchVulnerability': 7 }
Realis=['GNG','AO']
Month={1:"January", 2:"February", 3:"March", 4:"April",5:"May",6:"June",7:"July",8:"August",\
       9:"September",10:"October",11:"November",12:"December"}
Day={1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday",7:"Sunday"}
TimeDiff={'Databreach':30 ,'Phishing':30, 'Ransom':30,'DiscoverVulnerability':30,'PatchVulnerability':30 }

def load_model_realis(realistype):
    """load trained model for predicting realis"""
    modelfile = 'bestmodel/realis_'+realistype+'.h5'
    labelfile = 'bestmodel/realis_label_'+realistype+'.pkl'
    wordfile = 'bestmodel/realis_word_'+realistype+'.pkl'
    wordidx=pickle.load(open(wordfile, 'rb'))
    labelidx = pickle.load(open(labelfile, 'rb'))
    model=bilstm_realis.build_model(wordidx, labelidx, MAX_LENGTH=14)
    save_load_utils.load_all_weights(model, modelfile, include_optimizer=False)

    return model,wordidx,labelidx

def load_model_role(eventtype):
    """load trained model for predicting role"""
    flags = (True, True, True, True, True, False, False)

    wordfile='bestmodel/role_word_'+eventtype+'.pkl'
    labelfile='bestmodel/role_label_'+eventtype+'.pkl'
    featurefile='bestmodel/role_feature_'+eventtype+'.pkl'
    classweightfile='bestmodel/role_classweight_'+eventtype+'.pkl'
    modelfile = 'bestmodel/role_'+eventtype+'.h5'

    wordidx=pickle.load(open(wordfile, 'rb'))
    featureidx=pickle.load(open(featurefile, 'rb'))
    labelidx = pickle.load(open(labelfile, 'rb'))
    classweight=pickle.load(open(classweightfile,'rb'))
    allidx=wordidx, labelidx, featureidx
    model = bilstm_role_phrase.build_model(allidx,MaxLength[eventtype],classweight,flags)
    save_load_utils.load_all_weights(model, modelfile, include_optimizer=False)

    return model,wordidx,featureidx,labelidx

def role_predict(fname,dir,databreach_role_model,phishing_role_model,ransom_role_model,discover_role_model,patch_role_model,wordidx,featureidx,labelidx):
    """predict whether sub-sentence is capability or not"""
    filename = dir + fname + '_pred.content.nostop.label'
    fsentences, flabels = bilstm_role_phrase.process_input_phrase(filename)
    jfile = dir + fname + '.content.json'

    if os.path.isfile(jfile):
        each_sent_feature = parseJsontoFeatures.parse(jsonfile=jfile, labelfile=filename, options='role')
    else:
        print ('no content.json file', jfile)
        return

    result = {}

    for etype in MaxLength.keys():

        testset,test_labels=[],[]
        testset, test_labels = prepare.features_role_phrase(fsentences,each_sent_feature,testset,\
                                                        etype,flabels,test_labels,True)
        string_test={}
        word_etype_idx=wordidx[etype]
        label_etype_idx=labelidx[etype]
        feat_etype_idx=featureidx[etype]

        train_X, test_X, train_features, test_features, train_y, test_y = x2index.sampleRolePhrase2idx([], testset, [], test_labels,word_etype_idx,label_etype_idx,feat_etype_idx, flags)
        if len(test_X)==0:
            result[etype] = string_test
            continue

        train_mat,test_mat,train_y,test_y=prepare.mat_phrase(train_X,test_X,train_features,test_features,train_y,test_y,MaxLength[etype])
        if etype=='Databreach':
            y_pred = databreach_role_model.predict(test_mat)
        elif etype=='Phishing':
            y_pred = phishing_role_model.predict(test_mat)
        elif etype=='Ransom':
            y_pred = ransom_role_model.predict(test_mat)
        elif etype=='DiscoverVulnerability':
            y_pred = discover_role_model.predict(test_mat)
        elif etype=='PatchVulnerability':
            y_pred = patch_role_model.predict(test_mat)

        y_classes = y_pred.argmax(axis=-1)
        labelidxlist = list(label_etype_idx.keys())
        labelvaluelist = list(label_etype_idx.values())

        textlist, offsetlist, filelist = [], [], []
        for i in range(len(testset)):
            for j in range(len(testset[i])):
                if testset[i][j]:
                    textlist.append(testset[i][j]['text'])
                    offsetlist.append(testset[i][j]['firstoffset'])

        if len(textlist) == len(y_pred):
            for i in range(len(y_pred)):
                predicttag = labelidxlist[labelvaluelist.index(y_classes[i])]
                words=textlist[i].split()
                string_test[i]={'text':textlist[i],'offset':offsetlist[i],'pred':predicttag}
        result[etype]=string_test
    return result

def realis_predict(fname,dir,realis_GNG_model,realis_AO_model,wordidx,labelidx,realistype):

    filename = dir + fname + '_pred.content.label'

    jfile = dir + fname + '.content.json'
    if os.path.isfile(jfile):
        each_sent_feature = parseJsontoFeatures.parse(jsonfile=jfile, labelfile=filename, options='realis', useextra=False)
    else:
        print ('no content.json file', jfile)
        return

    fsentences, flabels = bilstm_realis.process_input_phrase(filename, realistype)
    testset, offsets = prepare.features_realis_sentence(fsentences, each_sent_feature)

    string_test={}
    word_idx=wordidx[Realis[realistype-1]]
    label_idx=labelidx[Realis[realistype-1]]

    train_X, test_X, train_y, test_y = x2index.sampleWordPhrase2idx([], testset, [], flabels, word_idx, label_idx)

    if len(test_X)==0:
        return string_test

    train_mat, test_mat = {}, {}
    train_mat['main_input'] = pad_sequences(train_X, maxlen=14, padding='post')
    test_mat['main_input'] = pad_sequences(test_X, maxlen=14, padding='post')

    if realistype==1:
        y_pred = realis_GNG_model.predict(test_mat)
    elif realistype==2:
        y_pred = realis_AO_model.predict(test_mat)

    y_classes = y_pred.argmax(axis=-1)
    labelidxlist = list(label_idx.keys())
    labelvaluelist = list(label_idx.values())

    textlist, filelist = {}, []
    for i in range(len(testset)):
        textlist[i]=[]
        for j in range(len(testset[i])):
            if testset[i][j]:
                textlist[i].append(testset[i][j]['text'])
    if len(textlist) == len(y_pred):
        for i in range(len(y_pred)):
            predicttag = labelidxlist[labelvaluelist.index(y_classes[i])]
            string_test[i]={'text':textlist[i],'offset':offsets[i]['offset'],'pred':predicttag}

    return string_test

def getpubdate(txt_input):
    """retrieved publication date"""
    txtcontent=utils.readFileEncode(txt_input,'utf8')
    datebegin=txtcontent.index("<date>")
    dateend=txtcontent.index("</date>")
    strdate=txtcontent[datebegin+7:dateend]
    spdate=strdate.split('_')
    pubdate=[int(x) for x in spdate]
    return pubdate

def similarity(verblist,eventType):
    global w2vmodel
    wordlist=Wordset[eventType]
    sim=[]
    for verb in verblist:
        wordsim=[]
        if verb in w2vmodel.wv.vocab:
            for word in wordlist:
                wordsim.append(w2vmodel.wv.similarity(verb.lower(), word.lower()))
            sim.append(max(wordsim))
        else:
            sim.append(0.0)
    return verblist[sim.index(max(sim))]

def avgsim(verb,eventType):
    global w2vmodel
    wordlist=Wordset[eventType]
    if verb.lower() not in w2vmodel.wv.vocab:
        return 0.0
    wordsim=0.0
    for word in wordlist:
        if word.lower() not in w2vmodel.wv.vocab:
            wordsim+=0.0
        else:
            wordsim+=w2vmodel.wv.similarity(verb.lower(), word.lower())
    avg=float(wordsim)/float(len(wordlist))

    return avg

def maxsim(verb,eventType):
    global w2vmodel
    wordlist=Wordset[eventType]

    if verb.lower() not in w2vmodel.wv.vocab:
        return 0.0
    wordsim=[]
    for word in wordlist:
        if word.lower() not in w2vmodel.wv.vocab:
            wordsim.append(0.0)
        else:
            wordsim.append(w2vmodel.wv.similarity(verb.lower(), word.lower()))
    max_sim=max(wordsim)

    return max_sim

def avg_sentence_vector(words):
    """function to average all words vectors in a given paragraph"""
    global w2vmodel
    featureVec = np.zeros((100,), dtype="float32")
    nwords = 0

    for word in words:
        if word in w2vmodel.wv.vocab:
            nwords = nwords+1
            featureVec = np.add(featureVec, w2vmodel[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def strdist(str1,str2):
    """distance between two strings"""
    s1 = avg_sentence_vector(str1)
    s2 = avg_sentence_vector(str2)

    if np.count_nonzero(s1)>1 and np.count_nonzero(s2)>1:
        dist=cosine(s1,s2)/2
    else:
        dist = 1.0
    return dist # same return 1.0

def iscoref(arg1,arg2,features):
    """is argument text 1 in coref of argument2"""
    sentno2, position2 = arg2['pair']
    startoffset=arg1['startOffset']
    endoffset=startoffset+len(" ".join(arg1['text']))
    if 'coref' in features[sentno2][position2]:
        coref2=features[sentno2][position2]['coref']
    else:
        return False
    for start, end in coref2:
        if start >= endoffset:
            return False
        elif end <= startoffset:
            return False
        elif start <= startoffset and startoffset <= end:
            return True
        elif start <= endoffset and endoffset <= end:
            return True
        elif startoffset <= start and start <= endoffset:
            return True
        elif startoffset <= end and end <= endoffset:
            return True

def txt2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for w in textnum.split():
        if w not in numwords:
            continue

        scale, increment = numwords[w]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

def strtotime(surface):
    month,day,year,date=0,0,0,0
    taken=[]
    r = re.compile("([0-9]+)([a-zA-Z]*)")
    for x in surface:
        if x.isdigit():
            if int(x)>=2000 and int(x)<2020:
                year=int(x)
                taken.append(x)
                continue
            elif int(x)>0 and int(x)<=31 and month!=0:
                date=int(x)
                continue
            elif int(x)>0 and int(x)<=12 and month==0:
                month=int(x)
                continue

        a = r.match(x)
        if a:
            b=a.groups()
            if int(b[0])>0 and int(b[0])<=31 and b[1] in ['th','rd','nd','st']:
                date=int(b[0])
                taken.append(x)
                continue

        found=False
        for y in Month.keys():
            if x[:3] == Month[y][:3]:
                month=y
                taken.append(x)
                found=True
                break
        if found:
            continue

        for y in Day.keys():
            if x[:3] == Day[y][:3]:
                day=y
                taken.append(x)
                break

    if date!=0:
        return [year,month,date]
    elif day!=0:
        return [year,month,day]

    return [year,month,date]

def durationtotime(surface1,pubdate):
    date1=[]
    strsurface=True
    surface=[]
    number=0
    for x in surface1:
        surface.append(x.lower())
    if 'this' in surface or 'today' in surface or 'now' in surface:
        date1 = pubdate
        strsurface=False
    elif 'last' in surface or 'ago' in surface or 'past' in surface:
        if 'year' in surface:
            date1 = [pubdate[0]-1,pubdate[1],pubdate[2]]
        elif 'years' in surface:
            strtxt = []
            for x in surface:
                if x != 'years' and x != 'ago' and x != 'past' and x != 'last':
                    if x.isdigit():
                        number = int(x)
                        break
                    else:
                        strtxt.append(x)
            if strtxt:
                number = txt2int(" ".join(strtxt))
            date1 = [pubdate[0]-number, pubdate[1], pubdate[2]]
        elif 'week' in surface:
            date1 = [pubdate[0],pubdate[1],pubdate[2] - 7]
        elif 'weeks' in surface:
            strtxt=[]
            for x in surface:
                if x != 'weeks' and x != 'ago':
                    strtxt.append(x)
            number = txt2int(" ".join(strtxt))
            date1 = [pubdate[0], pubdate[1], pubdate[2] - (7*number)]
        elif 'days' in surface:
            strtxt=[]
            for x in surface:
                if x != 'days' and x != 'ago' and x != 'past' and x!= 'last':
                    if x.isdigit():
                        number=int(x)
                        break
                    else:
                        strtxt.append(x)
            if strtxt:
                number = txt2int(" ".join(strtxt))
            date1 = [pubdate[0], pubdate[1], pubdate[2] - number]
        elif 'month' in surface:
            date1 = [pubdate[0], pubdate[1]-1, pubdate[2]]
        elif 'months' in surface:
            strtxt = []
            for x in surface:
                if x != 'months' and x != 'ago' and x != 'last' and x != 'past':
                    if x.isdigit():
                        number=int(x)
                        break
                    else:
                        strtxt.append(x)
            if strtxt:
                number = txt2int(" ".join(strtxt))

            if pubdate[1]-number == 0:
                pubdate[0]=pubdate[0]-1
                pubdate[1]=1
            date1 = [pubdate[0], pubdate[1]-number, pubdate[2]]
    if date1:
        strsurface=False
    return date1,strsurface

def adjust(date1,pubdate):
    year,month,date=date1
    if year == 0 and month !=0 and date != 0:
        year=pubdate[0]

    if date < 0:
        month = month-1
        date = 31+date
    if month < 0:
        year = year-1
        month = 12+month
    if date == 0 and month != 0 and year !=0:
        date = 1
    if date == 0 and month != 0 and year == 0:
        date = 1
        year = pubdate[0]

    return [year,month,date]

def timedifference(pubdate,events,eventid1,eventid2):
    """compute difference of time between two mentions"""
    surface1,surface2='',''
    for argid in events[eventid1]['argument']:
        if events[eventid1]['argument'][argid]['argumenttype']=='Time':
            surface1=events[eventid1]['argument'][argid]['text']

    for argid in events[eventid2]['argument']:
        if events[eventid2]['argument'][argid]['argumenttype']=='Time':
            surface2=events[eventid2]['argument'][argid]['text']

    # unit of time difference will be differ by event type
    # patch differ in month, ransoms can be same when events happened within a week (10148)

    #change string to calculated object
    diff=0
    if surface1 and surface2:
        if surface1==surface2:
            return diff

        # Duration
        date1, strsurface1 = durationtotime(surface1,pubdate)
        date2, strsurface2 = durationtotime(surface2,pubdate)

        # Date
        if strsurface1:
            date1=strtotime(surface1)
        if strsurface2:
            date2=strtotime(surface2)

        if date1 and date2:
            date1=adjust(date1,pubdate)
            date2=adjust(date2,pubdate)

        if 0 not in date1 and 0 not in date2:
            date1=datetime.date(date1[0],date1[1],date1[2])
            date2 = datetime.date(date2[0], date2[1], date2[2])
            diff = (date1-date2).days
        else:
            if date1[0]!=0 and date2[0]!=0:
                diff+=(abs(date1[0]-date2[0])*365)
            if date1[1]!=0 and date2[1]!=0:
                diff+=(abs(date1[1]-date2[1])*30)
            if date1[2]!=0 and date2[2]!=0:
                diff+=abs(date1[2]-date2[2])
    return diff

def findGov(verblist,argpos,tokens):
    """ find governor of arguments, if the governor is in list of verb then return """
    found=False
    verbpos=-1
    position=[]
    position.append(argpos)
    checkedlist=[]
    while not found and len(position)>0:
        positionlist=[]
        for y in position:
            if y not in checkedlist:
                checkedlist.append(y)
                for x in tokens[y]['gov_id']:
                    if x-1 in verblist:
                        verbpos=x-1
                        found=True
                        break
                    elif x-1 not in checkedlist:
                        positionlist.append(x-1)
                if found:
                    break
            position=positionlist
    return verbpos

def readPredicted(pfile):
    """ read input content into dict
    each line is fname text offset keytrigger predtrigger keyargument predargument"""

    ct=utils.readFileEncode(pfile,'utf8')
    linest=ct.split('\n')[:-1]
    predicted={}
    pred={}
    sentno=0
    label='O'
    mention=[]
    for i in range(len(linest)):
        if len(linest[i])<3 or not (linest[i][0].isdigit()):
            if mention:
                pred = {'label': label, 'mention': " ".join(mention)}
            if pred and pred['label'] != 'O':
                predicted[fname][sentno]['pred'].append(pred)

            label=''
            mention=[]
            pred={}
            sentno+=1
            continue

        wp=linest[i].split('\t')
        word={}

        fname=wp[0]
        word['text']=wp[1]
        word['offset']=int(wp[2])
        word['keytrigger']=wp[3]
        word['trigger']=wp[4]
        word['keyargument']=wp[5]
        word['argument']=wp[6]

        ### found new file, clear old file ###
        if fname not in predicted.keys(): #{fname:{sentno:[words:[{text:'', trigger:'', argument:''},{}], triggerlist:[], argumentlist:[]}, sentno:{}}

            ### reset for new file ###
            predicted[fname]={}
            sentno=0          
             
        ### if predicted redundant, choose argument and reset trigger ###
        if word['trigger'] !='O' and word['argument']!='O':
            word['trigger']='O'

        if word['trigger']=='O' and word['argument']=='O':
            if label !='O' and mention:
                pred={'label':label,'mention':" ".join(mention)}
                mention=[]
            label='O'
        elif word['trigger']!='O':
            if word['trigger'].startswith('B-') or word['trigger'][2:]!=label:
                if mention:
                    pred={'label':label,'mention':" ".join(mention)}
                mention=[]
            mention.append(word['text'])
            label=word['trigger'][2:]
        elif word['argument']!='O':
            if word['argument'].startswith('B-') or word['argument'][2:]!=label:
                if mention:
                    pred={'label':label,'mention':" ".join(mention)}
                mention=[]
            mention.append(word['text'])
            label=word['argument'][2:]

        if sentno not in predicted[fname]:
            predicted[fname][sentno]={}
            predicted[fname][sentno]['words']=[]  
            predicted[fname][sentno]['pred']=[]
        predicted[fname][sentno]['words'].append(word)

        if pred and pred['label']!='O':
            predicted[fname][sentno]['pred'].append(pred)
            pred={}

    for x in predicted.keys(): #fname
        for y in predicted[x].keys(): #sentno
            predicted[x][y]['triggertypes']=[]
            predicted[x][y]['newtriggertypes'] = []
            predicted[x][y]['argumenttypes']=[]
            predicted[x][y]['newargumenttypes'] = []
            for z in predicted[x][y]['pred']:
                if z['label'] in TriggerList:
                    predicted[x][y]['triggertypes'].append(z['label'])
                elif z['label'] in ArgumentList:
                    predicted[x][y]['argumenttypes'].append(z['label'])

    return predicted

def readFinal(pfile):
    """ read input content into dict
    each line is fname text offset keytrigger predtrigger keyargument predargument"""

    ct = utils.readFileEncode(pfile, 'utf8')
    linest = ct.split('\n')[:-1]
    predicted = {}
    pred = {}
    sentno = 0
    label = 'O'
    mention = []
    for i in range(len(linest)):
        if len(linest[i]) < 3:
            if mention:
                pred = {'label': label, 'mention': " ".join(mention)}
            if pred and pred['label'] != 'O':
                predicted[sentno]['pred'].append(pred)
            label = ''
            mention = []
            pred = {}
            sentno += 1
            continue

        wp = linest[i].split('\t')
        word = {}

        word['text'] = wp[0]
        word['offset'] = int(wp[1])
        predtype = wp[2]
        word['role'] = wp[3]
        word['realis'] = wp[4]
        if len(wp)==6:
            if wp[5].startswith('('):
                word['keycoref'] = wp[5].replace('(','')
                word['keycoref'] = word['keycoref'].replace(')','')
                word['keycoref'] = int(word['keycoref'])

        ### if predicted redundant, choose argument and reset trigger ###
        if predtype=='O':
            word['trigger']='O'
            word['argument']='O'
        elif predtype != 'O':
            if predtype[2:] in TriggerList:
                word['trigger']=predtype
                word['argument']='O'
            elif predtype[2:] in ArgumentList:
                word['argument']=predtype
                word['trigger']='O'

        if word['trigger'] == 'O' and word['argument'] == 'O':
            if label != 'O' and mention:
                pred = {'label': label, 'mention': " ".join(mention)}
                mention = []
            label = 'O'
        elif word['trigger'] != 'O':
            if word['trigger'].startswith('B-') or word['trigger'][2:] != label:
                if mention:
                    pred = {'label': label, 'mention': " ".join(mention)}
                mention = []
            mention.append(word['text'])
            label = word['trigger'][2:]
        elif word['argument'] != 'O':
            if word['argument'].startswith('B-') or word['argument'][2:] != label:
                if mention:
                    pred = {'label': label, 'mention': " ".join(mention)}
                mention = []
            mention.append(word['text'])
            label = word['argument'][2:]

        if sentno not in predicted:
            predicted[sentno] = {}
            predicted[sentno]['words'] = []
            predicted[sentno]['pred'] = []
        predicted[sentno]['words'].append(word)

        if pred and pred['label'] != 'O':
            predicted[sentno]['pred'].append(pred)
            pred = {}

    for x in predicted.keys():

        predicted[x]['triggertypes'] = []
        predicted[x]['argumenttypes'] = []
        for z in predicted[x]['pred']:
            if z['label'] in TriggerList:
                predicted[x]['triggertypes'].append(z['label'])
            elif z['label'] in ArgumentList:
                predicted[x]['argumenttypes'].append(z['label'])

    return predicted

def findargument(argtype,sentences,sentno,tokens):
    """Add free arguments to the sentence with some events"""
    if argtype=='Time':
        for i in range(len(tokens)):
            if tokens[i]['ner'][2:] in Date:
                for j in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][j]['text'] == tokens[i]['originalText'] and \
                            sentences[sentno]['words'][j]['argument']=='O':
                        if sentences[sentno]['words'][j]['trigger']!='O':
                            sentences[sentno]['words'][j]['trigger'] = 'O'
                        sentences[sentno]['words'][j]['aargument'] = tokens[i]['ner'][0]+'-Time'
                        sentences[sentno]['newargumenttypes'].append('Time')
                        break
    elif argtype=='Money':
        for i in range(len(tokens)):
            if tokens[i]['ner'][2:] == 'Money':
                for j in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][j]['text'] == tokens[i]['originalText'] and sentences[sentno]['words'][j]['argument']=='O':
                        if sentences[sentno]['words'][j]['trigger']!='O':
                            sentences[sentno]['words'][j]['trigger'] = 'O'
                        sentences[sentno]['words'][j]['aargument'] = tokens[i]['ner'][0] + '-Money'
                        sentences[sentno]['newargumenttypes'].append('Money')
                        break
    elif argtype in GeneralArg.keys():
        for i in range(len(tokens)):
            if tokens[i]['originalText'] in GeneralArg[argtype]:
                for j in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][j]['text'] == tokens[i]['originalText'] and sentences[sentno]['words'][j]['argument']=='O':
                        if sentences[sentno]['words'][j]['trigger']!='O':
                            sentences[sentno]['words'][j]['trigger'] = 'O'
                        sentences[sentno]['words'][j]['aargument'] = 'B-'+argtype
                        sentences[sentno]['newargumenttypes'].append(argtype)
                        break
                break

    return sentences

def findtrigger(eventType,argpos,sentences,sentno,tokens):
    """ force verb to be trigger, if not found trigger nearby """

    foundeventType = eventType
    # set range for context
    contextsize = 0
    lowerlim=0 if sentno-1<0 else sentno-contextsize
    upperlim=len(sentences) if sentno+contextsize+1>=len(sentences) else sentno+contextsize+1

    for i in range(lowerlim,upperlim):
        if i in sentences.keys():
            if eventType in sentences[i]['triggertypes']:
                # found must have trigger in the context -> do nothing
                return sentences
        
    #find closet verb in this sentence to be a trigger
    realverbpos,verbpos,verbpospair=[],[],{}
    verblist=[]
    realargpos=-1
    realnounpos=[]
    nounpospair={}

    i=0
    while True:
        # find real argument position
        left=argpos-i if argpos-i>=0 else 0
        right=argpos+i if argpos+i<len(tokens) else len(tokens)-1
        if tokens[left]['originalText']==sentences[sentno]['words'][argpos]['text']:
            realargpos=left
            break
        elif tokens[right]['originalText']==sentences[sentno]['words'][argpos]['text']:
            realargpos=right
            break
        elif left==0 and right ==len(tokens)-1:
            break
        i+=1

    for i in range(len(tokens)):
        #find verbs in this sentence
        if tokens[i]['pos'].startswith('N'):
            for j in range(len(sentences[sentno]['words'])):
                if sentences[sentno]['words'][j]['text'] == tokens[i]['originalText'] and sentences[sentno]['words'][j]['argument'] == 'O':
                    realnounpos.append(i)
                    nounpospair[i]=j
                    break
        if tokens[i]['pos'].startswith('V'):
            for j in range(len(sentences[sentno]['words'])):
                if sentences[sentno]['words'][j]['text'] == tokens[i]['originalText'] and sentences[sentno]['words'][j]['argument'] == 'O':
                    realverbpos.append(i)
                    verbpospair[i]=j
                    verblist.append(tokens[i]['originalText'])
                    break

    # each verb, measure distance, sort them by their distance
    dist = []
    for p in realverbpos:
        dist.append(abs(realargpos-p))
    verbdist=zip(realverbpos,dist)

    verbdist=sorted(verbdist, key=lambda x: x[1])

    useverb=False
    if len(verbdist)>0:
        gov = findGov(realverbpos,realargpos,tokens)
        useverb=True
    else:
        gov = findGov(realnounpos, realargpos, tokens)

    if gov!=-1:
        if useverb:
            j=verbpospair[gov]
        else:
            j=nounpospair[gov]

        if eventType == 'DiscoverVulnerability':
            # Vulnerability can be argument of patchvulnerability or discovervulnerability
            # find closet group and assign a new nugget
            discoversim = avgsim(sentences[sentno]['words'][j]['text'], 'DiscoverVulnerability')
            patchsim = avgsim(sentences[sentno]['words'][j]['text'], 'PatchVulnerability')
            #print (sentno, 'sim', discoversim, patchsim)
            if discoversim > patchsim:
                foundeventType = 'DiscoverVulnerability'
            else:
                foundeventType = 'PatchVulnerability'

        sentences[sentno]['words'][j]['gtrigger'] = foundeventType   #gtrigger
        sentences[sentno]['newtriggertypes'].append(foundeventType)

    verbpos=-1
    if len(verbdist)>0:
        verbpos, distance = verbdist[0]

    if verbpos!=-1 and verbpos!=gov:
        j = verbpospair[verbpos]
        # if nearest verb differs from governor verb
        if eventType == 'DiscoverVulnerability':
            # Vulnerability can be argument of patchvulnerability or discovervulnerability
            # find closet group and assign a new nugget
            discoversim = avgsim(sentences[sentno]['words'][j]['text'], 'DiscoverVulnerability')
            patchsim = avgsim(sentences[sentno]['words'][j]['text'], 'PatchVulnerability')
            if discoversim > patchsim:
                foundeventType = 'DiscoverVulnerability'
            else:
                foundeventType = 'PatchVulnerability'

        sentences[sentno]['words'][j]['dtrigger'] = foundeventType
        sentences[sentno]['newtriggertypes'].append(foundeventType)

    return sentences

def generalnountrigger(eventypelist,sentno,sentences,tokens):
    nouns=[]
    for eventtype in eventypelist:
        if eventtype == 'Databreach':
            nouns+=GeneralN['Databreach']
            nouns+=GeneralN['general']
        elif eventtype in ['Phishing', 'Ransom']:
            nouns.append(GeneralN['general'])
        elif eventtype == 'PatchVulnerability':
            nouns = GeneralN['PatchVulnerability']
    if len(nouns)>0:
        for i in range(len(tokens)):
            if tokens[i]['originalText'] in nouns:
                for j in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][j]['text'] == tokens[i]['originalText'] and \
                            sentences[sentno]['words'][j]['argument'] == 'O':
                        for k in range(sentno-1,sentno+2):
                            if k in sentences.keys():
                                if len(sentences[k]['triggertypes'])>0:
                                    sentences[sentno]['words'][j]['ntrigger'] = sentences[k]['triggertypes'][0]
                                    sentences[sentno]['newtriggertypes'].append(sentences[k]['triggertypes'][0])
                                    return sentences,True
    return sentences,False

def findanytrigger(sentences, sentno, tokens):
    """ force verb to be trigger, if not found trigger nearby """

    # compute score from argument list
    needNoun=True
    NeedVerb=['Person','Organization']
    score={'Databreach':0,'Phishing':0,'Ransom':0,'DiscoverVulnerability':0,'PatchVulnerability':0}
    for argtype in sentences[sentno]['argumenttypes']:
        if argtype in NeedVerb:
            needNoun=False
        for eventtype in score.keys():
            if argtype in Pair[eventtype]:
                score[eventtype]+=1

    event_w_score_arg=[]
    for eventtype in score.keys():
        if score[eventtype]>0:
            event_w_score_arg.append(eventtype)

    scoreH = {'Databreach': 0, 'Phishing': 0, 'Ransom': 0, 'DiscoverVulnerability': 0, 'PatchVulnerability': 0}
    # check other triggers in the file
    for i in sentences.keys():
        for k in sentences[i]['triggertypes']:
            scoreH[k] += 1

    event_w_score_trggr=[]
    for eventtype in scoreH.keys():
        if scoreH[eventtype]>0:
            event_w_score_trggr.append(eventtype)

    # intersection of event type which has highest score in counting arguments and counting other event types in file
    intersect_eventtype=[]
    for eventtype in score.keys():
        if eventtype in event_w_score_arg and eventtype in event_w_score_trggr:
            intersect_eventtype.append(eventtype)

    # General noun case
    sentences,found=generalnountrigger(intersect_eventtype,sentno,sentences,tokens)
    if found:
        return sentences
    realverbpos, verbpos, verbpospair = [], [], {}
    verblist = []

    # if 'Person' in argument should find nearby verb which has same type with highest sim
    # if it is the same type then check distance

    for i in range(len(tokens)):
        # all tokens which are verb or noun will be selected as a candidate
        if tokens[i]['pos'].startswith('V') or (needNoun and tokens[i]['pos'].startswith('N')) :
            for j in range(len(sentences[sentno]['words'])):
                if sentences[sentno]['words'][j]['text'] == tokens[i]['originalText'] and \
                        sentences[sentno]['words'][j]['argument'] == 'O':
                    realverbpos.append(i)
                    verbpospair[i] = j
                    verblist.append(tokens[i]['originalText'])
                    break
    if len(realverbpos)==0:
        return sentences
    elif len(intersect_eventtype)==0:
        return sentences

    sim = {}
    max_sim={}
    for i in realverbpos:
        sim[i]={}
        # find similarity score for each verb and event type
        for eventtype in intersect_eventtype:
            sim[i][eventtype]=maxsim(tokens[i]['originalText'], eventtype)
        sim[i]['maxsimevent'],max_sim[i]=max(sim[i].items(), key=operator.itemgetter(1))

    simrealverbpos=sorted(max_sim.items(), key=operator.itemgetter(1),reverse=True)

    # in case of multiple maximum, event will be selected following nearby eventtype
    # in case of the only one maximum, use it
    position,simvalue = simrealverbpos[0]
    if simvalue>=0.68:
        multiple=0
        scoreP = {'Databreach': 0, 'Phishing': 0, 'Ransom': 0, 'DiscoverVulnerability': 0, 'PatchVulnerability': 0}
        for p,s in simrealverbpos:
            if s==simvalue:
                multiple+=1
                scoreP[sim[p]['maxsimevent']]+=1
        foundeventtype=''
        if multiple>0:
            for i in range(sentno,-1,-1):
                if i in sentences.keys():
                    if len(sentences[i]['triggertypes'])>0:
                        for k in sentences[i]['triggertypes']:
                            if scoreP[k]>0:
                                # other event found in the past sentences is the same with maximum score
                                foundeventtype=k
                                break
            if foundeventtype!='':
                for p, s in simrealverbpos:
                    if s == simvalue and sim[p]['maxsimevent']==foundeventtype:
                        position=p
                        break

        highestsimevent=sim[position]['maxsimevent']
        j=verbpospair[position]
        sentences[sentno]['words'][j]['atrigger'] = highestsimevent
        sentences[sentno]['newtriggertypes'].append(highestsimevent)

    return sentences

def recheck(sentences,features):
    """ found must-have arguments need some triggers """
    for sentno in sentences.keys():
        thistrigger=sentences[sentno]['triggertypes']
        thisarguments=sentences[sentno]['argumenttypes']

        ## if must have arguments found, recheck if there is a trigger in nearby sentence
        if len(thisarguments)>0: # and len(sentences[sentno]['triggertypes'])==0:
            argpos=0
            if 'Money' in thisarguments and 'PaymentMethod' in thisarguments and 'Ransom' not in thistrigger:
                for i in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][i]['argument'][2:]=='Money':
                        argpos=i
                        break

                sentences = findtrigger('Ransom', argpos, sentences, sentno, features[sentno])

            if 'Patch' in thisarguments and 'PatchVulnerability' not in thistrigger:
                for i in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][i]['argument'][2:]=='Patch':
                        argpos=i
                        break
                sentences = findtrigger('PatchVulnerability', argpos, sentences, sentno, features[sentno])

            if 'Vulnerability' in thisarguments and 'Patch' not in thisarguments and 'DiscoverVulnerability' not in thistrigger:
                for i in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][i]['argument'][2:]=='Vulnerability':
                        argpos=i
                        break
                sentences=findtrigger('DiscoverVulnerability',argpos,sentences,sentno,features[sentno])

            if ('Data' in thisarguments or 'PII' in thisarguments) and 'Databreach' not in thistrigger:
                for i in range(len(sentences[sentno]['words'])):
                    if sentences[sentno]['words'][i]['argument'][2:]=='Data':
                        argpos=i
                        break
                    elif sentences[sentno]['words'][i]['argument'][2:]=='PII':
                        argpos=i
                        break

                sentences = findtrigger('Databreach', argpos, sentences, sentno, features[sentno])

        if len(thisarguments) > 0 and len(sentences[sentno]['newtriggertypes']) == 0  and len(sentences[sentno]['triggertypes']) == 0:
            ### the rest w/o specific arguments ###
            scoreH = {'Databreach': 0, 'Phishing': 0, 'Ransom': 0, 'DiscoverVulnerability': 0, 'PatchVulnerability': 0}
            # check other triggers in the file
            for i in sentences.keys():
                for k in sentences[i]['triggertypes']:
                    scoreH[k] += 1
            noofevent = 0
            for eventtype in scoreH.keys():
                if scoreH[eventtype] > 0:
                    noofevent += 1

            if noofevent == 1:
                # only one event type was found in this file
                foundeventtype, tmp = max(scoreH.items(), key=operator.itemgetter(1))
                latestlabel = ''

                sentences, found = generalnountrigger([foundeventtype], sentno, sentences, features[sentno])
                if 'PII' in sentences[sentno]['argumenttypes'] and foundeventtype == 'Databreach':
                    for j in range(len(sentences[sentno]['words'])):
                        if sentences[sentno]['words'][j]['argument'][2:]=='PII':
                            sentences = findtrigger(foundeventtype, j, sentences, sentno, features[sentno])
                            break

                elif ('File' in sentences[sentno]['argumenttypes'] or 'Website' in sentences[sentno]['argumenttypes'] or 'PII' in sentences[sentno]['argumenttypes']) and foundeventtype == 'Phishing':
                    for j in range(len(sentences[sentno]['words'])):
                        if sentences[sentno]['words'][j]['argument'][2:] == 'File' or sentences[sentno]['words'][j]['argument'][2:] == 'Website' :
                            sentences = findtrigger(foundeventtype, j, sentences, sentno, features[sentno])
                            break
                elif ('File' in sentences[sentno]['argumenttypes'] or 'Malware' in sentences[sentno]['argumenttypes'])and foundeventtype == 'Ransom':
                    for j in range(len(sentences[sentno]['words'])):
                        if sentences[sentno]['words'][j]['argument'][2:] == 'File' or sentences[sentno]['words'][j]['argument'][2:] == 'Malware' :
                            sentences = findtrigger(foundeventtype, j, sentences, sentno, features[sentno])
                            break
                else:
                    # no general noun
                    if not found:
                        for j in range(len(sentences[sentno]['words'])):
                            k = sentences[sentno]['words'][j]['argument']
                            if k != 'O' and k[2:] != latestlabel[2:] and k[2:] not in DoNothing[foundeventtype]:
                                # might find dep instead of gov for some types of argument e.g. Person
                                sentences = findtrigger(foundeventtype, j, sentences, sentno, features[sentno])

                            latestlabel = k
            else: #found multiple types of event
                sentences = findanytrigger(sentences, sentno, features[sentno])

        if 'Ransom' in thistrigger:
            findargument('Money',sentences,sentno,features[sentno])
        if 'PatchReleased' in thistrigger:
            findargument('Patch',sentences,sentno,features[sentno])
        if 'DiscoverVulnerability' in thistrigger:
            findargument('Vulnerability',sentences,sentno,features[sentno])

        if thistrigger:
            findargument('Time',sentences,sentno,features[sentno])

    return sentences

def compatible(predicted):
    """must have trigger, if found unwanted arguments then remove them.
       In the opposite, if no trigger but arguments were detected then do nothing
       In the others, if found trigger but no arguments, we may find conflict of trigger"""

    for sentno in predicted.keys():
        # if trigger is in this sentence, arguments will be checked
        trigger=set([])
        if len(predicted[sentno]['triggertypes'])>0:
            for x in predicted[sentno]['triggertypes']:
                trigger.add(x)
        if sentno-1 in predicted.keys():
            for x in predicted[sentno-1]['triggertypes']:
                trigger.add(x)
        if sentno+1 in predicted.keys():
            for x in predicted[sentno+1]['triggertypes']:
                trigger.add(x)
        if len(predicted[sentno]['newtriggertypes'])>0:
            for x in predicted[sentno]['newtriggertypes']:
                trigger.add(x)
        if sentno-1 in predicted.keys():
            for x in predicted[sentno-1]['newtriggertypes']:
                trigger.add(x)
        if sentno+1 in predicted.keys():
            for x in predicted[sentno+1]['newtriggertypes']:
                trigger.add(x)
        if len(trigger)==0:
            if len(predicted[sentno]['argumenttypes'])>0 or len(predicted[sentno]['newargumenttypes'])>0:
                for y in predicted[sentno]['words']:
                    if y['argument']!='O':
                        y['argument']='O'
                    if 'aargument' in y:
                        y['aargument']='O'
            continue

        passargument=copy.deepcopy(predicted[sentno]['argumenttypes'])
        passargument +=predicted[sentno]['newargumenttypes']

        for x in predicted[sentno]['argumenttypes']:
            for y in trigger:
                #print (x,Pair[y])
                if x in Pair[y]:
                    passargument.remove(x)
                    break
        for x in predicted[sentno]['newargumenttypes']:
            for y in trigger:
                #print (x,Pair[y])
                if x in Pair[y]:
                    passargument.remove(x)
                    break
        #leftover, no compatible trigger
        for x in passargument:
            if 'Purpose' in passargument:
                predicted[sentno]['argumenttypes'].remove('Purpose')
                for y in predicted[sentno]['words']:
                    if y['argument']=='Purpose':
                        y['newtag']=='O'
                        break

    return predicted

def combine(sentences):
    """combine new findings of triggers and arguments"""
    for sentno in range(len(sentences)):
        if sentno in sentences.keys():
            for i in range(len(sentences[sentno]['words'])):
                w = sentences[sentno]['words'][i]

                if w['trigger'] == 'O':

                    if 'gtrigger' in w:
                        w['trigger']='B-'+w['gtrigger']
                        sentences[sentno]['triggertypes'].append(w['trigger'][2:])
                    elif 'dtrigger' in w:
                        w['trigger']='B-' + w['dtrigger']
                        sentences[sentno]['triggertypes'].append(w['trigger'][2:])
                    elif 'atrigger' in w:
                        w['trigger']='B-' + w['atrigger']
                        sentences[sentno]['triggertypes'].append(w['trigger'][2:])
                    elif 'ntrigger' in w:
                        w['trigger']='B-' + w['ntrigger']
                        sentences[sentno]['triggertypes'].append(w['trigger'][2:])
                if w['argument']=='O':
                    if 'aargument' in w:
                        w['argument']=w['aargument']
                        sentences[sentno]['argumenttypes'].append(w['argument'][2:])
                if i-1>=0:
                    w0=sentences[sentno]['words'][i-1]
                    if w['trigger'].startswith('I-') and w0['trigger'][2:]!=w['trigger'][2:]:
                        w['trigger']='B-'+w['trigger'][2:]
                    if w['argument'].startswith('I-') and w0['argument'][2:]!=w['argument'][2:]:
                        w['argument']='B-'+w['argument'][2:]

    return sentences

def link(sentences,features):
    """ link between nugget and argument """
    allevent = {}
    eventid = 0
    # event{'event_id':eventid, nugget_pos:'',event_nugget:'',event_arg:''}
    for sentno in sentences.keys():
        # combine trigger with new trigger
        if len(sentences[sentno]['triggertypes']) > 0:
            allevent[sentno]={}
            oldlabel,nugget,label='',[],''
            for j in range(len(sentences[sentno]['words'])):
                if sentences[sentno]['words'][j]['trigger'].startswith('B-') :

                    if nugget:
                        event={}
                        event['event_id']=eventid
                        event['text'] = nugget
                        event['eventtype'] = label
                        event['startOffset']=sentences[sentno]['words'][nugget_pos]['offset']
                        event['pair']=nugget_pos
                        if 'realis' in sentences[sentno]['words'][nugget_pos]:
                            event['realis']=sentences[sentno]['words'][nugget_pos]['realis']
                        if 'keycoref' in sentences[sentno]['words'][nugget_pos]:
                            event['keycoref'] = sentences[sentno]['words'][nugget_pos]['keycoref']
                        event['argument']={}
                        allevent[sentno][eventid]=event
                        eventid+=1
                    nugget=[]
                    nugget_pos=j
                    nugget.append(sentences[sentno]['words'][j]['text'])
                    label=sentences[sentno]['words'][j]['trigger'][2:]

                elif sentences[sentno]['words'][j]['trigger'].startswith('I-'):
                    if oldlabel[2:]!=sentences[sentno]['words'][j]['trigger'][2:]:
                        if nugget:
                            event = {}
                            event['event_id'] = eventid
                            event['text'] = nugget
                            event['eventtype'] = label
                            event['startOffset'] = sentences[sentno]['words'][nugget_pos]['offset']
                            event['pair'] = nugget_pos
                            if 'realis' in sentences[sentno]['words'][nugget_pos]:
                                event['realis'] = sentences[sentno]['words'][nugget_pos]['realis']
                            if 'keycoref' in sentences[sentno]['words'][nugget_pos]:
                                event['keycoref'] = sentences[sentno]['words'][nugget_pos]['keycoref']
                            event['argument'] = {}
                            allevent[sentno][eventid]=event
                            eventid += 1
                        nugget = []
                        nugget_pos=j
                        nugget.append(sentences[sentno]['words'][j]['text'])
                        label = sentences[sentno]['words'][j]['trigger'][2:]
                    elif oldlabel[2:]==sentences[sentno]['words'][j]['trigger'][2:]:
                        nugget.append(sentences[sentno]['words'][j]['text'])
                oldlabel=sentences[sentno]['words'][j]['trigger']
            if nugget:
                event = {}
                event['event_id'] = eventid
                event['text'] = nugget #" ".join(nugget)
                event['eventtype'] = label
                event['startOffset'] = sentences[sentno]['words'][nugget_pos]['offset']
                event['pair'] = nugget_pos
                if 'realis' in sentences[sentno]['words'][nugget_pos]:
                    event['realis'] = sentences[sentno]['words'][nugget_pos]['realis']
                if 'keycoref' in sentences[sentno]['words'][nugget_pos]:
                    event['keycoref'] = sentences[sentno]['words'][nugget_pos]['keycoref']

                event['argument'] = {}
                allevent[sentno][eventid]=event
                eventid += 1

    argumentid=0
    for sentno in sentences.keys():
        # combine trigger with new trigger
        select_sent=-1
        tokens=features[sentno]
        if len(sentences[sentno]['argumenttypes']) == 0:
            continue

        if sentno in allevent.keys():
            if len(allevent[sentno].keys())>0:
                select_sent=sentno
        elif sentno-1 in allevent.keys():
            if len(allevent[sentno-1].keys())>0:
                select_sent = sentno-1
        if select_sent==-1:
            continue
        argumentlist={}
        oldlabel,argument = '',[]
        for j in range(len(sentences[sentno]['words'])):
            if sentences[sentno]['words'][j]['argument'].startswith('B-'):
                if argument:
                    arg={}
                    arg['argument_id']=argumentid
                    arg['text'] = argument
                    arg['startOffset']=sentences[sentno]['words'][arg_pos]['offset']
                    arg['pair']=(sentno,arg_pos)
                    arg['argumenttype']=label
                    if 'role' in sentences[sentno]['words'][arg_pos]:
                        arg['role'] = sentences[sentno]['words'][arg_pos]['role']
                    argumentlist[argumentid]=arg
                    argumentid+=1
                argument = []
                arg_pos = j
                argument.append(sentences[sentno]['words'][j]['text'])
                label=sentences[sentno]['words'][j]['argument'][2:]
            elif sentences[sentno]['words'][j]['argument'].startswith('I-'):
                if oldlabel[2:] != sentences[sentno]['words'][j]['argument'][2:]:
                    if argument:
                        arg = {}
                        arg['argument_id'] = argumentid
                        arg['text'] = argument #" ".join()
                        arg['startOffset'] = sentences[sentno]['words'][arg_pos]['offset']
                        arg['pair'] = (sentno,arg_pos)
                        arg['argumenttype'] = label
                        if 'role' in sentences[sentno]['words'][arg_pos]:
                            arg['role'] = sentences[sentno]['words'][arg_pos]['role']
                        argumentlist[argumentid]=arg
                        argumentid += 1
                    argument = []
                    arg_pos = j
                    argument.append(sentences[sentno]['words'][j]['text'])
                    label = sentences[sentno]['words'][j]['argument'][2:]
                elif oldlabel[2:] == sentences[sentno]['words'][j]['argument'][2:]:
                    argument.append(sentences[sentno]['words'][j]['text'])
            oldlabel = sentences[sentno]['words'][j]['argument']
        if argument:
            arg = {}
            arg['argument_id'] = argumentid
            arg['text'] = argument
            arg['startOffset'] = sentences[sentno]['words'][arg_pos]['offset']
            arg['pair'] = (sentno,arg_pos)
            arg['argumenttype'] = label
            if 'role' in sentences[sentno]['words'][arg_pos]:
                arg['role'] = sentences[sentno]['words'][arg_pos]['role']
            argumentlist[argumentid]=arg
            argumentid += 1

        for argid in argumentlist.keys():
            arg=argumentlist[argid]
            possiblelist = []
            for eventid in allevent[select_sent].keys():
                eventtype = allevent[select_sent][eventid]['eventtype']
                if arg['argumenttype'] in Pair[eventtype]:  # check distance
                    possiblelist.append(eventid)
            if len(possiblelist)==0:
                continue

            if len(possiblelist) == 1:
                for eventid in allevent[select_sent].keys():
                    if eventid == possiblelist[0]:
                        allevent[select_sent][eventid]['argument'][argid]=arg
            else:  # more than one choice, select from distance
                found=False
                for t in tokens:

                    if t['characterOffsetBegin'] == arg['startOffset'] or (
                            t['characterOffsetBegin'] > arg['startOffset'] and \
                            t['characterOffsetBegin'] < arg['startOffset'] + len(arg['text'])):

                        if 'triggerPosition' in t:
                            if t['triggerPosition']=='before' or t['triggerPosition']=='differsentence':
                                for eventid in possiblelist:
                                    if t['nearTrigger'] in allevent[select_sent][eventid]['text'] and \
                                            allevent[select_sent][eventid]['startOffset']<=t['characterOffsetBegin']:
                                        allevent[select_sent][eventid]['argument'][argid]=arg
                                        found = True
                                        break
                            elif t['triggerPosition']=='after':
                                for eventid in possiblelist:
                                    if t['nearTrigger'] in allevent[select_sent][eventid]['text'] and \
                                            allevent[select_sent][eventid]['startOffset']>=t['characterOffsetBegin']:
                                        allevent[select_sent][eventid]['argument'][argid]=arg
                                        found = True
                                        break
                            elif t['triggerPosition']=='beforeseparated' or t['triggerPosition']=='afterseparated':
                                for eventid in possiblelist:
                                    if t['nearTrigger'] in allevent[select_sent][eventid]['text']:
                                        allevent[select_sent][eventid]['argument'][argid]=arg
                                        found = True
                                        break
                    if found:
                        break

    return allevent

def roleassign(allevent,pred_result,sentences):
    """assign role to argument followed type of event"""

    for sentno in allevent.keys():
        for eventid in allevent[sentno].keys():
            event = allevent[sentno][eventid]
            arglist=RolePair[event['eventtype']]
            foundNumber=[]
            for argid in event['argument'].keys():
                arg=event['argument'][argid]
                if arg['argumenttype'] in arglist.keys():
                    if arg['argumenttype']=='Number':
                        foundNumber.append(argid)
                    if len(arglist[arg['argumenttype']])==1:
                        arg['role']=arglist[arg['argumenttype']][0]
                        s,k=arg['pair']
                        for argtext in arg['text']:
                            if argtext == sentences[s]['words'][k]['text']:
                                sentences[s]['words'][k]['role']=arglist[arg['argumenttype']][0]
                            k+=1

                    else:
                        eventtype = allevent[sentno][eventid]['eventtype']
                        arg_role = pred_result[eventtype]
                        if len(arg_role) > 0:
                            for i in arg_role.keys():
                                if arg_role[i]['offset']==arg['startOffset']:
                                    arg['role']=arg_role[i]['pred']
                                    s,k = arg['pair']
                                    for argtext in arg['text']:
                                        if argtext == sentences[s]['words'][k]['text']:
                                            sentences[s]['words'][k]['role'] = arg_role[i]['pred']
                                        k += 1
                                    break

            if event['eventtype'] == 'Databreach' and len(foundNumber)>0:
                for number_argid in foundNumber:
                    targetid=number_argid+1

                    if 'role' in allevent[sentno][eventid]['argument'][targetid].keys():
                        if allevent[sentno][eventid]['argument'][targetid]['role']=='Victim':
                            targetrole='Number-of-Victim'

                        elif allevent[sentno][eventid]['argument'][targetid]['role']=='Compromised-Data':
                            targetrole='Number-of-Compromised-Data'

                        allevent[sentno][eventid]['argument'][number_argid]['role'] = targetrole

    return allevent,sentences

def realisassign(allevent,pred_result,sentences):
    """assign role to argument followed type of event"""
    for sentno in allevent.keys():
        for eventid in allevent[sentno].keys():
            event = allevent[sentno][eventid]
            for i in pred_result.keys():

                if event['startOffset'] in pred_result[i]['offset']:
                    if pred_result[i]['pred']=='NotGeneral':
                        event['realis']='Actual'
                        k = event['pair']
                        for nugget in event['text']:
                            if nugget == sentences[sentno]['words'][k]['text']:
                                sentences[sentno]['words'][k]['realis'] = 'Actual'
                            k += 1
                    else:
                        event['realis']=pred_result[i]['pred']
                        k = event['pair']
                        for nugget in event['text']:
                            if nugget == sentences[sentno]['words'][k]['text']:
                                sentences[sentno]['words'][k]['realis'] = pred_result[i]['pred']
                            k += 1
                    break

    return allevent,sentences

def corefassign(events,sentences):
    """ assign corefid to event nugget"""

    for eventid in events.keys():
        sentno=events[eventid]['sentenceid']
        k = events[eventid]['pair']

        if len(events[eventid]['text']) == 1:
            oneword=True
        else:
            oneword=False

        for i in range(len(events[eventid]['text'])):
            if i==0 and oneword:
                sentences[sentno]['words'][k]['coref'] = '('+ str(events[eventid]['coref']) +')'
            elif i==0 and not oneword:
                sentences[sentno]['words'][k]['coref'] = '(' + str(events[eventid]['coref'])
            elif i==len(events[eventid]['text'])-1 and not oneword:
                sentences[sentno]['words'][k]['coref'] = str(events[eventid]['coref']) + ')'
            else:
                sentences[sentno]['words'][k]['coref'] = '-'
            k += 1

    return sentences

def corefdiffbyevent(pubdate,events,features,weight,threshold):
    """comprise features for clustering events"""
    clusterid=0
    maxdist=0.0
    nuggetdiff,argtype,nuggetdist={},{},{}
    argmiss,realisdiff,argcoref={},{},{}
    timedist={}
    subset={}
    for eventtype in TriggerList:
        subset[eventtype]=[]
    for eventid in events.keys():
        subset[events[eventid]['eventtype']].append(eventid)
    for eventtype in TriggerList:
        if len(subset[eventtype])==0:
            continue
        elif len(subset[eventtype])==1:
            eventid=subset[eventtype][0]
            events[eventid]['coref']=clusterid
            clusterid+=1
            continue
        e1pair={}
        e1=0
        for eventid1 in subset[eventtype]:
            e1pair[e1]=eventid1
            nuggetdiff[e1] = {}
            argtype[e1] = {}
            nuggetdist[e1] = {}
            argmiss[e1] = {}
            realisdiff[e1] = {}
            argcoref[e1] = {}
            timedist[e1]={}
            event1 = events[eventid1]
            argtypelist1 = []
            for argid in event1['argument'].keys():
                argtypelist1.append(event1['argument'][argid]['argumenttype'])
            e2=0
            for eventid2 in subset[eventtype]:
                if eventid1==eventid2:
                    nuggetdist[e1][e2]=0.0
                    nuggetdiff[e1][e2]=0.0
                    realisdiff[e1][e2]=0.0
                    argtype[e1][e2] = 0.0
                    argmiss[e1][e2] =0.0
                    argcoref[e1][e2]=0.0
                    timedist[e1][e2]=0.0
                    e2+=1
                    continue

                event2=events[eventid2]

                argtypelist2 = []
                for argid in event2['argument'].keys():
                    argtypelist2.append(event2['argument'][argid]['argumenttype'])

                # nugget surface match <2>
                word1=event1['text']
                word2=event2['text']
                nuggetdiff[e1][e2]=strdist(word1,word2)

                # nugget distance <2>
                nuggetdist[e1][e2]=abs(event1['sentenceid']-event2['sentenceid'])

                # realis match <2>
                if event1['realis']==event2['realis']:
                    realisdiff[e1][e2] = 0
                else:
                    realisdiff[e1][e2] = 1

                # argument type match
                intersectargtype=[]
                for arg1 in argtypelist1:
                    if arg1 in argtypelist2:
                        argtypelist2.remove(arg1)
                        intersectargtype.append(arg1)

                argtype[e1][e2] = len(intersectargtype)

                #argument mismatch in arg1 but not in arg2
                argmiss[e1][e2]=len(argtypelist1)-argtype[e1][e2]

                # argument time difference
                if 'Time' in argtypelist1 and 'Time' in argtypelist2:
                    tdiff=timedifference(pubdate,events,eventid1,eventid2)
                    if tdiff > 0: # TimeDiff[eventtype]:
                        timedist[e1][e2]=1
                    elif tdiff==0:
                        timedist[e1][e2]=0
                    else:
                        timedist[e1][e2]=0
                else:
                    timedist[e1][e2]=0

                # argument coref similarity
                argcoref[e1][e2], totalcoref= 0.0,0.0
                for argid in event1['argument'].keys():
                    for argid2 in event2['argument'].keys():
                        if iscoref(event1['argument'][argid],event2['argument'][argid2],features):
                            #is argument 1 in coref set of argument 2
                            argcoref[e1][e2]+=1
                        totalcoref+=1
                if totalcoref>0.0:
                    argcoref[e1][e2]=1.0-(float(argcoref[e1][e2])/float(totalcoref))
                else:
                    argcoref[e1][e2]=1.0
                e2 += 1
            e1+=1

        #normalization
        maxnuggetdist,maxintsct,maxmisarg=0.0,0.0,0.0
        minnuggetdist, minintsct, minmisarg = 100.0, 100.0, 100.0
        for e1 in range(len(subset[eventtype])):
            dist,intsct,misarg = [],[],[]
            for e2 in range(len(subset[eventtype])):
                dist.append(nuggetdist[e1][e2])
                intsct.append(argtype[e1][e2])
                misarg.append(argmiss[e1][e2])
            if maxnuggetdist < max(dist):
                maxnuggetdist=max(dist)
            if minnuggetdist < min(dist):
                minnuggetdist=min(dist)
            if maxintsct < max(intsct):
                maxintsct=max(intsct)
            if minintsct < min(intsct):
                minintsct=min(intsct)
            if maxmisarg < max(misarg):
                maxmisarg=max(misarg)
            if minmisarg < min(misarg):
                minmisarg=min(misarg)
        for e1 in range(len(subset[eventtype])):
            for e2 in range(len(subset[eventtype])):
                if maxnuggetdist!=0.0:
                    nuggetdist[e1][e2]=float(nuggetdist[e1][e2]-minnuggetdist)/float(maxnuggetdist-minnuggetdist)
                else:
                    nuggetdist[e1][e2] = 0.0
                if maxintsct!=0:
                    argtype[e1][e2]=float(argtype[e1][e2]-minintsct)/float(maxintsct-minintsct)
                else:
                    argtype[e1][e2]=0.0
                if maxmisarg!=0:
                    argmiss[e1][e2]=float(argmiss[e1][e2]-minmisarg)/float(maxmisarg-minmisarg)
                else:
                    argmiss[e1][e2]=0.0

        distmat=np.ndarray(shape=(len(subset[eventtype]),len(subset[eventtype])), dtype='float32')

        for e1 in range(len(subset[eventtype])):
            for e2 in range(len(subset[eventtype])):
                tmp=0.0
                if e1!=e2:
                    tmp+=nuggetdiff[e1][e2]*weight[0] # nugget surface distance
                    tmp+=realisdiff[e1][e2]*weight[1] # realis difference
                    tmp+=argtype[e1][e2]*weight[2] # number of same argument type
                    tmp+=argcoref[e1][e2]*weight[3] # coreference of arguments
                    tmp+=nuggetdist[e1][e2]*weight[4] # normalized number of sentence between nuggets
                    tmp+=argmiss[e1][e2]*weight[5] # number of argument mismatch
                    tmp+=timedist[e1][e2]*weight[6] # difference in time
                distmat[e1][e2]=tmp
                if maxdist<tmp:
                    maxdist=tmp

        cluster = AgglomerativeClustering(affinity='precomputed',linkage='average', distance_threshold=threshold, n_clusters=None) # , linkage='ward',
        cluster.fit_predict(distmat)
        noofcluster=max(cluster.labels_)
        for j in range(noofcluster+1):
            for i in range(len(cluster.labels_)):
                if cluster.labels_[i]==j:
                    events[e1pair[i]]['coref']=clusterid
            clusterid+=1

    return events

def rearrange(allevent):
    """rearrange structure of event ,move sentence index to be info"""

    events={}
    for sentno in allevent.keys():
        for eventid in allevent[sentno].keys():
            events[eventid]=allevent[sentno][eventid]
            events[eventid]['sentenceid']=sentno
    return events

def allfileoutput(predicted,outputname,options):
    """write trigger to file"""

    ft = codecs.open(outputname, 'w', 'utf8')
    for fname in predicted.keys():
        if options=='last':
            ft.write('#begin document (' + fname + ');\n')

        to_one_file_each(predicted[fname], ft,withkey=False,fname=fname, options=options)

        if options == 'last':
            ft.write('#end document\n\n')
    ft.close()

def to_one_file_coref(predicted,ft,fname):
    """write one file trigger to one file output"""

    for sentno in range(len(predicted)):
        if sentno in predicted.keys():
            for i in range(len(predicted[sentno]['words'])):
                w = predicted[sentno]['words'][i]

                # write trigger to output file
                ft.write(fname)
                ft.write('\t')
                ft.write(w['text'])
                ft.write('\t')
                ft.write(str(w['offset']))
                ft.write('\t')
                if 'coref' in w:
                    ft.write(w['coref'])
                else:
                    ft.write('-')
                ft.write('\n')
            ft.write('\n')

def to_one_file_each(predicted,ft,withkey,fname,options):
    #write one file trigger to one file output

    for sentno in range(len(predicted)):
        if sentno in predicted.keys():
            for i in range(len(predicted[sentno]['words'])):
                trigger,argument=False,False
                w = predicted[sentno]['words'][i]

                # write trigger to output file
                if withkey or options=='last':
                    ft.write(fname)
                    ft.write('\t')

                ft.write(w['text'])
                ft.write('\t')
                ft.write(str(w['offset']))
                ft.write('\t')
                if withkey:
                    if w['keytrigger'] !='O':
                        ft.write(w['keytrigger'])
                    elif w['keyargument']!='O':
                        ft.write(w['keyargument'])
                    else:
                        ft.write('O')
                    ft.write('\t')

                if w['trigger'] != 'O':
                    ft.write(w['trigger'])
                    trigger = True
                elif w['argument'] != 'O':
                    ft.write(w['argument'])
                    argument = True
                else:
                    ft.write('O')

                if options=='role':
                    ft.write('\t')
                    ft.write('O') #dummy role label
                elif options=='realis': #prepare for realis first round
                    ft.write('\t')
                    if 'role' in w:
                        ft.write(w['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if trigger:
                        ft.write('General')
                    else:
                        ft.write('O')
                elif options=='first': #finish first round General/NotGeneral
                    ft.write('\t')
                    if 'role' in w and argument:
                        ft.write(w['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'realis' in w and trigger:
                        ft.write(w['realis'])
                    else:
                        ft.write('O')
                elif options == 'second':  # finish second round Actual/Other
                    ft.write('\t')
                    if 'role' in w and argument:
                        ft.write(w['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'realis' in w and trigger:
                        ft.write(w['realis'])
                    else:
                        ft.write('O')
                elif options == 'last':
                    ft.write('\t')
                    if 'role' in w and argument:
                        ft.write(w['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'realis' in w and trigger:
                        ft.write(w['realis'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'coref' in w and trigger:
                        ft.write(w['coref'])
                    else:
                        ft.write('-')
                ft.write('\n')
            ft.write('\n')

def to_one_file_w_stopwords(predicted,features,ft,withkey,fname,options):
    #write prediction to one file output

    for sentno in range(len(features)):
        if sentno in features.keys():
            i=0
            for token in features[sentno]:
                thislabel={}
                if sentno in predicted.keys():
                    if i<len(predicted[sentno]['words']) and i>=0:
                        w = predicted[sentno]['words'][i]
                    else:
                        w = {'text':'','offset':-1,'trigger':'O','argument':'O','keytrigger':'O','keyargument':'O','role':'O','realis':'O'}

                    if w['text']==token['originalText'] and w['offset']==token['characterOffsetBegin']:
                        thislabel=w
                        i+=1

                    elif i-1 >= 0 and i<len(predicted[sentno]['words']):
                        w0=predicted[sentno]['words'][i-1]
                        if token['characterOffsetBegin']<w['offset']:
                            if w0['argument']=='O' and w0['trigger']=='O' and w['trigger']=='O' and w['argument']=='O':
                                # ordinary
                                thislabel=w
                            elif w['argument'].startswith('B-') or w['trigger'].startswith('B-'):
                                # at the beginning of the next label
                                thislabel={'text':'','offset':-1,'trigger':'O','argument':'O','keytrigger':'O','keyargument':'O','role':'O','realis':'O'}
                            elif (w['argument'].startswith('I-') and w['argument'][2:]==w0['argument'][2:]) or (w['trigger'].startswith('I-') and w['trigger'][2:]==w0['trigger'][2:]):
                                # in between
                                thislabel=w
                            elif (w['argument']=='O' and w0['argument'][2:]!=w['argument'][2:]) or (w['trigger']=='O' and w0['trigger'][2:]!=w['trigger'][2:]):
                                # at the end of previous label
                                thislabel = {'text': '', 'offset': -1, 'trigger': 'O', 'argument': 'O', 'keytrigger': 'O',
                                             'keyargument': 'O', 'role': 'O', 'realis': 'O'}

                    elif i==0:
                        thislabel={'text': '', 'offset': -1, 'trigger': 'O', 'argument': 'O', 'keytrigger': 'O',
                                             'keyargument': 'O', 'role': 'O', 'realis': 'O'}
                    else:
                        thislabel=w
                else:
                    thislabel = {'text': '', 'offset': -1, 'trigger': 'O', 'argument': 'O', 'keytrigger': 'O',
                                 'keyargument': 'O', 'role': 'O', 'realis': 'O'}

                trigger, argument = False, False
                if withkey:
                    ft.write(fname)
                    ft.write('\t')
                ft.write(token['originalText'])
                ft.write('\t')
                ft.write(str(token['characterOffsetBegin']))
                ft.write('\t')

                if withkey:
                    if thislabel['keytrigger'] != 'O':
                        ft.write(thislabel['keytrigger'])
                    elif thislabel['keyargument'] != 'O':
                        ft.write(thislabel['keyargument'])
                    else:
                        ft.write('O')
                    ft.write('\t')

                if thislabel['trigger']!='O':
                    ft.write(thislabel['trigger'])
                    trigger = True


                elif thislabel['argument'] != 'O':
                    ft.write(thislabel['argument'])
                    argument = True

                else:
                    ft.write('O')

                if options == 'role':
                    ft.write('\t')
                    ft.write('O')  # dummy role label

                elif options == 'realis':  # prepare for realis first round
                    ft.write('\t')
                    if 'role' in thislabel:
                        ft.write(thislabel['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if trigger:
                        ft.write('General')
                    else:
                        ft.write('O')

                elif options == 'first':  # finish first round General/NotGeneral
                    ft.write('\t')
                    if 'role' in thislabel and argument:
                        ft.write(thislabel['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'realis' in thislabel and trigger:
                        ft.write(thislabel['realis'])
                    else:
                        ft.write('O')
                elif options == 'second':  # finish second round Actual/Other
                    ft.write('\t')
                    if 'role' in thislabel and argument:
                        ft.write(thislabel['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'realis' in thislabel and trigger:
                        ft.write(thislabel['realis'])
                    else:
                        ft.write('O')
                elif options=='last':
                    ft.write('\t')
                    if 'role' in thislabel and argument:
                        ft.write(thislabel['role'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'realis' in thislabel and trigger:
                        ft.write(thislabel['realis'])
                    else:
                        ft.write('O')
                    ft.write('\t')
                    if 'coref' in thislabel and trigger:
                        ft.write(thislabel['coref'])
                    else:
                        ft.write('-')
                ft.write('\n')
            ft.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument("-input", default=None, help="combined output of TRIGGER ARGUMENT")
parser.add_argument("-dir", default=None, help="directory which contains raw file",required=True)
parser.add_argument("-output", default=None, help="combine output filename",required=True)

def main(argv):

    args=parser.parse_args()
    global w2vmodel
    w2vmodel = gensim.models.Word2Vec.load("../embeddings/Domain-Word2vec.model")

    wordidx, featureidx, labelidx = {}, {}, {}
    databreach_role_model,wordidx['Databreach'],featureidx['Databreach'],labelidx['Databreach']=load_model_role('Databreach')
    phishing_role_model, wordidx['Phishing'], featureidx['Phishing'], labelidx['Phishing'] = load_model_role('Phishing')
    ransom_role_model, wordidx['Ransom'], featureidx['Ransom'], labelidx['Ransom'] = load_model_role('Ransom')
    discover_role_model, wordidx['DiscoverVulnerability'], featureidx['DiscoverVulnerability'], labelidx['DiscoverVulnerability'] = load_model_role('DiscoverVulnerability')
    patch_role_model, wordidx['PatchVulnerability'], featureidx['PatchVulnerability'], labelidx['PatchVulnerability'] = load_model_role('PatchVulnerability')

    realis_GNG_model,wordidx['GNG'],labelidx['GNG'] = load_model_realis('GNG') #Generic vs Non-generic
    realis_AO_model,wordidx['AO'],labelidx['AO'] = load_model_realis('AO') #Actual vs Other"""

    # for each file
    # check if trigger and arguments are compatible
    # find possible trigger from arguments
    # find arguments from trigger
    # produce three output predict file for trigger, predict file for argument, combine trigger and argument

    predicted = readPredicted(args.input)
    predictedwstopwords={}
    for fname in predicted.keys():

        # read content.json for nlp features
        jfile = args.dir + fname + '.content.json'
        labelfile=args.dir+fname+'.content.label'

        features = parseJsontoFeatures.parse(jsonfile=jfile, labelfile=labelfile,options='argument')

        # find more label
        predicted[fname] = recheck(predicted[fname], features)

        # check compatible
        predicted[fname] = compatible(predicted[fname])
        predicted[fname] = combine(predicted[fname])

        # write modified nugget and argument to file -->  Facebook	415	B-Organization	O
        pred_output = args.dir + fname + '_pred.content.nostop.label'
        ft = codecs.open(pred_output, 'w', 'utf8')
        to_one_file_each(predicted[fname], ft, withkey=False, fname=fname, options='role')
        ft.close()

        features = parseJsontoFeatures.parse(jsonfile=jfile, labelfile=labelfile, options='argument')

        # find nugget-argument set
        allevent = link(predicted[fname],features)
        pred_result = role_predict(fname, args.dir, databreach_role_model, phishing_role_model, ransom_role_model, discover_role_model, patch_role_model, wordidx, featureidx, labelidx)
        allevent, predicted[fname] = roleassign(allevent, pred_result, predicted[fname])

        #write role w/o stopword to file and prepare for realis classification --> phishing	725	B-Phishing	O	General
        pred_output = args.dir + fname + '_pred.content.nostop.label'
        ft = codecs.open(pred_output, 'w', 'utf8')
        to_one_file_each(predicted[fname], ft, withkey=False, fname=fname, options='realis')
        ft.close()
        #write role w/stopword to file and prepare for realis classification --> phishing	725	B-Phishing	O	General
        pred_output_stopword = args.dir + fname + '_pred.content.label'
        ft = codecs.open(pred_output_stopword, 'w', 'utf8')
        to_one_file_w_stopwords(predicted[fname], features, ft, withkey=False, fname=fname, options='realis')
        ft.close()

        realis_pred=realis_predict(fname, args.dir, realis_GNG_model, realis_AO_model, wordidx, labelidx, realistype=1)
        allevent, predicted[fname] = realisassign(allevent, realis_pred, predicted[fname])

        pred_output = args.dir + fname + '_pred.content.label'
        ft = codecs.open(pred_output, 'w', 'utf8')
        to_one_file_w_stopwords(predicted[fname], features, ft, withkey=False, fname=fname, options='first')
        ft.close()

        realis_pred = realis_predict(fname, args.dir, realis_GNG_model, realis_AO_model, wordidx, labelidx, realistype=2)
        allevent, predicted[fname] = realisassign(allevent, realis_pred, predicted[fname])

        pred_output = args.dir + fname + '_final_pred.content.label'
        ft = codecs.open(pred_output, 'w', 'utf8')
        to_one_file_w_stopwords(predicted[fname], features, ft, withkey=False, fname=fname, options='second')
        ft.close()

        features = parseJsontoFeatures.parse(jsonfile=jfile, labelfile=labelfile, options='argument')
        txt_input = args.dir + fname + '.txt'
        pubdate=getpubdate(txt_input)
        pred_output = args.dir + fname + '_final_pred.content.label'
        predictedwstopwords[fname]=readFinal(pred_output)
        alleventwstopwords = link(predictedwstopwords[fname], features)
        events=rearrange(alleventwstopwords)

        threshold = 0.75
        weight = [0.141, 0.177, 0.14, 0.148, 0.112, 0.141, 0.141]

        events=corefdiffbyevent(pubdate,events,features,weight,threshold)
        predictedwstopwords[fname]=corefassign(events,predictedwstopwords[fname])


        pred_output = args.dir + fname + '_coref_pred.content.label'
        ft = codecs.open(pred_output, 'w', 'utf8')
        to_one_file_coref(predictedwstopwords[fname], ft, fname=fname)
        ft.close()

    allfileoutput(predictedwstopwords, args.output, options='last')

if __name__ == "__main__":
    main(sys.argv[1:])
