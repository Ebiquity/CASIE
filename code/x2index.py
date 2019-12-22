"""transform every feature into categorical values"""

import os
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random as rn
rn.seed(1)

os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
import tensorflow as tf
tfflags = tf.flags

FLAGS = tfflags.FLAGS

def args2idx(arguments):

    args2index = {w: i + 2 for i, w in enumerate(arguments)}
    args2index['-PAD-'] = 0  # The special value used for padding
    args2index['-OOV-'] = 1

    return args2index

def argtype2idx(train_sentences):

    arguments=set([])

    for s in train_sentences:
        for w in s:
            for x in w['nearArgumentType']:
                arguments.add(x)

    argtype2index = {w: i + 3 for i, w in enumerate(arguments)}
    argtype2index['-PAD-'] = 0  # The special value used for padding
    argtype2index['-OOV-'] = 1
    argtype2index[None] = 2

    return argtype2index

def chnk2idx(train_sentences):
    """build index for chunk feature
     Input:  train_sentences-input contained chunk features
        Output: chnk2index-dict of chunk index {chunktype: index}
    """
    chnk = set([])
 
    for s in train_sentences:
        for w in s:            
            chnk.add(w['chnk'])
    if '-' in chnk:
        chnk.remove('-')
 
    chnk2index = {w: i + 2 for i, w in enumerate(list(chnk))}
    chnk2index['-PAD-'] = 0  # The special value used for padding
    chnk2index['-OOV-'] = 1

    return chnk2index


def char2idx(train_sentences):
    """build index for character feature
     Input:  train_sentences-input contained surface word
        Output: char2index-dict of character index {character: index}
    """
    chars = set([])
    longest=0
    for s in train_sentences:
        for w in s:
            if longest < len(w['text']):
                longest = len(w['text'])
            for c in w["text"]:                                
                chars.add(c)


    char2index = {c: i + 2 for i, c in enumerate(list(chars))}
    char2index['-PAD-'] = 0  # The special value used for padding
    char2index['-OOV-'] = 1  # The special value used for OOVs
    
    return char2index,longest

def datetype2idx():
    datetype=['DATE','TIME','DURATION','SET', None]

    datetype2index = {w: i + 1 for i, w in enumerate(datetype)}
    datetype2index['-PAD-'] = 0

    return datetype2index

def dbpedianer2idx(sentences):
    """build index for name entity from dbpedia
     Input:  train_sentences-input contained name entity features from dbpedia
        Output: dbner2index-dict of dbpedia index {NERtype: index}
    """
    dbner = set([])
 
    for s in sentences:
        for w in s:  
            dbner.add(w['db_ner'])    

    dbner2index = {w: i + 2 for i, w in enumerate(list(dbner))}
    dbner2index['-PAD-'] = 0  # The special value used for padding
    dbner2index['-OOV-'] = 1

    return dbner2index 

def depcount2idx(train_sentences):
    """build index for number of dependency feature
    Input:  train_sentences-input contained dependency set
        Output: depcount2index-dict of chunk index {dependency: index}
    """
    depcount = set([])
 
    for s in train_sentences:
        for w in s:            
            if 'dep_set' in w:
                depcount.add(len(w['dep_set']))         
            else:
                depcount.add(0)         
 
    depcount2index = {w: i+1 for i, w in enumerate(list(depcount))}
    depcount2index['-PAD-'] = 0  # The special value used for padding

    return depcount2index

def dep2idx(train_sentences):
    """build index for dependency feature
    Input:  train_sentences-input contained dependency set
        Output: dep2index-dict of dependency index {dependency: index}
    """
    dep = set([])
 
    for s in train_sentences:
        for w in s:
            try:
                for x in w['dep_set']: 
                    dep.add(x.split(':')[0])         
            except KeyError:
                pass
    dep2index = {w: i + 2 for i, w in enumerate(list(dep))}
    dep2index['-PAD-'] = 0  # The special value used for padding
    dep2index['-OOV-'] = 1 

    return dep2index

def distance2idx():
    """build index for number of distance feature
     Input:  None
        Output: distance2index-dict of chunk index {distance: index}
    """
    encoded=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','y','z','aa','x']

    distance2index = {w: i + 2 for i, w in enumerate(encoded)}
    distance2index['-PAD-'] = 0  # The special value used for padding
    distance2index['-OOV-'] = 1 

    return distance2index
    
def event2idx(train_sentences):
    """build index for label feature
    Input:  train_sentences-input contained label
        Output: event2index-dict of event index {event: index}
    """
    nearEvent = set([])
 
    for s in train_sentences:
        for w in s:            
            nearEvent.add(w['nearEvent']) 
    if '-' in nearEvent:   
        nearEvent.remove('-')

    event2index = {w: i + 2 for i, w in enumerate(list(nearEvent))}
    event2index['-PAD-'] = 0  # The special value used for padding
    event2index['-OOV-'] = 1 

    return event2index

def entitytypeDep2idx(sentences):
    """build index for entity_dependency feature
    Input:  train_sentences-input contained entity and dependency set
        Output: entdep2index-dict of entity and dependency index {entity_dependency: index}
    """
    entdep = set([])
 
    for s in sentences:
        for w in s:  
            if "entity_deppath0" in w:
                for e in w["entity_deppath0"]:          
                    entdep.add(e)    

    entdep2index = {w: i + 2 for i, w in enumerate(list(entdep))}
    entdep2index['-PAD-'] = 0  # The special value used for padding
    entdep2index['-OOV-'] = 1 

    return entdep2index    

def entityType2idx(sentences):
    """build index for near entity feature
    Input:  train_sentences-input contained near entity feature
        Output: entityType2index-dict of chunk index {nearentity: index}
    """
    entityType = set([])
 
    for s in sentences:
        for w in s:  
            if 'nearEntityType' in w:
                for e in w['nearEntityType']:
                    entityType.add(e)

    entityType.discard('-')

    entityType2index = {w: i + 3 for i, w in enumerate(list(entityType))}
    entityType2index['-PAD-'] = 0  # The special value used for padding
    entityType2index['-OOV-'] = 1
    entityType2index[None] = 2
    return entityType2index 

def fiveevents2idx():

    events=["Phishing","DiscoverVulnerability","Databreach","PatchVulnerability","Ransom","O"]

    event2index = {w: i + 2 for i, w in enumerate(events)}
    event2index['-PAD-'] = 0  # The special value used for padding
    event2index['-OOV-'] = 1

    return event2index

def label2idx(train_tags,pad):
    tags = set([])

    for ts in train_tags:
        for t in ts:
            tags.add(t)

    if pad:
        tag2index = {t: i + 2 for i, t in enumerate(list(tags))}
        tag2index['-PAD-'] = 0  # The special value used to padding
        tag2index['-OOV-'] = 1
    else:
        tag2index = {t: i+1 for i, t in enumerate(list(tags))}
        tag2index['-OOV-'] = 1
    return tag2index


def neartrigger2idx(train_sentences):
    words = set([])

    for s in train_sentences:
        for w in s:
            words.add(w["nearTrigger"].lower())

    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs

    return word2index

def ner2idx(train_sentences):
    ner = set([])
 
    for s in train_sentences:
        for w in s:            
            ner.add(w['ner'])
    
    ner2index = {w: i + 2 for i, w in enumerate(list(ner))}
    ner2index['-PAD-'] = 0  # The special value used for padding
    ner2index['-OOV-'] = 1 

    return ner2index

def passive2idx():
    passive = ['yes','no']

    passive2index = {t: i + 2 for i, t in enumerate(list(passive))}
    passive2index['-PAD-'] = 0  # The special value used to padding
    passive2index['-OOV-'] = 1

    return passive2index


def phrase2word2idx(sentences):
    words = set([])
    maxlen = 0
    for s in sentences:
        for w in s:
            sw = w['text'].split()
            if len(sw) > maxlen:
                maxlen = len(sw)
            for y in sw:
                words.add(y)

    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1

    return word2index, maxlen


def pos2idx(train_sentences):
    pos = set([])
 
    for s in train_sentences:
        for w in s:            
            pos.add(w['pos'])
  
    pos2index = {w: i + 2 for i, w in enumerate(list(pos))}
    pos2index['-PAD-'] = 0  # The special value used for padding
    pos2index['-OOV-'] = 0

    return pos2index

def position2idx(sentences):
    position = set([])
 
    for s in sentences:
        for w in s:            
            position.add(w['triggerPosition']) 
    position.remove('-')   

    position2index = {w: i + 2 for i, w in enumerate(list(position))}
    position2index['-PAD-'] = 0  # The special value used for padding
    position2index['-OOV-'] = 1 

    return position2index

def generic2idx():
    realis=['NonGeneric','Generic']

    realis2index = {w: i+1 for i, w in enumerate(realis)}
    realis2index['-OOV-']=0

    return realis2index

def specific2idx():
    realis=['Actual','Other']

    realis2index = {w: i+1 for i, w in enumerate(realis)}
    realis2index['-OOV-'] = 0
    return realis2index

def rootparse2idx(sentences):
    rootparse = set([])
 
    for s in sentences:
        for w in s:
            if 'commonRootParse' in w:
                for c in w['commonRootParse']:
                    rootparse.add(c) 

    if '-' in rootparse:
        rootparse.remove('-')
    if '' in rootparse:
        rootparse.remove('')
    if "''" in rootparse:
        rootparse.remove("''")

    rootparse2index = {w: i + 2 for i, w in enumerate(list(rootparse))}
    rootparse2index['-PAD-'] = 0  # The special value used for padding
    rootparse2index['-OOV-'] = 1 

    return rootparse2index

def rootparseTrggr2idx(sentences):
    rootparse = set([])
 
    for s in sentences:
        for w in s:  
            if 'commonRootwTriggerParse' in w:          
                rootparse.add(w['commonRootwTriggerParse'])    
                
    if '-' in rootparse:
        rootparse.remove('-')
    if '' in rootparse:
        rootparse.remove('')
    if "''" in rootparse:
        rootparse.remove("''")
    rootparse2index = {w: i + 2 for i, w in enumerate(list(rootparse))}
    rootparse2index['-PAD-'] = 0  # The special value used for padding
    rootparse2index['-OOV-'] = 1 

    return rootparse2index

def role2idx(labels):
    role = set([])
 
    for s in labels:
        for w in s:  
            role.add(w)    

    role2index = {w: i + 2 for i, w in enumerate(list(role))}
    role2index['-PAD-'] = 0  # The special value used for padding
    role2index['-OOV-'] = 1

    return role2index

def secondarg2idx():
    secondarg=['whole-Capa','partial-Capa','whole-Pur','partial-Pur','partial-O-Capa','partial-O-Pur','whole-O']

    secondarg2index={w: i+1 for i,w in enumerate(secondarg)}
    secondarg2index['-PAD-']=0

    return secondarg2index

def secondargWord2idx():
    secondarg=['Purpose','O'] #,'I-Capabilities','I-Purpose','O']  'Capabilities',

    secondarg2index={w: i+1 for i,w in enumerate(secondarg)}
    secondarg2index['-PAD-']=0

    return secondarg2index

def subner2idx(sentences):
    subner = set([])
 
    for s in sentences:
        for w in s:  
            subner.add(w['finerner'])    

    subner2index = {w: i + 2 for i, w in enumerate(list(subner))}
    subner2index['-PAD-'] = 0  # The special value used for padding
    subner2index['-OOV-'] = 1

    return subner2index

def wikiner2idx(sentences):
    wkner = set([])
 
    for s in sentences:
        for w in s:  
            wkner.add(w['wk_ner'])    

    wkner2index = {w: i + 2 for i, w in enumerate(list(wkner))}
    wkner2index['-PAD-'] = 0  # The special value used for padding
    wkner2index['-OOV-'] = 1

    return wkner2index 


def word2idx(train_sentences):
    words, tags = set([]), set([])
 
    for s in train_sentences:
        for w in s:                
            words.add(w["text"].lower())

    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
 
    return word2index

def yesno2idx():
    yesno=['yes_no','yes_yes','no_yes','no_no']

    yesno2index = {w: i + 2 for i, w in enumerate(yesno)}
    yesno2index['-PAD-'] = 0  # The special value used for padding
    yesno2index['-OOV-'] = 1 

    return yesno2index

def sample2idx_bert(train_sentences,test_sentences,train_tags,test_tags,tag2index,word2index,features2index,extra2index,onlyArg):
    """ assign index to each instance with only Trigger options"""
    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
    train_dep0,test_dep0,train_dep1,test_dep1,train_dep2,test_dep2,train_depcount,test_depcount=[],[],[],[],[],[],[],[]
    train_extra,test_extra={},{}

    ### text section ### 
    for s in train_sentences:
        s_int = []
        for w in s:

            s_int.append(w['bert_emb']) 
        train_sentences_X.append(s_int)

    for s in test_sentences:
        s_int = []
        for w in s:

            s_int.append(w['bert_emb'])
        test_sentences_X.append(s_int)

    ### features section ###
    pos2index,ner2index,dep2index,distance2index,chnk2index,wkner2index,dbner2index,subner2index=features2index

    train_pos=sampleFeatureWord2idx(train_sentences, pos2index, 'pos')
    test_pos=sampleFeatureWord2idx(test_sentences, pos2index, 'pos')
    train_chnk=sampleFeatureWord2idx(train_sentences, chnk2index, 'chnk')
    test_chnk=sampleFeatureWord2idx(test_sentences, chnk2index, 'chnk')
    train_lvl=sampleFeatureWord2idx(train_sentences, distance2index, 'lvl')
    test_lvl=sampleFeatureWord2idx(test_sentences, distance2index, 'lvl')
    train_ner=sampleFeatureWord2idx(train_sentences, ner2index, 'ner')
    test_ner=sampleFeatureWord2idx(test_sentences, ner2index, 'ner')
    train_wkner=sampleFeatureWord2idx(train_sentences, wkner2index, 'wk_ner')
    test_wkner=sampleFeatureWord2idx(test_sentences, wkner2index, 'wk_ner')
    train_dbner=sampleFeatureWord2idx(train_sentences, dbner2index, 'db_ner')
    test_dbner=sampleFeatureWord2idx(test_sentences, dbner2index, 'db_ner')
    train_subner=sampleFeatureWord2idx(train_sentences, subner2index, 'finerner')
    test_subner=sampleFeatureWord2idx(test_sentences, subner2index, 'finerner')

    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][0].split(':')[0]])
            except :
                s_int.append(dep2index['-OOV-'])

        train_dep0.append(s_int)

    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][0].split(':')[0]])
            except :
                s_int.append(dep2index['-OOV-'])

        test_dep0.append(s_int)

    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][1].split(':')[0]])
            except :
                s_int.append(dep2index['-OOV-'])

        train_dep1.append(s_int)


    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][1].split(':')[0]])
            except:
                s_int.append(dep2index['-OOV-'])

        test_dep1.append(s_int)

    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][2].split(':')[0]])
            except:
                s_int.append(dep2index['-OOV-'])

        train_dep2.append(s_int)

    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][2].split(':')[0]])
            except:
                s_int.append(dep2index['-OOV-'])

        test_dep2.append(s_int)


    ########## extra features section ##################

    train_deppathtoTrigger0,test_deppathtoTrigger0,train_deppathtoTrigger1,test_deppathtoTrigger1,train_deppathtoTrigger2,test_deppathtoTrigger2=[],[],[],[],[],[]
    train_deppathtoTrigger3,test_deppathtoTrigger3,train_deppathtoTrigger4,test_deppathtoTrigger4=[],[],[],[]

    if onlyArg:

        event2index,position2index,rootparse2index,yesno2index = extra2index

        train_extra['nearEvent'] = sampleFeatureWord2idx(train_sentences, event2index, 'nearEvent')
        test_extra['nearEvent'] = sampleFeatureWord2idx(test_sentences, event2index, 'nearEvent')
        train_extra['nearTrigger'] = sampleFeatureWord2idx(train_sentences, word2index, 'nearTrigger')
        test_extra['nearTrigger'] = sampleFeatureWord2idx(test_sentences, word2index, 'nearTrigger')
        train_extra['distFromTrigger'] = sampleFeatureWord2idx(train_sentences, distance2index, 'distFromTrigger')
        test_extra['distFromTrigger'] = sampleFeatureWord2idx(test_sentences, distance2index, 'distFromTrigger')
        train_extra['triggerPosition'] = sampleFeatureWord2idx(train_sentences, position2index, 'triggerPosition')
        test_extra['triggerPosition'] = sampleFeatureWord2idx(test_sentences, position2index, 'triggerPosition')
        train_extra['isOnly1_isNearest'] = sampleFeatureWord2idx(train_sentences, yesno2index, 'isOnly1_isNearest')
        test_extra['isOnly1_isNearest'] = sampleFeatureWord2idx(test_sentences, yesno2index, 'isOnly1_isNearest')
        train_extra['commonRootwTriggerParse'] = sampleFeatureWord2idx(train_sentences, rootparse2index, 'commonRootwTriggerParse')
        test_extra['commonRootwTriggerParse'] = sampleFeatureWord2idx(test_sentences, rootparse2index, 'commonRootwTriggerParse')
        train_extra['depthOfCommonRootwTrigger'] = sampleFeatureWord2idx(train_sentences, distance2index,
                                                                          'depthOfCommonRootwTrigger')
        test_extra['depthOfCommonRootwTrigger'] = sampleFeatureWord2idx(test_sentences, distance2index,
                                                                         'depthOfCommonRootwTrigger')
        train_extra['deppathtoTriggerLength'] = sampleFeatureWord2idx(train_sentences, distance2index,
                                                                          'deppathtoTriggerLength')
        test_extra['deppathtoTriggerLength'] = sampleFeatureWord2idx(test_sentences, distance2index,
                                                                         'deppathtoTriggerLength')

        # 5. dependency path from the token to trigger
        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][0].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger0.append(s_int)
        train_extra["deppathtoTrigger0"]=train_deppathtoTrigger0
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][0].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger0.append(s_int)
        test_extra["deppathtoTrigger0"]=test_deppathtoTrigger0

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][1].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger1.append(s_int)
        train_extra["deppathtoTrigger1"]=train_deppathtoTrigger1
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][1].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger1.append(s_int)
        test_extra["deppathtoTrigger1"]=test_deppathtoTrigger1

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][2].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger2.append(s_int)
        train_extra["deppathtoTrigger2"]=train_deppathtoTrigger2
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][2].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger2.append(s_int)
        test_extra["deppathtoTrigger2"]=test_deppathtoTrigger2

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][3].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger3.append(s_int)
        train_extra["deppathtoTrigger3"]=train_deppathtoTrigger3
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][3].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger3.append(s_int)
        test_extra["deppathtoTrigger3"]=test_deppathtoTrigger3

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][4].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger4.append(s_int)
        train_extra["deppathtoTrigger4"]=train_deppathtoTrigger4
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][4].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger4.append(s_int)
        test_extra["deppathtoTrigger4"]=test_deppathtoTrigger4

    train_features=(train_pos,train_ner,train_dep0,train_dep1,train_dep2,train_lvl,train_chnk,train_wkner,train_dbner,train_subner)
    test_features=(test_pos,test_ner,test_dep0,test_dep1,test_dep2, test_lvl,test_chnk,test_wkner,test_dbner,test_subner)

    ### label section ###
    train_tags_y=labelEachWord(train_tags, tag2index)
    test_tags_y = labelEachWord(test_tags, tag2index)

    return train_sentences_X, test_sentences_X, train_features, test_features, train_extra, test_extra, train_tags_y, test_tags_y

def sample2idx(train_sentences,test_sentences,train_tags,test_tags,word2index,tag2index,features2index,extra2index,onlyArg):
    """ assign index to each instance with only Trigger options
        Input:   train_sentences,test_sentences-train/test data
                train_tags, test_tags-label list of train/test data
                word2index-dict of word index
                tag2index-dict of label index
                features2index-dict of features index
                extra2index-dict of extra features index
                onlyNugget-choice of nugget detection
                onlyArg-choice of argument detection                 
                flags-set of flags for choice of features used
                attn-choice of attention used
                MAX_LENGTH-length of each sample output
        Output: train_sentences_X, test_sentences_X-index of word of train/test
                train_features, test_features-index of feature set of train/test
                train_extra, test_extra-index of extra features of train/test
                train_tags_y, test_tags_y-index of label of train/test
                train_attn_weight, test_attn_weight-weight vector of train/test
    """

    pos2index,ner2index,dep2index,distance2index,chnk2index,wkner2index,dbner2index,subner2index=features2index
    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
    train_extra,test_extra={},{}
    train_features,test_features={},{}

    ### text section ### 
    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w['text'].lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
 
        train_sentences_X.append(s_int)
 
    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w['text'].lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
 
        test_sentences_X.append(s_int)
    
    ### features section ###
    train_features['pos']=sampleFeatureWord2idx(train_sentences, pos2index, 'pos')
    test_features['pos']=sampleFeatureWord2idx(test_sentences, pos2index, 'pos')
    train_features['chnk']=sampleFeatureWord2idx(train_sentences, chnk2index, 'chnk')
    test_features['chnk']=sampleFeatureWord2idx(test_sentences, chnk2index, 'chnk')
    train_features['lvl']=sampleFeatureWord2idx(train_sentences, distance2index, 'lvl')
    test_features['lvl']=sampleFeatureWord2idx(test_sentences, distance2index, 'lvl')
    train_features['ner']=sampleFeatureWord2idx(train_sentences, ner2index, 'ner')
    test_features['ner']=sampleFeatureWord2idx(test_sentences, ner2index, 'ner')
    train_features['wikiner']=sampleFeatureWord2idx(train_sentences, wkner2index, 'wk_ner')
    test_features['wikiner']=sampleFeatureWord2idx(test_sentences, wkner2index, 'wk_ner')
    train_features['dbpedianer']=sampleFeatureWord2idx(train_sentences, dbner2index, 'db_ner')
    test_features['dbpedianer']=sampleFeatureWord2idx(test_sentences, dbner2index, 'db_ner')
    train_features['subner']=sampleFeatureWord2idx(train_sentences, subner2index, 'finerner')
    test_features['subner']=sampleFeatureWord2idx(test_sentences, subner2index, 'finerner')

    train_dep0=[]
    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][0].split(':')[0]])
            except :
                s_int.append(dep2index['-OOV-'])
        train_dep0.append(s_int)
    train_features['dep0'] = train_dep0

    test_dep0=[]
    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][0].split(':')[0]])
            except :
                s_int.append(dep2index['-OOV-'])
        test_dep0.append(s_int)
    test_features['dep0'] = test_dep0

    train_dep1=[]
    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][1].split(':')[0]])
            except :
                s_int.append(dep2index['-OOV-'])
        train_dep1.append(s_int)
    train_features['dep1'] = train_dep1

    test_dep1=[]
    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][1].split(':')[0]])
            except:
                s_int.append(dep2index['-OOV-'])
        test_dep1.append(s_int)
    test_features['dep1'] = test_dep1

    train_dep2=[]
    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][2].split(':')[0]])
            except:
                s_int.append(dep2index['-OOV-'])
        train_dep2.append(s_int)
    train_features['dep2'] = train_dep2

    test_dep2=[]
    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(dep2index[w['dep_set'][2].split(':')[0]])
            except:
                s_int.append(dep2index['-OOV-'])
        test_dep2.append(s_int)
    test_features['dep2'] = test_dep2

    ########## extra features section ##################

    train_deppathtoTrigger0,test_deppathtoTrigger0,train_deppathtoTrigger1,test_deppathtoTrigger1,train_deppathtoTrigger2,test_deppathtoTrigger2=[],[],[],[],[],[]
    train_deppathtoTrigger3,test_deppathtoTrigger3,train_deppathtoTrigger4,test_deppathtoTrigger4=[],[],[],[]


    if onlyArg:
        event2index,position2index,rootparse2index,yesno2index = extra2index
        train_features['nearEvent'] = sampleFeatureWord2idx(train_sentences, event2index, 'nearEvent')
        test_features['nearEvent'] = sampleFeatureWord2idx(test_sentences, event2index, 'nearEvent')
        train_features['nearTrigger'] = sampleFeatureWord2idx(train_sentences, word2index, 'nearTrigger')
        test_features['nearTrigger'] = sampleFeatureWord2idx(test_sentences, word2index, 'nearTrigger')
        train_features['distFromTrigger'] = sampleFeatureWord2idx(train_sentences, distance2index, 'distFromTrigger')
        test_features['distFromTrigger'] = sampleFeatureWord2idx(test_sentences, distance2index, 'distFromTrigger')
        train_features['triggerPosition'] = sampleFeatureWord2idx(train_sentences, position2index, 'triggerPosition')
        test_features['triggerPosition'] = sampleFeatureWord2idx(test_sentences, position2index, 'triggerPosition')
        train_features['isOnly1_isNearest'] = sampleFeatureWord2idx(train_sentences, yesno2index, 'isOnly1_isNearest')
        test_features['isOnly1_isNearest'] = sampleFeatureWord2idx(test_sentences, yesno2index, 'isOnly1_isNearest')
        train_features['commonRootwTriggerParse'] = sampleFeatureWord2idx(train_sentences, rootparse2index, 'commonRootwTriggerParse')
        test_features['commonRootwTriggerParse'] = sampleFeatureWord2idx(test_sentences, rootparse2index, 'commonRootwTriggerParse')
        train_features['depthOfCommonRootwTrigger'] = sampleFeatureWord2idx(train_sentences, distance2index,
                                                                          'depthOfCommonRootwTrigger')
        test_features['depthOfCommonRootwTrigger'] = sampleFeatureWord2idx(test_sentences, distance2index,
                                                                         'depthOfCommonRootwTrigger')
        train_features['deppathtoTriggerLength'] = sampleFeatureWord2idx(train_sentences, distance2index,
                                                                          'deppathtoTriggerLength')
        test_features['deppathtoTriggerLength'] = sampleFeatureWord2idx(test_sentences, distance2index,
                                                                         'deppathtoTriggerLength')

        # 5. dependency path from the token to trigger
        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][0].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger0.append(s_int)
        train_extra["deppathtoTrigger0"]=train_deppathtoTrigger0
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][0].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger0.append(s_int)
        test_extra["deppathtoTrigger0"]=test_deppathtoTrigger0

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][1].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger1.append(s_int)
        train_extra["deppathtoTrigger1"]=train_deppathtoTrigger1
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][1].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger1.append(s_int)
        test_extra["deppathtoTrigger1"]=test_deppathtoTrigger1

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][2].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger2.append(s_int)
        train_extra["deppathtoTrigger2"]=train_deppathtoTrigger2
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][2].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger2.append(s_int)
        test_extra["deppathtoTrigger2"]=test_deppathtoTrigger2

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][3].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger3.append(s_int)
        train_extra["deppathtoTrigger3"]=train_deppathtoTrigger3
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][3].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger3.append(s_int)
        test_extra["deppathtoTrigger3"]=test_deppathtoTrigger3

        for s in train_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][4].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            train_deppathtoTrigger4.append(s_int)
        train_extra["deppathtoTrigger4"]=train_deppathtoTrigger4
 
        for s in test_sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(dep2index[w['deppathtoTrigger'][4].split(':')[0]])
                except :
                    s_int.append(dep2index['-OOV-'])

            test_deppathtoTrigger4.append(s_int)
        test_extra["deppathtoTrigger4"]=test_deppathtoTrigger4

    ### label section ###
    train_tags_y=labelEachWord(train_tags, tag2index)
    test_tags_y = labelEachWord(test_tags, tag2index)

    return train_sentences_X, test_sentences_X, train_features, test_features, train_extra, test_extra, train_tags_y, test_tags_y

def sampleRolePhrase2idx(train_sentences, test_sentences, train_tags, test_tags, word2index, tag2index,featuresidx):
    """prepare feature into index for role-phrase
    Used by: role_phrase.py"""

    ner2index, subner2index, args2index, distance2index, wikiner2index, dbpedianer2index, passive2index, pos2index = featuresidx
    train_features,test_features={},{}

    ### samples section ###
    train_phrase_X = samplePhrase2Phrase(train_sentences, word2index)
    test_phrase_X = samplePhrase2Phrase(test_sentences, word2index)
    train_features["ner"]=sampleFeatureSentence2idx(train_sentences,ner2index,'ner')
    test_features["ner"] = sampleFeatureSentence2idx(test_sentences, ner2index, 'ner')
    train_features["wikiner"] = sampleFeatureSentence2idx(train_sentences, wikiner2index, 'wk_ner')
    test_features["wikiner"] = sampleFeatureSentence2idx(test_sentences, wikiner2index, 'wk_ner')
    train_features["dbpedianer"] = sampleFeatureSentence2idx(train_sentences, dbpedianer2index, 'db_ner')
    test_features["dbpedianer"] = sampleFeatureSentence2idx(test_sentences, dbpedianer2index, 'db_ner')
    train_features["entitylabel"] = sampleFeatureSentence2idx(train_sentences, args2index, 'entitylabel')
    test_features["entitylabel"] = sampleFeatureSentence2idx(test_sentences, args2index, 'entitylabel')
    train_features["finerner"] = sampleFeatureSentence2idx(train_sentences, subner2index, 'finerner')
    test_features["finerner"] = sampleFeatureSentence2idx(test_sentences, subner2index, 'finerner')
    train_features["distfromtrigger"] = sampleFeatureSentence2idx(train_sentences, distance2index, 'distFromTrigger')
    test_features["distfromtrigger"] = sampleFeatureSentence2idx(test_sentences, distance2index, 'distFromTrigger')
    train_features["leftargument"] = sampleFeatureSentence2idx(train_sentences, args2index, 'leftargument')
    test_features["leftargument"] = sampleFeatureSentence2idx(test_sentences, args2index, 'leftargument')
    train_features["rightargument"] = sampleFeatureSentence2idx(train_sentences, args2index, 'rightargument')
    test_features["rightargument"] = sampleFeatureSentence2idx(test_sentences, args2index, 'rightargument')

    ### label section ###
    train_tags_y = labelEachSentence(train_tags,tag2index)
    test_tags_y = labelEachSentence(test_tags,tag2index)

    return train_phrase_X,test_phrase_X,train_features,test_features,train_tags_y,test_tags_y

def sampleFeatureSentence2idx(train_sentences, feature2index, featurename, keyerrortype='-OOV-'):
    """transform feature to index"""
    train_feature = []
    for s in train_sentences:
        for w in s:
            try:
                train_feature.append(feature2index[w[featurename]])
            except KeyError:
                train_feature.append(feature2index[keyerrortype])

    return train_feature

def sampleFeatureWord2idx(train_sentences, feature2index, featurename, keyerrortype='-OOV-'):
    """transform feature to index"""
    train_feature = []
    for s in train_sentences:
        s_int=[]
        for w in s:
            try:
                s_int.append(feature2index[w[featurename]])
            except KeyError:
                s_int.append(feature2index[keyerrortype])
        train_feature.append(s_int)

    return train_feature

def sampleFeatureListSentence2idx(train_sentences, feature2index, featurename, idx, keyerrortype='-OOV-'):
    """transform feature list to index"""
    train_feature = []
    for s in train_sentences:
        for w in s:
            try:
                train_feature.append(feature2index[w[featurename][idx]])
            except KeyError:
                train_feature.append(feature2index[keyerrortype])

    return train_feature

def sampleFeatureListWord2idx(train_sentences, feature2index, featurename, idx, keyerrortype='-OOV-'):
    """transform feature list to index"""
    train_feature = []
    for s in train_sentences:
        s_int=[]
        for w in s:
            if len(w[featurename])>=idx+1:
                try:
                    s_int.append(feature2index[w[featurename][idx]])
                except KeyError:
                    s_int.append(feature2index[keyerrortype])
            else:
                s_int.append(feature2index[None])

        train_feature.append(s_int)
    return train_feature

def sampleWord2Word(train_sentences,word2index):
    train_sentences_X=[]

    for s in train_sentences:
        s_int = []

        for w in s:

            try:
                s_int.append(word2index[w['text'].lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        train_sentences_X.append(s_int)

    return train_sentences_X

def samplePhrase2Phrase(train_sentences,word2index):
    train_sentences_X=[]

    for s in train_sentences:
        for p in s:

            words=p['text'].split()
            s_int = []
            for w in words:
                try:

                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])

            train_sentences_X.append(s_int)

    return train_sentences_X



def labelEachWord(train_tags,label2index):
    """produce tag index for each sample, labels packed into sentence, one word one label"""
    train_tags_y=[]
    for s in train_tags:
        s_int=[]
        for t in s:
            try:
                s_int.append(label2index[t])
            except KeyError:
                s_int.append(label2index['-OOV-'])

        train_tags_y.append(s_int)

    return train_tags_y

def labelEachSentence(train_tags,label2index):
    """produce tag index for each sample(phrase/sentence), one sentence one label"""
    train_tags_y=[]
    for s in train_tags:
        for t in s:
            try:
                train_tags_y.append(label2index[t])
            except KeyError:
                train_tags_y.append(label2index['-OOV-'])

    return train_tags_y

def sampleWordPhrase2idx(train_sentences, test_sentences, train_tags, test_tags,word2index, label2index):
    """Used by: realis_identify.py"""
    ### samples section ###
    train_sentence_X=sampleWord2Word(train_sentences,word2index)
    test_sentence_X = sampleWord2Word(test_sentences, word2index)

    ### label section ###
    train_tags_y=labelEachSentence(train_tags,label2index)
    test_tags_y = labelEachSentence(test_tags, label2index)

    return train_sentence_X, test_sentence_X, train_tags_y, test_tags_y

