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
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords 
import string
import operator
import tensorflow as tf

tfflags = tf.flags
FLAGS = tfflags.FLAGS

def mat_phrase(train_X, test_X, train_features, test_features, train_y, test_y, MAX_LENGTH):
    """For transfer/domain word2vec, pad sequence and pack them into dict for input to neural networks
       Used by: role_phrase.py"""

    train_mat,test_mat={},{}
    train_X = pad_sequences(train_X, maxlen=MAX_LENGTH, padding='post')
    train_mat['main_input']=train_X
    test_X = pad_sequences(test_X, maxlen=MAX_LENGTH, padding='post')
    test_mat['main_input']=test_X

    for x in train_features.keys():
        y=x.lower()+'_input'
        train_mat[y]=np.array(train_features[x])
        test_mat[y]=np.array(test_features[x])

    return train_mat,test_mat,train_y,test_y

def mat(train_X, test_X, train_y, test_y,  MAX_LENGTH=10, train_features=None, test_features=None, train_extra=None, test_extra=None):
    """For transfer/domain word2vec, pad sequence and pack them into dict for input to neural networks
    Input:   train_X,test_X-train/test data
                train_features, test_features-features set of train/test data
                train_extra, test_extra-extra features set of train/test data
                train_attn_weight, test_attn_weight-weight matrix for attention layer of train/test data
                train_y, test_y-label list of train/test data
                MAX_LENGTH-length of each sample output
                flags-set of flags for choice of features used
                attn-choice of attention used
        Output: train_mat,test_mat-dict of train/test data set
                train_y,test_y-array of label of train/test data
        Used by: nug_arg_detection.py
    """

    train_y = pad_sequences(train_y, maxlen=MAX_LENGTH, padding='post')
    test_y = pad_sequences(test_y, maxlen=MAX_LENGTH, padding='post')

    train_mat,test_mat={},{}
    train_mat["main_input"]=pad_sequences(train_X, maxlen=MAX_LENGTH, padding='post')
    test_mat["main_input"]=pad_sequences(test_X, maxlen=MAX_LENGTH, padding='post')

    if train_features:
        for x in train_features.keys():
            y=x.lower()+'_input'
            if y=='char_input' and usechar:
                train_mat[y]=train_features[x]
                test_mat[y]=test_features[x]
            else:
                train_mat[y]=pad_sequences(train_features[x], maxlen=MAX_LENGTH, padding='post')
                test_mat[y]=pad_sequences(test_features[x], maxlen=MAX_LENGTH, padding='post')

    if train_extra:
        for x in train_extra.keys():
            y = x.lower() + '_input'
            train_mat[y] = pad_sequences(train_extra[x], maxlen=MAX_LENGTH, padding='post')
            test_mat[y] = pad_sequences(test_extra[x], maxlen=MAX_LENGTH, padding='post')


    return train_mat,test_mat,train_y,test_y

def mat_bert(train_X, test_X, train_features, test_features, train_extra, test_extra,train_y, test_y,  MAX_LENGTH, onlyArg):
    """For BERT word2vec, pad sequence and pack them into dict for input to neural networks
    Input:   train_X,test_X-train/test data
                train_features, test_features-features set of train/test data
                train_extra, test_extra-extra features set of train/test data
                train_attn_weight, test_attn_weight-weight matrix for attention layer of train/test data
                train_y, test_y-label list of train/test data
                MAX_LENGTH-length of each sample output
        Output: train_mat,test_mat-dict of train/test data set
                train_y,test_y-array of label of train/test data
        Used by: nug_arg_detection_bert.py
    """
    train_mat,test_mat={},{}

    for i in range(len(train_X)):
        for j in range(len(train_X[i])):
            itype=type(train_X[i][j][0])
            if len(train_X[i][j])!=768:
                print('diff length:',i,j)
            for k in range(len(train_X[i][j])):
                if itype!=type(train_X[i][j][k]):
                   print (itype,type(train_X[i][j][k]))

    train_X = pad_sequences(train_X, maxlen=MAX_LENGTH, padding='post',dtype='float32')
    test_X = pad_sequences(test_X, maxlen=MAX_LENGTH, padding='post',dtype='float32')
    train_y = pad_sequences(train_y, maxlen=MAX_LENGTH, padding='post')
    test_y = pad_sequences(test_y, maxlen=MAX_LENGTH, padding='post')

    train_pos,train_ner,train_dep0,train_dep1,train_dep2, train_lvl,train_chnk,train_wkner,train_dbner,train_subner=train_features
    test_pos,test_ner,test_dep0,test_dep1,test_dep2, test_lvl,test_chnk,test_wkner,test_dbner,test_subner=test_features
    train_mat['bert_input']=train_X
    test_mat['bert_input']=test_X

    train_pos = pad_sequences(train_pos, maxlen=MAX_LENGTH, padding='post')
    test_pos = pad_sequences(test_pos, maxlen=MAX_LENGTH, padding='post')
    train_mat["pos_input"]=train_pos
    test_mat["pos_input"]=test_pos

    train_ner = pad_sequences(train_ner, maxlen=MAX_LENGTH, padding='post')
    test_ner = pad_sequences(test_ner, maxlen=MAX_LENGTH, padding='post')
    train_mat["ner_input"]=train_ner
    test_mat["ner_input"]=test_ner

    train_dep0 = pad_sequences(train_dep0, maxlen=MAX_LENGTH, padding='post')
    test_dep0 = pad_sequences(test_dep0, maxlen=MAX_LENGTH, padding='post')
    train_mat["dep0_input"]=train_dep0
    test_mat["dep0_input"]=test_dep0

    train_dep1 = pad_sequences(train_dep1, maxlen=MAX_LENGTH, padding='post')
    test_dep1 = pad_sequences(test_dep1, maxlen=MAX_LENGTH, padding='post')
    train_mat["dep1_input"]=train_dep1
    test_mat["dep1_input"]=test_dep1

    train_dep2 = pad_sequences(train_dep2, maxlen=MAX_LENGTH, padding='post')
    test_dep2 = pad_sequences(test_dep2, maxlen=MAX_LENGTH, padding='post')
    train_mat["dep2_input"]=train_dep2
    test_mat["dep2_input"]=test_dep2

    train_lvl = pad_sequences(train_lvl, maxlen=MAX_LENGTH, padding='post')
    test_lvl = pad_sequences(test_lvl, maxlen=MAX_LENGTH, padding='post')
    train_mat["lvl_input"]=train_lvl
    test_mat["lvl_input"]=test_lvl

    train_chnk = pad_sequences(train_chnk, maxlen=MAX_LENGTH, padding='post')
    test_chnk = pad_sequences(test_chnk, maxlen=MAX_LENGTH, padding='post')
    train_mat["chnk_input"]=train_chnk
    test_mat["chnk_input"]=test_chnk

    train_wkner = pad_sequences(train_wkner, maxlen=MAX_LENGTH, padding='post')
    test_wkner = pad_sequences(test_wkner, maxlen=MAX_LENGTH, padding='post')
    train_mat["wikiner_input"]=train_wkner
    test_mat["wikiner_input"]=test_wkner

    train_dbner = pad_sequences(train_dbner, maxlen=MAX_LENGTH, padding='post')
    test_dbner = pad_sequences(test_dbner, maxlen=MAX_LENGTH, padding='post')
    train_mat["dbpedianer_input"]=train_dbner
    test_mat["dbpedianer_input"]=test_dbner

    train_subner = pad_sequences(train_subner, maxlen=MAX_LENGTH, padding='post')
    test_subner = pad_sequences(test_subner, maxlen=MAX_LENGTH, padding='post')
    train_mat["subner_input"]=train_subner
    test_mat["subner_input"]=test_subner
    
    if onlyArg:
        for x in train_extra.keys():
            train_extra[x] = pad_sequences(train_extra[x], maxlen=MAX_LENGTH, padding='post')
            test_extra[x] = pad_sequences(test_extra[x], maxlen=MAX_LENGTH, padding='post')
            y=x.lower()+'_input'
            train_mat[y]=train_extra[x]
            test_mat[y]=test_extra[x]

    return train_mat,test_mat,train_y,test_y

AmbiguousList={'Phishing':['Person','Organization','Website'],'Ransom':['Person','Organization','Website','Money'],\
               'Databreach':['Person','Organization','Website','Number'],'DiscoverVulnerability':['Person','Organization'],\
               'PatchVulnerability':['Person','Organization','Website','Device','Product','Version']}

def features_role_phrase(fsentences, each_sent_feature, trainset, eventtype, flabels, train_labels):
    """prepare feature for role-phrase
    Used by:role_phrase.py"""
    sample, labels = [], []
    for s in range(len(fsentences)):  # each sentence  {0:[{token},{token},{}],1:[]}
        now = 0
        for w in range(len(fsentences[s])):
            word = {}
            for y in range(now, len(each_sent_feature[s])):
                if fsentences[s][w]["headentity"].lower() == each_sent_feature[s][y]["originalText"].lower() \
                        and fsentences[s][w]["offset"] == each_sent_feature[s][y]["characterOffsetBegin"]:

                    word["text"] = fsentences[s][w]['surface']
                    word['offset'] = fsentences[s][w]['offset']
                    word['firstoffset'] = fsentences[s][w]['firstoffset']
                    word["finerner"] = each_sent_feature[s][y]["finerner"][2:]
                    word["entitylabel"] = fsentences[s][w]["entitylabel"]  # argidx
                    word["pos"] = each_sent_feature[s][y]["pos"]  # posidx
                    word["ner"] = each_sent_feature[s][y]["ner"][2:]  # neridx
                    word['wk_ner'] = each_sent_feature[s][y]["wk_ner"]
                    word['db_ner'] = each_sent_feature[s][y]["db_ner"]
                    word['finerner'] = each_sent_feature[s][y]["finerner"]

                    if each_sent_feature[s][y]["chunk"]:
                        lowestidx = each_sent_feature[s][y]["chunk"].index(max(each_sent_feature[s][y]["chunk"]))
                        word["lvl"] = each_sent_feature[s][y]["depthchunk"][lowestidx]
                        word["chnk"] = each_sent_feature[s][y]["chunk"][lowestidx]



                    if 'nearestverb' in each_sent_feature[s][y]:
                        word["nearestverb"] = each_sent_feature[s][y]["nearestverb"]  # wordidx
                        word["nearestverbpos"] = each_sent_feature[s][y]["nearestverbpos"]  # posidx
                        for d in each_sent_feature[s][y]["deppathtonearestverb"]:
                            if 'subj' in d or 'obj' in d:
                                word["deppathtonearestverb"] = d  # depidx
                                break
                        else:
                            word[
                                "deppathtonearestverb"] = '-'  # each_sent_feature[s][y]["deppathtonearestverb"][0] #depidx
                        word["dist2nearestverb"] = each_sent_feature[s][y]["dist2nearestverb"]  # distanceidx
                        word["passive"] = each_sent_feature[s][y]["passive"]  # passiveidx
                    else:
                        word["nearestverb"] = '-'
                        word["nearestverbpos"] = '-'
                        word["deppathtonearestverb"] = '-'
                        word["dist2nearestverb"] = '-'
                        word["passive"] = '-'

                    if 'nearEvent' in each_sent_feature[s][y]:
                        word["nearEvent"] = each_sent_feature[s][y]["nearEvent"]
                        word["nearTrigger"] = each_sent_feature[s][y]["nearTrigger"]
                        word["distFromTrigger"] = each_sent_feature[s][y]["distFromTrigger"]
                        word["triggerPosition"] = each_sent_feature[s][y]["triggerPosition"]
                    else:
                        word["nearEvent"] = '-'
                        word["nearTrigger"] = '-'
                        word["distFromTrigger"] = '-'
                        word["triggerPosition"] = '-'

                    if 'leftargument' in each_sent_feature[s][y]:
                        word['leftargument'] = each_sent_feature[s][y]['leftargument']
                    else:
                        word['leftargument'] = '-'

                    if 'rightargument' in each_sent_feature[s][y]:
                        word['rightargument'] = each_sent_feature[s][y]['rightargument']
                    else:
                        word['rightargument'] = '-'
                    now = y
                    if word['nearEvent'] in eventtype:
                        if word['entitylabel'] in AmbiguousList[word['nearEvent']]:
                            sample.append(word)
                            labels.append(flabels[s][w])
                    break

    trainset.append(sample)
    train_labels.append(labels)
    return trainset, train_labels

def features_realis_sentence(fsentences, each_sent_feature):
    """prepare feature for realis classification, models #3 use only word vector
       select only nugget and its context with size 7 from each side
       Used by: realis_identify.py"""

    sentences = []
    offsets = []
    for s in range(len(fsentences)):  # each sentence  {0:[{token},{token},{}],1:[]}

        # {'surface': " ", 'realislabel': Actual, 'offset': 123, 'eventtype': Databreach}
        for w in range(len(fsentences[s])):

            sentence = []
            for y in range(len(each_sent_feature[s])):
                if fsentences[s][w]['offset'][0]==each_sent_feature[s][y]['characterOffsetBegin']:
                    lower=0 if y-7<0 else y-7
                    upper=len(each_sent_feature[s]) if y+7 > len(each_sent_feature[s]) else y+7
                    for i in range(lower,upper,1):
                        word={'text':each_sent_feature[s][i]['originalText']}
                        sentence.append(word)
                    break
            if sentence:
                sentences.append(sentence)
                offsets.append({'offset':fsentences[s][w]['offset'],'index':[s]}) #sentence no
    return sentences,offsets

def features(fsentences,each_sent_feature,sentences,onlyArg,onlyNugget):
    """ remove stopwords and combine features with text
        Input:   fsentences-train/test data from annotation which stopwords were removed
                each_sent_features-features set of train/test data from Stanford CoreNLP output
                sentences-output list 
                onlyArg, onlyNugget-flags for argument/nugget detection                
                flags-set of flags for choice of features used
                neoptions-choice of name entity
                attn-choice of attention used
        Output: train_mat,test_mat-dict of train/test data set
                train_y,test_y-array of label of train/test data
        Used by: nug_arg_detection.py
    """
    usedep, usener, usechnk, useextra, usepos, usecrf, uselstm, usesubner, usegov = False,False,False,False,False,False,False,False,False

    for s in range(len(fsentences)): #each sentence  {0:[{token},{token},{}],1:[]}
        now=0
        sentence=[]

        for w in range(len(fsentences[s])):                     
            word={}

            for y in range(now,len(each_sent_feature[s])):
                if fsentences[s][w]["originalText"]==each_sent_feature[s][y]["originalText"] and \
                        fsentences[s][w]['offset']==each_sent_feature[s][y]['characterOffsetBegin']:
                    word["index"] = each_sent_feature[s][y]["index"]
                    word["text"]=fsentences[s][w]["originalText"]
                    word["characterOffsetBegin"] = each_sent_feature[s][y]["characterOffsetBegin"]


                    word["pos"]=each_sent_feature[s][y]["pos"]

                    word["ner"]=each_sent_feature[s][y]["ner"]
                    word['wk_ner']=each_sent_feature[s][y]["wk_ner"]
                    word['db_ner']=each_sent_feature[s][y]["db_ner"]
                    word['finerner']=each_sent_feature[s][y]["finerner"]

                    try:
                        word["dep_set"]=each_sent_feature[s][y]["dep_set"]
                    except KeyError:
                        pass

                    if each_sent_feature[s][y]["chunk"]:
                        if onlyArg:
                            lowestidx=each_sent_feature[s][y]["chunk"].index(max(each_sent_feature[s][y]["chunk"]))
                            word["lvl"] = each_sent_feature[s][y]["depthchunk"][lowestidx]
                            word["chnk"] = each_sent_feature[s][y]["chunk"][lowestidx]
                        elif onlyNugget:
                            highestidx=each_sent_feature[s][y]["chunk"].index(min(each_sent_feature[s][y]["chunk"]))
                            word["lvl"] = each_sent_feature[s][y]["depthchunk"][highestidx]
                            word["chnk"] = each_sent_feature[s][y]["chunk"][highestidx]
                    else:
                        word["chnk"] = '-'
                        word["lvl"] = '-'

                    if onlyArg:
                        if 'nearEvent' in each_sent_feature[s][y]:
                            word["nearEvent"]=each_sent_feature[s][y]["nearEvent"]              
                            word["nearTrigger"]=each_sent_feature[s][y]["nearTrigger"]               
                            word["distFromTrigger"]=each_sent_feature[s][y]["distFromTrigger"]              
                            word["triggerPosition"]=each_sent_feature[s][y]["triggerPosition"]
                        else:
                            word["nearEvent"]='-'
                            word["nearTrigger"]='-'
                            word["distFromTrigger"]='-'
                            word["triggerPosition"]='-'
                            
                        if 'deppathtoTrigger' in each_sent_feature[s][y]:
                            word["deppathtoTrigger"]=each_sent_feature[s][y]["deppathtoTrigger"]
                            word["deppathtoTriggerLength"]=each_sent_feature[s][y]["deppathtoTriggerLength"]               
                        else:
                            word["deppathtoTrigger"]='-'
                            word["deppathtoTriggerLength"]='-'

                        if 'commonRootwTriggerParse' in each_sent_feature[s][y]:
                            word["commonRootwTriggerParse"]=each_sent_feature[s][y]["commonRootwTriggerParse"]
                            word["depthOfCommonRootwTrigger"]=each_sent_feature[s][y]["depthOfCommonRootwTrigger"]
                        else:
                            word["commonRootwTriggerParse"]='-'
                            word["depthOfCommonRootwTrigger"]='-'
                        
                        if 'isOnly1ItsType' in each_sent_feature[s][y]:
                            word["isOnly1_isNearest"]=each_sent_feature[s][y]["isOnly1ItsType"]+'_'+each_sent_feature[s][y]["isNearestItsType"]
 
                        else:
                            word["isOnly1_isNearest"]='-'
                    
                    now=y
                    sentence.append(word)
                    break
        sentences.append(sentence)

    return sentences

def features_bert_emb(fsentences,each_sent_feature,sentences,onlyArg,onlyTrigger,bert_emb):
    """ bert embedding from file and input to function
        Used by: nug_arg_detection_bert.py"""

    for s in range(len(fsentences)): #each sentence  {0:[{token},{token},{}],1:[]}
        now=0
        sentence=[]

        nowbert=0
        nowunk=0
        for w in range(len(fsentences[s])):                     
            word={}
            sent_emb=bert_emb[s]

            for z in range(nowbert,len(sent_emb["tokens"]),1):

                if fsentences[s][w]["originalText"].lower()==sent_emb["tokens"][z]['originalText'].lower():

                    if 'layers' in sent_emb["tokens"][z]:    
                        word['bert_emb']=sent_emb["tokens"][z]['layers'][3]
                        nowbert=z+1
                        break
                    else:
                        word['bert_emb']=np.random.normal(0,np.sqrt(0.25),768)

            for y in range(now,len(each_sent_feature[s])):
                if fsentences[s][w]["originalText"].lower()==each_sent_feature[s][y]["originalText"].lower():                    
                    word["text"]=fsentences[s][w]["originalText"]
                    word["index"] = each_sent_feature[s][y]["index"]
                    word["characterOffsetBegin"] = each_sent_feature[s][y]["characterOffsetBegin"]
                    
                    word["pos"]=each_sent_feature[s][y]["pos"] 
                    
                    word["ner"]=each_sent_feature[s][y]["ner"]
                    word['wk_ner']=each_sent_feature[s][y]["wk_ner"] 
                    word['db_ner']=each_sent_feature[s][y]["db_ner"] 
                    
                    word['finerner']=each_sent_feature[s][y]["finerner"] 
                    
                    try:
                        word["dep_set"]=each_sent_feature[s][y]["dep_set"]
                    except KeyError:
                        pass
                    
                    try:
                        word["gov_set"] = each_sent_feature[s][y]["gov_rel"]
                    except KeyError:
                        pass
                    
                    if each_sent_feature[s][y]["chunk"]:
                        if onlyArg:
                            lowestidx=each_sent_feature[s][y]["chunk"].index(max(each_sent_feature[s][y]["chunk"]))
                            word["lvl"] = each_sent_feature[s][y]["depthchunk"][lowestidx]
                            word["chnk"] = each_sent_feature[s][y]["chunk"][lowestidx]
                        elif onlyTrigger:
                            highestidx=each_sent_feature[s][y]["chunk"].index(min(each_sent_feature[s][y]["chunk"]))
                            word["lvl"] = each_sent_feature[s][y]["depthchunk"][highestidx]
                            word["chnk"] = each_sent_feature[s][y]["chunk"][highestidx]

                    else:
                        word["chnk"] = '-'
                        word["lvl"] = '-'
                            
                    if onlyArg:
                        if 'nearEvent' in each_sent_feature[s][y]:
                            word["nearEvent"]=each_sent_feature[s][y]["nearEvent"]              
                            word["distFromTrigger"]=each_sent_feature[s][y]["distFromTrigger"]
                            word["triggerPosition"]=each_sent_feature[s][y]["triggerPosition"]
                        else:
                            word["nearEvent"]='-'
                            word["distFromTrigger"]='-'
                            word["triggerPosition"]='-'
                            
                        if 'deppathtoTrigger' in each_sent_feature[s][y]:
                            word["deppathtoTrigger"]=each_sent_feature[s][y]["deppathtoTrigger"]
                            word["deppathtoTriggerLength"]=each_sent_feature[s][y]["deppathtoTriggerLength"]               
                        else:
                            word["deppathtoTrigger"]='-'
                            word["deppathtoTriggerLength"]='-'

                        if 'commonRootwTriggerParse' in each_sent_feature[s][y]:
                            word["commonRootwTriggerParse"]=each_sent_feature[s][y]["commonRootwTriggerParse"]
                            word["depthOfCommonRootwTrigger"]=each_sent_feature[s][y]["depthOfCommonRootwTrigger"]
                        else:
                            word["commonRootwTriggerParse"]='-'
                            word["depthOfCommonRootwTrigger"]='-'
                        
                        if 'isOnly1ItsType' in each_sent_feature[s][y]:
                            word["isOnly1_isNearest"]=each_sent_feature[s][y]["isOnly1ItsType"]+'_'+each_sent_feature[s][y]["isNearestItsType"]
 
                        else:
                            word["isOnly1_isNearest"]='-'

                    now=y+1
                    sentence.append(word)
                    break
            if 'bert_emb' not in word:
                for unk in range(nowunk,len(sent_emb["tokens"]),1):
                    if sent_emb["tokens"][unk]=='[UNK]':
                        word['bert_emb']=sent_emb["tokens"][unk]['layers'][3]
            if 'bert_emb' not in word:
                word['bert_emb'] = np.random.normal(0, np.sqrt(0.25), 768)
        sentences.append(sentence)

    return sentences

def most_common_len(train_X):
    len_list={}

    for sent in train_X:
         length=len(sent)
         if length not in len_list:
             len_list[length]=1
         len_list[length]+=1
    maxfreq=0
    length=0
    for x in len_list.keys():
        if maxfreq<len_list[x]:
            maxfreq=len_list[x]
            length=x
    return length

def avg_len(train_X):

    words=0
    for x in train_X:
        words+=len(x)
    length=words/len(train_X)
    return length

def wv_embedded(wordidx,w2vmodel):
    word_vectors = Word2Vec.load(w2vmodel)
    
    NUM_WORDS=len(word_vectors.wv.vocab)-1
    EMBEDDING_DIM=word_vectors.wv.vector_size
    lenword=len(wordidx)
    vocabulary_size=min(lenword+1,NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in wordidx.items():
        try:          
            embedding_matrix[i] = word_vectors[word] #embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

    del(word_vectors)
    return embedding_matrix,EMBEDDING_DIM,vocabulary_size

def wvf_embedded(wordidx,w2vfmodel):
    f=open(w2vfmodel)
    content=f.readlines()
    num_words=int(content[0].split()[0])
    embedding_dim=int(content[0].split()[1])
    e={}
    for line in content[1:-1]:
        v=line.split()
        c=np.asarray(v[1:], dtype='float32')
        w=v[0]
        e[w]=c
    f.close()

    vocabulary_size=min(len(wordidx)+1,num_words)
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in wordidx.items():
        if i>=num_words:
            continue 
        try:            
            embedding_matrix[i] = e[word]
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),embedding_dim)

    return embedding_matrix,embedding_dim,vocabulary_size


