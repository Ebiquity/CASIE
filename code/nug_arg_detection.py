"""Event nugget and argument detection system
   build a neural classifier (Bi-LSTM-CRF) (w/attention for Argument detection)
   Input: file '.content' one token per line
   Output: predicted file with classification score"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import random as rn
rn.seed(1)
import os
os.environ['PYTHONHASHSEED'] = '0'
import sys
sys.path.append('../')


import utils
import parseJsontoFeatures
import x2index
import prepare
import layers
import report

import json
import getopt
import codecs
import pickle
import numpy as np
import argparse


from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Embedding, Activation, Concatenate, Flatten
from keras.layers import Dropout
from keras.optimizers import Adam

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras_self_attention import SeqSelfAttention

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


NuggetList10=["I-Phishing","B-Phishing","I-DiscoverVulnerability","B-DiscoverVulnerability","B-Databreach","I-Databreach",\
              "B-PatchVulnerability","I-PatchVulnerability","B-Ransom","I-Ransom"]
ArgumentList=["B-Website","I-Website","B-Patch","I-Data","I-Money","B-Time","B-Organization","B-Device","I-GPE","B-File",\
              "B-Version","B-Person","B-Software","I-Organization","I-Software","B-GPE","I-Time","I-Person","B-Vulnerability",\
              "B-Data","I-Patch","I-Version","B-Money","I-PaymentMethod","B-PaymentMethod","B-CVE","I-System","I-Tool",\
              "I-Vulnerability","I-Device","B-Tool","B-System","I-File","I-Number","B-Number","B-PII","I-PII","B-Malware",\
              "I-Malware","B-Capabilities","I-Capabilities","I-Purpose","B-Purpose"]



def process_input(fname,onlynugget,onlyarg):
    """ label file split into data and label
        Input:  filename-list of file to be processed
                onlynugget-set to true if detect nuggets
                onlyarg-set to true if detect arguments
        Output: sentence-for each sentence is list of dict of surface word of all files [[{'originalText': ,}]]
                label-list of label
    """
    content=utils.readFileEncode(fname,'utf8')
    lines = content.split('\n')[:-1]
    sentences=[]
    labels=[]
    sent=[]
    label=[]
    for i in range(len(lines)):
        if len(lines[i])>3:
            words=lines[i].split('\t')
            word={'originalText':words[0],'offset':int(words[1])}
            sent.append(word)
            if onlynugget:
                if words[2] in NuggetList10:
                    label.append(words[2])                   
                else:
                    label.append('O')
            elif onlyarg:
                if words[2] in ArgumentList:

                    if 'Software' in words[2]:
                        label.append(words[2][0:2]+'System')
                    else:
                        label.append(words[2])
                else:
                    label.append('O')
        else:
            if len(sent)>0 and len(label)>0:        
                sentences.append(sent)
                labels.append(label)                                 
                sent=[]
                label=[]
            elif len(sent)==0 and i < len(lines)-1:
                sentences.append([])
                labels.append([])
    
    return sentences,labels


def build_model(allidx,MAX_LENGTH,onlyArg):
    """ build model for event detection task
        Input:  allidx-dict of index of word, label, features, extra features
                MAX_LENGTH-length of sentence
                classweight-weight by frequency for each class
                onlyArg-set true if detect argument
                onlyNugget-set true if detect nugget
                flags-collection of flags (use/not use) for feature sets
                w2v-choice of word vector to be used
                attn-choice of attention to be used
        Output: model and print out model summary
    """
    wordidx, labelidx, featuresidx, extraidx=allidx
    posidx, neridx, depidx, distanceidx, chnkidx, wikineridx, dbpedianeridx, subneridx = featuresidx

    main_input = Input(shape=(MAX_LENGTH,), name='main_input', dtype='int32')
    inputNodes=[main_input]

    w2vmodel="../embeddings/Domain-Word2vec.model"

    embedding_matrix,EMBEDDING_DIM,vocabulary_size=prepare.wv_embedded(wordidx,w2vmodel)
        
    x = Embedding(output_dim=EMBEDDING_DIM, weights=[embedding_matrix],input_dim=vocabulary_size, input_length=MAX_LENGTH, mask_zero=False)(main_input)
    numnode=int(EMBEDDING_DIM/2)

    # pos embedding
    inputNodes,pos_layer=layers.embedlayer(inputNodes,"pos_input",posidx,MAX_LENGTH)
    x=Concatenate()([x,pos_layer])
    numnode+=int(len(posidx)/2)

    # ner embedding
    inputNodes,ner_layer=layers.embedlayer(inputNodes,"ner_input",neridx,MAX_LENGTH)
    x=Concatenate()([x,ner_layer])
    numnode+=int(len(neridx)/2)

    inputNodes,wikiner_layer=layers.embedlayer(inputNodes,"wikiner_input",wikineridx,MAX_LENGTH)
    x=Concatenate()([x,wikiner_layer])
    numnode+=int(len(wikineridx)/2)

    inputNodes,dbpedianer_layer=layers.embedlayer(inputNodes,"dbpedianer_input",dbpedianeridx,MAX_LENGTH)
    x=Concatenate()([x,dbpedianer_layer])
    numnode+=int(len(dbpedianeridx)/2)

    # dep embedding
    inputNodes,dep0_layer=layers.embedlayer(inputNodes,"dep0_input",depidx,MAX_LENGTH)
    x=Concatenate()([x,dep0_layer])
    numnode+=int(len(depidx)/2)

    inputNodes,dep1_layer=layers.embedlayer(inputNodes,"dep1_input",depidx,MAX_LENGTH)
    x=Concatenate()([x,dep1_layer])
    numnode+=int(len(depidx)/2)

    inputNodes,dep2_layer=layers.embedlayer(inputNodes,"dep2_input",depidx,MAX_LENGTH)
    x=Concatenate()([x,dep2_layer])
    numnode+=int(len(depidx)/2)

    # chnk embedding
    inputNodes,lvl_layer=layers.embedlayer(inputNodes,"lvl_input",distanceidx,MAX_LENGTH)
    x=Concatenate()([x,lvl_layer])
    numnode+=int(len(distanceidx)/2)

    inputNodes,chnk_layer=layers.embedlayer(inputNodes,"chnk_input",chnkidx,MAX_LENGTH)
    x=Concatenate()([x,chnk_layer])
    numnode+=int(len(chnkidx)/2)

    # wikiclass embedding
    inputNodes,subner_layer=layers.embedlayer(inputNodes,"subner_input",subneridx,MAX_LENGTH)
    x=Concatenate()([x,subner_layer])
    numnode+=int(len(subneridx)/2)

    if onlyArg:
        neartrigger_input = Input(shape=(MAX_LENGTH,), name='neartrigger_input', dtype='int32')
        inputNodes.append(neartrigger_input)
        neartrigger_layer = Embedding(output_dim=EMBEDDING_DIM, weights=[embedding_matrix],input_dim=vocabulary_size, \
                                      input_length=MAX_LENGTH, mask_zero=False)(neartrigger_input)
        x=Concatenate()([x,neartrigger_layer])
        numnode+=50
        inputNodes,x,numnode=layers.extralayer(inputNodes,x,numnode,extraidx,featuresidx,MAX_LENGTH)

    lstm_out = Bidirectional(LSTM(numnode,  dropout=0.2,return_sequences=True))(x)
    numnode=int((numnode+len(labelidx))*2/3)

    if onlyArg:
        lstm_out = SeqSelfAttention(attention_activation='tanh', attention_width=5)(lstm_out)

    lstm_out = Dropout(0.2)(lstm_out)
    out = Dense(numnode)(lstm_out)

    crf = CRF(len(labelidx), sparse_target=False)  # CRF layer
    main_output=crf(out)
    loss=crf_loss #crf.loss_function
    acc=[crf_accuracy]

    model = Model(inputs=inputNodes, outputs=main_output)    
    model.compile(loss=loss,optimizer=Adam(0.001),metrics=acc)
    model.summary()

    return model


parser = argparse.ArgumentParser()
parser.add_argument("-trainfile", default=None, help="train file list", required=True)
parser.add_argument("-testfile", default=None, help="test file list", required=True)
parser.add_argument("-directory", default=None, help="directory contained train and test files", required=True)
parser.add_argument("-epochs", default=1, help="maximum number of epochs for training", type=int)
parser.add_argument("-nugget", default=False, help="set this options for nugget detection", action="store_true")
parser.add_argument("-argument", default=False, help="set this options for argument detection", action="store_true")
parser.add_argument("-outputfile", default=None, help="specify output predict file name")

def main(argv):
    args=parser.parse_args()
            
    if args.nugget:
        parseOpt='trigger'
    elif args.argument:
        parseOpt='argument'
        useextra=True

    filechange_position={}
    if args.trainfile!='' and args.testfile!='' :
        ######### read train files #############
        lines=utils.readFile(args.trainfile)         
        listfile=lines.split('\n')[:-1]
        trainset=[]                
        train_labels=[]
        for fname in listfile:  
            filename=args.directory+fname+'.content.nostop.label'
            fsentences,flabels=process_input(filename,args.nugget,args.argument)
            jfile=fname+'.content.json'
            jfile=os.path.join(args.directory,jfile)

            if os.path.isfile(jfile):
                each_sent_feature=parseJsontoFeatures.parse(jfile,filename,options=parseOpt)
            else:
                print ('no content.json file' % jfile)
                continue
              
            trainset=prepare.features(fsentences,each_sent_feature,trainset,args.argument,args.nugget)

            for l in flabels:
                train_labels.append(l)

        ######## read test files ##############
        lines=utils.readFile(args.testfile)         
        listfile=lines.split('\n')[:-1]
        testset=[]                
        test_labels=[]
        fileth=0

        for fname in listfile:
            filechange_position[fileth]={}
            filechange_position[fileth]['position']=len(testset)
            filechange_position[fileth]['filename']=fname
            fileth+=1
            filename=args.directory+fname+'.content.nostop.label'
            fsentences,flabels=process_input(filename,args.nugget,args.argument)
            jfile=fname+'.content.json'
            jfile=os.path.join(args.directory,jfile)            
            if os.path.isfile(jfile):
                each_sent_feature=parseJsontoFeatures.parse(jfile,filename,options=parseOpt)
            else:
                print ('no content.json file' % jfile)
                continue
              
            testset=prepare.features(fsentences,each_sent_feature,testset,args.argument,args.nugget)

            for l in flabels:
                test_labels.append(l)


        wordidx=x2index.word2idx(trainset)
        labelidx=x2index.label2idx(train_labels,True)
        posidx=x2index.pos2idx(trainset)
        neridx = x2index.ner2idx(trainset)
        wikineridx = x2index.wikiner2idx(trainset)
        dbpedianeridx = x2index.dbpedianer2idx(trainset)
        depidx=x2index.dep2idx(trainset)
        chnkidx=x2index.chnk2idx(trainset)
        subneridx=x2index.subner2idx(trainset)
        distanceidx=x2index.distance2idx()
        extraidx={}

        if args.argument:
            eventidx=x2index.event2idx(trainset)            
            positionidx=x2index.position2idx(trainset)
            rootparseidx=x2index.rootparseTrggr2idx(trainset)
            yesnoidx=x2index.yesno2idx()            
            extraidx=(eventidx,positionidx,rootparseidx,yesnoidx)

        featuresidx=(posidx,neridx,depidx,distanceidx,chnkidx,wikineridx,dbpedianeridx,subneridx)

        MAX_LENGTH = len(max(trainset, key=len))
        
        train_X, test_X, train_features, test_features, train_extra, test_extra, train_y, test_y = x2index.sample2idx(\
            trainset, testset, train_labels, test_labels, wordidx, labelidx, featuresidx, extraidx, args.argument)


        train_mat,test_mat,train_y,test_y=prepare.mat(train_X, test_X, train_y, test_y, MAX_LENGTH, \
                                                      train_features, test_features, train_extra, test_extra)

        allidx=wordidx, labelidx, featuresidx, extraidx
        model=build_model(allidx,MAX_LENGTH,args.argument)
        model.fit(train_mat,layers.to_categorical_word(train_y, len(labelidx)), batch_size=64, epochs=args.epochs, \
                  validation_split=0.2, shuffle=False)

        scores = model.evaluate(test_mat, layers.to_categorical_word(test_y, len(labelidx)))
        print ("{}:{}".format(model.metrics_names[1],scores[1] * 100))

        y_pred = model.predict(test_mat)
        y_classes = y_pred.argmax(axis=-1)

        report.classification_multilabel_multioutput(labelidx, test_y, y_classes)
        report.to_file_multioutput(args.outputfile, labelidx, y_pred, filechange_position, test_y,testset,y_classes)

         
if __name__ == "__main__":
    main(sys.argv[1:])
