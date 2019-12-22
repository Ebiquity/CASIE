import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import sys
sys.path.append('../')
import utils
import json
import getopt
import os
import codecs
import parseJsontoFeatures
import x2index
import chunk
import prepare
import layers
import report

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Embedding, Activation, Concatenate, Conv1D, MaxPooling1D, Flatten, MaxPooling2D
from keras.layers import Dropout, Layer
from keras.optimizers import Adam, SGD
from keras_contrib.utils import save_load_utils
from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report
from keras import backend as K
from itertools import product
import pickle

#ambiguous role choice argument
ArgumentList3=["Organization","Device","Person","Software","System","Website","File","Version","O","Money","Number"]

def process_input_phrase(fname):
    """ label file split into data and label """
    content=utils.readFileEncode(fname,'utf8')
    lines = content.split('\n')[:-1]
    
    sentences,phrases,labels=[],[],[]
    phrase,label,text={},[],[]
    oldtype,oldoffset,firstoffset,thislabel,thisrole='',0,0,0,''
    for i in range(len(lines)):
        if len(lines[i])>3:
            words=lines[i].split('\t')
            #select only samples that were labeled as in ArgumentList3            
            
            if words[2][2:] in ArgumentList3:
                if words[2].startswith('B-'):
                    if text:
                        phrase={'surface':" ".join(text),'entitylabel':thislabel,'headentity':text[-1],\
                                'offset':oldoffset,'firstoffset':firstoffset}
                        phrases.append(phrase)
                        label.append(thisrole)
                    text=[]
                    text.append(words[0])
                    firstoffset=int(words[1])
                    thislabel=words[2][2:]
                    thisrole=words[3][2:]
                elif words[2].startswith('I-'):
                    if words[2][2:]==oldtype[2:]:
                        text.append(words[0])
                    elif words[2][2:]!=oldtype[2:]:
                        if text:
                            phrase={'surface':" ".join(text),'entitylabel':thislabel,'headentity':text[-1], \
                                    'offset':oldoffset,'firstoffset':firstoffset}
                            phrases.append(phrase)
                            label.append(thisrole)
                        text=[]
                        text.append(words[0])
                        firstoffset = int(words[1])
                        thislabel = words[2][2:]
                        thisrole = words[3][2:]
                oldoffset = int(words[1])
            oldtype=words[2]

        else:
            if text:
                phrase={'surface':" ".join(text),'entitylabel':thislabel,'headentity':text[-1], \
                        'offset':oldoffset,'firstoffset':firstoffset}
                phrases.append(phrase)
                label.append(thisrole)
                text=[]

            if len(phrases)>0 and len(label)>0:
                sentences.append(phrases)
                labels.append(label)
                phrases=[]
                label=[]
            elif len(phrases)==0 and i < len(lines)-1:
                sentences.append([])
                labels.append([])

    return sentences,labels

def build_model(allidx,MAX_LENGTH,classweight):

    """ build model for event detection task
    Input:  allidx-dict of index of word, label, features, extra features
                MAX_LENGTH-length of sentence
                classweight-weight by frequency for each class
                flags-collection of flags (use/not use) for feature sets
        Output: model and print out model summary
    """
    wordidx, roleidx, featuresidx = allidx
    main_input = Input(shape=(MAX_LENGTH,), name='main_input', dtype='int32')
    inputNodes = [main_input]

    w2vmodel = "../embeddings/Domain-Word2vec.model"
    embedding_matrix, EMBEDDING_DIM, vocabulary_size = prepare.wv_embedded(wordidx, w2vmodel)

    x = Embedding(output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_dim=vocabulary_size,
                  input_length=MAX_LENGTH, mask_zero=True)(main_input)

    numnode = int(EMBEDDING_DIM / 2)

    neridx, subneridx, argsidx, distanceidx, wikineridx, dbpedianeridx, passiveidx, posidx = featuresidx

    x = Dense(numnode)(x)
    x = Dropout(0.2)(x)
    numnode = int(numnode/2)
    x = Dense(numnode)(x)
    x = Dropout(0.2)(x)
    numnode = int(numnode / 2)
    x = layers.NonMasking()(x)
    x = Flatten()(x)

    inputNodes, distfromtrigger_layer = layers.embedlayer(inputNodes=inputNodes, layername="distfromtrigger_input",x_idx=distanceidx, MAX_LENGTH=1)
    distfromtrigger_layer= layers.NonMasking()(distfromtrigger_layer)
    distfromtrigger_layer= Flatten()(distfromtrigger_layer)
    x = Concatenate()([x, distfromtrigger_layer])
    numnode += int(len(distanceidx) / 2)

    inputNodes, leftargument_layer = layers.embedlayer(inputNodes=inputNodes, layername="leftargument_input", x_idx=argsidx, MAX_LENGTH=1)
    leftargument_layer = layers.NonMasking()(leftargument_layer)
    leftargument_layer = Flatten()(leftargument_layer)
    x = Concatenate()([x, leftargument_layer])
    numnode += int(len(argsidx) / 2)

    inputNodes, rightargument_layer = layers.embedlayer(inputNodes=inputNodes, layername="rightargument_input", x_idx=argsidx, MAX_LENGTH=1)
    rightargument_layer = layers.NonMasking()(rightargument_layer)
    rightargument_layer = Flatten()(rightargument_layer)
    x = Concatenate()([x, rightargument_layer])
    numnode += int(len(argsidx) / 2)

    inputNodes, ner_layer = layers.embedlayer(inputNodes=inputNodes, layername="ner_input", x_idx=neridx, MAX_LENGTH=1)
    ner_layer = layers.NonMasking()(ner_layer)
    ner_layer = Flatten()(ner_layer)
    x = Concatenate()([x, ner_layer])
    numnode += int(len(neridx) / 2)

    inputNodes,wikiner_layer=layers.embedlayer(inputNodes=inputNodes, layername="wikiner_input", x_idx=wikineridx, MAX_LENGTH=1)
    wikiner_layer = layers.NonMasking()(wikiner_layer)
    wikiner_layer = Flatten()(wikiner_layer)
    x=Concatenate()([x,wikiner_layer])
    numnode+=int(len(wikineridx)/2)

    inputNodes,dbpedianer_layer=layers.embedlayer(inputNodes=inputNodes, layername="dbpedianer_input", x_idx=dbpedianeridx, MAX_LENGTH=1)
    dbpedianer_layer = layers.NonMasking()(dbpedianer_layer)
    dbpedianer_layer = Flatten()(dbpedianer_layer)
    x=Concatenate()([x,dbpedianer_layer])
    numnode+=int(len(dbpedianeridx)/2)

    inputNodes, arg_layer = layers.embedlayer(inputNodes=inputNodes, layername="entitylabel_input", x_idx=argsidx, MAX_LENGTH=1)
    arg_layer = layers.NonMasking()(arg_layer)
    arg_layer = Flatten()(arg_layer)
    x = Concatenate()([x, arg_layer])
    numnode += int(len(argsidx) / 2)

    inputNodes, subner_layer = layers.embedlayer(inputNodes=inputNodes, layername="finerner_input",
                                              x_idx=subneridx, MAX_LENGTH=1)
    subner_layer= layers.NonMasking()(subner_layer)
    subner_layer= Flatten()(subner_layer)
    x = Concatenate()([x, subner_layer])
    numnode += int(len(subneridx) / 2)


    numnode = int((numnode + len(roleidx)) * 2 / 3)
    x = Dense(numnode)(x)
    x = Dropout(0.2)(x)
    main_output = Dense(len(roleidx),activation='softmax')(x)
    loss = layers.WeightedCategoricalCrossEntropy(classweight)
    acc = ['accuracy']

    model = Model(inputs=inputNodes, outputs=main_output)
    model.compile(loss=loss, optimizer=Adam(0.001), metrics=acc)
    model.summary()

    return model

parser = argparse.ArgumentParser()
parser.add_argument("-trainfile", default=None, help="train file list", required=True)
parser.add_argument("-testfile", default=None, help="test file list", required=True)
parser.add_argument("-directory", default=None, help="directory contained train and test files", required=True)
parser.add_argument("-epochs", default=1, help="number of epochs for training", type=int, required=True)
parser.add_argument("-eventtype", default=None, help="pick a choice of eventtype, whole means choose all events", choices=['Databreach','Phishing','Ransom','DiscoverVulnerability','PatchVulnerability','whole'],required=True)
parser.add_argument("-outputfile", default=1, help="output predicted filename", required=True)

def main(argv):
    args=parser.parse_args()

    etype=[]    
    if args.eventtype=='whole':
        etype=etype+['PatchVulnerability','DiscoverVulnerability','Ransom','Databreach','Phishing']
    else:
        etype.append(args.eventtype)
        
    filechange_position={}
    if args.trainfile!='' and args.testfile!='' :
        ### read train file ###
        lines=utils.readFile(args.trainfile)         
        listfile=lines.split('\n')[:-1]
        trainset=[]                
        train_labels=[]
        for fname in listfile:            
            filename=args.directory+fname+'.content.nostop.label'
            fsentences,flabels=process_input_phrase(filename) 
            jfile=fname+'.content.json'
            jfile=os.path.join(args.directory,jfile)            
            if os.path.isfile(jfile):
                each_sent_feature=parseJsontoFeatures.parse(jfile,filename,'role')
            else:
                print ('no content.json file', jfile)
                continue

            trainset,train_labels=prepare.features_role_phrase(fsentences,each_sent_feature,trainset,etype,flabels,train_labels)

        ### read test files ###
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
            fsentences,flabels=process_input_phrase(filename) 
            jfile=fname+'.content.json'
            jfile=os.path.join(args.directory,jfile)            
            if os.path.isfile(jfile):
                each_sent_feature=parseJsontoFeatures.parse(jfile,filename,'role')
            else:
                print ('no content.json file', jfile)
                continue
              
            testset,test_labels=prepare.features_role_phrase(fsentences,each_sent_feature,testset,etype,flabels,test_labels)

        wordidx,MAX_LENGTH=x2index.phrase2word2idx(trainset) #input
        labelidx=x2index.role2idx(train_labels) #output

        passiveidx=x2index.passive2idx()
        distanceidx = x2index.distance2idx()
        argsidx=x2index.args2idx(ArgumentList3)
        neridx = x2index.ner2idx(trainset)
        wikineridx = x2index.wikiner2idx(trainset)
        dbpedianeridx = x2index.dbpedianer2idx(trainset)
        subneridx = x2index.subner2idx(trainset)
        posidx = x2index.pos2idx(trainset)

        featuresidx = (neridx, subneridx, argsidx, distanceidx, wikineridx, dbpedianeridx, passiveidx, posidx)

        train_X, test_X, train_features, test_features, train_y, test_y = x2index.sampleRolePhrase2idx(trainset, testset,\
                                                                            train_labels, test_labels, wordidx, labelidx,featuresidx)

        ### count frequency for each role for classweight ###
        classweight={}
        freqeach={}
        for x in labelidx.keys():
            freqeach[labelidx[x]]=0
        for x in train_y:                    
            freqeach[x]+=1         
        for x in labelidx.keys():
            if freqeach[labelidx[x]]==0:
                classweight[labelidx[x]]=0.001
            else:
                classweight[labelidx[x]]=1.0/float(freqeach[labelidx[x]])
        allidx=wordidx, labelidx, featuresidx

        ### call neural ###
        train_mat,test_mat,train_y,test_y=prepare.mat_phrase(train_X,test_X,train_features,test_features,train_y,test_y,MAX_LENGTH)
        model=build_model(allidx,MAX_LENGTH,classweight)
        model.fit(train_mat, layers.to_categorical_sentence(train_y, len(labelidx)), batch_size=32, epochs=args.epochs,
              validation_split=0.2, shuffle=False)

        """modelname="role_"+args.eventtype+".h5"
        model.save(modelname)
        classweightname="role_classweight_"+args.eventtype+".pkl"
        pickle.dump(classweight, open(classweightname, 'wb'))
        labelname="role_label_"+args.eventtype+'.pkl'
        pickle.dump(labelidx, open(labelname, 'wb'))
        wordname="role_word_"+args.eventtype+'.pkl'
        pickle.dump(wordidx, open(wordname, 'wb'))
        featurename="role_feature_"+args.eventtype+'.pkl'
        pickle.dump(featuresidx, open(featurename, 'wb'))"""

        y_pred = model.predict(test_mat)
        scores = model.evaluate(test_mat, layers.to_categorical_sentence(test_y,len(labelidx)))
        print ("{}:{}".format(model.metrics_names[1],scores[1] * 100))
        y_classes = y_pred.argmax(axis=-1)

        predfile="role_predict.txt"
        scorefile = "role_score.txt"

        report.to_file_classification_multilabel_oneoutput(scorefile,labelidx, test_y, y_classes)
        report.to_file_oneinoneout(predfile, wordidx, labelidx, y_pred, filechange_position, testset, test_y,y_classes)

if __name__ == "__main__":
    main(sys.argv[1:])        
        


