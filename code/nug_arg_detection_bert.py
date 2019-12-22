from __future__ import absolute_import
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import sys
sys.path.append('../')

import json
import getopt
import os
import codecs

import utils
import parseJsontoFeatures
import x2index
import prepare
import layers

import pickle
import numpy as np

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Embedding, Activation, Concatenate, Flatten
from keras.layers import Dropout, Reshape, Permute
from keras.optimizers import Adam

from sklearn.metrics import classification_report

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from keras_self_attention import SeqSelfAttention

from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.logging.set_verbosity(tf.logging.ERROR)
flags = tf.app.flags
FLAGS = flags.FLAGS

NuggetList10=["I-Phishing","B-Phishing","I-DiscoverVulnerability","B-DiscoverVulnerability","B-Databreach","I-Databreach","B-PatchVulnerability","I-PatchVulnerability","B-Ransom","I-Ransom"]
ArgumentList=["B-Website","I-Website","B-Patch","I-Data","I-Money","B-Time","B-Organization","B-Device","I-GPE","B-File","B-Version","B-Person","B-Software","I-Organization","I-Software","B-GPE","I-Time","I-Person","B-Vulnerability","B-Data","I-Patch","I-Version","B-Money","I-PaymentMethod","B-PaymentMethod","B-CVE","I-System","I-Tool","I-Vulnerability","I-Device","B-Tool","B-System","I-File","I-Number","B-Number","B-PII","I-PII","B-Malware","I-Malware","B-Capabilities","I-Capabilities","I-Purpose","B-Purpose"]


def process_input(fname,onlytrigger,onlyarg):
    """ label file split into data and label """
    content=utils.readFileEncode(fname,'utf8')
    lines = content.split('\n')[:-1]
    sentences=[]
    labels=[]
    sent=[]
    label=[]

    for i in range(len(lines)):
        if len(lines[i])>3:
            words=lines[i].split('\t')
            word={'originalText':words[0]}            
            sent.append(word)

            if onlytrigger:
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

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)



def build_model(labelidx, featuresidx, extraidx, MAX_LENGTH, onlyArg):

    bert_input = Input(shape=(MAX_LENGTH, 768), name='bert_input', dtype='float32')
    x=Bidirectional(LSTM(384,  dropout=0.2,return_sequences=True))(bert_input)
    numnode=170
    inputNodes = [bert_input]
    numnode = int(768 / 2)

    posidx, neridx, depidx, distanceidx, chnkidx, wikineridx, dbpedianeridx, subneridx = featuresidx

    inputNodes, pos_layer = layers.embedlayer(inputNodes, "pos_input", posidx, MAX_LENGTH)
    x = Concatenate()([x, pos_layer])
    numnode += int(len(posidx) / 2)


    inputNodes, dep0_layer = layers.embedlayer(inputNodes, "dep0_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, dep0_layer])
    numnode += int(len(depidx) / 2)

    inputNodes, dep1_layer = layers.embedlayer(inputNodes, "dep1_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, dep1_layer])
    numnode += int(len(depidx) / 2)

    inputNodes, dep2_layer = layers.embedlayer(inputNodes, "dep2_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, dep2_layer])
    numnode += int(len(depidx) / 2)

    inputNodes, lvl_layer = layers.embedlayer(inputNodes, "lvl_input", distanceidx, MAX_LENGTH)
    x = Concatenate()([x, lvl_layer])
    numnode += int(len(distanceidx) / 2)

    inputNodes, chnk_layer = layers.embedlayer(inputNodes, "chnk_input", chnkidx, MAX_LENGTH)
    x = Concatenate()([x, chnk_layer])
    numnode += int(len(chnkidx) / 2)


    inputNodes, wikiner_layer = layers.embedlayer(inputNodes, "wikiner_input", wikineridx, MAX_LENGTH)
    x = Concatenate()([x, wikiner_layer])
    numnode += int(len(wikineridx) / 2)

    inputNodes, dbpedianer_layer = layers.embedlayer(inputNodes, "dbpedianer_input", dbpedianeridx, MAX_LENGTH)
    x = Concatenate()([x, dbpedianer_layer])
    numnode += int(len(dbpedianeridx) / 2)

    inputNodes, ner_layer = layers.embedlayer(inputNodes, "ner_input", neridx, MAX_LENGTH)
    x = Concatenate()([x, ner_layer])
    numnode += int(len(neridx) / 2)

    inputNodes, subner_layer = layers.embedlayer(inputNodes, "subner_input", subneridx, MAX_LENGTH)
    x = Concatenate()([x, subner_layer])
    numnode += int(len(subneridx) / 2)


    if onlyArg:

        inputNodes, x, numnode = layers.extralayer(inputNodes, x, numnode, extraidx, featuresidx, MAX_LENGTH)

    numnode = int((numnode + len(labelidx)) * 2 / 3)
    lstm_out = Bidirectional(LSTM(numnode,  dropout=0.2,return_sequences=True))(x)

    if onlyArg:
        lstm_out = SeqSelfAttention(attention_activation='tanh', attention_width=5)(lstm_out)

    td = TimeDistributed(Dense(numnode))(lstm_out)
    crf = CRF(len(labelidx), sparse_target=False)  # CRF layer
    main_output = crf(td)
    loss = crf_loss  # crf.loss_function
    acc = [crf_accuracy]

    model = Model(inputs=inputNodes, outputs=main_output)
    model.compile(loss=loss, optimizer=Adam(0.001), metrics=acc)
    model.summary()

    return model


flags.DEFINE_string("trainfile", None,"File contained a list of training data files (no extension)")    
flags.DEFINE_string("testfile", None, "File contained a list of testing data files (no extension)")
flags.DEFINE_string("directory", None, "Folder name contained training and testing data files")
flags.DEFINE_bool("nugget", False, "set this options for trigger detection")
flags.DEFINE_bool("argument", False, "set this options for argument detection")
flags.DEFINE_integer("epochs", 1, "set number of epochs for training model")

def main(argv):
  
    onlyArg,onlyTrigger,parseOpt=False,False,False
    if FLAGS.nugget:
        onlyTrigger=True
        onlyArg=False
        parseOpt='trigger'
    if FLAGS.argument:
        onlyArg=True
        onlyTrigger=False
        parseOpt='argument'

    filechange_position={}


    if FLAGS.trainfile:
        ######### read train files #############
        lines=utils.readFile(FLAGS.trainfile)         
        listfile=lines.split('\n')[:-1]
        trainset,train_bert_emb=[],[]
        train_labels=[]
        for fname in listfile:
            filename=FLAGS.directory+fname+'.content.nostop.label'
            fsentences,flabels=process_input(filename,onlyTrigger,onlyArg) 
            jfile=fname+'.content.json'
            jfile=os.path.join(FLAGS.directory,jfile)            
            if os.path.isfile(jfile):
                each_sent_feature=parseJsontoFeatures.parse(jfile,filename,parseOpt)
            else:
                print ('no content.json file' ,str(jfile))
                continue
            bfile=fname+'.pkl'
            bertfile=os.path.join(FLAGS.directory,bfile)
            train_bert_emb=pickle.load(open(bertfile,"rb"))
            if train_bert_emb:
                bert_emb=train_bert_emb["sentences"]

            else:
                print ('cannot find bert')
            trainset=prepare.features_bert_emb(fsentences,each_sent_feature,trainset,onlyArg,onlyTrigger,bert_emb)

            for l in flabels:
                train_labels.append(l)
    if FLAGS.testfile:
        ######## read test files ##############
        lines=utils.readFile(FLAGS.testfile)         
        listfile=lines.split('\n')[:-1]
        testset,test_bert_emb=[],[]
        test_labels=[]
        fileth=0
        for fname in listfile:
            filechange_position[fileth]={}
            filechange_position[fileth]['position']=len(testset)
            filechange_position[fileth]['filename']=fname
            fileth+=1
            filename=FLAGS.directory+fname+'.content.nostop.label'
            fsentences,flabels=process_input(filename,onlyTrigger,onlyArg) 
            jfile=fname+'.content.json'
            jfile=os.path.join(FLAGS.directory,jfile)            
            if os.path.isfile(jfile):
                each_sent_feature=parseJsontoFeatures.parse(jfile,filename,parseOpt)
            else:
                print ('no content.json file' ,str(jfile))
                continue

            bfile=fname+'.pkl'
            bertfile=os.path.join(FLAGS.directory,bfile)
            test_bert_emb=pickle.load(open(bertfile,"rb"))         
            bert_emb=test_bert_emb["sentences"]
            testset=prepare.features_bert_emb(fsentences,each_sent_feature,testset,onlyArg,onlyTrigger,bert_emb)
            for l in flabels:
                test_labels.append(l)

        wordidx=x2index.word2idx(trainset)
        labelidx=x2index.label2idx(train_labels,True)

        posidx=x2index.pos2idx(trainset)
        neridx=x2index.ner2idx(trainset)
        wikineridx = x2index.wikiner2idx(trainset)
        dbpedianeridx = x2index.dbpedianer2idx(trainset)

        depidx=x2index.dep2idx(trainset)
        chnkidx=x2index.chnk2idx(trainset)
        distanceidx=x2index.distance2idx()
        subneridx=x2index.subner2idx(trainset)

        extraidx={}
              
        if onlyArg:
            eventidx=x2index.event2idx(trainset)            
            positionidx=x2index.position2idx(trainset)
            rootparseidx=x2index.rootparseTrggr2idx(trainset)
            yesnoidx=x2index.yesno2idx()            
            extraidx=(eventidx,positionidx,rootparseidx,yesnoidx)

        featuresidx=(posidx,neridx,depidx,distanceidx,chnkidx,wikineridx,dbpedianeridx,subneridx)

        MAX_LENGTH = len(max(trainset, key=len))
        train_X, test_X, train_features, test_features, train_extra, test_extra, train_y, test_y = x2index.sample2idx_bert( \
            trainset, testset, train_labels, test_labels, wordidx, labelidx, featuresidx, extraidx, onlyArg)
        
        train_mat,test_mat,train_y,test_y=prepare.mat_bert(train_X, test_X, train_features, test_features, \
                            train_extra, test_extra, train_y, test_y, MAX_LENGTH, onlyArg)

        callbacks = [EarlyStopping(monitor='val_loss', patience=5), \
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        model=build_model(labelidx,featuresidx,extraidx,MAX_LENGTH,onlyArg)
        model.fit(train_mat,to_categorical(train_y, len(labelidx)), batch_size=64, epochs=FLAGS.epochs, validation_split=0.2, shuffle=False, callbacks=callbacks)

        scores = model.evaluate(test_mat, to_categorical(test_y, len(labelidx)))
        print ("{}:{}".format(model.metrics_names[1],scores[1] * 100))
        """
        if onlyTrigger:
            model.save("bestlstm_trigger_fnload.h5")
            pickle.dump(classweight, open("classweight_trigger.pkl", 'wb'))
            pickle.dump(labelidx, open("triggeridx.pkl", 'wb')) 
        else:
            model.save("bestlstm_argument_fnload.h5")
            pickle.dump(classweight, open("classweight_argument.pkl", 'wb'))
            pickle.dump(labelidx, open("argumentidx.pkl", 'wb')) 
        """
        y_pred = model.predict(test_mat)
        y_classes = y_pred.argmax(axis=-1)
        
        fileth=0
        predfile = "pred_output.txt"
        f=codecs.open(predfile,"w","utf8")
        labelidxlist=list(labelidx.keys())
        labelvaluelist=list(labelidx.values())
        nowfile=''

        for i in range(len(y_pred)):
            if fileth < len(filechange_position)-1:
                if i<filechange_position[fileth+1]['position'] and i>=filechange_position[fileth]['position']:
                    pass
                else:
                    fileth+=1 
                nowfile=filechange_position[fileth]['filename']           
            for j in range(len(y_pred[i])):
                sampletag=test_y[i][j]
                if sampletag==0:
                    continue 
                sampleword=testset[i][j]['text']                
                predicttag=y_classes[i][j]               
                f.write(str(nowfile))
                f.write('\t')
                f.write(sampleword)
                f.write('\t')
                f.write(str(testset[i][j]['characterOffsetBegin']))
                f.write('\t')
                f.write(labelidxlist[labelvaluelist.index(sampletag)])
                f.write('\t')
                f.write(labelidxlist[labelvaluelist.index(predicttag)])
                f.write('\n')
            f.write('\n')
        f.close()
            
        target_names=[]
        labels_idx=[]
        for label in labelidx.keys():
            if label!='O' and label!='-PAD-':
                target_names.append(label)
                labels_idx.append(labelidx[label])
        sorted_labels = sorted(target_names,key=lambda name: (name[1:], name[0]))
        idxsort=[]
        for label in sorted_labels:        
            idxsort.append(labelidx[label])
        print (classification_report(test_y.ravel(), y_classes.ravel(), labels=idxsort,  target_names=sorted_labels))
        
         
if __name__ == "__main__":
    tf.app.run()
