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
import report

import argparse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Embedding, Activation, Flatten
from keras.layers import Dropout, Layer, GRU, Reshape
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from sklearn.metrics import classification_report

EventList=["I-Phishing","B-Phishing","I-DiscoverVulnerability","B-DiscoverVulnerability","B-Databreach","I-Databreach","B-PatchVulnerability","I-PatchVulnerability","B-Ransom","I-Ransom"]

def process_input_phrase(fname,labeloption):
    content = utils.readFileEncode(fname, 'utf8')
    lines = content.split('\n')[:-1]

    sentences, phrases, labels = [], [], []
    phrase, label, text = {}, [], []
    oldtype, offsets, oldevent = '', [], ''
    for i in range(len(lines)):
        if len(lines[i]) > 3:
            words = lines[i].split('\t')
            # select only samples that were labeled as in ArgumentList3
            if words[2] in EventList:
                if words[2].startswith('B-'):
                    if text:
                        phrase = {'surface': " ".join(text), 'realislabel': oldtype, 'offset': offsets, 'eventtype': oldevent}
                        if labeloption==1: #generic vs specific
                            if oldtype=='Other' or oldtype=='Actual':
                                label.append("NotGeneric")
                            else:
                                label.append("Generic")
                            phrases.append(phrase)
                        elif labeloption==2: # not general -> actual vs other
                            if oldtype=='Generic':
                                pass
                            else:
                                label.append(oldtype)
                                phrases.append(phrase)
                        text,offsets = [],[]
                    text.append(words[0])
                    offsets.append(int(words[1]))
                elif words[2].startswith('I-'):
                    if words[2][2:] == oldevent:
                        text.append(words[0])
                        offsets.append(int(words[1]))
                    elif words[2][2:] != oldevent:
                        if text:
                            phrase = {'surface': " ".join(text), 'realislabel': oldtype, 'offset': offsets, 'eventtype': oldevent}
                            if labeloption == 1:  # generic vs specific
                                if oldtype == 'Other' or oldtype == 'Actual':
                                    label.append("NotGeneric")
                                else:
                                    label.append("Generic")
                                phrases.append(phrase)
                            elif labeloption == 2:  # not generic -> actual vs other
                                if oldtype == 'General':
                                    pass
                                else:
                                    label.append(oldtype)
                                    phrases.append(phrase)
                            text,offsets = [],[]
                        text.append(words[0])
                        offsets.append(int(words[1]))

                oldtype = words[4]
                oldevent = words[2][2:]

        else:
            if text:
                phrase = {'surface': " ".join(text), 'realislabel': oldtype, 'offset': offsets, 'eventtype': oldevent}
                if labeloption == 1:  # generic vs specific
                    if oldtype == 'Other' or oldtype == 'Actual':
                        label.append("NotGeneric")
                    else:
                        label.append("Generic")
                    phrases.append(phrase)
                elif labeloption == 2:  # not general -> actual vs other
                    if oldtype == 'Generic':
                        pass
                    else:
                        label.append(oldtype)
                        phrases.append(phrase)
                text,offsets = [],[]

            if len(phrases) > 0 and len(label) > 0:
                sentences.append(phrases)
                labels.append(label)
                phrases = []
                label = []
            elif len(phrases) == 0 and i < len(lines) - 1:
                sentences.append([])
                labels.append([])

    return sentences, labels

def build_model(wordidx, labelidx, MAX_LENGTH):

    """ build model for realis classification task
        Input:  allidx-dict of index of word, label, features, extra features
                MAX_LENGTH-length of sentence
                classweight-weight by frequency for each class
                flags-collection of flags (use/not use) for feature sets
        Output: model and print out model summary
    """

    main_input = Input(shape=(MAX_LENGTH,), name='main_input', dtype='int32')
    inputNodes = [main_input]

    w2vmodel = "../embeddings/Domain-Word2vec.model"
    embedding_matrix, EMBEDDING_DIM, vocabulary_size = prepare.wv_embedded(wordidx, w2vmodel)

    x = Embedding(output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_dim=vocabulary_size,
                  input_length=MAX_LENGTH, mask_zero=True)(main_input)

    numnode = int(EMBEDDING_DIM / 2)
    lstm_out = Dense(numnode)(x)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = layers.NonMasking()(lstm_out)
    lstm_out = Flatten()(lstm_out)
    lstm_out = Dense(len(labelidx))(lstm_out)


    main_output = Activation('softmax')(lstm_out)
    acc = ['accuracy']

    model = Model(inputs=inputNodes, outputs=main_output)
    model.compile(loss=binary_crossentropy, optimizer=SGD(0.001), metrics=acc)
    model.summary()

    return model

parser = argparse.ArgumentParser()
parser.add_argument("-trainfile", default=None, help="train file list", required=True)
parser.add_argument("-testfile", default=None, help="test file list", required=True)
parser.add_argument("-directory", default=None, help="directory contained train and test files", required=True)
parser.add_argument("-epochs", default=1, help="number of epochs for training", type=int)
parser.add_argument("-label", default=None, help="choices of label; 1.non-generic vs generic 2.actual vs other", type=int, required=True)

def main(argv):
    args = parser.parse_args()

    filechange_position = {}
    sentences={}
    if args.trainfile != '' and args.testfile != '':
        ### read train file ###
        lines = utils.readFile(args.trainfile)
        listfile = lines.split('\n')[:-1]
        trainset = []
        train_labels = []
        for fname in listfile:

            labelfile = args.directory + fname + ".content.label"
            fsentences, flabels = process_input_phrase(labelfile,args.label)
            jfile = fname + '.content.json'
            jfile = os.path.join(args.directory, jfile)
            if os.path.isfile(jfile):
                each_sent_feature = parseJsontoFeatures.parse(jfile, labelfile, 'realis')
            else:
                print ('no content.json file', jfile)
                continue
            #only word w/o nlp features
            ftrainset,foffsets = prepare.features_realis_sentence(fsentences, each_sent_feature)
            trainset=trainset+ftrainset
            train_labels=train_labels+flabels

        ### read test files ###
        lines = utils.readFile(args.testfile)
        listfile = lines.split('\n')[:-1]
        testset = []
        test_labels = []
        fileth = 0
        for fname in listfile:
            filechange_position[fileth] = {}
            filechange_position[fileth]['position'] = len(testset)
            filechange_position[fileth]['filename'] = fname
            fileth += 1
            labelfile = args.directory + fname + ".content.label"
            fsentences, flabels = process_input_phrase(labelfile,args.label)

            jfile = fname + '.content.json'
            jfile = os.path.join(args.directory, jfile)
            if os.path.isfile(jfile):
                each_sent_feature = parseJsontoFeatures.parse(jfile, labelfile, 'realis')
            else:
                print ('no content.json file', jfile)
                continue

            ftestset,foffsets = prepare.features_realis_sentence(fsentences, each_sent_feature)

            sentences[fname]={'raw':ftestset,'offset':foffsets}
            testset=testset+ftestset
            test_labels=test_labels+flabels

        wordidx = x2index.word2idx(trainset)
        MAX_LENGTH=len(max(trainset, key=len))

        if args.label==1:
            labelidx = x2index.generic2idx()  # output
        elif args.label==2:
            labelidx = x2index.specific2idx()  # output



        train_X, test_X, train_y, test_y = x2index.sampleWordPhrase2idx(trainset,testset,train_labels,test_labels,wordidx,labelidx)

        ### call neural ###
        train_mat,test_mat={},{}
        train_mat['main_input'] = pad_sequences(train_X, maxlen=MAX_LENGTH, padding='post')
        test_mat['main_input'] = pad_sequences(test_X, maxlen=MAX_LENGTH, padding='post')
        #only word
        model = build_model(wordidx, labelidx, MAX_LENGTH)
        model.fit(train_mat, layers.to_categorical_sentence(train_y, len(labelidx)), batch_size=32, epochs=args.epochs,
                  validation_split=0.2, shuffle=False)

        """if args.label==1:
            modelname = 'realis_GNG.h5'
            labelname = 'realis_label_GNG.pkl'
            wordname = 'realis_word_GNG.pkl'
        elif args.label==2:
            modelname = 'realis_AO.h5'
            labelname = 'realis_label_AO.pkl'
            wordname = 'realis_word_AO.pkl'
        
        model.save(modelname)
        pickle.dump(labelidx, open(labelname, 'wb'))
        pickle.dump(wordidx, open(wordname, 'wb'))"""

        scores = model.evaluate(test_mat, layers.to_categorical_sentence(test_y, len(labelidx)))
        print ("{}:{}".format(model.metrics_names[1], scores[1] * 100))


if __name__ == "__main__":
    main(sys.argv[1:])
