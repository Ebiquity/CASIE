from __future__ import absolute_import
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
import extract_features_onthefly as efof
#import argparse


import pickle
import numpy as np

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import extract_features_onthefly as efof
import tensorflow as tf

""" prepare pickle for later training """
""" pickle structure """
""" one content file produce one pickle file         """
"""        {id: filename, 
          sentenceid: [:{index:, 
                         tokens:[
                                 index: , 
                                 originalText: , 
                                 layer:[{0:},{1:},{2:},{3:}]]}]}
        """
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.logging.set_verbosity(tf.logging.ERROR)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("filelist", None,"File contained a list of files (no extension)")    
flags.DEFINE_string("directory", None, "Folder name contained training and testing data files")

""" remove stopwords and combine features with text"""
def build_bert_embedding(jfile,filename):
    out_emb={}
    out_emb["filename"]=filename
    out_emb["sentences"]=[]
    content=utils.loadJsontoDict(jfile)    
    for sent in content["sentences"]: #each sentence  {0:[{token},{token},{}],1:[]}        
        sent_emb={}
        sent_emb['index']=sent["index"]
        sent_emb['tokens']=[]
        sent_text=[]

        for token in sent["tokens"]:
            sent_text.append(token["originalText"])
        #print (sent_text)
        raw_bert_embedding={}
        raw_bert_embedding=efof.get_emb_sentence(" ".join(sent_text))
        """ 
        for i in range(len(raw_bert_embedding["features"])):
            print(raw_bert_embedding["features"][i]["token"],end=" ")
        print()
        """   
        if raw_bert_embedding:     
            bert_embedding=bert_to_word(sent_text,raw_bert_embedding)
        else:
            out_emb["sentences"].append(sent_emb)
            print ("no bert embedding for this sentence",sent["index"])
            continue
        """
        for i in range(len(bert_embedding)):
            print(bert_embedding[i]["token"],bert_embedding[i]['values'][0][0],end=" ")
        print()        
        """
        nowbert=0

        for token in sent["tokens"]: 
            word={}
            word['originalText']=token['originalText']                    
            word['index']=token['index']
            #print (token['originalText'])                    
            for i in range(nowbert,len(bert_embedding),1):
                #print (bert_embedding[i]["token"])
                if bert_embedding[i]["token"].lower()==token['originalText'].lower():
                     emb={}
                     for k in range(len(bert_embedding[i]["values"])):
                         emb[k]=bert_embedding[i]["values"][k]
                     #print (bert_embedding[i]["values"][k][0],emb[0][0],emb[1][0])
                     word['layers']=emb
                     nowbert=i+1
                     break
 
            
            sent_emb['tokens'].append(word)
        nowunk=0
        for word in sent_emb['tokens']:
            if 'layers' not in word:
                for j in range(nowunk,len(bert_embedding),1):
                     if bert_embedding[j]['token']=='[UNK]':
                          emb={}
                          for k in range(len(bert_embedding[i]["values"])):
                              emb[k]=bert_embedding[j]['values'][k]
                          print ('unk: ',emb[0][0],emb[1][0],emb[2][0],emb[3][0],word["originalText"].lower())
                          word['layers']=emb
                          nowunk=j+1
                          break               
        for word in sent_emb['tokens']:       
            if 'layers' not in word:
                print('no embedding for ',word['originalText'])       
            
        out_emb["sentences"].append(sent_emb)
    return out_emb

"""combine bert wordpieces back to original text"""
def bert_to_word(sent_text,bert_embedding):
   
    out_emb=[]
    j=0
    wait_emb=[]
    layers=len(FLAGS.layers.split(','))
    print ('layers',layers)
    for k in range(0,layers):
        wait_emb.append(np.zeros(768))
    wait=''

    ## start from 1 to skip [CLS] ##
    for i in range(1,len(bert_embedding["features"]),1):        
        if j>=len(sent_text):
            break
        if '#' in bert_embedding["features"][i]["token"]:
            if bert_embedding["features"][i]["token"].count('#')==1:             
                wait+=bert_embedding["features"][i]["token"]
            else:
                wait+=bert_embedding["features"][i]["token"].replace('#','')
            ## average
            for k in range(0,layers):
                wait_emb[k]=(wait_emb[k]+np.array(bert_embedding["features"][i]["layers"][k]["values"]))/2
        ## unknown vocab ##
        elif bert_embedding["features"][i]["token"]=='[UNK]':
            ## clear old data
            if wait:
                out={"token":wait, "values":wait_emb}
                out_emb.append(out)
            wait='[UNK]'
            for k in range(0,layers):
                wait_emb[k]=np.array(bert_embedding["features"][i]["layers"][k]["values"])
            out={"token":wait, "values":wait_emb}
            out_emb.append(out)
            wait=''
            wait_emb=[]
            for k in range(0,layers):
                wait_emb.append(np.zeros(768))
            j+=1
        else:
            if wait:
                for k in range(0,layers):
                    wait_emb[k]=(wait_emb[k]+np.array(bert_embedding["features"][i]["layers"][k]["values"]))/2
            else:
                for k in range(0,layers):
                    wait_emb[k]=wait_emb[k]+np.array(bert_embedding["features"][i]["layers"][k]["values"])
            wait+=bert_embedding["features"][i]["token"]
        #print (wait,sent_text[j].lower())
        ## compare with original token
        if wait == sent_text[j].lower():
            out={"token":wait, "values":wait_emb}
            #print(out['token'],out['values'][0][0],out['values'][1][0],end=" ")
            out_emb.append(out)
            wait=''
            wait_emb=[]
            for k in range(0,layers):
                wait_emb.append(np.zeros(768))
            j+=1
        ## reset and move to next word
        elif wait not in sent_text[j].lower():
            wait='[UNK]'
            for k in range(0,layers):
                wait_emb[k]=np.array(bert_embedding["features"][i]["layers"][k]["values"])
            out={"token":wait, "values":wait_emb}
            out_emb.append(out)
            wait=''
            wait_emb=[]
            for k in range(0,layers):
                wait_emb.append(np.zeros(768))
            j+=1  

    return out_emb


def main(argv):
  

    if FLAGS.filelist and FLAGS.directory:
        ######### read train files #############
        lines=utils.readFile(FLAGS.filelist)         
        listfile=lines.split('\n')[:-1]
        for fname in listfile:                        
            print (fname)
            jfile=fname+'.content.json' #'.content.ner.json'
            jfile=os.path.join(FLAGS.directory,jfile)            
            if os.path.isfile(jfile):
                
                bert_emb=build_bert_embedding(jfile,fname)
                outfile=jfile.replace('.content.json','.pkl')
                pickle.dump(bert_emb, open(outfile, 'wb'))
                """
                train=pickle.load(open(outfile,'rb'))
                print (train['filename'])
                for s in train['sentences']:
                    print ('sentenceid:',s['index'])
                    for y in s['tokens']:
                        print ('tokenid:',y['index'],'originalText:',y['originalText'])
                        for l in y['layers']:
                            print ('layers:',y['layers'][l][0])
                """
if __name__ == "__main__":
    tf.app.run()
