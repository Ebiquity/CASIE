import os
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from sklearn.metrics import classification_report
import codecs

import utils
import parseJsontoFeatures

EventList=["Phishing","DiscoverVulnerability","Databreach","PatchVulnerability","Ransom"]

def cuthead(txtfile):
    """ find header size """
    txt=utils.readFileEncode(txtfile,'utf-8')
    return txt.index("<text>")+7

def classification_multilabel_multioutput(labelidx, test_y, y_classes):
    target_names = []
    labels_idx = []
    for label in labelidx.keys():
        if label != 'O' and label != '-PAD-':
            target_names.append(label)
            labels_idx.append(labelidx[label])
    sorted_labels = sorted(target_names, key=lambda name: (name[1:], name[0]))
    idxsort = []
    for label in sorted_labels:
        idxsort.append(labelidx[label])
    print (classification_report(test_y.ravel(), y_classes.ravel(), labels=idxsort, target_names=sorted_labels))

def classification_multilabel_oneoutput(labelidx,test_y, y_classes):
    target_names=[]
    labels_idx=[]
    for label in labelidx.keys():
        if label!='O' and label!='-PAD-' and label!='-OOV-':
            target_names.append(label) #name
            labels_idx.append(labelidx[label]) #number
    sorted_labels = sorted(target_names) #,key=lambda name: (name[1:], name[0]))
    idxsort=[]
    for label in sorted_labels:
        idxsort.append(labelidx[label])
    print (classification_report(test_y, y_classes, labels=idxsort,  target_names=sorted_labels))

def to_file_classification_multilabel_oneoutput(outfile,labelidx,test_y, y_classes):
    f = codecs.open(outfile, "w", "utf8")
    target_names=[]
    labels_idx=[]
    for label in labelidx.keys():
        if label!='O' and label!='-PAD-' and label!='-OOV-':
            target_names.append(label) #name
            labels_idx.append(labelidx[label]) #number
    sorted_labels = sorted(target_names) #,key=lambda name: (name[1:], name[0]))
    idxsort=[]
    for label in sorted_labels:
        idxsort.append(labelidx[label])
    f.write(classification_report(test_y, y_classes, labels=idxsort,  target_names=sorted_labels))
    f.close()

def to_file_multiinoneout(outfile,labelidx,y_pred,filechange_position,testset,test_y,y_classes):
    f = codecs.open(outfile, "w", "utf8")

    labelidxlist = list(labelidx.keys())
    labelvaluelist = list(labelidx.values())
    fileth, nowfile = 0, 0

    for i in range(len(y_pred)):
        if fileth < len(filechange_position) - 1:
            if i < filechange_position[fileth + 1]['position'] and i >= filechange_position[fileth]['position']:
                pass
            else:
                fileth += 1
            nowfile = filechange_position[fileth]['filename']
        sampleword = []
        for j in range(len(y_pred[i])):

            sampleword.append(testset[i][j]['text'])

        sampletag = test_y[i]
        predicttag = y_classes[i]

        f.write(str(nowfile))
        f.write('\t')
        f.write(" ".join(sampleword))
        f.write('\t')
        f.write(labelidxlist[labelvaluelist.index(sampletag)])
        f.write('\t')
        f.write(labelidxlist[labelvaluelist.index(predicttag)])
        f.write('\n')
    f.write('\n')
    f.close()

def to_file_oneinoneout(outfile, wordidx, labelidx, y_pred, filechange_position, testset, test_y, y_classes):

    labelidxlist = list(labelidx.keys())
    labelvaluelist = list(labelidx.values())
    fileth, nowfile = 0, 0

    textlist,offsetlist,filelist=[],[],[]
    for i in range(len(testset)):
        if fileth < len(filechange_position) - 1:
            if i < filechange_position[fileth + 1]['position'] and i >= filechange_position[fileth]['position']:
                pass
            else:
                fileth += 1

            nowfile = filechange_position[fileth]['filename']
        for j in range(len(testset[i])):
            if testset[i][j]:
                textlist.append(testset[i][j]['text'])
                offsetlist.append(testset[i][j]['offset'])
                filelist.append(nowfile)

    if len(textlist)==len(y_pred):
        f = codecs.open(outfile, "w", "utf8")
        for i in range(len(y_pred)):

            sampletag = test_y[i]
            predicttag = y_classes[i]

            f.write(str(filelist[i]))
            f.write('\t')
            f.write(textlist[i])
            f.write('\t')
            f.write(str(offsetlist[i]))
            f.write('\t')
            f.write(labelidxlist[labelvaluelist.index(sampletag)])
            f.write('\t')
            f.write(labelidxlist[labelvaluelist.index(predicttag)])
            f.write('\n')

        f.write('\n')
        f.close()
    else:
        print ('different length')

def to_realis(labelidx,y_pred,y_classes,filechange_position,testset,test_y,sentences):
    result={}
    fileth = 0
    labelidxlist = list(labelidx.keys())
    labelvaluelist = list(labelidx.values())

    for i in filechange_position.keys():
        # initialize result dictionary
        result[filechange_position[i]['filename']]={}

    for fileno in sentences.keys():

        for i in range(len(sentences[fileno]['raw'])):
            result[fileno][i] = {'offset':sentences[fileno]['offset'][i]['offset'],'sentno':sentences[fileno]['offset'][i]['index']}
            print (sentences[fileno]['raw'][i])
            print (sentences[fileno]['offset'][i])


    k=0
    for i in range(len(y_pred)):

        if fileth < len(filechange_position)-1:
            if i < filechange_position[fileth + 1]['position'] and i >= filechange_position[fileth]['position']:
                pass
            else:
                fileth += 1
                k = 0
            nowfile = filechange_position[fileth]['filename']
        print (nowfile)

        sampletag = test_y[i]
        predicttag = y_classes[i]
        result[nowfile][k]['pred']=labelidxlist[labelvaluelist.index(predicttag)]
        result[nowfile][k]['key'] =labelidxlist[labelvaluelist.index(sampletag)]
        k+=1
    for fileno in result.keys():
        print (fileno,result[fileno])
    return result

def realis_to_ann(dir,result):
    for fileno in result.keys():
        aid=1
        annfile=dir+fileno+'_pred.ann'
        content = utils.readFileEncode(annfile, 'utf8')
        token, event, relationlist, attrlist = ann2xml.readAnn(content)
        txtfile=dir+fileno+'.txt'
        head=cuthead(txtfile)
        f=codecs.open(annfile,'a','utf8')
        for eventid in event.keys():
            triggerid=event[eventid]['triggertokenid']
            annoffset=int(token[triggerid]['startOffset'])-head
            for i in range(len(result[fileno])):
                for k in result[fileno][i]['offset']:
                    if k==annoffset:
                        f.write('A'+str(aid)+'\t'+'Realis'+' '+eventid+' '+result[fileno][i]['pred']+'\n')
                        aid+=1
                        break
        f.close()

def to_file_multioutput(outfile,labelidx,y_pred,filechange_position,test_y,testset,y_classes):
    fileth = 0
    f = codecs.open(outfile, "w", "utf8")
    labelidxlist = list(labelidx.keys())
    labelvaluelist = list(labelidx.values())
    nowfile = ''

    for i in range(len(y_pred)):
        if fileth < len(filechange_position) - 1:
            if i < filechange_position[fileth + 1]['position'] and i >= filechange_position[fileth]['position']:
                pass
            else:
                fileth += 1
            nowfile = filechange_position[fileth]['filename']
        for j in range(len(y_pred[i])):
            sampletag = test_y[i][j]
            if sampletag == 0:
                continue
            sampleword = testset[i][j]['text']
            predicttag = y_classes[i][j]
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

def to_nugget(labelidx,y_pred,y_classes,filechange_position,testset,test_y):
    result={} # result {filename: {sentno:[wordno:{text: ,key: , pred}
    fileth = 0
    labelidxlist = list(labelidx.keys())
    labelvaluelist = list(labelidx.values())

    for i in filechange_position.keys():
        # initialize result dictionary
        result[filechange_position[i]['filename']]={}

    wordid=0
    for i in range(len(y_pred)):
        # assign prediction to result dictionary
        if fileth < len(filechange_position) - 1:
            if i < filechange_position[fileth + 1]['position'] and i >= filechange_position[fileth]['position']:
                pass
            else:
                fileth += 1
                wordid=0
            nowfile = filechange_position[fileth]['filename']

        for j in range(len(y_pred[i])):

            sampletag = test_y[i][j]
            if sampletag == 0: #"-PAD-"
                continue

            sampleword = testset[i][j]['text']
            predicttag = y_classes[i][j]
            result[nowfile][wordid]={'text':sampleword,'offset':testset[i][j]['characterOffsetBegin'],'key':labelidxlist[labelvaluelist.index(sampletag)],'pred':labelidxlist[labelvaluelist.index(predicttag)]}
            wordid+=1
    return result

def nugget_to_ann(dir,result):
    for fileno in result.keys():
        tokid=1

        txtfile=dir+fileno+'.txt'
        jfile=dir+fileno+'.content.json'
        content = utils.loadJsontoDict(jfile)
        sentences=content['sentences']
        head=cuthead(txtfile)
        annfile=dir+fileno+'_pred.ann'
        f=codecs.open(annfile,'w','utf8')
        sample, offset = [], []
        for wordno in result[fileno].keys():
            predlabel=result[fileno][wordno]['pred']
            if predlabel.startswith('B-'):
                if sample:
                    text=" ".join(sample)
                    startoffset=offset[0]+head
                    endoffset=offset[-1]+len(sample[-1])+head
                    f.write('T'+str(tokid)+'\t'+label+' '+str(startoffset)+' '+str(endoffset)+'\t'+text+'\n')
                    tokid+=1
                sample, offset = [], []
                sample.append(result[fileno][wordno]['text'])
                offset.append(result[fileno][wordno]['offset'])
                label=predlabel[2:]
            elif predlabel.startswith('I-'):
                if predlabel[2:] != oldlabel[2:]:

                    if sample:
                        text = " ".join(sample)
                        startoffset = offset[0] + head
                        endoffset = offset[-1] + len(sample[-1]) + head
                        f.write('T'+str(tokid)+'\t'+label+' '+str(startoffset)+' '+str(endoffset)+'\t'+text+'\n')
                        tokid += 1
                    sample,offset=[],[]
                    sample.append(result[fileno][wordno]['text'])
                    offset.append(result[fileno][wordno]['offset'])
                    label = predlabel[2:]
                else:
                    sample.append(result[fileno][wordno]['text'])
                    offset.append(result[fileno][wordno]['offset'])
            oldlabel=result[fileno][wordno]['pred']
        if sample:
            text = " ".join(sample)
            startoffset = offset[0] + head
            endoffset = offset[-1] + len(sample[-1]) + head
            f.write('T'+str(tokid)+'\t'+label+' '+str(startoffset)+' '+str(endoffset)+'\t'+text+'\n')
            tokid += 1

        f.close()

def to_argument(labelidx,y_pred,y_classes,filechange_position,testset,test_y):
    result={} # result {filename: {sentno:[wordno:{text: ,key: , pred}
    fileth = 0
    labelidxlist = list(labelidx.keys())
    labelvaluelist = list(labelidx.values())

    for i in filechange_position.keys():
        # initialize result dictionary
        result[filechange_position[i]['filename']]={}

    wordid=0
    for i in range(len(y_pred)):
        # assign prediction to result dictionary
        if fileth < len(filechange_position) - 1:
            if i < filechange_position[fileth + 1]['position'] and i >= filechange_position[fileth]['position']:
                pass
            else:
                fileth += 1
                wordid=0
            nowfile = filechange_position[fileth]['filename']

        for j in range(len(y_pred[i])):

            sampletag = test_y[i][j]
            if sampletag == 0: #"-PAD-"
                continue

            sampleword = testset[i][j]['text']
            predicttag = y_classes[i][j]
            result[nowfile][wordid]={'text':sampleword,'offset':testset[i][j]['characterOffsetBegin'],\
                                     'key':labelidxlist[labelvaluelist.index(sampletag)], \
                                     'pred':labelidxlist[labelvaluelist.index(predicttag)], \
                                     'nearevent':testset[i][j]['nearEvent'], \
                                     'neartrigger':testset[i][j]['nearTrigger'], \
                                     'triggerposition':testset[i][j]['triggerPosition']}
            wordid+=1
    return result

def argument_to_ann(dir,result):
    for fileno in result.keys():
        eventid=1
        tokid=1
        jfile=dir+fileno+'.content.json'
        content = utils.loadJsontoDict(jfile)
        sentences=content['sentences']

        txtfile = dir + fileno + '.txt'
        head=cuthead(txtfile)

        annfile=dir+fileno+'_pred.ann'
        content = utils.readFileEncode(annfile, 'utf8')
        token, event, relationlist, attrlist = ann2xml.readAnn(content)

        idx,event=[],{}

        for tokenid in token.keys():
            # find the last token id from ann
            idx.append(int(tokenid.replace('T','')))
            event[eventid] = {}
            event[eventid]['triggertokenid']=tokenid
            event[eventid]['name']=token[tokenid]['label']
            event[eventid]['arguments']=[]
            eventid+=1
        tokid=max(idx)+1

        f=codecs.open(annfile,'a','utf8')
        sample, offset = [], []
        oldlabel='O'
        for wordno in result[fileno].keys():
            predlabel=result[fileno][wordno]['pred']
            if predlabel.startswith('B-'):
                if sample:
                    text=" ".join(sample)
                    startoffset=offset[0]+head
                    endoffset=offset[-1]+len(sample[-1])+head
                    f.write('T'+str(tokid)+'\t'+label+' '+str(startoffset)+' '+str(endoffset)+'\t'+text+'\n')
                    for eventid in event.keys():
                        trggrtokenid=event[eventid]['triggertokenid']
                        if result[fileno][wordno]['triggerposition']=='before':
                            if startoffset>int(token[trggrtokenid]['endOffset']):
                                if result[fileno][wordno]['nearevent']==token[trggrtokenid]['label'] and result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                                    event[eventid]['arguments'].append({'argname':label, 'value':tokid})
                                    break
                        elif result[fileno][wordno]['triggerposition']=='after':
                            if endoffset > int(token[trggrtokenid]['startOffset']):
                                if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                        result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                                    event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                                    break
                        else:
                            if startoffset > int(token[tokenid]['endOffset']):
                                if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                        result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                                    event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                                    break
                    tokid+=1

                sample, offset = [], []
                sample.append(result[fileno][wordno]['text'])
                offset.append(result[fileno][wordno]['offset'])
                label=predlabel[2:]
            elif predlabel.startswith('I-'):
                if predlabel[2:] != oldlabel[2:]:

                    if sample:
                        text = " ".join(sample)
                        startoffset = offset[0] + head
                        endoffset = offset[-1] + len(sample[-1]) + head
                        f.write('T'+str(tokid)+'\t'+label+' '+str(startoffset)+' '+str(endoffset)+'\t'+text+'\n')
                        for eventid in event.keys():
                            trggrtokenid = event[eventid]['triggertokenid']
                            if result[fileno][wordno]['triggerposition'] == 'before':
                                if startoffset > int(token[trggrtokenid]['endOffset']):
                                    if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                            result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                                        event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                                        break
                            elif result[fileno][wordno]['triggerposition'] == 'after':
                                if endoffset > int(token[trggrtokenid]['startOffset']):
                                    if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                            result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                                        event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                                        break
                            else:
                                if startoffset > int(token[tokenid]['endOffset']):
                                    if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                            result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                                        event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                                        break
                        tokid += 1
                    sample,offset=[],[]
                    sample.append(result[fileno][wordno]['text'])
                    offset.append(result[fileno][wordno]['offset'])
                    label = predlabel[2:]
                else:
                    sample.append(result[fileno][wordno]['text'])
                    offset.append(result[fileno][wordno]['offset'])
            oldlabel=result[fileno][wordno]['pred']
        if sample:
            text = " ".join(sample)
            startoffset = offset[0] + head
            endoffset = offset[-1] + len(sample[-1]) + head
            f.write('T'+str(tokid)+'\t'+label+' '+str(startoffset)+' '+str(endoffset)+'\t'+text+'\n')
            for eventid in event.keys():
                trggrtokenid = event[eventid]['triggertokenid']
                if result[fileno][wordno]['triggerposition'] == 'before':
                    if startoffset > int(token[trggrtokenid]['endOffset']):
                        if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                            event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                            break
                elif result[fileno][wordno]['triggerposition'] == 'after':
                    if endoffset > int(token[trggrtokenid]['startOffset']):
                        if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                            event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                            break
                else:
                    if startoffset > int(token[tokenid]['endOffset']):
                        if result[fileno][wordno]['nearevent'] == token[trggrtokenid]['label'] and \
                                result[fileno][wordno]['neartrigger'] in token[trggrtokenid]['text']:
                            event[eventid]['arguments'].append({'argname': label, 'value': tokid})
                            break

        for eventid in event.keys():
            f.write('E' + str(eventid) + '\t' + event[eventid]['name'] + ':' + event[eventid]['triggertokenid'])
            for arg in event[eventid]['arguments']:
                f.write(' '+arg['argname']+':'+'T'+str(arg['value']))
            f.write('\n')

        f.close()

