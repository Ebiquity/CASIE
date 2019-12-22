import utils
import copy
import sys
import getopt
import argparse

NuggetList=["I-DDoS","B-DDoS","I-Phishing","B-Phishing","I-DiscoverVulnerability","B-DiscoverVulnerability","B-Databreach","I-Databreach","B-PatchVulnerability","I-PatchVulnerability","B-Ransom","I-Ransom","O"]
ArgumentList=["B-Website","I-Website","B-Patch","I-Data","I-Money","B-Time","B-Organization","B-Device","I-GPE","B-File","B-Version","B-Person","I-Organization","B-GPE","I-Time","I-Person","B-Vulnerability","B-Data","I-Patch","I-Version","B-Money","I-PaymentMethod","B-PaymentMethod","B-CVE","I-System","I-Vulnerability","I-Device","B-System","I-File","I-Number","B-Number","B-PII","I-PII","B-Malware","I-Malware","B-Capabilities","I-Capabilities","I-Purpose","B-Purpose"]
RoleList=["B-Attacker","I-Attacker","B-Victim","I-Victim","B-Tool","I-Tool","B-Discoverer","B-Vulnerable_System_Owner","I-Vulnerable_System_Owner","B-Legitimate-Inst","I-Supported_Platform","I-Vulnerable_System","I-Discoverer","B-Victim","I-Legitimate-Inst","B-Vulnerable_System","I-Releaser","B-Releaser","B-Supported_Platform"]
       
def typescore(lines,choice):
    right={}
    keys={}
    predicted={}
    if choice=='nugget':
        selectedlist=NuggetList
    elif choice == 'argument':
        selectedlist=ArgumentList
    elif choice == 'role':
        selectedlist=RoleList

    for x in selectedlist:
        if x[2:] not in right and x!='O':
            right[x[2:]]=0
            keys[x[2:]]=0
            predicted[x[2:]]=0
    right['O']=0
    keys['O']=0
    predicted['O']=0
    oldpredicted=''
    for i in range(len(lines)):
        if len(lines[i])>3:
            
            words=lines[i].split('\t')[1:]
            if words[1].startswith('B-') and words[1] in selectedlist:
                keys[words[1][2:]]+=1 #for recall
                if words[1][2:]==words[2][2:]:  # B-X B-X
                    right[words[1][2:]]+=1                    
            elif words[1].startswith('I-') and words[1] in selectedlist and words[2][2:]!=oldpredicted:
                if words[1][2:]==words[2][2:]: # I-X I-X or I-X B-X
                    right[words[1][2:]]+=1
                    
            elif words[1]=='O' and words[2].startswith('B-') and words[2] in selectedlist: # O B-Y
                keys['O']+=1
            elif words[1]=='O' and words[2]=='O':
                right['O']+=1
                keys['O']+=1

            #for precision
            if words[2].startswith('B-') and words[2] in selectedlist:
                predicted[words[2][2:]]+=1 
            elif words[2]=='O':
                predicted['O']+=1
            elif words[2].startswith('I-') and words[2] in selectedlist and words[2][2:]!=oldpredicted:
                predicted[words[2][2:]]+=1 
            oldpredicted=words[2][2:]

    tr,tk,tp=0,0,0
    for x in right.keys():
        if x=='O':
            continue
        tr+=right[x]
        tp+=predicted[x]
        tk+=keys[x]
        precision= float(right[x])/float(predicted[x])
        recall= float(right[x])/float(keys[x])
        f1=2*precision*recall/(precision+recall)
    miprec=float(tr)/float(tp)
    mirec=float(tr)/float(tk)


def intersect(str1,str2):
    """ count # of intersection between gold and predict in words """
    words1=str1.split()
    words2=str2.split()
    empty=''

    # find substring of words1 in words2, remove if it is found
    if len(words1)>=len(words2):
        oldlen=len(words2)

        for x in words1:            
            for y in words2:
                if x==y:
                    empty=y                 
                    break

            if empty!='': words2.remove(empty)
            empty=''

        return oldlen-len(words2)
    elif len(words1)<len(words2):
        oldlen=len(words1)

        for x in words2:            
            for y in words1:
                if x==y:
                    empty=y                 
                    break

            if empty!='': words1.remove(empty)
            empty=''

        return oldlen-len(words1)    

def span(gold,predicted,selectedlist):
   #for span w/o mention type

    nogold,nopredict,dice=0.0,0.0,0.0

    for fname in gold.keys():
        for sentno in gold[fname].keys():
            for x in  gold[fname][sentno]:
                nogold+=1
            for y in predicted[fname][sentno]:
                nopredict+=1

    for fname in gold.keys():
        for sentno in gold[fname].keys():
            for x in gold[fname][sentno]:                
                goldtokens=x['trigger']
                if sentno in predicted[fname]:
                    maxred=-1
                    redidx=0
                    idx=0
                    for y in predicted[fname][sentno]:
                        red=intersect(y['trigger'],goldtokens)
                        #print red, y['trigger'],goldtokens
                        if red>maxred:
                            maxred=red
                            redidx=idx
                        idx+=1
                    if maxred>0:
                        x['intersect']=maxred
                        x['predlen']=len(predicted[fname][sentno][redidx]['trigger'].split())
                        del predicted[fname][sentno][redidx]

    for fname in gold.keys():
        for sentno in gold[fname].keys():
            for x in gold[fname][sentno]:
                tp=x['intersect']
                tg=len(x['trigger'].split())
                ts=x['predlen']
                dice+=float(2*tp)/float(tg+ts)
              


    p,r,f=0.0,0.0,0.0
    p=float(dice/nopredict)
    r=float(dice/nogold)
    f=float(2*p*r)/(p+r)


def mention(gold,predicted,selectedlist,O):
    """ compute Precision, Recall, F1 for each class followed evaluation metrics; 'type' from TAC Event Nugget Detection Task """
    nopredict={}
    nogold={}
    mdice={}

    # initialize raw score
    for x in selectedlist:
        if x[2:] not in nogold and x!='O':
            nogold[x[2:]]=0
            nopredict[x[2:]]=0
            mdice[x[2:]]=0
            
    
    nogold['O']=0
    nopredict['O']=0
    mdice['O']=0

    # count number of gold and predict label
    for fname in gold.keys():
        for sentno in gold[fname].keys():
            for x in  gold[fname][sentno]:
                nogold[x['eventtype']]+=1
            for y in predicted[fname][sentno]:
                nopredict[y['eventtype']]+=1
        
    
    for fname in gold.keys():
        for sentno in gold[fname].keys():
            # clear correctly predict O 
            for x in gold[fname][sentno]:
                if x['eventtype']=='O':
                    idx=0
                    found=False
                    for y in predicted[fname][sentno]:
                        if y['trigger']==x['trigger'] and y['eventtype']=='O':
                            x['intersect']=1
                            x['predlen']=1
                            found=True
                            break
                        idx+=1
                    if found:
                        del predicted[fname][sentno][idx]

                if x['eventtype']!='O':
                    goldtokens=x['trigger']
                    if sentno in predicted[fname]:
                        maxred=-1
                        redidx=0
                        idx=0
                        for y in predicted[fname][sentno]:
                         
                            if y['eventtype']==x['eventtype']:
                                red=intersect(y['trigger'],goldtokens)
                            
                                if red>maxred:
                                    maxred=red
                                    redidx=idx
                        
                            idx+=1
                        if maxred>0:
                            x['intersect']=maxred
                            x['predlen']=len(predicted[fname][sentno][redidx]['trigger'].split())
                            del predicted[fname][sentno][redidx]

    
    for fname in gold.keys():
        for sentno in gold[fname].keys():
            for x in gold[fname][sentno]:

                tp=x['intersect']
                tg=len(x['trigger'].split())
                ts=x['predlen']
                mdice[x['eventtype']]+=float(2*tp)/float(tg+ts)
                  
    tdice,tnogold,tnopred=0.0,0.0,0.0
    
    for c in sorted(mdice.keys()):
        p,r,f=0.0,0.0,0.0
        if c=='O':
            if not O:
                continue
        tdice+=mdice[c]
        tnogold+=nogold[c]
        tnopred+=nopredict[c]
        if nopredict[c]>0:
            p=float(mdice[c]/nopredict[c])*100
        if nogold[c]>0:  
            r=float(mdice[c]/nogold[c])*100
        if p>0 or r>0:
            f=float(2*p*r)/(p+r)
        print ('{0:20s}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}'.format(c,mdice[c],nogold[c],nopredict[c],p,r,f))
    mp=tdice/tnopred*100
    mr=tdice/tnogold*100
    mf=2*mp*mr/(mp+mr)
    mavg='micro avg'
    print ('{0:20s}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}'.format('micro avg',tdice,tnogold,tnopred,mp,mr,mf))


def confusion_label(gold,predicted,selectedlist,raw):
    """ compute Confusion matrix """
    nogold={}
    nopredict={}
    cfm={} 
    # initialize raw score
    
    for x in selectedlist: 
        event=x if len(x[2:])==0 else x[2:]
        nogold[event]=0
        nopredict[event]=0
        cfm[event]={}
        for y in selectedlist:
            event2=y if len(y[2:])==0 else y[2:]
            cfm[event][event2]=0
               

    # count 
    for fname in raw.keys():
        for sentno in raw[fname].keys():
            for wordno in raw[fname][sentno].keys():
                if raw[fname][sentno][wordno]['gold'].startswith('B-'):
                    if raw[fname][sentno][wordno]['pred']!='O':
                        cfm[raw[fname][sentno][wordno]['gold'][2:]][raw[fname][sentno][wordno]['pred'][2:]]+=1
                    else:
                        cfm[raw[fname][sentno][wordno]['gold'][2:]][raw[fname][sentno][wordno]['pred']]+=1
                elif raw[fname][sentno][wordno]['gold']=='O':
                    if raw[fname][sentno][wordno]['pred']!='O':
                        cfm[raw[fname][sentno][wordno]['gold']][raw[fname][sentno][wordno]['pred'][2:]]+=1
                    else:
                        cfm[raw[fname][sentno][wordno]['gold']][raw[fname][sentno][wordno]['pred']]+=1

    print ('\t',end=" ")
    for x in cfm.keys():        
        print (x,end="\t")
    print ('total')
    for x in cfm.keys():  
        print (x,end="\t")      
        total=0
        for y in cfm[x].keys():     
            total+=cfm[x][y]
            print (cfm[x][y],end="\t")
        print (total)



def confusion_token(gold,predicted,selectedlist,raw):
    """ compute Confusion matrix """
    nogold={}
    nopredict={}
    cfm={} 
    # initialize raw score
    
    for x in selectedlist: 
        event=x if len(x[2:])==0 else x[2:]
        nogold[event]=0
        nopredict[event]=0
        cfm[event]={}
        for y in selectedlist:
            event2=y if len(y[2:])==0 else y[2:]
            cfm[event][event2]=0
               

    # count 
    for fname in raw.keys():
        for sentno in raw[fname].keys():
            for wordno in raw[fname][sentno].keys():
                if raw[fname][sentno][wordno]['pred']!='O':
                    if raw[fname][sentno][wordno]['gold']=='O':
                        cfm[raw[fname][sentno][wordno]['gold']][raw[fname][sentno][wordno]['pred'][2:]]+=1
                    else:
                        cfm[raw[fname][sentno][wordno]['gold'][2:]][raw[fname][sentno][wordno]['pred'][2:]]+=1
                else:
                    if raw[fname][sentno][wordno]['gold']=='O':
                        cfm[raw[fname][sentno][wordno]['gold']][raw[fname][sentno][wordno]['pred']]+=1
                    else:
                        cfm[raw[fname][sentno][wordno]['gold'][2:]][raw[fname][sentno][wordno]['pred']]+=1

    print ('\t',end=" ")    
    for x in cfm.keys():        
        print (x,end="\t")
    print ('total')
    for x in cfm.keys():  
        print (x,end="\t")      
        total=0
        for y in cfm[x].keys():     
            total+=cfm[x][y]
            print (cfm[x][y],end="\t")
        print (total)

def confusion_1(gold,predicted,selectedlist,raw):
    """ compute Confusion matrix """
    nogold={}
    nopredict={}
    cfm={} 
    # initialize raw score
    
    for x in selectedlist: 
        event=x if len(x[2:])==0 else x[2:]
        nogold[event]=0
        nopredict[event]=0
        cfm[event]={}
        for y in selectedlist:
            event2=y if len(y[2:])==0 else y[2:]
            cfm[event][event2]=0
               
    # count number of gold and predict label
    for fname in gold.keys():
        for sentno in gold[fname].keys():
            for x in  gold[fname][sentno]:
                nogold[x['eventtype']]+=1
            for y in predicted[fname][sentno]:
                nopredict[y['eventtype']]+=1
        
    # count correctly predict    
    for fname in gold.keys():
        for sentno in gold[fname].keys():
            for x in gold[fname][sentno]:
                eventtype=x['eventtype']
                goldtokens=x['trigger']
                if sentno in predicted[fname]:
                    maxred=-1
                    redidx=0
                    idx=0
                    for y in predicted[fname][sentno]:
                        
                        if y['eventtype']==eventtype:
                            red=intersect(y['trigger'],goldtokens)
                            
                            if red>maxred:
                                maxred=red
                                redidx=idx                                                    
                        idx+=1
                    if maxred>0:
                        x['intersect']=maxred
                        x['predlen']=len(predicted[fname][sentno][redidx]['trigger'].split())
                        wordlist=predicted[fname][sentno][redidx]['trigger'].split()
                        for wordno in raw[fname][sentno].keys():
                            if raw[fname][sentno][wordno]['text'] in wordlist and raw[fname][sentno][wordno]['pred'][2:]==eventtype:
                                raw[fname][sentno][wordno]['remove']=True
                        cfm[eventtype][eventtype]+=1
                        del predicted[fname][sentno][redidx]

    # count miss
    for fname in predicted.keys():
        for sentno in predicted[fname].keys():

            for x in predicted[fname][sentno]:
                #print (x)
                wordlist=x['trigger'].split()
                for wordno in raw[fname][sentno].keys():
                    if raw[fname][sentno][wordno]['text'] in wordlist and raw[fname][sentno][wordno]['pred'][2:]==x['eventtype']:
                        raw[fname][sentno][wordno]['remove']=True                               
                        
                cfm[raw[fname][sentno][wordno]['gold'][2:]][raw[fname][sentno][wordno]['pred'][2:]]+=1
            else:                
                for wordno in raw[fname][sentno].keys():
                    if 'remove' not in raw[fname][sentno][wordno].keys() and raw[fname][sentno][wordno]['gold'] != raw[fname][sentno][wordno]['pred']:
                        cfm[raw[fname][sentno][wordno]['gold'][2:]][raw[fname][sentno][wordno]['pred'][2:]]+=1

    for x in cfm.keys():        
        print (x,end="\t")
    print ()
    for x in cfm.keys():  
        print (x,end="\t")      
        for y in cfm[x].keys():     
            print (cfm[x][y],end="\t")
        print ()


def collect(lines,choice):
    """ read predict file and put in dict """
    gold={}
    predicted={}
    raw={}

    if choice=='nugget':
        selectedlist=NuggetList
    elif choice=='argument':
        selectedlist=ArgumentList
    elif choice=='role':
        selectedlist=RoleList


    sentno=0
    tp,ep,tg,eg='','','',''
    oldpredicted=''
    fname=''
    wordno=0
    for i in range(len(lines)):
        words=lines[i].split('\t')

        # sentence ends
        if len(words)<=0:
            if eg:                    
                tmp={}
                tmp['eventtype']=eg
                tmp['trigger']=tg.strip()
                tmp['intersect']=0
                tmp['predlen']=0
                gold[fname][sentno].append(tmp) 
            
                eg,tg='',''
            if ep:                    
                tmp={}
                tmp['eventtype']=ep
                tmp['trigger']=tp.strip()
                predicted[fname][sentno].append(tmp) 
                
                ep,tp='',''         
            sentno+=1
            wordno=0
        # in sentence
        elif len(words)>3:
            
            if words[0]!=fname:
                fname=words[0]
            if fname not in gold:
                gold[fname]={}
                sentno=0
                predicted[fname]={}
                raw[fname]={}
            if sentno not in gold[words[0]]:
                gold[fname][sentno]=[]
                predicted[fname][sentno]=[]
                raw[fname][sentno]={}
                wordno=0
            
            # start a new label of gold annotation
            if words[3].startswith('B-') and words[3] in selectedlist:
                if eg:                    
                    tmp={}
                    tmp['eventtype']=eg
                    tmp['trigger']=tg.strip()
                    tmp['intersect']=0
                    tmp['predlen']=0
                    gold[fname][sentno].append(tmp) 
                    
                tg=words[1]+' '
                eg=words[3][2:]
            elif words[3].startswith('I-') and words[3] in selectedlist:
                tg+=words[1]+' '
            
            elif words[3]=='O':
                
                if eg:
                    tmp={}
                    tmp['eventtype']=eg
                    tmp['trigger']=tg.strip()
                    tmp['intersect']=0
                    tmp['predlen']=0
                    gold[fname][sentno].append(tmp) 

                tg=words[1]+' '
                eg=words[3]
            
            # start a new label of prediction
            if words[4].startswith('B-') and words[4] in selectedlist:
                if ep:                    
                    tmp={}
                    tmp['eventtype']=ep
                    tmp['trigger']=tp.strip()
                    predicted[fname][sentno].append(tmp) 
                    
                tp=words[1]+' '
                ep=words[4][2:]
                oldpredicted=words[4][2:]

            # start a new label of prediction, but the label starts with 'I'
            elif words[4].startswith('I-') and words[4] in selectedlist and words[4][2:]!=oldpredicted:
                if ep:                    
                    tmp={}
                    tmp['eventtype']=ep
                    tmp['trigger']=tp.strip()
                    predicted[fname][sentno].append(tmp) 
                    
                tp=words[1]+' '
                ep=words[4][2:]
                oldpredicted=words[4][2:]

            # still be the same label
            elif words[4].startswith('I-') and words[4] in selectedlist and words[4][2:]==oldpredicted:
                tp+=words[1]+' '

            elif words[4]=='O':
                
                if ep:                    
                    tmp={}
                    tmp['eventtype']=ep
                    tmp['trigger']=tp.strip()
                    predicted[fname][sentno].append(tmp) 
                    
                tp=words[1]+' '
                ep=words[4]
                 
                oldpredicted=words[4]

            # keep raw predict data into dict
            word={}
            word['text']=words[1]
            word['gold']=words[3]
            word['pred']=words[4]
            raw[fname][sentno][wordno]=word
            wordno+=1

    return gold,predicted,selectedlist,raw

def confusion_role(lines):
    gold={}
    for line in lines:
        if len(line)>3:
            words=line.split('\t')
            if words[3] not in gold.keys():
                gold[words[3]]={}
            if words[4] not in gold[words[3]].keys():
                gold[words[3]][words[4]]=0
            gold[words[3]][words[4]]+=1
    print (gold)


parser = argparse.ArgumentParser()
parser.add_argument("-predictedfile", default=None, help="predicted file output",required=True)
parser.add_argument("-options", default=None, choices=['nugget','argument','role'],help="specify one choice of [trigger, argument, role]",required=True)
parser.add_argument("-metric", default=None, choices=['f1','confusion_token','confusion_label','confusion_role'],required=True)
parser.add_argument("-O",default=False, action="store_true")
 
def main():
    args=parser.parse_args()

    content=utils.readFileEncode(args.predictedfile,'utf8')
    lines=content.split('\n')[:-1]

    if args.metric=='f1':
        gold, predicted, selectedlist, raw = collect(lines, args.options)
        mention(gold,predicted,selectedlist,args.O)
    elif args.metric=='confusion_role':
        confusion_role(lines)
    elif args.metric=='confusion_token':
        gold, predicted, selectedlist, raw = collectlabels(lines, args.options)
        confusion_token(gold,predicted,selectedlist,raw)    
    elif args.metric=='confusion_label':
        confusion_label(gold,predicted,selectedlist,raw)    
    print ('=========================================')

if __name__ == "__main__":
    main()
