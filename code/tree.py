"""utility function for traveling in dependency relation tree and parse tree"""

def build_deptree(enhDep):
    deptree={}
    for x in enhDep:
        if x['governorGloss']=='ROOT':
            continue
        if x['governorGloss']+'_@_'+str(x['governor']) not in deptree:
            deptree[x['governorGloss']+'_@_'+str(x['governor'])]=[]
        deptree[x['governorGloss']+'_@_'+str(x['governor'])].append(x['dependentGloss']+'_@_'+str(x['dependent']))
    for x in deptree.keys():
        #print x,deptree[x]
        for y in deptree[x]:
            if y in deptree.keys() and x in deptree[y]: #loop happened
                noy=y.split('_@_') 
                nox=x.split('_@_')
                if nox[1]<noy[1]: 
                    deptree[y].remove(x)
                else:
                    deptree[x].remove(y)

    return deptree

def go_up(tree,node,visited):
    for x in tree.keys():
        if node in tree[x]:
            if x in visited:
                visited[x]+=1
            else: visited[x]=1
            return x,visited         
    return None,visited

def go_down(tree,node,path,visited):
    rem=[]
    for x in tree.keys():
        if x==node:
            if x in visited:
                visited[x]+=1
            else: visited[x]=1
            for y in tree[x]:
                if y not in path:
                    rem.append(y)
            return rem,visited
    return None,visited

def find_path(tree,start,end,path,visited):

    path.append(start)
    if start==end:
        return start

    node,visited=go_up(tree,start,visited)     
    tmp,visited=go_down(tree,start,path,visited)
    if node and tmp:
        tmp.append(node)
    elif node and not tmp:
        tmp=[node]
    if tmp:
        for x in tmp:
            if x in path:
                if x in visited:
                    visited[x]+=1
                else: visited[x]=1
                continue

            newpath=find_path(tree,x,end,path,visited)            
            if end in newpath:
                return path
            elif x not in tree.keys():                
                path.pop()
            if x in tree.keys():
                if visited[x]>=2*len(tree[x]):
                    path.pop()
    return path

def todeppath(path,enhDep):
    deppath=[]

    for y in range(len(path)-1):
        start=int(path[y].split('_@_')[1])
        end=int(path[y+1].split('_@_')[1])
        for x in enhDep:
            if x['governor']==start and x['dependent']==end:
                deppath.append(x['dep'])
                break
            elif x['governor']==end and x['dependent']==start:
                deppath.append(x['dep'])
                break
    return deppath

def build_parsetree(cfp,tokens):
    parsetree={}
    
    tokenlist=[]
    for y in tokens:
        tokenlist.append(y["originalText"]+'_@_'+str(y["index"]))
    tmp=cfp.split()
    idx=0
    for i in range(len(tmp)):          
        pure=tmp[i].replace(')','')
        for y in tokenlist:
            if y.startswith(pure) and pure:                
                if tmp[i].endswith(')'):                     
                    #print pure,tmp[i],y
                    tmp[i]=tmp[i].replace(pure,y) # tmp[i]=hacker)) y='hacker_2' pure=hacker
                else:
                    tmp[i]=y
                tokenlist.remove(y)
                break
        else:  
            tmp[i]=tmp[i]+'_@_'+str(idx)
            idx+=1

    cfp=" ".join(tmp)

    tmp=cfp.split()
    for i in range(len(tmp)):
        if ')_@_' in tmp[i]:
            tmp[i]=tmp[i].split('_@_')[0]
    cfp=" ".join(tmp)

    # build parse tree
    while '(' in cfp:
        paren=0
        leftlist=[]
        for x in range(len(cfp)):
            if cfp[x].startswith('('):
                paren+=1  
                leftlist.append(x)

            if cfp[x].endswith(')'):
                paren-=1
                left=leftlist.pop()
                oldstr=cfp[left:x+1]
                tmp=cfp[left+1:x].split()
                newstr=tmp[0]
                parsetree[tmp[0]]=[]
                for i in range(1,len(tmp)):
                    parsetree[tmp[0]].append(tmp[i])
                cfp=cfp.replace(oldstr,newstr)
                break

    return parsetree

def commonroot(parsetree,path):
    for x in path: 
        for y in parsetree.keys():
            if x in parsetree[y]: 
                if y not in path:
                    return x
                else:
                    break
        else:
            return x

def go_end(node,parsetree,chnk):
    for x in parsetree[node]:
        if x not in parsetree.keys():            
            chnk.append(x)
        elif x in parsetree.keys():
            go_end(x,parsetree,chnk)

    return chnk
    
def list_chunk(parsetree):
    nplist,vplist,depthNP,depthVP=[],[],[],[]
    for x in parsetree.keys():
        chnk=[]
        if x.startswith('NP'):
            chnk=go_end(x,parsetree,chnk)
            if len(chnk)>1:
                depthNP.append(min(len(find_path(parsetree,x,'ROOT_@_0',[],{}))-1,25))
                nplist.append(chnk)            
        elif x.startswith('VP'):
            chnk=go_end(x,parsetree,chnk)
            if len(chnk)>1:
                depthVP.append(min(len(find_path(parsetree,x,'ROOT_@_0',[],{}))-1,25))
                vplist.append(chnk)
    return nplist,vplist,depthNP,depthVP

def verbsamedept(head,verblist,deptree):
    dist=[]
    for x in verblist:
        tmp=findverb(x+1,head,deptree,1,[])
        if tmp:
            dist.append(tmp)
    if dist:
        mind=min(dist)
        return verblist[dist.index(mind)], mind
    else:
        return -1,-1

def findverb(verb,head,deptree,d,visited):
    found=False
    next=''
    for k in deptree.keys():
        idx=int(k.split('_@_')[1])
               
        if head in deptree[k] and k not in visited:
            if verb==idx:
                found=True
                return d
            else:
                next=k
                visited.append(next)
                return findverb(verb,next,deptree,d+1,visited)


def subtree(node,tree,tokenlist,outstr):
    """ gather all node in subtree with root as node """
    if node in tree.keys():
        for x in tree[node]:
            if '_@_' in x:
                nd,idx=x.split('_@_')                
                if nd in tokenlist:
                    outstr.append(x)
                elif 'S_@_' in x:
                    return outstr
                else:
                    subtree(x,tree,tokenlist,outstr)
    return outstr


def selectSubS(parsetree,tokens):
    """ find substring in sentence that starts with S node in parsetree """
    outstr=[]
    slist=[]
    tokenlist=[]
    for x in tokens:
        tokenlist.append(x['originalText'])
    for node in parsetree.keys():
        nd,idx=node.split('_@_')
        if nd=='S' and idx != '1':
            for subS in parsetree[node]:
                if 'NP' in subS:
                    break
            else:
                slist.append(node)
       
    for node in slist: #loop through every substring            
        stree=subtree(node,parsetree,tokenlist,[])


        b,i=0,0
        while (i < len(stree)):
            if stree[i].startswith(','):
                outstr.append(stree[b:i])
                b=i+1
            i+=1
        outstr.append(stree[b:i])

    
    return outstr
           



