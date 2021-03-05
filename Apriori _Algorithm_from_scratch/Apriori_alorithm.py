#!/usr/bin/env python
# coding: utf-8

# In[1]:


datalist = []
def takeInput():
    print()
    fileNum = input("Please choose an input file from 1 to 5: ")
    inputFile = "Datam"+ str(fileNum) +".csv"
    print("Please input the minimum support and confidence values.")
    minSup = int(input("Minimum Support: "))
    minConf = int(input("Minimum Confidence: "))
    file = open(inputFile, "r")
    fLine = (file.read()).split("\n")
    for l in fLine:
        datalist.append(sorted(l.split(", "))) 
    return minSup, minConf


# In[2]:


def containInList(list1,list2):
    #if list 2 is in list 1
    result = True
    for item in list2:
        #print(item)
        if item not in list1: 
            result =  False
    return result


# In[3]:


def calSuport(datalist,items):
    support = 0
    for i in range(0,len(datalist)):
        row = datalist[i]
        #print(row)
        #print(items)
        if containInList(row,items):
            support = support + 1
    support=support*100/len(datalist)        
    return support


# In[4]:


def combinationR(List, r):
    if r == 0: 
        return [[]]     
    combination = [] 
    for i in range(0, len(List)):         
        primary = List[i]
        secondaries = List[(i+1):]
        for j in combinationR(secondaries, r-1):
            combination.append([primary]+j)
    return combination 


# In[5]:


def supSingleItem(c):
    itemset = {-1}
    for i in range(len(c)):
        for j in range(len(c[i])):
            itemset.add(c[i][j])
    itemset.remove(-1)
    #print(itemset)
    return itemset


# In[6]:


def Apriori(D,minSup):
    Lout = []
    i = 0
    L = D
    condition = True
    while condition:
        i = i + 1
        sinIt = list(supSingleItem(L))
        Lnew = []
        C = combinationR(sinIt, i)
        #print(C3)
        for item in C:
            supp = calSuport(D,item)
            #print(item," : ",supp)
            if supp >= minSup:
                Lnew.append(item)
                print("Support({}------->{})".format(item,supp))
        L = Lnew
        Lout.append(Lnew)
        condition = len(Lnew) > 1
        
    return Lout


# In[7]:


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


# In[34]:


def get_association_rules_apriori():
    minSup, minConf = takeInput()
    x=Apriori(datalist,minSup)
    for i in range(1,len(x)):
         for item in x[i]:
                pSet = powerset(item)
                for p in pSet:
                    left_side = set(p)
                    lhs=list(left_side)
                    right_side = set(item) - left_side
                    rhs=list(right_side)
                    if len(lhs) and len(rhs):
                        calSuport1=calSuport(datalist,lhs)
                        if calSuport1 == 0:
                            continue
                        confidence = calSuport(datalist,item) * 100 / calSuport1
                        if confidence < minConf:
                            continue
                        print("Confidence({} ---------> {} :  {})".format(lhs, rhs, confidence))


# In[9]:


def Brute_force(D,minSup):
    Lout = []
    i = 0
    L = D
    sinIt = list(supSingleItem(L))
    condition = True
    while condition:
        i = i + 1
        Lnew = []
        C = combinationR(sinIt, i)
        #print(C3)
        for item in C:
            supp = calSuport(D,item)
            #print(item," : ",supp)
            if supp >= minSup:
                Lnew.append(item)
                print("Support({}------->{})".format(item,supp))
        L = Lnew
        Lout.append(Lnew)
        condition = len(Lnew) > 1
        
    return Lout


# In[10]:


def get_association_rules_bruteforce():
    minSup, minConf = takeInput()
    x=Brute_force(datalist,minSup)
    for i in range(1,len(x)):
         for item in x[i]:
                pSet = powerset(item)
                for p in pSet:
                    left_side = set(p)
                    lhs=list(left_side)
                    right_side = set(item) - left_side
                    rhs=list(right_side)
                    if len(lhs) and len(rhs):
                        calSuport1=calSuport(datalist,lhs)
                        if calSuport1 == 0:
                            continue
                        confidence = calSuport(datalist,item) * 100 / calSuport1
                        if confidence < minConf:
                            continue
                        print("Confidence({} ---------> {} :  {})".format(lhs, rhs, confidence))


# In[33]:


import time
start = time.time()
get_association_rules_apriori()
end = time.time()
print("Time required using apriori method {}".format(end - start))


# In[32]:


import time
start = time.time()
get_association_rules_bruteforce()
end = time.time()
print("Time required using brute force method {}".format(end - start))


# In[ ]:




