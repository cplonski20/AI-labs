"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""
import math
from re import S
import numpy as np

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    alphaw = 1e-5
    alphatag = 1e-5
    tagdic = {}
    tagwordtagdic = {}
    wordtagdic = {}
    tagpairs = {}

    #getting tagpair counts
    for i in train:
        for b in range(1,len(i)):
            if(i[b-1][1] in tagpairs):
                tagpairs[i[b - 1][1]][i[b][1]] = tagpairs[i[b-1][1]].get(i[b][1], 0) + 1
            else:
                tagpairs[i[b - 1][1]] = {}
                tagpairs[i[b - 1][1]][i[b][1]] = 1
    for i in train:
        for b in i:
            # getting dict of words with tags
            if b[0] not in wordtagdic:
                wordtagdic[b[0]] = {}
                wordtagdic[b[0]][b[1]] = 1
            else:
                wordtagdic[b[0]][b[1]] = wordtagdic[b[0]].get(b[1], 0) + 1
            
            #getting dic of tags MAY REMOVE LATER IF NOT USED
            tagdic[b[1]] = tagdic.get(b[1],0) + 1

            # getting dict of tags with words
            if b[1] not in tagwordtagdic:
                tagwordtagdic[b[1]] = {}
                tagwordtagdic[b[1]][b[0]] = 1
            else:
                tagwordtagdic[b[1]][b[0]] = tagwordtagdic[b[1]].get(b[0], 0) + 1
    
    
    #word tag dict for hapax
    hapaxList = []
    alphawords = {}
    for word in wordtagdic.keys():
        for tag in wordtagdic[word].keys():
            if sum(wordtagdic[word].values()) == 1:
                alphawords[tag] = alphawords.get(tag, 0) + 1
                hapaxList.append((word,tag))
    temperinosum = sum(alphawords.values())
    for key in alphawords.keys():
        alphawords[key] = alphawords[key]/temperinosum

    # if contains'-' NOUN, if cotain num declare NUM, ing, ed, 's, ly, tion, sion
    Dash = {}

    Num = {}

    ing = {}

    ed = {}

    apos = {}

    ly = {}
    
    tion = {}

    sion = {}

    ate = {}
    
    fy = {}

    est = {}
    #"er","ment","age","al","ence","ism","ive","ous","able","ful","ery","less","ing","ly","'s","ed", and one prefix "$"
    sList = {}

    en = {}
    
    logy = {}

    ism = {}
    
    dollar= {}
    for a in hapaxList:
        if '-' in a[0]:
            Dash[a[1]] = Dash.get(a[1], 0) + 1
        if any(char.isdigit() for char in a[0]):
            Num[a[1]] = Num.get(a[1], 0) + 1
        if a[0].endswith('ing'):
            ing[a[1]] = ing.get(a[1], 0) + 1
        if a[0].endswith("ed"):
            ed[a[1]] = ed.get(a[1], 0) + 1
        if a[0].endswith("'s"):
            apos[a[1]] = apos.get(a[1], 0) + 1
        if a[0].endswith('ly'):
            ly[a[1]] = ly.get(a[1], 0) + 1
        if a[0].endswith('tion'):
            tion[a[1]] = tion.get(a[1], 0) + 1
        if a[0].endswith('sion'):
            sion[a[1]] = sion.get(a[1], 0) + 1
        if a[0].endswith('ate'): # ar
            ate[a[1]] = ate.get(a[1], 0) + 1
        if a[0].endswith("fy"):
            fy[a[1]] = fy.get(a[1], 0) + 1
        if a[0].endswith("est"):
            est[a[1]] = est.get(a[1], 0) + 1
        if a[0].endswith("s"):
            sList[a[1]] = sList.get(a[1], 0) + 1
        if a[0].endswith("en"):
            en[a[1]] = en.get(a[1], 0) + 1
        if a[0].endswith("logy"):
            logy[a[1]] = logy.get(a[1], 0) + 1
        if a[0].endswith("ism"):
            ism[a[1]] = ism.get(a[1], 0) + 1
        
    sumdash = sum(Dash.values())
    for key in Dash.keys():
        Dash[key] = math.log(Dash[key]/sumdash)

    sumNUM = sum(Num.values())
    for key in Num.keys():
        Num[key] = math.log(Num[key]/sumNUM)

    sumING = sum(ing.values())
    for key in ing.keys():
        ing[key] = math.log(ing[key]/sumING)

    sumED = sum(ed.values())
    for key in ed.keys():
        ed[key] = math.log(ed[key]/sumED)

    sumAPOS = sum(apos.values())
    for key in apos.keys():
        apos[key] = math.log(apos[key]/sumAPOS)

    sumLY = sum(ly.values())
    for key in ly.keys():
        ly[key] = math.log(ly[key]/sumLY)

    sumTION = sum(tion.values())
    for key in tion.keys():
        tion[key] = math.log(tion[key]/sumTION)

    sumSION = sum(sion.values())
    for key in sion.keys():
        sion[key] = math.log(sion[key]/sumSION)

    sumATE= sum(ate.values())
    for key in ate.keys():
        ate[key] = math.log(ate[key]/sumATE)

    sumFY = sum(fy.values())
    for key in fy.keys():
        fy[key] = math.log(fy[key]/sumFY)

    sumEST = sum(est.values())
    for key in est.keys():
        est[key] = math.log(est[key]/sumEST)

    sSUM= sum(sList.values())
    for key in sList.keys():
        sList[key] = math.log(sList[key]/sSUM)

    enSUM= sum(en.values())
    for key in en.keys():
        en[key] = math.log(en[key]/enSUM)

    logySUM= sum(logy.values())
    for key in logy.keys():
        logy[key] = math.log(logy[key]/logySUM)

    ismSUM= sum(ism.values())
    for key in ism.keys():
        ism[key] = math.log(ism[key]/ismSUM)

    # turn tagpairs into log probabilities
    for key in tagpairs.keys():
        n = sum(tagpairs[key].values())
        v = len(tagpairs[key])
        for key2 in tagpairs[key].keys():
            count = tagpairs[key][key2]
            tagpairs[key][key2] =  math.log((alphatag + count)/(n + alphatag*(v + 1)))     #math.log(tagpairs[key][key2] / pairtot)
        tagpairs[key]['UNKNOWN'] = math.log(alphatag/(n + alphatag*(v + 1)))

    for key in tagwordtagdic.keys():
        alphareal = alphawords[key] if key in alphawords else 1e-10
        alphareal = alphareal *alphaw
        n = sum(tagwordtagdic[key].values())
        v = len(tagwordtagdic[key])
        for key2 in tagwordtagdic[key].keys():
            count = tagwordtagdic[key][key2]
            tagwordtagdic[key][key2] =  math.log((alphareal + count)/(n + alphareal*(v + 1)))

        tagwordtagdic[key]['UNKNOWN'] = math.log(alphareal/(n + alphareal*(v + 1)))
    
    tagmap = [i for i in tagdic.keys()]
    numTags = len(tagdic)
    del tagdic

    toReturn = []
    for sent in test:

        trellprob = np.zeros((len(sent),numTags))
        trellback = np.zeros((len(sent),numTags))

        for (idx,word) in enumerate(sent):
            if idx == 0: #start edge case
                for i in range(numTags):
                    trellprob[idx][i] += tagwordtagdic[tagmap[i]][word] if word in tagwordtagdic[tagmap[i]] else tagwordtagdic[tagmap[i]]['UNKNOWN']
            else:
                for i in range(numTags):
                    toMax = np.zeros(numTags)
                    for b in range(numTags): 
                        if(tagmap[b] == 'END'):
                             toMax[b] = -100000
                             continue
                        emission = tagwordtagdic[tagmap[i]][word] if word in tagwordtagdic[tagmap[i]] else tagwordtagdic[tagmap[i]]['UNKNOWN']
                        transfer = tagpairs[tagmap[b]][tagmap[i]] if tagmap[i] in tagpairs[tagmap[b]] else tagpairs[tagmap[b]]['UNKNOWN']   #SUS LINE?
                        previous = trellprob[idx - 1][b]
                        toMax[b] = previous + transfer + emission

                    #For specific combo of tag and word 
                    # word not in wordtagdic.keys():
                    toAdd = 0
                    finalvar = math.log(1e-10)
                    if word not in tagwordtagdic[tagmap[i]]:
                        if '-' in word:
                            toAdd = Dash.get(tagmap[i], finalvar)

                        elif any(char.isdigit() for char in word):
                            toAdd = Num.get(tagmap[i], finalvar)

                        elif word.endswith('ing'):
                            toAdd = ing.get(tagmap[i], finalvar)

                        elif word.endswith("ed"):
                            toAdd = ed.get(tagmap[i], finalvar)

                        elif word.endswith("'s"):
                            toAdd = apos.get(tagmap[i], finalvar)

                        elif word.endswith('ly'):
                            toAdd = ly.get(tagmap[i], finalvar)

                        elif word.endswith('tion'):
                            toAdd = tion.get(tagmap[i], finalvar)

                        elif word.endswith('sion'):
                            toAdd = sion.get(tagmap[i], finalvar)

                        elif word.endswith('sion'):
                            toAdd = sion.get(tagmap[i], finalvar)

                        elif a[0].endswith('ate'): # ar
                            toAdd = ate.get(tagmap[i], finalvar)

                        elif a[0].endswith("fy"):
                            toAdd = fy.get(tagmap[i], finalvar)

                        elif a[0].endswith("est"):
                            toAdd = est.get(tagmap[i], finalvar)

                        elif a[0].endswith("s"):
                            toAdd = sList.get(tagmap[i], finalvar)

                    # print(toAdd)
                    trellback[idx][i] = np.argmax(toMax)
                    trellprob[idx][i] = np.amax(toMax) + toAdd
        
        reverseidx = []
        # print(trellprob)
        reverseidx.append(int(np.argmax(trellprob[len(sent) - 1][:])))
        for idx in range(len(sent) - 1, 0, -1):
            # temp = int(reverseidx[-1])
            reverseidx.append(int(trellback[idx][reverseidx[-1]]))
        
        toPush = [tagmap[reverseidx[b]] for b in range(len(reverseidx) - 1, -1, -1)]
        # toPush = toPush[::-1]
        for i in range(len(sent)):
            toPush[i] = tuple((sent[i], toPush[i]))
        
        toReturn.append(toPush)

    return toReturn