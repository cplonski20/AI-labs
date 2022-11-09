"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
# from typing import KeysView
import numpy as np
import math
def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    alphatag = .00001
    alphawords = .00001
    tagdic = {}
    tagwordict = {}
    # wordict = {}
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
            # if b[0] not in wordict:
            #     wordict[b[0]] = {}
            #     wordict[b[0]][b[1]] = 1
            # else:
            #     wordict[b[0]][b[1]] = wordict[b[0]].get(b[1], 0) + 1

            #getting dic of tags MAY REMOVE LATER IF NOT USED
            tagdic[b[1]] = tagdic.get(b[1],0) + 1
            
            # getting dict of tags with words
            if b[1] not in tagwordict:
                tagwordict[b[1]] = {}
                tagwordict[b[1]][b[0]] = 1
            else:
                tagwordict[b[1]][b[0]] = tagwordict[b[1]].get(b[0], 0) + 1
    
    # turn tagpairs into log probabilities
    for key in tagpairs.keys():
        n = sum(tagpairs[key].values())
        v = len(tagpairs[key])
        for key2 in tagpairs[key].keys():
            count = tagpairs[key][key2]
            tagpairs[key][key2] =  math.log((alphatag + count)/(n + alphatag*(v + 1)))     #math.log(tagpairs[key][key2] / pairtot)
        tagpairs[key]['UNKNOWN'] = math.log(alphatag/(n + alphatag*(v + 1)))

    for key in tagwordict.keys():
        n = sum(tagwordict[key].values())
        v = len(tagwordict[key])
        for key2 in tagwordict[key].keys():
            count = tagwordict[key][key2]
            tagwordict[key][key2] =  math.log((alphawords + count)/(n + alphawords*(v + 1)))

        tagwordict[key]['UNKNOWN'] = math.log(alphawords/(n + alphawords*(v + 1)))
    
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
                    trellprob[idx][i] += tagwordict[tagmap[i]][word] if word in tagwordict[tagmap[i]] else tagwordict[tagmap[i]]['UNKNOWN']
            else:
                for i in range(numTags):
                    toMax = np.zeros(numTags)
                    for b in range(numTags): 
                        if(tagmap[b] == 'END'):
                             toMax[b] = -100000
                             continue
                        emission = tagwordict[tagmap[i]][word] if word in tagwordict[tagmap[i]] else tagwordict[tagmap[i]]['UNKNOWN']
                        transfer = tagpairs[tagmap[b]][tagmap[i]] if tagmap[i] in tagpairs[tagmap[b]] else tagpairs[tagmap[b]]['UNKNOWN']   #SUS LINE?
                        previous = trellprob[idx - 1][b]
                        toMax[b] = previous + transfer + emission
                    trellback[idx][i] = np.argmax(toMax)
                    trellprob[idx][i] = np.amax(toMax)
        
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