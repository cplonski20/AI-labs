"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tagdic = {}
    wordict = {}


    for i in train:
        for b in i:
            if b[0] not in wordict:
                wordict[b[0]] = {}
                wordict[b[0]][b[1]] = 1
            else:
                wordict[b[0]][b[1]] = wordict[b[0]].get(b[1], 0) + 1
            tagdic[b[1]] = tagdic.get(b[1],0) + 1

    maxTag = max(tagdic, key=tagdic.get)

    toReturn = []
    for i in test:
        toPush  = []
        for b in range(len(i)):
            if i[b] in wordict:
                maxWord = max(wordict[i[b]], key=wordict[i[b]].get)
                toPush.append((i[b], maxWord))
            else:
                toPush.append((i[b], maxTag))
        toReturn.append(toPush)
        
    return toReturn