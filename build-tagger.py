# python3.5 build-tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    prevTagToTagCounts = {}
    wordToPrevTagCounts = {}
    with open(train_file, "r") as tf:
        for line in tf:
            if len(line) != 0:
                processLine(line, prevTagToTagCounts, wordToPrevTagCounts)

    wordToPrevTagCounts['<UNK>'] = {}

    createAndSaveModels(model_file, prevTagToTagCounts, wordToPrevTagCounts)
    print('Finished...')

def createPosTagTransitionModel(poss, prevTagToTagCounts):
    
    numOfPos = len(poss)
    prevTagToTagCountsMatrix = np.zeros((numOfPos, numOfPos), dtype='float')

    # Transfer counts to matrix
    for i in range(0, numOfPos):
        for j in range(0, numOfPos):
            if poss[j] in prevTagToTagCounts[poss[i]]:
                prevTagToTagCountsMatrix[i][j] += prevTagToTagCounts[poss[i]][poss[j]]

    return computeProb(prevTagToTagCountsMatrix, kneserNey)

def createWordEmissionModel(words, poss, wordToPrevTagCounts):

    numOfPos = len(poss)
    numOfWords = len(words)
    wordToPrevTagCountsMatrix = np.zeros((numOfPos, numOfWords), dtype='float')
    # Transfer counts to matrix
    for i in range(0, numOfWords):
        for j in range(0, numOfPos):
            if poss[j] in wordToPrevTagCounts[words[i]]:
                wordToPrevTagCountsMatrix[j][i] += wordToPrevTagCounts[words[i]][poss[j]]

    return computeProb(wordToPrevTagCountsMatrix, kneserNey)

def laplace(mat):
    # With Laplace Smoothing
    mat += 1
    return mat

def computeProb(mat, fn=None):
    if fn != None:
        mat = fn(mat)
    sumMat = mat.sum(axis=1, dtype='float').reshape(len(mat), 1)
    return np.nan_to_num(np.divide(mat, sumMat))

def kneserNey(mat):
    # With Kneser Ney Smoothing
    discount = 0.75
    existenceMat = mat.copy()
    existenceMat[existenceMat>0] = 1
    invertedExistenceMat = np.ones(mat.shape, dtype='float') - existenceMat

    numOfDistinctPrecedingEvents = existenceMat.sum(axis=1, dtype='float').reshape(mat.shape[0], 1)
    numOfDistinctEvents = existenceMat.sum(axis=0, dtype='float').reshape(1, mat.shape[1])

    # This will account for <UNK> and <s> this is wrong though P(<s>|X) = 0 since <s> cannot be after any word. But well it works
    numOfDistinctEvents[numOfDistinctEvents==0] = 1

    continuationMat = np.multiply(invertedExistenceMat, numOfDistinctEvents)
    continuationMat = np.multiply(numOfDistinctPrecedingEvents*discount, continuationMat)

    continuationMat = np.divide(continuationMat, continuationMat.sum(axis=1, dtype='float').reshape(continuationMat.shape[0], 1))

    mat[mat>0] -= discount
    mat += continuationMat

    return mat

def kneserNeyInterpolated(mat):
    # With Kneser Ney Smoothing with interpolation
    discount = 0.75

    existenceMat = mat.copy()
    existenceMat[existenceMat>0] = 1

    numOfDistinctPrecedingEvents = existenceMat.sum(axis=1, dtype='float').reshape(mat.shape[0], 1)
    numOfDistinctEvents = existenceMat.sum(axis=0, dtype='float').reshape(1, mat.shape[1])

    # This will account for <UNK> and <s> this is wrong though P(<s>|X) = 0 since <s> cannot be after any word. But well it works
    numOfDistinctEvents[numOfDistinctEvents==0] = 1

    continuationMat = np.multiply(np.ones(mat.shape, dtype='float'), numOfDistinctEvents)
    continuationMat = np.multiply(numOfDistinctPrecedingEvents*discount, continuationMat)

    continuationMat = np.divide(continuationMat, continuationMat.sum(axis=1, dtype='float').reshape(continuationMat.shape[0], 1))

    mat[mat>0] -= discount
    mat += continuationMat

    return mat


def processLine(line, prevTagToTagCounts, wordToPrevTagCounts):
    arr = line.rstrip().split(" ")
    prevPos = "<s>"

    for i in range(0,len(arr)):
        wordAndPosDelimited = (str)(arr[i])
        wordAndPos = wordAndPosDelimited.rsplit("/", 1)
        word = wordAndPos[0]
        pos = wordAndPos[1]
        addToTagCounts(pos, prevPos, prevTagToTagCounts)
        addToWordCounts(word, pos, wordToPrevTagCounts)
        prevPos = pos
    
    addToTagCounts('</s>', prevPos, prevTagToTagCounts)
    return

def createAndSaveModels(model_file, prevTagToTagCounts, wordToPrevTagCounts):
    poss = list(prevTagToTagCounts.keys())
    poss.sort()
    words = list(wordToPrevTagCounts.keys())
    words.sort()

    posTagTransitionMatrix = createPosTagTransitionModel(poss, prevTagToTagCounts)
    wordEmissionMatrix = createWordEmissionModel(words, poss, wordToPrevTagCounts)

    saveModels(poss, posTagTransitionMatrix, words, wordEmissionMatrix, model_file)
    return

def saveModels(poss, posTagTransitionMatrix, words, wordEmissionMatrix, model_file):
    with open(model_file, 'w') as mf:
        mf.write('%s\n' % str(len(poss)))
        mf.write('%s\n' % str(len(words)))
        for pos in poss:
            mf.write("%s " % pos)
        mf.write("\n")
        for word in words:
            mf.write("%s " % word)
        mf.write("\n")

        np.savetxt(mf, posTagTransitionMatrix)
        np.savetxt(mf, wordEmissionMatrix)

    # # Comment out if not debugging. Prints out the matrices for debugging purposes
    # with open('posTagTransitionMatrix', 'w') as posTagTransitionMatrixFile:
    #     np.savetxt(posTagTransitionMatrixFile, posTagTransitionMatrix)
    # with open('wordEmissionMatrix', 'w') as wordEmissionMatrixFile:
    #     np.savetxt(wordEmissionMatrixFile, wordEmissionMatrix)
    return

def addToTagCounts(pos, prevPos, prevTagToTagCounts):
    if prevPos not in prevTagToTagCounts:
        prevTagToTagCounts[prevPos] = {}
    if pos not in prevTagToTagCounts:
        prevTagToTagCounts[pos] = {}
    if pos not in prevTagToTagCounts[prevPos]:
        prevTagToTagCounts[prevPos][pos] = 0

    prevTagToTagCounts[prevPos][pos] += 1

    return

def addToWordCounts(word, pos, wordToPrevTagCounts):
    if word not in wordToPrevTagCounts:
        wordToPrevTagCounts[word] = {}
    if pos not in wordToPrevTagCounts[word]:
        wordToPrevTagCounts[word][pos] = 0

    wordToPrevTagCounts[word][pos] += 1

    return

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
