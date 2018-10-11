# python3.5 run-tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
from io import StringIO

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.

    # Get the models
    with open(model_file, 'r') as mf:
        numOfPos = int(mf.readline())
        numOfWords = int(mf.readline())

        poss = mf.readline().rstrip().split(" ")
        words = mf.readline().rstrip().split(" ")

        if len(poss) != numOfPos:
            raise ValueError
        
        if len(words) != numOfWords:
            raise ValueError
        
        posTagTransitionMatrix = np.loadtxt((mf.readline() for i in range(0, numOfPos)), delimiter=" ", dtype='float')
        wordEmissionMatrix = np.loadtxt((mf.readline() for i in range(0, numOfPos)), delimiter=" ", dtype="float")
        
    # # Comment out if not debugging. Prints out the matrices for debugging purposes
    # with open('posTagTransitionMatrixRebuilt', 'w') as posTagTransitionMatrixFile:
    #     np.savetxt(posTagTransitionMatrixFile, posTagTransitionMatrix, fmt='%.2f')
    # with open('wordEmissionMatrixRebuilt', 'w') as wordEmissionMatrixFile:
    #     np.savetxt(wordEmissionMatrixFile, wordEmissionMatrix, fmt='%.2f')
    
    with open(test_file, 'r') as tf:
        with open(out_file, 'w') as of:
            for line in tf:
                of.write(tag_line(line, poss, words, posTagTransitionMatrix, wordEmissionMatrix))
    print('Finished...')
    return

def tag_line(line, poss, words, posTagTransitionMatrix, wordEmissionMatrix):
    tokens = line.rstrip().split(" ")

    viterbiMatrix = np.zeros((len(poss), len(tokens)), dtype='float')
    retraceMatrix = np.zeros((len(poss), len(tokens)), dtype='int') - 1

    # Set up the viterbiMatrix
    viterbiMatrix[:,0] =  np.multiply(wordEmissionMatrix[:, words.index(tokens[0] if tokens[0] in words else '<UNK>')], posTagTransitionMatrix[poss.index('<s>'), :])

    for i in range(1, len(tokens)):
        word_index = words.index(tokens[i] if tokens[i] in words else '<UNK>')
        for j in range(0, len(poss)):
            possibleValuesGivenPrevPos = np.multiply(viterbiMatrix[:,i-1], posTagTransitionMatrix[:,j])
            maximizingPrevPos = np.argmax(possibleValuesGivenPrevPos)

            retraceMatrix[j,i] = maximizingPrevPos
            viterbiMatrix[j,i] = possibleValuesGivenPrevPos[maximizingPrevPos] * wordEmissionMatrix[j, word_index]
    

    result = "\n"
    tagIndex = np.argmax(np.multiply(viterbiMatrix[:,-1], posTagTransitionMatrix[:, poss.index('</s>')]))
    result = tokens[-1] + '/' + poss[tagIndex] + result

    for i in range(0, len(tokens)-1):
        tagIndex = retraceMatrix[tagIndex, len(tokens)-2-i+1]
        result = tokens[len(tokens)-2-i] + '/' + poss[tagIndex] + ' ' + result

    return result

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
