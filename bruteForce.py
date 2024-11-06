import sys
import numpy as np
import time


def readFile(filename):
    with open(filename, 'r') as f:
        content = f.read().splitlines()
        size = len(content)
        inputArray = np.zeros((size,size))
        for i in range(size):
            values = content[i].split(',')
            for j in range(size):
                inputArray[i][j] = int(values[j])
        return inputArray

def getMaxMinValuations(valuations, n):
    maxValuation = valuations[0][0]
    minValuation = valuations[0][0]
    for i in range(n):
        for j in range(n):
            if(valuations[i][j] > maxValuation):
                maxValuation = valuations[i][j]
            if(valuations[i][j] < minValuation):
                minValuation = valuations[i][j]
    return minValuation,maxValuation

def getMaxDegree(seatGraph, n):
    maxDegree = 0
    for i in range(n):
        degree = 0
        for j in range(n):
            degree += seatGraph[i][j]
        if degree > maxDegree:
            maxDegree = degree
    return maxDegree

def getNeighbours(u, seatGraph):
    neighbours = np.zeros(int(sum(seatGraph[u])))
    i = 0
    for v in range(n):
        if seatGraph[u][v] == 1:
            neighbours[i] = v
            i += 1
    return neighbours

def calculateUtility(permutation, valuations, seatGraph, n):
    utility  = 0
    for p in range(n):
        for q in range(n):
            utility += valuations[p][q]*seatGraph[permutation[p]][permutation[q]]
    return utility

def createPermutation(permutation, remaining, currentTotalUtility, valuations, seatGraph, n):
    totalUtility = currentTotalUtility
    if len(remaining)==0:
        return calculateUtility(permutation, valuations, seatGraph, n)
    else:
        for i in remaining:
            permutation2 = permutation.copy()
            permutation2.append(i)
            remaining2 = remaining.copy()
            remaining2.remove(i)
            utility = createPermutation(permutation2, remaining2, totalUtility, valuations, seatGraph, n)
            if utility > totalUtility:
                totalUtility = utility
        return totalUtility

def bruteForce(valFile, seatFile):
    valuations = readFile(valFile)
    seatGraph = readFile(seatFile)

    n = len(valuations)
    minValuation = getMaxMinValuations(valuations, n)
    maxDegree = getMaxDegree(seatGraph, n)
    minUtility = minValuation[0]*maxDegree
    start_total_time = time.time()

    remaining = []
    for i in range(n):
        remaining.append(i)

    permutation = []
    totalUtility = createPermutation(permutation, remaining, minUtility*maxDegree, valuations, seatGraph, n)
    start_total_time = time.time() - start_total_time
    output = {}
    output['Total time'] = start_total_time
    output['Build time'] = 0
    output['Solve time'] = start_total_time
    output['Total utility'] = totalUtility
    return output

if __name__ == '__main__':
    valFile = sys.argv[1]
    seatFile = sys.argv[2]
    output = bruteForce(valFile, seatFile)
    print(output)