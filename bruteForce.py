import gurobipy as gp
from gurobipy import GRB
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

def getMaxMinValuations(valuations):
    maxValuation = valuations[0][0]
    minValuation = valuations[0][0]
    for i in range(n):
        for j in range(n):
            if(valuations[i][j] > maxValuation):
                maxValuation = valuations[i][j]
            if(valuations[i][j] < minValuation):
                minValuation = valuations[i][j]
    return minValuation,maxValuation

def getMaxDegree(seatGraph):
    maxDegree = 0
    for i in range(n):
        degree = 0
        for j in range(n):
            degree += seatGraph[i][j]
        if degree > maxDegree:
            maxDegree = degree
    return maxDegree

def getNeighbours(u):
    neighbours = np.zeros(int(sum(seatGraph[u])))
    i = 0
    for v in range(n):
        if seatGraph[u][v] == 1:
            neighbours[i] = v
            i += 1
    return neighbours

valFile = sys.argv[1]
seatFile = sys.argv[2]
valuations = readFile(valFile)
seatGraph = readFile(seatFile)

n = len(valuations)
minValuation, maxValuation = getMaxMinValuations(valuations)
absMaxValuation = max(abs(minValuation), abs(maxValuation))
maxDegree = getMaxDegree(seatGraph)
minUtility = minValuation*maxDegree
maxUtility = maxValuation*maxDegree
absMaxutility = max(abs(minUtility), abs(maxUtility))
epsilon = 0.1

start_total_time = time.time()

remaining = []
for i in range(n):
    remaining.append(i)

def calculateUtility(permutation):
    utility  = 0
    for p in range(n):
        for q in range(n):
            utility += valuations[p][q]*seatGraph[permutation[p]][permutation[q]]
    return utility

def createPermutation(permutation, remaining, currentTotalUtility):
    totalUtility = currentTotalUtility
    if len(remaining)==0:
        return calculateUtility(permutation)
    else:
        for i in remaining:
            permutation2 = permutation.copy()
            permutation2.append(i)
            remaining2 = remaining.copy()
            remaining2.remove(i)
            utility = createPermutation(permutation2, remaining2, totalUtility)
            if utility > totalUtility:
                totalUtility = utility
        return totalUtility

permutation = []
totalUtility = createPermutation(permutation, remaining, minUtility*maxDegree)

print("Totaltime: " + str((time.time() - start_total_time)) + " s")
print('Utility = ', totalUtility)