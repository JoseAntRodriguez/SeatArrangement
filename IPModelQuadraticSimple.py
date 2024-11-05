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

model = gp.Model("SeatArrangement")
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
model.addConstrs(x.sum(p, [v for v in range(n)]) == 1 for p in range(n))
model.addConstrs(x.sum([p for p in range(n)], v) == 1 for v in range(n))

y = model.addVars(n, n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="y")
for p in range(n):
    for u in range(n):
        utility = 0
        for v in range(n):
                for q in range(n):
                    utility += x[q,v]*valuations[p][q]*seatGraph[u][v]
        model.addConstr(y[p,u] == utility)

u = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="u")
for p in range(n):
    model.addConstr(u[p] == sum(y[p,u]*x[p,u] for u in range(n)))

model.setObjective(u.sum(p for p in range(n)), GRB.MAXIMIZE)
model.optimize()

print("Totaltime: " + str((time.time() - start_total_time)) + " s")

for p in range(n):
    line = ''
    for v in range(n):
        line += str(x[p,v].X)+','
    print(line)

print('\n')

line = ''
for p in range(n):
    line += str(u[p].X)+','
print(line)