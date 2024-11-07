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

def IPModelQuadraticSimple(valFile, seatFile):
    valuations = readFile(valFile)
    seatGraph = readFile(seatFile)

    n = len(valuations)
    minValuation, maxValuation = getMaxMinValuations(valuations, n)
    maxDegree = getMaxDegree(seatGraph, n)
    minUtility = minValuation*maxDegree
    maxUtility = maxValuation*maxDegree

    start_total_time = time.time()
    start_build_time = start_total_time

    model = gp.Model("SeatArrangement")
    model.setParam('OutputFlag', 0)
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
    start_build_time = time.time() - start_build_time
    start_solve_time = time.time()
    model.optimize()
    start_solve_time = time.time() - start_solve_time
    start_total_time = time.time() - start_total_time
    totalUtility = 0
    for p in range(n):
        totalUtility += u[p].X
    output = {}
    output['Total time'] = start_total_time
    output['Build time'] = start_build_time
    output['Solve time'] = start_solve_time
    output['Objective'] = totalUtility
    return output

if __name__ == '__main__':
    valFile = sys.argv[1]
    seatFile = sys.argv[2]
    output = IPModelQuadraticSimple(valFile, seatFile)
    print(output)

'''
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
'''