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

def getNeighbours(u, n, seatGraph):
    neighbours = np.zeros(int(sum(seatGraph[u])))
    i = 0
    for v in range(n):
        if seatGraph[u][v] == 1:
            neighbours[i] = v
            i += 1
    return neighbours

def IPModelQuadratic(valFile, seatFile, utilityType, objective):
    valuations = readFile(valFile)
    seatGraph = readFile(seatFile)

    n = len(valuations)
    minValuation, maxValuation = getMaxMinValuations(valuations,n)
    absMaxValuation = max(abs(minValuation), abs(maxValuation))
    maxDegree = getMaxDegree(seatGraph,n)
    minUtility = minValuation*maxDegree
    if utilityType == 'B' or utilityType == 'W':
        minUtility = minValuation
    maxUtility = maxValuation*maxDegree
    if utilityType == 'B' or utilityType == 'W':
        maxUtility = maxValuation
    absMaxutility = max(abs(minUtility), abs(maxUtility))
    epsilon = 0.1

    start_total_time = time.time()
    start_build_time = start_total_time

    model = gp.Model("SeatArrangement")
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    model.addConstrs(x.sum(p, [v for v in range(n)]) == 1 for p in range(n))
    model.addConstrs(x.sum([p for p in range(n)], v) == 1 for v in range(n))

    # uv stands for utility at vertex
    uv = model.addVars(n, n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="uv")
    for p in range(n):
        for u in range(n):
            if utilityType == 'B':
                for u in range(n):
                    neighbours = getNeighbours(u,n,seatGraph)
                    z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p")
                    for v in range(len(neighbours)):
                        totalSum = 0
                        for q in range(n):
                            totalSum += valuations[p][q]*x[q,neighbours[v]]
                        model.addConstr(z_p[v] == totalSum)
                    y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p")
                    model.addConstrs(y_p[v] >= 1-(uv[p,u]-z_p[v]) for v in range(len(neighbours)))
                    model.addConstrs(y_p[v] <= 1-(uv[p,u]-z_p[v])/(2*absMaxValuation) for v in range(len(neighbours)))
                    model.addConstrs(uv[p,u] >= z_p[v] for v in range(len(neighbours)))
                    model.addConstr(y_p.sum(v for v in range(len(neighbours))) >= 1)
            elif utilityType == 'W':
                for u in range(n):
                    neighbours = getNeighbours(u,n,seatGraph)
                    z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p")
                    for v in range(len(neighbours)):
                        totalSum = 0
                        for q in range(n):
                            totalSum += valuations[p][q]*x[q,neighbours[v]]
                        model.addConstr(z_p[v] == totalSum)
                    y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p")
                    model.addConstrs(y_p[v] >= 1-(z_p[v]-uv[p,u]) for v in range(len(neighbours)))
                    model.addConstrs(y_p[v] <= 1-(z_p[v]-uv[p,u])/(2*absMaxValuation) for v in range(len(neighbours)))
                    model.addConstrs(uv[p,u] <= z_p[v] for v in range(len(neighbours)))
                    model.addConstr(y_p.sum(v for v in range(len(neighbours))) >= 1)
            else: # S-utility
                utility = 0
                for v in range(n):
                    for q in range(n):
                        utility += x[q,v]*valuations[p][q]*seatGraph[u][v]
                model.addConstr(uv[p,u] == utility)

    util = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="u")
    for p in range(n):
        model.addConstr(util[p] == sum(uv[p,u]*x[p,u] for u in range(n)))

    def envyFreenessStability(p,q):
        exchange_p = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_p")
        if utilityType == 'B' or utilityType == 'W':
            pass
        else: # S-utility
            for u in range(n):
                utility_v = 0
                for r in range(n):
                    for v in range(n):
                        if r == q:
                            utility_v += x[p,v]*valuations[p][q]*seatGraph[u][v]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,v]*valuations[p][r]*seatGraph[u][v]
                model.addConstr(exchange_p[u] == utility_v)
                #exchange_p[u] = utility_v
        exchange_p_util = sum(exchange_p[u]*x[q,u] for u in range(n))
        #exchange_p_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_p_util")
        #model.addConstr(exchange_p_util == sum(exchange_p[u]*x[q,u] for u in range(n)))
        exchange_q = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_q")
        if utilityType == 'B' or utilityType == 'W':
            pass
        else: # S-utility
            for u in range(n):
                utility_v = 0
                for r in range(n):
                    for v in range(n):
                        if r == p:
                            utility_v += x[q,v]*valuations[q][p]*seatGraph[u][v]
                        elif r == q:
                            utility_v += 0
                        else:
                            utility_v += x[r,v]*valuations[q][r]*seatGraph[u][v]
                model.addConstr(exchange_q[u] == utility_v)
                #exchange_q[u] = utility_v
        exchange_q_util = sum(exchange_q[u]*x[p,u] for u in range(n))
        #exchange_q_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_q_util")
        #model.addConstr(exchange_q_util == sum(exchange_q[u]*x[p,u] for u in range(n)))
        if objective == 'EFA':
            model.addConstr(exchange_p_util <= util[p])
            model.addConstr(exchange_q_util <= util[q])
        else: #STA
            e_p = model.addVar(vtype=GRB.BINARY, name="e_p")
            model.addConstr(e_p >= (exchange_p_util - util[p])/absMaxutility)
            model.addConstr(e_p <= 1 - (util[p] - exchange_p_util - epsilon)/absMaxutility)
            e_q = model.addVar(vtype=GRB.BINARY, name="e_q")
            model.addConstr(e_q >= (exchange_q_util - util[q])/absMaxutility)
            model.addConstr(e_q <= 1 - (util[q] - exchange_q_util - epsilon)/absMaxutility)
            model.addConstr(e_p + e_q <= 1)

    if objective == 'MWA':
        model.setObjective(util.sum(p for p in range(n)), GRB.MAXIMIZE)
    elif objective == 'MUA':
        minVal = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="minVal")
        model.addConstrs(minVal <= util[p] for p in range(n))
        model.setObjective(minVal, GRB.MAXIMIZE)
    else: # EFA or STA
        for p in range(n):
            for q in range(p+1,n):
                envyFreenessStability(p,q)
    start_build_time = time.time() - start_build_time
    start_solve_time = time.time()
    model.optimize()
    start_solve_time = time.time() - start_solve_time
    start_total_time = time.time() - start_total_time
    totalUtility = 0
    for p in range(n):
        totalUtility += util[p].X
    output = {}
    output['Total time'] = start_total_time
    output['Build time'] = start_build_time
    output['Solve time'] = start_solve_time
    output['Objective'] = totalUtility
    output['Nodes'] = model.nodeCount
    for p in range(n):
        line = ''
        for v in range(n):
            line += str(x[p,v].X)+','
        print(line)

    print('\n')

    line = ''
    for p in range(n):
        line += str(util[p].X)+','
    print(line)
    return output

if __name__ == '__main__':
    valFile = sys.argv[1]
    seatFile = sys.argv[2]
    utilityType = sys.argv[3]
    objective = sys.argv[4]
    output = IPModelQuadratic(valFile, seatFile, utilityType, objective)
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
    line += str(util[p].X)+','
print(line)
'''