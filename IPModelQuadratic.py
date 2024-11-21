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
    model.setParam('OutputFlag', 0)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    x.BranchPriority = 100
    model.addConstrs(x.sum(p, [v for v in range(n)]) == 1 for p in range(n))
    model.addConstrs(x.sum([p for p in range(n)], v) == 1 for v in range(n))

    # uv stands for utility at vertex
    uv = model.addVars(n, n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="uv")
    uv.BranchPriority = -10
    for p in range(n):
        for u in range(n):
            if utilityType == 'B':
                neighbours = getNeighbours(u,n,seatGraph)
                z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p")
                z_p.BranchPriority = -10
                for v in range(len(neighbours)):
                    totalSum = 0
                    for q in range(n):
                        totalSum += valuations[p][q]*x[q,neighbours[v]]
                    model.addConstr(z_p[v] == totalSum)
                y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p")
                y_p.BranchPriority = -10
                model.addConstrs(y_p[v] >= 1-(uv[p,u]-z_p[v]) for v in range(len(neighbours)))
                model.addConstrs(y_p[v] <= 1-(uv[p,u]-z_p[v])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(uv[p,u] >= z_p[v] for v in range(len(neighbours)))
                model.addConstr(y_p.sum(v for v in range(len(neighbours))) >= 1)
            elif utilityType == 'W':
                neighbours = getNeighbours(u,n,seatGraph)
                z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p")
                z_p.BranchPriority = -10
                for v in range(len(neighbours)):
                    totalSum = 0
                    for q in range(n):
                        totalSum += valuations[p][q]*x[q,neighbours[v]]
                    model.addConstr(z_p[v] == totalSum)
                y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p")
                y_p.BranchPriority = -10
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
    util.BranchPriority = -10
    for p in range(n):
        model.addConstr(util[p] == sum(uv[p,u]*x[p,u] for u in range(n)))

    def envyFreenessStability(p,q):
        exchange_p = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_p-"+str(p)+"-"+str(q))
        exchange_p.BranchPriority = -10
        exchange_p_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name='exchange_p_util')
        exchange_p_util.BranchPriority = -10
        if utilityType == 'B':
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                z_p_exchange = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p_exchange")
                z_p_exchange.BranchPriority = -10
                for v in range(len(neighbours)):
                    utility_v = 0
                    for r in range(n):
                        if r == q:
                            utility_v += x[p,neighbours[v]]*valuations[p][q]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,neighbours[v]]*valuations[p][r]
                    model.addConstr(z_p_exchange[v] == utility_v)
                y_p_exchange = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p_exchange")
                y_p_exchange.BranchPriority = -10
                model.addConstrs(y_p_exchange[v] >= 1-(exchange_p[u]-z_p_exchange[v]) for v in range(len(neighbours)))
                model.addConstrs(y_p_exchange[v] <= 1-(exchange_p[u]-z_p_exchange[v])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_p[u] >= z_p_exchange[v] for v in range(len(neighbours)))
                model.addConstr(y_p_exchange.sum(v for v in range(len(neighbours))) >= 1)
        elif utilityType == 'W':
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                z_p_exchange = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p_exchange")
                z_p_exchange.BranchPriority = -10
                for v in range(len(neighbours)):
                    utility_v = 0
                    for r in range(n):
                        if r == q:
                            utility_v += x[p,neighbours[v]]*valuations[p][q]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,neighbours[v]]*valuations[p][r]
                    model.addConstr(z_p_exchange[v] == utility_v)
                y_p_exchange = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p_exchange")
                y_p_exchange.BranchPriority = -10
                model.addConstrs(y_p_exchange[v] >= 1-(z_p_exchange[v]-exchange_p[u]) for v in range(len(neighbours)))
                model.addConstrs(y_p_exchange[v] <= 1-(z_p_exchange[v]-exchange_p[u])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_p[u] <= z_p_exchange[v] for v in range(len(neighbours)))
                model.addConstr(y_p_exchange.sum(v for v in range(len(neighbours))) >= 1)
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
        model.addConstr(exchange_p_util == sum(exchange_p[u]*x[q,u] for u in range(n)))
        exchange_q = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_q")
        exchange_q.BranchPriority = -10
        exchange_q_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name='exchange_q_util')
        exchange_q_util.BranchPriority = -10
        if utilityType == 'B':
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                z_q_exchange = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_q_exchange")
                z_q_exchange.BranchPriority = -10
                for v in range(len(neighbours)):
                    utility_v = 0
                    for r in range(n):
                        if r == p:
                            utility_v += x[q,neighbours[v]]*valuations[q][p]
                        elif r == q:
                            utility_v += 0
                        else:
                            utility_v += x[r,neighbours[v]]*valuations[q][r]
                    model.addConstr(z_q_exchange[v] == utility_v)
                y_q_exchange = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_q_exchange")
                y_q_exchange.BranchPriority = -10
                model.addConstrs(y_q_exchange[v] >= 1-(exchange_q[u]-z_q_exchange[v]) for v in range(len(neighbours)))
                model.addConstrs(y_q_exchange[v] <= 1-(exchange_q[u]-z_q_exchange[v])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_q[u] >= z_q_exchange[v] for v in range(len(neighbours)))
                model.addConstr(y_q_exchange.sum(v for v in range(len(neighbours))) >= 1)
        elif utilityType == 'W':
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                z_q_exchange = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_q_exchange")
                z_q_exchange.BranchPriority = -10
                for v in range(len(neighbours)):
                    utility_v = 0
                    for r in range(n):
                        if r == p:
                            utility_v += x[q,neighbours[v]]*valuations[q][p]
                        elif r == q:
                            utility_v += 0
                        else:
                            utility_v += x[r,neighbours[v]]*valuations[q][r]
                    model.addConstr(z_q_exchange[v] == utility_v)
                y_q_exchange = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_q_exchange")
                y_q_exchange.BranchPriority = -10
                model.addConstrs(y_q_exchange[v] >= 1-(z_q_exchange[v]-exchange_q[u]) for v in range(len(neighbours)))
                model.addConstrs(y_q_exchange[v] <= 1-(z_q_exchange[v]-exchange_q[u])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_q[u] <= z_q_exchange[v] for v in range(len(neighbours)))
                model.addConstr(y_q_exchange.sum(v for v in range(len(neighbours))) >= 1)
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
        model.addConstr(exchange_q_util == sum(exchange_q[u]*x[p,u] for u in range(n)))
        if objective == 'EFA':
            model.addConstr(exchange_p_util <= util[p])
            model.addConstr(exchange_q_util <= util[q])
        else: #STA
            e_p = model.addVar(vtype=GRB.BINARY, name="e_p")
            e_p.BranchPriority = -10
            model.addConstr(e_p >= (exchange_p_util - util[p])/(2*absMaxutility))
            model.addConstr(e_p <= 1 - (util[p] - exchange_p_util - epsilon)/(2*absMaxutility))
            e_q = model.addVar(vtype=GRB.BINARY, name="e_q")
            e_q.BranchPriority = -10
            model.addConstr(e_q >= (exchange_q_util - util[q])/(2*absMaxutility))
            model.addConstr(e_q <= 1 - (util[q] - exchange_q_util - epsilon)/(2*absMaxutility))
            model.addConstr(e_p + e_q <= 1)

    if objective == 'MWA':
        model.setObjective(util.sum(p for p in range(n)), GRB.MAXIMIZE)
    elif objective == 'MUA':
        minVal = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="minVal")
        minVal.BranchPriority = -10
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
    objectiveValue = 0
    if objective == 'MWA':
        for p in range(n):
            objectiveValue += util[p].X
    elif objective == 'MUA':
        objectiveValue = minVal.X
    else: # EFA or STA
        if model.Status != GRB.INFEASIBLE:
            objectiveValue = 1
    output = {}
    output['Total time'] = start_total_time
    output['Build time'] = start_build_time
    output['Solve time'] = start_solve_time
    output['Objective'] = objectiveValue
    output['Nodes'] = model.nodeCount
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