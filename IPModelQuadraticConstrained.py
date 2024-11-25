import gurobipy as gp
from gurobipy import GRB
import sys
import numpy as np
import time


def readFile(filename, seatGraph=False):
    with open(filename, 'r') as f:
        content = f.read().splitlines()
        size = len(content)
        inputArray = np.zeros((size,size))
        for i in range(size):
            values = content[i].split(',')
            size2 = size
            if seatGraph:
                size2 = 2
            for j in range(size2):
                inputArray[i][j] = int(values[j])
        return inputArray

def IPModelQuadratic(valFile, seatFile, utilityType, objective):
    valuations = readFile(valFile, False)
    seatGraph = readFile(seatFile, True)

    n = len(valuations)
    maxUtility = 2
    if utilityType == 'B' or utilityType == 'W':
        maxUtility = 1
    epsilon = 0.1

    start_total_time = time.time()
    start_build_time = start_total_time

    model = gp.Model("SeatArrangement")
    model.setParam('OutputFlag', 0)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    for p in range(n):
        for u in range(n):
            x[p,u].BranchPriority = 100
    model.addConstrs(x.sum(p, [v for v in range(n)]) == 1 for p in range(n))
    model.addConstrs(x.sum([p for p in range(n)], v) == 1 for v in range(n))

    # uv stands for utility at vertex
    uv = model.addVars(n, n, vtype=GRB.INTEGER, lb=0, ub=maxUtility, name="uv")
    for p in range(n):
        for u in range(n):
            if utilityType == 'B':
                z_p = model.addVars(2, vtype=GRB.BINARY, name="z_p")
                for v in range(2):
                    totalSum = 0
                    for q in range(n):
                        totalSum += valuations[p][q]*x[q,seatGraph[u][v]]
                    model.addConstr(z_p[v] == totalSum)
                y_p = model.addVars(2, vtype=GRB.BINARY, name="y_p")
                model.addConstrs(y_p[v] >= 1-(uv[p,u]-z_p[v]) for v in range(2))
                model.addConstrs(y_p[v] <= 1-(uv[p,u]-z_p[v])/2 for v in range(2))
                model.addConstrs(uv[p,u] >= z_p[v] for v in range(2))
                model.addConstr(y_p.sum(v for v in range(2)) >= 1)
            elif utilityType == 'W':
                z_p = model.addVars(2, vtype=GRB.BINARY, name="z_p")
                for v in range(2):
                    totalSum = 0
                    for q in range(n):
                        totalSum += valuations[p][q]*x[q,seatGraph[u][v]]
                    model.addConstr(z_p[v] == totalSum)
                y_p = model.addVars(2, vtype=GRB.BINARY, name="y_p")
                model.addConstrs(y_p[v] >= 1-(z_p[v]-uv[p,u]) for v in range(2))
                model.addConstrs(y_p[v] <= 1-(z_p[v]-uv[p,u])/2 for v in range(2))
                model.addConstrs(uv[p,u] <= z_p[v] for v in range(2))
                model.addConstr(y_p.sum(v for v in range(2)) >= 1)
            else: # S-utility
                utility = 0
                for q in range(n):
                    utility += x[q,seatGraph[u][0]]*valuations[p][q] + x[q,seatGraph[u][1]]*valuations[p][q]
                model.addConstr(uv[p,u] == utility)

    util = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=maxUtility, name="u")
    for p in range(n):
        model.addConstr(util[p] == sum(uv[p,u]*x[p,u] for u in range(n)))
        # New constraints to account for the fact that valuations are binary and the seat graph is a cycle seat graph
        model.addConstr(util[p] >= 0)
        model.addConstr(util[p] <= maxUtility)

    def envyFreenessStability(p,q):
        exchange_p = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=maxUtility, name="exchange_p-"+str(p)+"-"+str(q))
        exchange_p_util = model.addVar(vtype=GRB.INTEGER, lb=0, ub=maxUtility, name='exchange_p_util')
        # New constraints to account for the fact that valuations are binary and the seat graph is a cycle seat graph
        model.addConstr(exchange_p_util >= 0)
        model.addConstr(exchange_p_util <= maxUtility)
        if utilityType == 'B':
            for u in range(n):
                z_p_exchange = model.addVars(2, vtype=GRB.BINARY, name="z_p_exchange")
                for v in range(2):
                    utility_v = 0
                    for r in range(n):
                        if r == q:
                            utility_v += x[p,seatGraph[u][v]]*valuations[p][q]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,seatGraph[u][v]]*valuations[p][r]
                    model.addConstr(z_p_exchange[v] == utility_v)
                y_p_exchange = model.addVars(2, vtype=GRB.BINARY, name="y_p_exchange")
                model.addConstrs(y_p_exchange[v] >= 1-(exchange_p[u]-z_p_exchange[v]) for v in range(2))
                model.addConstrs(y_p_exchange[v] <= 1-(exchange_p[u]-z_p_exchange[v])/2 for v in range(2))
                model.addConstrs(exchange_p[u] >= z_p_exchange[v] for v in range(2))
                model.addConstr(y_p_exchange.sum(v for v in range(2)) >= 1)
        elif utilityType == 'W':
            for u in range(n):
                z_p_exchange = model.addVars(2, vtype=GRB.BINARY, name="z_p_exchange")
                for v in range(2):
                    utility_v = 0
                    for r in range(n):
                        if r == q:
                            utility_v += x[p,seatGraph[u][v]]*valuations[p][q]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,seatGraph[u][v]]*valuations[p][r]
                    model.addConstr(z_p_exchange[v] == utility_v)
                y_p_exchange = model.addVars(2, vtype=GRB.BINARY, name="y_p_exchange")
                model.addConstrs(y_p_exchange[v] >= 1-(z_p_exchange[v]-exchange_p[u]) for v in range(2))
                model.addConstrs(y_p_exchange[v] <= 1-(z_p_exchange[v]-exchange_p[u])/2 for v in range(2))
                model.addConstrs(exchange_p[u] <= z_p_exchange[v] for v in range(2))
                model.addConstr(y_p_exchange.sum(v for v in range(2)) >= 1)
        else: # S-utility
            for u in range(n):
                utility_v = 0
                for v in range(2):
                    for r in range(n):
                        if r == q:
                            utility_v += x[p,seatGraph[u][v]]*valuations[p][q]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,seatGraph[u][v]]*valuations[p][r]
                model.addConstr(exchange_p[u] == utility_v)
        model.addConstr(exchange_p_util == sum(exchange_p[u]*x[q,u] for u in range(n)))
        exchange_q = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=maxUtility, name="exchange_q")
        exchange_q_util = model.addVar(vtype=GRB.INTEGER, lb=0, ub=maxUtility, name='exchange_q_util')
        # New constraints to account for the fact that valuations are binary and the seat graph is a cycle seat graph
        model.addConstr(exchange_q_util >= 0)
        model.addConstr(exchange_q_util <= maxUtility)
        if utilityType == 'B':
            for u in range(n):
                z_q_exchange = model.addVars(2, vtype=GRB.BINARY, name="z_q_exchange")
                for v in range(2):
                    utility_v = 0
                    for r in range(n):
                        if r == p:
                            utility_v += x[q,seatGraph[u][v]]*valuations[q][p]
                        elif r == q:
                            utility_v += 0
                        else:
                            utility_v += x[r,seatGraph[u][v]]*valuations[q][r]
                    model.addConstr(z_q_exchange[v] == utility_v)
                y_q_exchange = model.addVars(2, vtype=GRB.BINARY, name="y_q_exchange")
                model.addConstrs(y_q_exchange[v] >= 1-(exchange_q[u]-z_q_exchange[v]) for v in range(2))
                model.addConstrs(y_q_exchange[v] <= 1-(exchange_q[u]-z_q_exchange[v])/2 for v in range(2))
                model.addConstrs(exchange_q[u] >= z_q_exchange[v] for v in range(2))
                model.addConstr(y_q_exchange.sum(v for v in range(2)) >= 1)
        elif utilityType == 'W':
            for u in range(n):
                z_q_exchange = model.addVars(2, vtype=GRB.BINARY, name="z_q_exchange")
                for v in range(2):
                    utility_v = 0
                    for r in range(n):
                        if r == p:
                            utility_v += x[q,seatGraph[u][v]]*valuations[q][p]
                        elif r == q:
                            utility_v += 0
                        else:
                            utility_v += x[r,seatGraph[u][v]]*valuations[q][r]
                    model.addConstr(z_q_exchange[v] == utility_v)
                y_q_exchange = model.addVars(2, vtype=GRB.BINARY, name="y_q_exchange")
                model.addConstrs(y_q_exchange[v] >= 1-(z_q_exchange[v]-exchange_q[u]) for v in range(2))
                model.addConstrs(y_q_exchange[v] <= 1-(z_q_exchange[v]-exchange_q[u])/2 for v in range(2))
                model.addConstrs(exchange_q[u] <= z_q_exchange[v] for v in range(2))
                model.addConstr(y_q_exchange.sum(v for v in range(2)) >= 1)
        else: # S-utility
            for u in range(n):
                utility_v = 0
                for v in range(2):
                    for r in range(n):
                        if r == p:
                            utility_v += x[q,seatGraph[u][v]]*valuations[q][p]
                        elif r == q:
                            utility_v += 0
                        else:
                            utility_v += x[r,seatGraph[u][v]]*valuations[q][r]
                model.addConstr(exchange_q[u] == utility_v)
        model.addConstr(exchange_q_util == sum(exchange_q[u]*x[p,u] for u in range(n)))
        if objective == 'EFA':
            model.addConstr(exchange_p_util <= util[p])
            model.addConstr(exchange_q_util <= util[q])
        else: #STA
            e_p = model.addVar(vtype=GRB.BINARY, name="e_p")
            model.addConstr(e_p >= (exchange_p_util - util[p])/(2*maxUtility))
            model.addConstr(e_p <= 1 - (util[p] - exchange_p_util - epsilon)/(2*maxUtility))
            e_q = model.addVar(vtype=GRB.BINARY, name="e_q")
            model.addConstr(e_q >= (exchange_q_util - util[q])/(2*maxUtility))
            model.addConstr(e_q <= 1 - (util[q] - exchange_q_util - epsilon)/(2*maxUtility))
            model.addConstr(e_p + e_q <= 1)
    
    # New constraints to account for the fact that valuations are binary and the seat graph is a cycle seat graph
    for u in range(n-1):
        if seatGraph[u][0] > u:
            for v in range(int(seatGraph[u][0]), int(seatGraph[u][1]+1)):
                model.addConstrs(x.sum([q for q in range(p)], v) <= x[p,u] + 2*(1-x[p,u]) - epsilon for p in range(n))

    if objective == 'MWA':
        model.setObjective(util.sum(p for p in range(n)), GRB.MAXIMIZE)
    elif objective == 'MUA':
        minVal = model.addVar(vtype=GRB.INTEGER, lb=0, ub=maxUtility, name="minVal")
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