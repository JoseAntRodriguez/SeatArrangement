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

def IPModel(valFile, seatFile, utilityType, objective):
    valuations = readFile(valFile)
    seatGraph = readFile(seatFile)

    n = len(valuations)
    minValuation, maxValuation = getMaxMinValuations(valuations, n)
    absMaxValuation = max(abs(minValuation), abs(maxValuation))
    maxDegree = getMaxDegree(seatGraph, n)
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
    model.addConstrs(x.sum(p, [v for v in range(n)]) == 1 for p in range(n))
    model.addConstrs(x.sum([p for p in range(n)], v) == 1 for v in range(n))

    y = model.addVars(n, n, n, n, vtype=GRB.BINARY, name="y")
    for p in range(n):
        for q in range(n):
            for u in range(n):
                for v in range(n):
                    model.addConstr(y[p,q,u,v] >= (x[p,u] + x[q,v] - 1)*seatGraph[u][v])
                    model.addConstr(y[p,q,u,v] <= x[p,u]*seatGraph[u][v])
                    model.addConstr(y[p,q,u,v] <= x[q,v]*seatGraph[u][v])

    # Need to add z matrix because if both x[p,u] and x[q,v], and valuations[p][q] is negative, then the first
    # constraint says that y[p,q,u,v] must be greater than -valuations[p][q] (a positive number), while the
    # other two say that y[p,q,u,v] must be smaller than or equal to 0
    z = model.addVars(n, n, n, n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z")
    for p in range(n):
        for q in range(n):
            for u in range(n):
                for v in range(n):
                    model.addConstr(z[p,q,u,v] == y[p,q,u,v]*valuations[p][q])

    util = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="u")
    for p in range(n):
        if utilityType == 'B':
            u_p = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="u_p")
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p")
                model.addConstrs(z_p[v] == (z.sum(p,[q for q in range(n)],u,neighbours[v])) for v in range(len(neighbours)))
                y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p")
                model.addConstrs(y_p[v] >= 1-(u_p[u]-z_p[v]) for v in range(len(neighbours)))
                model.addConstrs(y_p[v] <= 1-(u_p[u]-z_p[v])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(u_p[u] >= z_p[v] for v in range(len(neighbours)))
                model.addConstr(y_p.sum(v for v in range(len(neighbours))) >= 1)
            model.addConstr(util[p] == u_p.sum(u for u in range(n)))
        elif utilityType == 'W':
            u_p = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="u_p")
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z_p")
                model.addConstrs(z_p[v] == (z.sum(p,[q for q in range(n)],u,neighbours[v])) for v in range(len(neighbours)))
                y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y_p")
                model.addConstrs(y_p[v] >= 1-(z_p[v]-u_p[u]) for v in range(len(neighbours)))
                model.addConstrs(y_p[v] <= 1-(z_p[v]-u_p[u])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(u_p[u] <= z_p[v] for v in range(len(neighbours)))
                model.addConstr(y_p.sum(v for v in range(len(neighbours))) >= 1)
            model.addConstr(util[p] == u_p.sum(u for u in range(n)))
        else: # S-utility
            model.addConstr(util[p] == z.sum(p, [q for q in range(n)], [u for u in range(n)], [v for v in range(n)]))

    def envyFreenessStability(p,q):
        name1 = "exchange_p_"+str(p)
        exchange_p = model.addVars(n, n, n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name=name1)
        exchange_p_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name='exchange_p_util')
        for r in range(n):
            for u in range(n):
                for v in range(n):
                    if r == q:
                        model.addConstr(exchange_p[r,u,v] == y[q,p,u,v]*valuations[p][q])
                    elif r == p:
                        model.addConstr(exchange_p[r,u,v] == 0)
                    else:
                        model.addConstr(exchange_p[r,u,v] == y[q,r,u,v]*valuations[p][r])
        if utilityType == 'B':
            exchange_u_p = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_u_p")
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                exchange_z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="exchange_z_p")
                model.addConstrs(exchange_z_p[v] == (exchange_p.sum([r for r in range(n)],u,neighbours[v])) for v in range(len(neighbours)))
                exchange_y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="exchange_y_p")
                model.addConstrs(exchange_y_p[v] >= 1-(exchange_u_p[u]-exchange_z_p[v]) for v in range(len(neighbours)))
                model.addConstrs(exchange_y_p[v] <= 1-(exchange_u_p[u]-exchange_z_p[v])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_u_p[u] >= exchange_z_p[v] for v in range(len(neighbours)))
                model.addConstr(exchange_y_p.sum(v for v in range(len(neighbours))) >= 1)
            model.addConstr(exchange_p_util == exchange_u_p.sum(u for u in range(n)))
        elif utilityType == 'W':
            exchange_u_p = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_u_p")
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                exchange_z_p = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="exchange_z_p")
                model.addConstrs(exchange_z_p[v] == (exchange_p.sum([r for r in range(n)],u,neighbours[v])) for v in range(len(neighbours)))
                exchange_y_p = model.addVars(len(neighbours), vtype=GRB.BINARY, name="exchange_y_p")
                model.addConstrs(exchange_y_p[v] >= 1-(exchange_z_p[v]-exchange_u_p[u]) for v in range(len(neighbours)))
                model.addConstrs(exchange_y_p[v] <= 1-(exchange_z_p[v]-exchange_u_p[u])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_u_p[u] <= exchange_z_p[v] for v in range(len(neighbours)))
                model.addConstr(exchange_y_p.sum(v for v in range(len(neighbours))) >= 1)
            model.addConstr(exchange_p_util == exchange_u_p.sum(u for u in range(n)))
        else: # S-utility
            model.addConstr(exchange_p_util == exchange_p.sum([r for r in range(n)], [u for u in range(n)], [v for v in range(n)]))
        
        name1 = "exchange_q_"+str(q)
        exchange_q = model.addVars(n, n, n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name=name1)
        exchange_q_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name='exchange_q_util')
        for r in range(n):
            for u in range(n):
                for v in range(n):
                    if r == p:
                        model.addConstr(exchange_q[r,u,v] == y[p,q,u,v]*valuations[q][p])
                    elif r == q:
                        model.addConstr(exchange_q[r,u,v] == 0)
                    else:
                        model.addConstr(exchange_q[r,u,v] == y[p,r,u,v]*valuations[q][r])
        if utilityType == 'B':
            exchange_u_q = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_u_q")
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                exchange_z_q = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="exchange_z_q")
                model.addConstrs(exchange_z_q[v] == (exchange_q.sum([r for r in range(n)],u,neighbours[v])) for v in range(len(neighbours)))
                exchange_y_q = model.addVars(len(neighbours), vtype=GRB.BINARY, name="exchange_y_q")
                model.addConstrs(exchange_y_q[v] >= 1-(exchange_u_q[u]-exchange_z_q[v]) for v in range(len(neighbours)))
                model.addConstrs(exchange_y_q[v] <= 1-(exchange_u_q[u]-exchange_z_q[v])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_u_q[u] >= exchange_z_q[v] for v in range(len(neighbours)))
                model.addConstr(exchange_y_q.sum(v for v in range(len(neighbours))) >= 1)
            model.addConstr(exchange_q_util == exchange_u_q.sum(u for u in range(n)))
        elif utilityType == 'W':
            exchange_u_q = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_u_p")
            for u in range(n):
                neighbours = getNeighbours(u,n,seatGraph)
                exchange_z_q = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="exchange_z_q")
                model.addConstrs(exchange_z_q[v] == (exchange_q.sum([r for r in range(n)],u,neighbours[v])) for v in range(len(neighbours)))
                exchange_y_q = model.addVars(len(neighbours), vtype=GRB.BINARY, name="exchange_y_q")
                model.addConstrs(exchange_y_q[v] >= 1-(exchange_z_q[v]-exchange_u_q[u]) for v in range(len(neighbours)))
                model.addConstrs(exchange_y_q[v] <= 1-(exchange_z_q[v]-exchange_u_q[u])/(2*absMaxValuation) for v in range(len(neighbours)))
                model.addConstrs(exchange_u_q[u] <= exchange_z_q[v] for v in range(len(neighbours)))
                model.addConstr(exchange_y_q.sum(v for v in range(len(neighbours))) >= 1)
            model.addConstr(exchange_q_util == exchange_u_q.sum(u for u in range(n)))
        else: # S-utility
            model.addConstr(exchange_q_util == exchange_q.sum([r for r in range(n)], [u for u in range(n)], [v for v in range(n)]))
        if objective == 'EFA':
            model.addConstr(exchange_p_util <= util[p])
            model.addConstr(exchange_q_util <= util[q])
        else: #STA
            e_p = model.addVar(vtype=GRB.BINARY, name="e_p")
            model.addConstr(e_p >= (exchange_p_util - util[p])/(2*absMaxutility))
            model.addConstr(e_p <= 1 - (util[p] - exchange_p_util - epsilon)/(2*absMaxutility))
            e_q = model.addVar(vtype=GRB.BINARY, name="e_q")
            model.addConstr(e_q >= (exchange_q_util - util[q])/(2*absMaxutility))
            model.addConstr(e_q <= 1 - (util[q] - exchange_q_util - epsilon)/(2*absMaxutility))
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
    output = IPModel(valFile, seatFile, utilityType, objective)
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