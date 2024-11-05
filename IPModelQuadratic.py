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
utilityType = sys.argv[3]
objective = sys.argv[4]

n = len(valuations)
minValuation, maxValuation = getMaxMinValuations(valuations)
absMaxValuation = max(abs(minValuation), abs(maxValuation))
maxDegree = getMaxDegree(seatGraph)
minUtility = minValuation*maxDegree
if utilityType == 'B' or utilityType == 'W':
    minUtility = minValuation
maxUtility = maxValuation*maxDegree
if utilityType == 'B' or utilityType == 'W':
    maxUtility = maxValuation
absMaxutility = max(abs(minUtility), abs(maxUtility))
epsilon = 0.1

start_total_time = time.time()

model = gp.Model("SeatArrangement")
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
model.addConstrs(x.sum(p, [v for v in range(n)]) == 1 for p in range(n))
model.addConstrs(x.sum([p for p in range(n)], v) == 1 for v in range(n))

# uv stands for utility at vertex
uv = model.addVars(n, n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="uv")
for p in range(n):
    for u in range(n):
        if utilityType == 'B':
            neighbours = getNeighbours(u)
            if len(neighbours) > 0:
                z = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z")
                y = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y")
                for i in range(len(neighbours)):
                    v = neighbours[i]
                    model.addConstr(z[i] == sum(x[q,v]*valuations[p][q] for q in range(n)))
                    model.addConstr(uv[p,u] >= z[i])
                    model.addConstr(y[i] >= 1-(uv[p,u]-z[i]))
                    model.addConstr(y[i] <= 1-(uv[p,u]-z[i])/(2*absMaxValuation))
                model.addConstr(y.sum(i for i in range(len(neighbours))) >= 1)
            else:
                model.addConstr(uv[p,u] == 0)
        elif utilityType == 'W':
            neighbours = getNeighbours(u)
            if len(neighbours) > 0:
                z = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z")
                y = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y")
                for i in range(len(neighbours)):
                    v = neighbours[i]
                    model.addConstr(z[i] == sum(x[q,v]*valuations[p][q] for q in range(n)))
                    model.addConstr(uv[p,u] <= z[i])
                    model.addConstr(y[i] >= 1-(z[i]-uv[p,u]))
                    model.addConstr(y[i] <= 1-(z[i]-uv[p,u])/(2*absMaxValuation))
                model.addConstr(y.sum(i for i in range(len(neighbours))) >= 1)
            else:
                model.addConstr(uv[p,u] == 0)
        else: # S-utility
            utility = 0
            for v in range(n):
                for q in range(n):
                    utility += x[q,v]*valuations[p][q]*seatGraph[u][v]
            model.addConstr(uv[p,u] == utility)

util = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="u")
for p in range(n):
    model.addConstr(util[p] == sum(uv[p,u]*x[p,u] for u in range(n)))

def exchangeUtility(exchangeVars, p, q):
    for u in range(n):
        if utilityType == 'B':
            neighbours = getNeighbours(u)
            if len(neighbours) > 0:
                z = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z")
                y = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y")
                for i in range(len(neighbours)):
                    v = neighbours[i]
                    utility_v = 0
                    for r in range(n):
                        if r == q:
                            utility_v += x[p,v]*valuations[p][q]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,v]*valuations[p][r]
                    model.addConstr(z[i] == utility_v)
                    model.addConstr(exchangeVars[u] >= z[i])
                    model.addConstr(y[i] >= 1-(exchangeVars[u]-z[i]))
                    model.addConstr(y[i] <= 1-(exchangeVars[u]-z[i])/(2*absMaxValuation))
                model.addConstr(y.sum(i for i in range(len(neighbours))) >= 1)
            else:
                model.addConstr(exchangeVars[u] == 0)
        else: # W-utility
            neighbours = getNeighbours(u)
            if len(neighbours) > 0:
                z = model.addVars(len(neighbours), vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="z")
                y = model.addVars(len(neighbours), vtype=GRB.BINARY, name="y")
                for i in range(len(neighbours)):
                    v = neighbours[i]
                    utility_v = 0
                    for r in range(n):
                        if r == q:
                            utility_v += x[p,v]*valuations[p][q]
                        elif r == p:
                            utility_v += 0
                        else:
                            utility_v += x[r,v]*valuations[p][r]
                    model.addConstr(z[i] == utility_v)
                    model.addConstr(exchangeVars[u] <= z[i])
                    model.addConstr(y[i] >= 1-(z[i]-exchangeVars[u]))
                    model.addConstr(y[i] <= 1-(z[i]-exchangeVars[u])/(2*absMaxValuation))
                model.addConstr(y.sum(i for i in range(len(neighbours))) >= 1)
            else:
                model.addConstr(exchangeVars[u] == 0)

def envyFreenessStability(p,q):
    exchange_p = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_p")
    if utilityType == 'B' or utilityType == 'W':
        exchangeUtility(exchange_p, p, q)
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
        exchangeUtility(exchange_q, q, p)
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
    line += str(util[p].X)+','
print(line)

#for v in model.getVars():
#    print('')
#    print(f"{v.VarName} {v.X:g}")