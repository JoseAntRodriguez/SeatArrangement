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
                if utilityType == 'B':
                    #model.addConstr(z[p,q,u,v] == y[p,q,u,v]*valuations[p][q] + minValuation*(1-y[p,q,u,v]))
                    model.addConstr(z[p,q,u,v] == y[p,q,u,v]*valuations[p][q])
                elif utilityType == 'W':
                    #model.addConstr(z[p,q,u,v] == y[p,q,u,v]*valuations[p][q] + maxValuation*(1-y[p,q,u,v]))
                    model.addConstr(z[p,q,u,v] == y[p,q,u,v]*valuations[p][q])
                else: # S-utility
                    model.addConstr(z[p,q,u,v] == y[p,q,u,v]*valuations[p][q])

def bUtility(varArray, utilValue):
    w = model.addVars(n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="w")
    model.addConstrs(w[q] == (varArray.sum(q,[u for u in range(n)],[v for v in range(n)])) for q in range(n))
    w2 = model.addVars(n, vtype=GRB.BINARY, name="w2")
    model.addConstrs(w2[q] >= 1-(utilValue-w[q]) for q in range(n))
    model.addConstrs(w2[q] <= 1-(w[q]-utilValue)/absMaxValuation for q in range(n))
    model.addConstr(w2.sum(q for q in range(n)) >= 1)

def wUtility(varArray, utilValue):
    w = model.addVars(n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="w")
    model.addConstrs(w[q] == (varArray.sum(q,[u for u in range(n)],[v for v in range(n)])) for q in range(n))
    w2 = model.addVars(n, vtype=GRB.BINARY, name="w2")
    model.addConstrs(w2[q] >= 1-(w[q]-utilValue) for q in range(n))
    model.addConstrs(w2[q] <= 1-(utilValue-w[q])/absMaxValuation for q in range(n))
    model.addConstr(w2.sum(q for q in range(n)) >= 1)

u = model.addVars(n, vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="u")
for p in range(n):
    if utilityType == 'B':
        #bUtility(z[p], u[p])
        w = model.addVars(n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="w")
        model.addConstrs(w[q] == (z.sum(p,q,[u for u in range(n)],[v for v in range(n)])) + minValuation*(1-y.sum(p,q,[u for u in range(n)],[v for v in range(n)])) for q in range(n))
        w2 = model.addVars(n, vtype=GRB.BINARY, name="w2")
        model.addConstrs(w2[q] >= 1-(u[p]-w[q]) for q in range(n))
        model.addConstrs(w2[q] <= 1-(u[p]-w[q])/absMaxValuation for q in range(n))
        model.addConstrs(u[p] >= w[q] for q in range(n))
        model.addConstr(w2.sum(q for q in range(n)) >= 1)
    elif utilityType == 'W':
        #wUtility(z[p], u[p])
        w = model.addVars(n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name="w")
        model.addConstrs(w[q] == (z.sum(p,q,[u for u in range(n)],[v for v in range(n)])) + maxValuation*(1-y.sum(p,q,[u for u in range(n)],[v for v in range(n)])) for q in range(n))
        w2 = model.addVars(n, vtype=GRB.BINARY, name="w2")
        model.addConstrs(w2[q] >= 1-(w[q]-u[p]) for q in range(n))
        model.addConstrs(w2[q] <= 1-(u[p]-w[q])/absMaxValuation for q in range(n))
        #model.addConstrs(u[p] <= w[q] for q in range(n))
        model.addConstr(w2.sum(q for q in range(n)) >= 1)
    else: # S-utility
        model.addConstr(u[p] == z.sum(p, [q for q in range(n)], [u for u in range(n)], [v for v in range(n)]))

def envyFreenessStability(p,q):
    name1 = "exchange_p_"+str(p)
    exchange_p = model.addVars(n, n, n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name=name1)
    for r in range(n):
        for u_1 in range(n):
            for v in range(n):
                if r == q:
                    model.addConstr(exchange_p[r,u_1,v] == y[q,p,u_1,v]*valuations[p][q])
                    #exchange_p[r,u_1,v] = y[q,p,u_1,v]*valuations[p][q]
                elif r == p:
                    model.addConstr(exchange_p[r,u_1,v] == 0)
                    #exchange_p[r,u_1,v] = 0
                else:
                    model.addConstr(exchange_p[r,u_1,v] == y[q,r,u_1,v]*valuations[p][r])
                    #exchange_p[r,u_1,v] = y[q,r,u_1,v]*valuations[p][r]
    if utilityType == 'B':
        exchange_p_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_p_util")
        bUtility(exchange_p, exchange_p_util)
    elif utilityType == 'W':
        exchange_p_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_p_util")
        wUtility(exchange_p, exchange_p_util)
    else: # S-utility
        exchange_p_util = exchange_p.sum([r for r in range(n)], [u_1 for u_1 in range(n)], [v for v in range(n)])
    
    name1 = "exchange_q_"+str(q)
    exchange_q = model.addVars(n, n, n, vtype=GRB.INTEGER, lb=minValuation, ub=maxValuation, name=name1)
    for r in range(n):
        for u_1 in range(n):
            for v in range(n):
                if r == p:
                    model.addConstr(exchange_q[r,u_1,v] == y[p,q,u_1,v]*valuations[q][p])
                    #exchange_q[r,u_1,v] = y[p,q,u_1,v]*valuations[q][p]
                elif r == q:
                    model.addConstr(exchange_q[r,u_1,v] == 0)
                    #exchange_p[r,u_1,v] = 0
                else:
                    model.addConstr(exchange_q[r,u_1,v] == y[p,r,u_1,v]*valuations[q][r])
                    #exchange_q[r,u_1,v] = y[p,r,u_1,v]*valuations[q][r]
    if utilityType == 'B':
        exchange_q_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_q_util")
        bUtility(exchange_p, exchange_q_util)
    elif utilityType == 'W':
        exchange_q_util = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="exchange_q_util")
        wUtility(exchange_p, exchange_q_util)
    else: # S-utility
        exchange_q_util = exchange_q.sum([r for r in range(n)], [u_1 for u_1 in range(n)], [v for v in range(n)])
    if objective == 'EFA':
        model.addConstr(exchange_p_util <= u[p])
        model.addConstr(exchange_q_util <= u[q])
    else: #STA
        e_p = model.addVar(vtype=GRB.BINARY, name="e_p")
        model.addConstr(e_p >= (exchange_p_util - u[p])/absMaxutility)
        model.addConstr(e_p <= 1 - (u[p] - exchange_p_util - epsilon)/absMaxutility)
        e_q = model.addVar(vtype=GRB.BINARY, name="e_q")
        model.addConstr(e_q >= (exchange_q_util - u[q])/absMaxutility)
        model.addConstr(e_q <= 1 - (u[q] - exchange_q_util - epsilon)/absMaxutility)
        model.addConstr(e_p + e_q <= 1)

if objective == 'MWA':
    model.setObjective(u.sum(p for p in range(n)), GRB.MAXIMIZE)
elif objective == 'MUA':
    minVal = model.addVar(vtype=GRB.INTEGER, lb=minUtility, ub=maxUtility, name="minVal")
    model.addConstrs(minVal <= u[p] for p in range(n))
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
    line += str(u[p].X)+','
print(line)

#for v in model.getVars():
#    print('v')
#    print(f"{v.VarName} {v.X:g}")