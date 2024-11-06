import random
import numpy as np
import sys

def createGraph(n, options):
    if 'cycle' in options:
        return createCycleGraph(n)
    if 'path' in options:
        return createPathGraph(n)
    else:
        return createRandomGraph(n)

def createCycleGraph(n):
    vertices = [i for i in range(n)]
    graph = np.zeros((n,n))
    while len(vertices) > 0:
        cycleVertices = []
        if len(vertices) <= 5:
            cycleVertices = vertices
        else:
            validNumberVertices = [i for i in range(3,len(vertices)-2)]
            validNumberVertices.append(len(vertices))
            nVertices = random.sample(validNumberVertices,1)[0]
            cycleVertices = vertices[:nVertices]
        for i in range(len(cycleVertices)):
            j = (i+1) % len(cycleVertices)
            graph[cycleVertices[i]][cycleVertices[j]] = 1
            graph[cycleVertices[j]][cycleVertices[i]] = 1
        del vertices[:len(cycleVertices)]
    return graph

'''
def createCycleGraphAlternative(n):
    vertices = [i for i in range(n)]
    graph = np.zeros((n,2))
    while len(vertices) > 0:
        cycleVertices = []
        if len(vertices) <= 5:
            cycleVertices = vertices
        else:
            validNumberVertices = [i for i in range(3,len(vertices)-2)]
            validNumberVertices.append(len(vertices))
            nVertices = random.sample(validNumberVertices,1)[0]
            cycleVertices = vertices[:nVertices]
        graph[cycleVertices[0]][0] = cycleVertices[1]
        graph[cycleVertices[0]][1] = cycleVertices[len(cycleVertices)-1]
        graph[cycleVertices[len(cycleVertices)-1]][0] = cycleVertices[len(cycleVertices)-2]
        graph[cycleVertices[len(cycleVertices)-1]][1] = cycleVertices[0]
        for i in range(1,len(cycleVertices)-1):
            graph[cycleVertices[i]][0] = cycleVertices[i-1]
            graph[cycleVertices[i]][1] = cycleVertices[i+1]
        del vertices[:len(cycleVertices)]
    return graph
'''

def createCycleGraphAlternative(graph):
    n = graph.shape[0]
    graphAlternative = np.zeros((n,2))
    for i in range(n):
        assigned = 0
        for j in range(n):
            if graph[i][j] == 1:
                graphAlternative[i][assigned] = j
                assigned += 1
    return graphAlternative

def createPathGraph(n):
    vertices = [i for i in range(n)]
    graph = np.zeros((n,n))
    while len(vertices) > 0:
        pathVertices = []
        if len(vertices) <= 4:
            pathVertices = vertices
        else:
            validNumberVertices = [i for i in range(2,len(vertices)-1)]
            validNumberVertices.append(len(vertices))
            nVertices = random.sample(validNumberVertices,1)[0]
            pathVertices = vertices[:nVertices]
        for i in range(len(pathVertices)-1):
            j = i+1
            graph[pathVertices[i]][pathVertices[j]] = 1
            graph[pathVertices[j]][pathVertices[i]] = 1
        del vertices[:len(pathVertices)]
    return graph

def createRandomGraph(n):
    graph = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            adjacentNumber = random.randint(0,1)
            graph[i][j] = adjacentNumber
            graph[j][i] = adjacentNumber
    return graph


def main(n, filename, options):
    graph = createGraph(n,options)

    with open(filename+'.csv', 'w') as f:
        for i in range(n):
            line = ''
            for j in range(n):
                line += str(int(graph[i][j]))
                if j < n-1:
                    line += ','
            if i < n-1:
                line += '\n'
            f.write(line)
    
    with open(filename+'.dzn', 'w') as f:
        f.write('G = array2d(n,n,[')
        for i in range(n):
            line = ''
            for j in range(n):
                line += str(int(graph[i][j]))
                line += ','
            f.write(line)
        f.write(']);')
    
    if 'cycle' in options:
        graphAlternative = createCycleGraphAlternative(graph)
        with open(filename+'Alternative.csv', 'w') as f:
            for i in range(n):
                line = ''
                for j in range(2):
                    line += str(int(graphAlternative[i][j]))
                    if j < 1:
                        line += ','
                if i < n-1:
                    line += '\n'
                f.write(line)
        
        with open(filename+'Alternative.dzn', 'w') as f:
            f.write('G = array2d(n,0..1,[')
            for i in range(n):
                line = ''
                for j in range(2):
                    line += str(int(graphAlternative[i][j]))
                    line += ','
                f.write(line)
            f.write(']);')

if __name__ == '__main__':
    n = int(sys.argv[1])
    filename = sys.argv[2]
    options = []
    for i in range(3, len(sys.argv)):
        options.append(sys.argv[i])
    main(n, filename, options)