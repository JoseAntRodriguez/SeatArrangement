from minizinc import Instance, Model, Solver
from bruteForce import bruteForce
from IPModel import IPModel
from IPModelQuadratic import IPModelQuadratic
from os import listdir

def getResultsFromCP(CPModel, valuations, seatGraph, utilityType, goal):
    model = Model(CPModel)
    model.add_file(valuations)
    model.add_file(seatGraph)

    gecode = Solver.lookup('gecode')
    instance = Instance(gecode, model)
    if utilityType == 'B':
        instance["uType"] = 2
    elif utilityType == 'W':
        instance["uType"] = 3
    else:
        instance["uType"] = 1
    if goal == 'MWA':
        instance["objective"] = 1
    elif goal == 'MUA':
        instance["objective"] = 2
    elif goal == 'EFA':
        instance["objective"] = 3
    else:
        instance["objective"] = 4
    result = instance.solve()
    output = {}
    if goal == 'STA' or goal == 'EFA':
        if result.status.has_solution():
            output['Objective'] = 1
        else:
            output['Objective'] = 0
    else:
        output['Objective'] = result['objective']
    return output

def divideFilesBySize(files, directory):
    sizes = []
    for file in files:
        elements = file.split('_')
        size = elements[1]
        if size not in sizes:
            sizes.append(size)
    filesBySizes = {}
    for size in sizes:
        filesBySizes[size] = []
    for file in files:
        elements = file.split('_')
        size = elements[1]
        filesBySizes[size].append(directory+file)
    return filesBySizes

def writeFile(models, filename, data, instancesNumber):
    with open(filename, 'w') as f:
        line = ''
        for i in range(len(models)):
            line += models[i]
            if i < len(models) - 1:
                line += ','
            else:
                line += '\n'
        f.write(line)

        for i in range(instancesNumber):
            line = ''
            for j in range(len(models)):
                line += str(data[models[j]][i])
                if j < len(models) - 1:
                    line += ','
                else:
                    line += '\n'
            f.write(line)

seatGraphDirectory = 'Instances/ArbitraryGraphs/'
valuationsDirectory = 'Instances/ArbitraryValuations/'
resultsDirectory = 'Results/'
models = ['IP4Subscripts','IPQuadratic','CPN-ary','CP4Subscripts','CPQuadratic']
goals = ['MWA', 'MUA', 'EFA', 'STA']
uTypes = ['B', 'S', 'W']

seatGraphFiles = listdir(seatGraphDirectory)
valuationsFiles = listdir(valuationsDirectory)

seatGraphFilesBySize = divideFilesBySize(seatGraphFiles, seatGraphDirectory)
valuationsFilesBySize = divideFilesBySize(valuationsFiles, valuationsDirectory)
sizes = seatGraphFilesBySize.keys()

for goal in goals:
    for uType in uTypes:
        print('Goal:', goal, "Utility type:", uType)
        for size in sizes:
            seatGraphFilesSize = seatGraphFilesBySize[size]
            valuationsFilesSize = valuationsFilesBySize[size]

            seatGraphFilesSizeCSV = [file for file in seatGraphFilesSize if '.csv' in file]
            seatGraphFilesSizeDZN = [file for file in seatGraphFilesSize if '.dzn' in file]

            valuationsFilesSizeCSV = [file for file in valuationsFilesSize if '.csv' in file]
            valuationsFilesSizeDZN = [file for file in valuationsFilesSize if '.dzn' in file]

            instancesNumber = len(seatGraphFilesSizeCSV)*len(valuationsFilesSizeCSV)

            objectiveArray = []
            totalTimeDict = {}
            for model in models:
                totalTimeDict[model] = []
            buildTimeDict = {}
            for model in models:
                buildTimeDict[model] = []
            solveTimeDict = {}
            for model in models:
                solveTimeDict[model] = []
            nodesDict = {}
            for model in models:
                if 'CP' in model or 'IP' in model:
                    nodesDict[model] = []
            peakDepthDict = {}
            for model in models:
                if 'CP' in model:
                    peakDepthDict[model] = []
            
            for i in range(len(seatGraphFilesSizeCSV)):
                for j in range(len(valuationsFilesSizeCSV)):
                    objectives = []
                    for model in models:
                        output = {}
                        if model == 'IP4Subscripts':
                            output = IPModel(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i], uType, goal)
                        elif model == 'IPQuadratic':
                            output = IPModelQuadratic(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i], uType, goal)
                        elif model == 'CPN-ary':
                            output = getResultsFromCP('./CPModelN-aryVariables.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i], uType, goal)
                        elif model == 'CP4Subscripts':
                            output = getResultsFromCP('./CPModelBinaryVariables.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i], uType, goal)
                        elif model == 'CPQuadratic':
                            output = getResultsFromCP('./CPModelBinaryVariablesQuadratic.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i], uType, goal)
                        objectives.append(output['Objective'])
                    for k in range(len(objectives)):
                        if objectives[k] != objectives[0]:
                            print('Instance number', i*len(seatGraphFilesSizeCSV) + j + 1, 'of size', size)
                            print(objectives)
                            break