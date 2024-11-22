from minizinc import Instance, Model, Solver
from bruteForce import bruteForce
from IPModelConstrained import IPModel
from IPModelQuadraticConstrained import IPModelQuadratic
from os import listdir
import numpy as np
import time

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
    start_total_time = time.time()
    result = instance.solve()
    start_total_time = time.time() - start_total_time
    output = {}
    output['Total time'] = start_total_time
    output['Build time'] = 0
    output['Solve time'] = start_total_time
    if goal == 'STA' or goal == 'EFA':
        if result.status.has_solution():
            output['Objective'] = 1
        else:
            output['Objective'] = 0
    else:
        output['Objective'] = result['objective']
    output['Nodes'] = result.statistics['nodes']
    output['Peak depth'] = result.statistics['peakDepth']
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

for size in sizes:
    seatGraphFilesSize = seatGraphFilesBySize[size]
    valuationsFilesSize = valuationsFilesBySize[size]

    seatGraphFilesSizeCSV = [file for file in seatGraphFilesSize if '.csv' in file]
    seatGraphFilesSizeDZN = [file for file in seatGraphFilesSize if '.dzn' in file]

    valuationsFilesSizeCSV = [file for file in valuationsFilesSize if '.csv' in file]
    valuationsFilesSizeDZN = [file for file in valuationsFilesSize if '.dzn' in file]

    instancesNumber = len(seatGraphFilesSizeCSV)*len(valuationsFilesSizeCSV)

    for goal in goals:
        for uType in uTypes:
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
                    print('Instance number', i*len(seatGraphFilesSizeCSV) + j + 1, 'of size', size)
                    for model in models:
                        output = {}
                        if model == 'bruteForce': # Only works for S-utility and MWA
                            output = bruteForce(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i])
                        elif model == 'IP4Subscripts':
                            output = IPModel(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i], uType, goal)
                        elif model == 'IPQuadratic':
                            output = IPModelQuadratic(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i], uType, goal)
                        elif model == 'CPN-ary':
                            output = getResultsFromCP('./CPModelN-aryVariablesConstrained.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i], uType, goal)
                        elif model == 'CP4Subscripts':
                            output = getResultsFromCP('./CPModelBinaryVariablesConstrained.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i], uType, goal)
                        elif model == 'CPQuadratic':
                            output = getResultsFromCP('./CPModelBinaryVariablesQuadraticConstrained.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i], uType, goal)
                        totalTimeDict[model].append(output['Total time'])
                        buildTimeDict[model].append(output['Build time'])
                        solveTimeDict[model].append(output['Solve time'])
                        if 'CP' in model or 'IP' in model:
                            nodesDict[model].append(output['Nodes'])
                        if 'CP' in model:
                            peakDepthDict[model].append(output['Peak depth'])
                        if len(objectiveArray) < len(seatGraphFilesSizeCSV)*i + (j+1):
                            objectiveArray.append(output['Objective'])
            
            lastPartOfFilename = goal + '_' + 'uType' + '.csv'

            objectiveFilename = 'Results_'+size+'_ObjectiveValue_'+lastPartOfFilename
            with open(resultsDirectory+objectiveFilename, 'w') as f:
                content = ''
                for i in range(instancesNumber):
                    content += str(objectiveArray[i])
                    if i < instancesNumber - 1:
                        content += ','
                f.write(content)
            
            totalTimeFilename = resultsDirectory+'Results_'+size+'_TotalTime_'+lastPartOfFilename
            writeFile(models, totalTimeFilename, totalTimeDict, instancesNumber)

            buildTimeFilename = resultsDirectory+'Results_'+size+'_BuildTime_'+lastPartOfFilename
            writeFile(models, buildTimeFilename, buildTimeDict, instancesNumber)

            solveTimeFilename = resultsDirectory+'Results_'+size+'_SolveTime_'+lastPartOfFilename
            writeFile(models, solveTimeFilename, solveTimeDict, instancesNumber)

            cpIpModels = [model for model in models if 'CP' in model or 'IP' in model]
            nodesFilename = resultsDirectory+'Results_'+size+'_Nodes_'+lastPartOfFilename
            writeFile(cpIpModels, nodesFilename, nodesDict, instancesNumber)

            cpModels = [model for model in models if 'CP' in model]
            peakDepthFilename = resultsDirectory+'Results_'+size+'_PeakDepth_'+lastPartOfFilename
            writeFile(cpModels, peakDepthFilename, peakDepthDict, instancesNumber)