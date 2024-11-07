from minizinc import Instance, Model, Solver
from bruteForce import bruteForce
from IPModelSimple import IPModelSimple
from IPModelQuadraticSimple import IPModelQuadraticSimple
from os import listdir
import numpy as np
import time

def getResultsFromCP(CPModel, valuations, seatGraph):
    model = Model(CPModel)
    model.add_file(valuations)
    model.add_file(seatGraph)

    gecode = Solver.lookup('gecode')
    instance = Instance(gecode, model)
    start_total_time = time.time()
    result = instance.solve()
    start_total_time = time.time() - start_total_time
    output = {}
    output['Total time'] = start_total_time
    output['Build time'] = 0
    output['Solve time'] = start_total_time
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
        f.write(content)

seatGraphDirectory = 'Instances/GraphSmall/'
valuationsDirectory = 'Instances/ValSmall/'
resultsDirectory = 'Results/'
models = ['bruteForce','IP4Subscripts','IPQuadratic','CPN-ary','CP4Subscripts','CPQuadratic']

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
        if 'CP' in model:
            nodesDict[model] = []
    peakDepthDict = {}
    for model in models:
        if 'CP' in model:
            peakDepthDict[model] = []
    
    for i in range(len(valuationsFilesSizeCSV)):
        for j in range(len(valuationsFilesSizeCSV)):
            for model in models:
                output = {}
                if model == 'bruteForce':
                    output = bruteForce(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i])
                elif model == 'IP4Subscripts':
                    output = IPModelSimple(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i])
                elif model == 'IPQuadratic':
                    output = IPModelQuadraticSimple(valuationsFilesSizeCSV[j], seatGraphFilesSizeCSV[i])
                elif model == 'CPN-ary':
                    output = getResultsFromCP('./CPModelN-aryVariablesSimple.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i])
                elif model == 'CP4Subscripts':
                    output = getResultsFromCP('./CPModelBinaryVariablesSimple.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i])
                elif model == 'CPQuadratic':
                    output = getResultsFromCP('./CPModelBinaryVariablesQuadraticSimple.mzn', valuationsFilesSizeDZN[j], seatGraphFilesSizeDZN[i])
                totalTimeDict[model].append(output['Total time'])
                buildTimeDict[model].append(output['Build time'])
                solveTimeDict[model].append(output['Solve time'])
                if 'CP' in model:
                    nodesDict[model].append(output['Nodes'])
                    peakDepthDict[model].append(output['Peak depth'])
                if len(objectiveArray) < 10*(i+1) + (j+1):
                    objectiveArray.append(output['Objective'])
    
    objectiveFilename = 'Results_'+size+'_ObjectiveValue_MWA_S.csv'
    with open(resultsDirectory+objectiveFilename, 'w') as f:
        content = ''
        for i in range(instancesNumber):
            content += str(objectiveArray[i])
            if i < instancesNumber - 1:
                content += ','
        f.write(content)
    
    totalTimeFilename = resultsDirectory+'Results_'+size+'_TotalTime_MWA_S.csv'
    writeFile(models, totalTimeFilename, totalTimeDict, instancesNumber)

    buildTimeFilename = resultsDirectory+'Results_'+size+'_BuildTime_MWA_S.csv'
    writeFile(models, buildTimeFilename, buildTimeDict, instancesNumber)

    solveTimeFilename = resultsDirectory+'Results_'+size+'_SolveTime_MWA_S.csv'
    writeFile(models, solveTimeFilename, solveTimeDict, instancesNumber)

    cpModels = [model for model in models if 'CP' in models]

    nodesFilename = resultsDirectory+'Results_'+size+'_Nodes_MWA_S.csv'
    writeFile(cpModels, nodesFilename, nodesDict, instancesNumber)

    peakDepthFilename = resultsDirectory+'Results_'+size+'_PeakDepth_MWA_S.csv'
    writeFile(cpModels, peakDepthFilename, peakDepthDict, instancesNumber)