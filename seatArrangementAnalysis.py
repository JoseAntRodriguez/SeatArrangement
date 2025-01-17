import re
import numpy as np
from os import listdir

def is_number(s):
    pattern = re.compile(r'^-?\d+(\.\d+)?$')
    return bool(pattern.match(s))

directories = []
objective = ''
utility = ''
interest = ''
outputFile = ''

files = []
sizes = []
data = {}
output = {}
models = []

for directory in directories:
    newFiles = listdir(directory)
    for dataFile in newFiles:
        if objective in dataFile and utility in dataFile and interest in dataFile:
            files.append(directory+'/'+dataFile)
            size = int(dataFile.split('_')[1])
            if size not in sizes:
                sizes.append(size)
sizes.sort()

for i in range(len(files)):
    with open(files[i], 'r')as f:
        lines = f.read().splitlines()
        if is_number(lines[0]) == False:
            newModels = lines[0].split(',')
            for model in newModels:
                if model not in models:
                    models.append(model)
        else:
            models.append('data')
models.sort()

for model in models:
    data[model] = {}
    output[model] = {}
    for size in sizes:
        data[model][size] = []
        output[model][size] = ''

for i in range(len(files)):
    with open(files[i], 'r')as f:
        size = int(files[i].split('_')[1])
        lines = f.read().splitlines()
        newModels = []
        if is_number(lines[0]) == False:
            newModels = lines[0].split(',')
            lines = lines[1:]
        else:
            newModels = ['data']
        for j in range(len(lines)):
            splitLine = lines[j].split(',')
            splitList = []
            for k in range(len(splitLine)):
                data[newModels[k]][size].append(float(splitLine[k]))

for model in models:
    for size in sizes:
        if len(data[model][size]) > 0:
            average = sum(data[model][size])/len(data[model][size])
            output[model][size] = str(round(average, 2))

with open(outputFile, 'w')as f:
    header = 'sizes,'+','.join(models)+'\n'
    f.write(header)
    for i in range(len(sizes)):
        line = str(sizes[i])+','
        for j in range(len(models)):
            line += output[models[j]][sizes[i]]
            if j < len(models) - 1:
                line += ','
        line += '\n'
        f.write(line)