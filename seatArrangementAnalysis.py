import re
import numpy as np

def is_number(s):
    pattern = re.compile(r'^-?\d+(\.\d+)?$')
    return bool(pattern.match(s))

files = []
sizes = []
data = {}
outputFile = ''

models = []

for i in range(len(files)):
    with open(files[i], 'r')as f:
        lines = f.read().splitlines()
        if is_number(lines[0]) == False:
            models = lines[0].split(',')
            lines = lines[1:]
        else:
            models.append('data')
        fileData = np.zeros((len(lines), len(lines[0].split(','))))
        for j in range(len(lines)):
            splitLine = lines[j].split(',')
            splitList = []
            for k in range(len(splitLine)):
                fileData[j,k] = float(splitLine[k])
        data[sizes[i]] = fileData

output = np.zeros((len(sizes), len(models)))

for i in range(len(sizes)):
    for j in range(len(models)):
        output[i,j]  = np.average(data[sizes[i]][:,j])

with open(outputFile, 'w')as f:
    header = 'sizes,'+','.join(models)+'\n'
    f.write(header)
    for i in range(len(sizes)):
        line = str(sizes[i])+','
        for j in range(len(models)):
            line += str(output[i,j])
            if j < len(models) - 1:
                line += ','
        line += '\n'
        f.write(line)