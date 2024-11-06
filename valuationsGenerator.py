import random
import numpy as np
import sys

def createValuations(n, options):
    valuations = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            valuation = 0
            if 'binary' in options:
                valuation = random.randint(0,1)
            else:
                valuation = random.randint(-100,100)
            if (i != j and ('symmetric' not in options or i > j)):
                if 'symmetric' in options:
                    if i > j:
                        valuations[i][j] = valuation
                        valuations[j][i] = valuation
                else:
                    valuations[i][j] = valuation
    if 'big' in options:
        bigValuations = random.randint(1,n)
        for k in range(bigValuations):
            i = random.randint(0,n-1)
            j = i
            while j == i:
                j = random.randint(0,n-1)
            valuations[i][j] = random.randint(900,1100)
            if 'symmtric' in options:
                valuations[j][i] = valuations[i][j]
    return valuations

def main(n, filename, options):
    valuations = createValuations(n,options)

    with open(filename+'.csv', 'w') as f:
        for i in range(n):
            line = ''
            for j in range(n):
                line += str(int(valuations[i][j]))
                if j < n-1:
                    line += ','
            if i < n-1:
                line += '\n'
            f.write(line)
    
    with open(filename+'.dzn', 'w') as f:
        f.write('k='+str(n)+';\n')
        f.write('V = array2d(n,n,[')
        for i in range(n):
            line = ''
            for j in range(n):
                line += str(int(valuations[i][j]))
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