#ANALYZE CONSTANTS
#this code uses the extracted information about constants used for quantization in the network and changes their format such that it will be easier to copy them into the C header

with open('quantizeConstants.txt') as f:
    lines = f.readlines()
i=-1
for line in lines:
    if line.startswith('[') or line.startswith(' '):
        if line.startswith('['):
            i+=1
            constant = []
            line = line[1:]
        for item in line.split(' '):
            if ']' in item:
                item = item[:-2]
            elif '\n' in item:
                item = item[:-1]
            if item != '':
                constant.append(int(item))
        if ']' in line:
            print(i)
            print(constant)
            print('length: ' +str(len(constant)))


