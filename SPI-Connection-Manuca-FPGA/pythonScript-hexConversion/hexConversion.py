import re

#This script converts hex values in a given file as follows and saves the result in a new file. The rest of the file stays identical.

# 0xFD --> 1111 1101 --> 1011 1111 --> 0xBF
# 0x58 --> 0101 1000 --> 0001 1010 --> 0x1A

scale = 16 #equals to hexadecimal
num_of_bits = 8
#file_org = 'array_add.h'
#file_new = 'array_add_new.h'
file_org = 'Hycube_SPI_exec_end_fft_121.ino' #file to read from
file_new = 'Hycube_SPI_exec_end_fft_121_new.ino' #file to write to


#get file content
with open(file_org) as f:
    lines = f.readlines()
    f.close()


#iterate over content
with open(file_new, 'w') as f:
    for line in lines:
        if '0x' in line: #only look at lines that conatain hex values
            result = [_.start() for _ in re.finditer('0x', line)] # find all positions of hex indicator
            print(result)
            for pos in result: #iterate over positions and replace old hex with new hex value (new order)
                hexData = line[pos+2:pos+4] #get hex value
                binaryData = bin(int(hexData, scale))[2:].zfill(num_of_bits)[::-1] #convert hex to bin and reverse order
                #print(binaryData[:4])
                newHexData1 = hex(int(binaryData[:4], 2)) # reconvert reversed binary to hex
                #print(newHexData1)
                newHexData2 = hex(int(binaryData[4:], 2)) # reconvert reversed binary to hex
                line = line[:pos+2] + newHexData1[2] + newHexData2[2] + line[pos+4:] # replace old hex with new one
            f.write(line)
        else: f.write(line)
    f.close()
        

