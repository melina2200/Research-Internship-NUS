#ANALYZE NETWORK
#This code extracts information about the padding layers and convolution layer from the layer information extracted by tvm and stores them in a dictionary to condense the information
import json


with open('layers.txt') as f:
    lines = f.readlines()

padding_dict = {}
conv_dict = {}
for line in lines:
    if 'nn.pad' in line:
        pos = int(line.find('%'))
        line = line[pos:]
        #print(line)
        pos = line.find(' ')
        number = line[:int(pos)]
        #print(number)
        pos = line.find('Tensor[')
        line = line[int(pos):]
        #print(line)
        pos1 = line.find('(')
        pos2 = line.find(')')
        shape = line[pos1:pos2+1]
        #print(shape)
        #print(number)
        padding_dict[number] = {}
        padding_dict[number]["shape"] = shape
        pos1 = shape.find(',')
        dim1 = int(shape[1:pos1])
        #print('dim1 ' +dim1)
        pos2 = shape.find(',',pos1+1)
        dim2 = int(shape[pos1+2:pos2])
        #print('dim2 ' +dim2)
        pos1 = shape.find(',',pos2+1)
        dim3 = int(shape[pos2+2:pos1])
        #print('dim3 ' +dim3)
        dim4 = int(shape[pos1+2:-1])
        #print('dim4 ' +dim4)
        padding_dict[number]["dim1"] = dim1
        padding_dict[number]["dim2"] = dim2
        padding_dict[number]["dim3"] = dim3
        padding_dict[number]["dim4"] = dim4

json_string = json.dumps(padding_dict, indent=4)
print(json_string)

for line in lines:
    if 'nn.conv2d' in line:
        pos = int(line.find('%'))
        line = line[pos:]
        pos = line.find(' ')
        number = line[:int(pos)]
        #print(line)
        conv_dict[number] = {}

        pos = line.find('nn.conv2d')
        line = line[int(pos):]
        pos = int(line.find('%'))
        line = line[pos:]
        pos = line.find(',')
        input = line[:int(pos)]
        #print(input)
        conv_dict[number]["input_layer"] = input

        pos = line.find('param_')
        line = line[int(pos):]
        #print(line)
        pos = line.find(',')
        input_param = line[:int(pos)]
        #print(input_param)
        conv_dict[number]["input_param"] = input_param

        if 'strides=' in line:
            pos = line.find('strides=')
            line = line[int(pos)+8:]
            #print(line)
            pos1 = line.find('[')
            pos2 = line.find(']')
            stride = line[pos1:pos2+1]
            #print(stride)
            conv_dict[number]["stride"] = stride
            pos = stride.find(',')
            stride1 = int(stride[1:pos])
            #print(stride1)
            stride2 = int(stride[pos+2:-1])
            #print(stride2)
            conv_dict[number]["stride_dim1"] = stride1
            conv_dict[number]["stride_dim2"] = stride2

        pos = line.find('channels=')
        line = line[int(pos)+9:]
        #print(line)
        pos = line.find(',')
        channels = line[:int(pos)]
        #print(channels)
        conv_dict[number]["channels"] = channels

        pos = line.find('kernel_size=')
        line = line[int(pos)+12:]
        #print(line)
        pos1 = line.find('[')
        pos2 = line.find(']')
        kernel = line[pos1:pos2+1]
        #print(kernel)
        pos = kernel.find(',')
        kernel1 = int(kernel[1:pos])
        #print(kernel1)
        kernel2 = int(kernel[pos+2:-1])
        ##print(kernel2)
        conv_dict[number]["kernel"] = kernel
        conv_dict[number]["kernel_dim1"] = kernel1
        conv_dict[number]["kernel_dim2"] = kernel2

        pos = line.find('Tensor[')
        line = line[int(pos):]
        pos1 = line.find('(')
        pos2 = line.find(')')
        output_shape = line[pos1:pos2+1]
        #print(output_shape)
        conv_dict[number]["output_shape"] = output_shape

        #print("________")

json_string = json.dumps(conv_dict, indent=4)
print(json_string)

