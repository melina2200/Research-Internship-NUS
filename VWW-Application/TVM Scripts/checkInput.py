#This script can be used to compare results from the C implementation with results from the TVM implementation

import numpy as np


def im2col(x, weight_dim, stride=1):
    # First figure out what the size of the output should be
    x = x.transpose(0, 3,1,2)
    N, C, H, W = np.shape(x)
    assert (H - weight_dim) % stride == 0
    assert (W  - weight_dim) % stride == 0
    out_height = int((H - weight_dim) / stride + 1)
    out_width = int((W - weight_dim) / stride + 1)
    print("out_height, out_width")
    print(out_height, out_width)
    i0 = np.repeat(np.arange(weight_dim), weight_dim)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(weight_dim), weight_dim * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), weight_dim * weight_dim).reshape(-1, 1)
    cols = x[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(weight_dim * weight_dim * C, -1)
    return cols


#COMPARE INPUT
with open('/home/melina/Documents/TVM-weightExtraction/tvm/tmp/test_/input____topo-index:0____output-num:0_.txt') as f:
    lines = f.readlines()
    print(len(lines))
    input = lines[0].split(',')
    my_array = np.reshape(np.asarray(input, np.int8), (1,96,96,1))
    print(np.shape(my_array))


from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
image_path = '/home/melina/Documents/TFLite-Pose/people.jpg'

resized_image = Image.open(image_path).resize((96, 96))
#plt.imshow(resized_image)
#plt.show()
image_data = np.asarray(resized_image).astype("int8")

# Add a dimension to the image so that we have NHWC format layout
image_data = np.expand_dims(image_data, axis=0)
#1,96,96,1
image_data = (0.21 * image_data[:,:,:,:1]) + (0.72 * image_data[:,:,:,1:2]) + (0.07 * image_data[:,:,:,-1:])
image_data_int8 = image_data.astype("int8")
print(np.shape(image_data_int8))

if (image_data_int8==my_array).all():
    print("True")
else:
    print("False")



#COMPARE PADDING

with open('/home/melina/Documents/TVM-weightExtraction/tvm/tmp/test_/tvmgen_default_fused_nn_pad____topo-index:1____output-num:0_.txt') as f:
    lines = f.readlines()
    print(len(lines))
    input = lines[0].split(',')
    my_array = np.reshape(np.asarray(input, np.int8), (1,97,97,1))
    print(np.shape(my_array))

image_data_padded = np.ones((1,97,97,1))*(-2)
image_data_padded[:,:-1,:-1,:] = image_data
print("input", image_data_padded.shape)
image_data_int8 = image_data_padded.astype("int8")

if (image_data_int8==my_array).all():
    print("True")
else:
    print("False")


#COMPARE AFTER CONV AND QUANT

with open('/home/melina/Documents/TVM-weightExtraction/tvm/tmp/test_/tvmgen_default_fused_nn_conv2d_subtract_add_cast_multiply_add_right_shift_cast_add_clip_cast_cl_1d56134be8a5d0eb_____topo-index:8____output-num:0_.txt') as f:
    lines = f.readlines()
    print(len(lines))
    input = lines[0].split(',')
    my_array = np.reshape(np.asarray(input, np.int32), (1,48,48,8))
    print(np.shape(my_array))
    print(np.max(my_array))


with open('/home/melina/Documents/TVM-weightExtraction/tvm/tmp/test_/tvmgen_default_fused_nn_conv2d_subtract_add_cast_multiply_add_right_shift_cast_add_clip_cast_cl_1d56134be8a5d0eb__1____topo-index:16____output-num:0_.txt') as f:
    lines = f.readlines()
    print(len(lines))
    input = lines[0].split(',')
    my_array3 = np.reshape(np.asarray(input, np.int32), (1,48,48,8))
    print(np.shape(my_array3))
    print(np.max(my_array3))


with open('/home/melina/Documents/TVM-weightExtraction/tvm/tmp/test_/convCPPoutput.txt') as f:
    lines = f.readlines()
    print(len(lines))
    #input = lines[0].split(',')
    my_array2 = np.reshape(np.asarray(lines, np.int32), (2304,8))[:,0].reshape(48,48).transpose()
    print(np.shape(my_array2))


if (my_array2==my_array[:,:,:,0]).all():
    print("True")
else:
    print("False")
    sumIdentical = np.sum(my_array[:,:,:,0] == my_array2)
    print(sumIdentical/(48*48))


if (my_array2==my_array3[:,:,:,0]).all():
    print("True")
else:
    print("False")
    sumIdentical = np.sum(my_array3[:,:,:,0] == my_array2)
    print(sumIdentical/(48*48))

if (my_array[0,:,:,0].transpose()==my_array[0,:,:,0]).all():
    print("True")
else:
    print("False")
    sumIdentical = np.sum(my_array2.transpose() == my_array2)
    print(sumIdentical/(48*48))
