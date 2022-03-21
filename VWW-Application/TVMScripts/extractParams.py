# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#extracting model weights and parameters from tflite file

#im2col if already padded/padding = 0
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



######################################################################
# Utils for downloading and extracting zip files
# ----------------------------------------------
import os

def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)


######################################################################
# Load pretrained TFLite model
# ----------------------------
# Load mobilenet V1 TFLite model provided by Google
from tvm.contrib.download import download_testdata

model_dir = '/home/melina/Documents/TrainedModel/PersonDetect-1000000-IP96'

# Now we can open tflite file
tflite_model_file = os.path.join(model_dir, "person_detection_model.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Load a test image
# -----------------
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
image_path = '/home/melina/Documents/TFLite-Pose/people.jpg'
image_path = '/home/melina/Documents/TFLite-Pose/house.jpeg'

#Image preprocessing
resized_image = Image.open(image_path).resize((96, 96))
plt.imshow(resized_image)
plt.show()
image_data = np.asarray(resized_image).astype("int8")

# Add a dimension to the image so that we have NHWC format layout
image_data = np.expand_dims(image_data, axis=0)
#make image grayscale
image_data = (0.21 * image_data[:,:,:,:1]) + (0.72 * image_data[:,:,:,1:2]) + (0.07 * image_data[:,:,:,-1:])

#pad image
image_data_padded = np.ones((1,97,97,1))*(-2)
image_data_padded[:,:-1,:-1,:] = image_data
print("input", image_data_padded.shape)
image_data_int8 = image_data_padded.astype("int8")
print("image", image_data_int8[0,:,:,0])
image_data_int8 = im2col(image_data_int8, 3, stride=2)
print("image flattend", image_data_int8)

######################################################################
# Compile the model with relay
# ----------------------------

# TFLite input tensor name, shape and type
input_tensor = "input"
input_shape = (1, 96, 96, 1)
input_dtype = "int8"

# Parse TFLite model and convert it to a Relay module
from tvm import relay, transform
from pathlib import Path

directory_params = '/home/melina/Documents/TrainedModel/PersonDetect-1000000-IP96/params'
Path(directory_params).mkdir(parents=True, exist_ok=True)

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

mod1 = relay.transform.InferType()(mod)
mod = relay.qnn.transform.CanonicalizeOps()(mod1)

for k in params.keys():
    print(k, params[k].shape)
    if k == '_param_5':
        print(params[k].numpy()[:,:,:,0])
        #print(params[k].numpy()[:,:,:,:].reshape(9,8))
    file = directory_params+ '/' + k + '.txt'
    #print(file)
    parameters = params[k].numpy()
    #print(parameters.ndim)
    #print(k, params[k].shape)
    if parameters.ndim == 4:
        shape = (parameters.shape)
        parameters = np.reshape(parameters,(shape[0]*shape[1],shape[2]*shape[3]))
        #print(parameters.shape)
    else:
        print(parameters.shape)
        parameters = parameters[np.newaxis, :]
        print(parameters.shape)
        #parameters = parameters.reshape(parameters,(1,shape[0]))
        #print(parameters.shape)
    np.savetxt(file, parameters, delimiter=',', fmt="%d")

#save input data to file
file = directory_params+ '/input.txt'
np.savetxt(file, image_data_int8.transpose(), delimiter=',', fmt="%d")
file = directory_params+ '/image.txt'
np.savetxt(file, image_data.reshape(96,96).astype("int8"), delimiter=',', fmt="%d")
# Build the module against to x86 CPU
target = "llvm"
with transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)


#############################################################
# Try using debugger
import tvm
from tvm import te
from tvm.contrib import graph_executor as runtime
from tvm.contrib.debugger import debug_executor as runtime

print(type(lib["default"]))
dev = tvm.cpu(0)

from tvm.contrib import graph_executor

dtype = "int8"
#module = graph_executor.GraphModule(lib["default"](dev))
module = runtime.GraphModuleDebug(lib["debug_create"]("default", dev), [dev], lib.get_graph_json(), "./tmp/" + "/test_")
module.set_input(input_tensor, tvm.nd.array(image_data.astype(dtype)))
module.run()
tvm_output = module.get_output(0).numpy()
print(tvm_output)

data = relay.load_param_dict(bytearray(open("./tmp"  + "/test_" + "/_tvmdbg_device_CPU_0/output_tensors.params", "rb").read()))
for keys in data.keys():
    data_shape = data[keys].shape
    if len(data_shape)>3:
        if data_shape[3] < 9 and data_shape[2] < 49 and data_shape[1] > 47:
            print(data[keys].shape)
            print(keys)
            data_k = data[keys].numpy().reshape(-1)
            np.savetxt('./tmp' + "/test_" + '/' + str(keys) + '_.txt', [data_k], delimiter=',', fmt="%d")
######################################################################
# Execute on TVM
# --------------
# import tvm
# from tvm import te
# from tvm.contrib import graph_executor as runtime

# # Create a runtime executor module
# module = runtime.GraphModule(lib["default"](tvm.cpu()))

# # Feed input data
# module.set_input(input_tensor, tvm.nd.array(image_data))

# # Run
# module.run()

# # Get output
# tvm_output = module.get_output(0).numpy()

######################################################################
# Display results
# ---------------

# Load label file
# label_file_url = "".join(
#     [
#         "https://raw.githubusercontent.com/",
#         "tensorflow/tensorflow/master/tensorflow/lite/java/demo/",
#         "app/src/main/assets/",
#         "labels_mobilenet_quant_v1_224.txt",
#     ]
# )
# label_file = "labels_mobilenet_quant_v1_224.txt"
# label_path = download_testdata(label_file_url, label_file, module="data")

# # List of 1001 classes
# with open(label_path) as f:
#     labels = f.readlines()

# # Convert result to 1D data
# predictions = np.squeeze(tvm_output)

# # Get top 1 prediction
# prediction = np.argmax(predictions)

# # Convert id to class name and show the result
# print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])


# #############################################################################################
# visualize the exact execution flow

import graphviz
import numpy
import tvm


# import pylab
def visualize(expr, collapse_small=True, node_attr_dict = {}):
    def collect_ops(node):
        ops = set()
        def visitor(e):
            if isinstance(e, tvm.ir.Op):
                ops.add(e.name)
        tvm.relay.analysis.post_order_visit(node, visitor)
        return ops

    # node_dict maps a Relay node to an index (node ID)
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

    node_dict = {}
    tvm.relay.analysis.post_order_visit(expr, lambda x: _traverse_expr(x, node_dict))

    relayviz_nodes = []

    dot = graphviz.Digraph(format='svg', )
    dot.attr('node', shape = 'box')

    def to_str(node):
        if isinstance(node, tvm.relay.Constant):
            return repr(node).lstrip('Constant(')[:-1]
        else:
            raise NotImplementedError("to_str:" + repr(node))

    def is_small_const(c):
        if not (collapse_small and isinstance(c, tvm.relay.Constant)):
            return False
        if isinstance(c.data, tvm.runtime.ndarray.NDArray):
            return numpy.prod(c.data.shape) < 10
        return True

    # Sort by node ID
    for node, node_id in sorted(node_dict.items(), key=lambda x: x[1]):
        # print('Node:')
        # print(node)
        if isinstance(node, tvm.relay.Function):
            dot.node(str(node_id), 'Function', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.body]), str(node_id))
        elif isinstance(node, tvm.relay.Var):
            if node.type_annotation is not None:
                if hasattr(node.type_annotation, 'shape'):
                    shape = tuple([int(x) for x in node.type_annotation.shape])
                    dtype = node.type_annotation.dtype
                    typstr = 'Tensor[{}, {}]'.format(shape, dtype)
                else:
                    typstr = str(node.type_annotation)
            else:
                typstr = '?'
            d = dict(shape = 'ellipse')
            d.update(node_attr_dict.get(node, {}))
            dot.node(str(node_id),
                     '{}: {}'.format(
                         node.name_hint, typstr
                     ), **d)
        elif isinstance(node, tvm.relay.Tuple):
            dot.node(str(node_id), 'Tuple[...])', **node_attr_dict.get(node, {}))
            for field in node.fields:
                dot.edge(str(node_dict[field]), str(node_id))
        elif isinstance(node, tvm.relay.Constant):
            if str(node.data.dtype) == 'int64':
                print(node)
                print(str(node_id), 'Constant({}, {})'.format(node.data.shape, node.data.dtype),
                        **node_attr_dict.get(node, {}))
                print(node.data)
            if not is_small_const(node): # small consts are shown in ops
                dot.node(str(node_id), 'Constant({}, {})'.format(node.data.shape, node.data.dtype),
                        **node_attr_dict.get(node, {}))
                
        elif isinstance(node, tvm.relay.Call):
            args_with_edge = []
            arg_str_list = []
            for arg in node.args:
                if is_small_const(arg):
                    arg_str_list.append(to_str(arg))
                else:
                    arg_str_list.append('·')
                    args_with_edge.append(arg)
            arg_str = ', '.join(arg_str_list)
            if isinstance(node.op, tvm.ir.Op):
                name = node.op.name
                attrs = {k:getattr(node.attrs, k) for k in node.attrs.keys()} if hasattr(node.attrs, 'keys') else {}
                #attrs = inspect.getmembers(node.attrs)
                attr_str_list = [k+'='+(str(v) if len(str(v))<20 else "...") for k, v in attrs.items()]
                if attr_str_list:
                    attr_str = '| '+ ', '.join(attr_str_list)
                else:
                    attr_str = ''
            else:
                ops = collect_ops(node)
                if ops:
                    name = '_'.join(ops)
                else:
                    name = '...'
                attr_str = ''
            s = f'{name}({arg_str}{attr_str})'
            dot.node(str(node_id), s, **node_attr_dict.get(node, {}))
            for arg in args_with_edge:
                dot.edge(str(node_dict[arg]), str(node_id))
        elif isinstance(node, tvm.ir.Op):
            # dot.node(str(node_id), 'Op {}'.format(node.name))
            pass # covered in call
        elif isinstance(node, tvm.relay.TupleGetItem):
            dot.node(str(node_id), 'TupleGetItem(idx={})'.format(node.index), **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.tuple_value]), str(node_id))
        elif isinstance(node, tvm.relay.Let):
            dot.node(str(node_id), 'Let(XX)', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.value]), str(node_id))
            dot.edge(str(node_id), str(node_dict[node.var]))
        else:
            raise RuntimeError(
                'Unknown node type. node_id: {}, node: {}'.format(node_id, type(node)))

    return dot
############################################################################################

def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

node_dict = {}
tvm.relay.analysis.post_order_visit(mod['main'], lambda x: _traverse_expr(x, node_dict))

#print(mod['main'])
#vis = visualize(mod['main'])
#vis.view()
