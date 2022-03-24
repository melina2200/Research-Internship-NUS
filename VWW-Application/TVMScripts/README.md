# TVM Scripts

These scripts are used for extracting model parameters from the trained TFLite model with the help of TVM. To run these scripts it is necessary to install TVM.

1. Install TVM: https://tvm.apache.org/docs/install/from_source.html 

2. Create a CONDA environment and export Python and TVM accordingly

3. Run this example to make sure TVM is installed correctly and you can run Tensorflow: https://tvm.apache.org/docs/how_to/compile_models/from_tflite.html#sphx-glr-how-to-compile-models-from-tflite-py

As soon as you finished these steps you can use the scripts to extract the model parameters.

The [extractParams script](extractParams.py) is based on the from_tflite.py script provided by TVM and is used to extract the model weights.
It can also be used to run the model with TVM and a given test image and extract interim results if the model that can be used as a reference/control result for the C Implementation. 
Furthermore it allows the display of the detailed layer structure in an svg file.

The [checkInput script](checkInput.py) can be used to compare results, e.g. from the TVM execution and the inference of the model of the C Implementation

The [analyzeNetwork script](analyzeNetwork.py) can be used to extract information about the padding and convolution layers from the [layers dictionary](layers.txt).

The [analyzeConstants script](analyzeConstants.py) can be used to extract the correct qunatization parameters for multiplying and adding from the [quantizeConstants txt file](quantizeConstants.txt) and save them to the [quantizeParams txt file](quantizeParams.txt).

All the txt files in this folder are copied from various outputs generated by the [checkInput script](checkInput.py).