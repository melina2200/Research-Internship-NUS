# Visual Wake Word Application

The Visual Wake Work Application shall detect people in images in a lightweight and efficient manner such that the application can be deployed on a microcontroller. The following image shows the steps that have been necessary to implement this application.

![Roadmap](https://github.com/melina2200/Research-Internship-NUS/blob/main/VWW-Application/img/roadmap.png?raw=true)

**(1)** In the first step the model had to be trained. I used the [COCO Dataset](https://cocodataset.org/#home) for training. The model is based on the [MobileNets](https://arxiv.org/pdf/1704.04861.pdf) model. The [TrainPersonDetect Jupyter Notebook](TrainPersonDetect.ipynb) contains the script to train the model for the Person Detection Algorithm. I used it in Google Colab. Maybe there need to be done some modifications if you run it as a jupyter notebook. The script is based on the Training of the [Person Detection Example of the TF Lite Micro Repo](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/person_detection).

The final ouput of the trained model can be found in the TrainedModel folders for [298.000 Iterations](Trained-Model-298000Iter) and for [1.000.000 Iterations](Trained-Model-1000000Iter). These models have been trained with an input image size of 96x96 Pixels (grayscale).

**(2)** In the second step I had to extract the trained model weight, the bias and the quantization parameters that are needed for the implementation in C of this network.[The TVMScripts folder](TVMScripts) contains all scripts used for extracting model parameters from trained TFLite models with the help of TVM. To run these scripts it is necessary to install TVM, more information in the ReadMe File within that folder.

**(3)** In the third steo I implemented the model in nativeC code. This step is necessary to be able to modify the layer functions for a simpler acceleration with the Hycube CGRA. [The C Implementation folder](CImplementation) contains the implemetation of Visual Wake Word Application in C, all model parameters needed have been extracted from the Trained Model (1000000 Iterations) and stored in header files. 

**(4)** In the next step both implementations (TFLite and nativeC) are deployed onto the Manuca Microcontroller with the help of the [mbedOS Framework](https://os.mbed.com/mbed-os/). 

**(5)** In the future the model shall be accelerated with the HyCube CGRA to achieve even faster and more efficient results. 


## Other Files

[The NodeRepresentation](NodeRepresentation) contains the Layer-Node Representation of the Visual Wake Word Network

The [Person Detection Algorithm Excel File](Person_Detection_Algorithm.xlsx) contains detailed information about the layer structure of the implemented model.


## VWW Model
The Visual Wake Word Model for Person Detection is based on the [MobileNets](https://arxiv.org/pdf/1704.04861.pdf) model. The following picture shows the layer structure of the 31 layers used in this model. 

<img src="https://github.com/melina2200/Research-Internship-NUS/blob/main/VWW-Application/img/layerStructure.png?raw=true" width="350">

The model makes additionally use of qunatization to minimize computational complexity. In the following the layers used in the model will be described in depth.

### Convolution (Depthwise and Pointwise)
The model makes use of depthwise seperable convolution. In this process the 'normal' convolution is seperated in two convolution steps, the depthwise and the pointwise convolution. This separation divides a convolution kernel into two smaller kernels leading to a reduction in multiplications and therefore computational complexity. In the example given below instead of doing 3x3x3 = 27 multiplications we will now do 3x3 = 9 multiplications in the first step and them 3 in the second, leading to a total of 12 multiplications.


![SeparableConv](https://github.com/melina2200/Research-Internship-NUS/blob/main/VWW-Application/img/separableConv.png?raw=true)


### Average Pooling
The Average Pooling layer can be found at the end of the model. It is used to downsample the detection of features in a feature map. Here it is averaged over a 2x2 square in the feature map.

<img src="https://github.com/melina2200/Research-Internship-NUS/blob/main/VWW-Application/img/averagePooling.png?raw=true" width="450">

### Softmax
The Softmax layer is used as last activation function to normalize the logits/numbers output to a probability disctribution.

<img src="https://github.com/melina2200/Research-Internship-NUS/blob/main/VWW-Application/img/softmax.png?raw=true" width="550">

### Quantization

The model used in this project is fully quntized to 8-bit-integer variables. The parameters that stay constant after training and during inference (weights, bias,...) are directly quantized to int8 variables based on the largest and smallest float value. The following image shows a visualization of this prozess. The largest and smallest float vlaue (The values in this visualization are exemplary.) get mapped to 0 (samllest int8 value) and 255 (largest int8) value. The other values get mapped in between.

<img src="https://github.com/melina2200/Research-Internship-NUS/blob/main/VWW-Application/img/int8Quant.png?raw=true" width="550">

Activation values are different in each inference and can therefore not be quantized before the inference. After each layer a seperate quantization step will be done to make sure the resulting values during the matrix multiplication can be saved in an int8 format. For this step it is necessary to determine the parameters needed for the addition, multiplication and shift operation which are all part of the quantization step. These values are determined by running a few infernces (50-100 examples) such that the approximate range of the values can be assessed and the quantization parameters specified. The process of quantization after one (e.g. convolution) layer is visualized in the following image. The quantization parameters are stored in int16 or int32 format.

<img src="https://github.com/melina2200/Research-Internship-NUS/blob/main/VWW-Application/img/int8QuantActivations.png?raw=true" width="900">

