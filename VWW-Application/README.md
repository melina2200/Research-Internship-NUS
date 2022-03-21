# Visual Wake Word Application

## C Implementation

[This folder](CImplementation) contains the implemetation of Visual Wake Word Application in C, all model parameters needed have been extracted from the Trained Model (1000000 Iterations) and stored in header files

## TVM Scripts

[This folder](TVMScripts) contains all scripts used for extracting model parameters from trained TFLite models with the help of TVM. To run these scripts it is necessary to install TVM, more information in the ReadMe File within that folder.

## Node Representation

[This folder](NodeRepresentation) contains the Layer-Node Representation of the Visual Wake Word Network

## Trained Model 298000Iter

[This folder](Trained-Model-298000Iter) contains the Model files of the Visual Wakeword Application trained for 298,000 iterations with input size 96x96

## Trained Model 1000000Iter

 [This folder](Trained-Model-1000000Iter) contains the Model files of the Visual Wakeword Application trained for 1,000,000 iterations with input size 96x96


## other Files

The [Person Detection Algorithm Excel File](Person_Detection_Algorithm.xlsx) contains detailed information about the layer structure of the implemented model.

The [TrainPersonDetect Jupyter Notebook](TrainPersonDetect.ipynb) contains the script to train the model for the Person Detection Algorithm. I used it in Google Colab. Maybe there need to be done some modifications if you run it as a jupyter notebook.
