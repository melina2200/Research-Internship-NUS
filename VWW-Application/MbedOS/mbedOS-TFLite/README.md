# TensorFlow Lite Mbed Implementation

This folder contains the TFLite implementation of the VWW application.

**How to generate the binary**

* Download [mbedOS](https://os.mbed.com/mbed-os/) and the respective [compiler toolchain](https://os.mbed.com/docs/mbed-os/v6.15/build-tools/install-and-set-up.html). I used GCC ARM to generate the TFLite binary with the following commands in the terminal.


The first command is used to set the current folder as the build folder.
```
mbed config root .
```
The next command will deploy the mbedOS folder. Make sure to change the PeripheralPins file as described above before compiling.
```
mbed deploy
```
Now compile the current script for Manuca Board (NUCLEO_F767ZI) with the GCC compiler:
```
mbed compile -m NUCLEO_F767ZI -t GCC_ARM
```
**How to modify the code**

The entry point for the application is the [main.cc](tensorflow/lite/micro/examples/person_detection/main.cc) script. It calls the setup and loop function. The setup function initializes the model as well as the needed functions from the TFLite library to run the model. It builds an interpreter to run the model with and allocates memory to the TensorArena. The loop function loads the image to classify and evaluates it based on the ML model. It then displays the results.
