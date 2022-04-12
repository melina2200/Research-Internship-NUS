# TensorFlow Lite Mbed Implementation

This folder contains the TFLite implementation of the VWW application.

**How to generate the binary**

* Download [mbed studio](https://os.mbed.com/studio/) and the respective [compiler toolchain](https://os.mbed.com/docs/mbed-os/v6.15/build-tools/install-and-set-up.html). I used GCC ARM to generate the TFLite binary with the following commands in the terminal.


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

