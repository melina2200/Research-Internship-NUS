# SPI connection between Manuca Microcontroller and FPGA

The [Hex Conversion Folder](pythonScript-hexConversion) contains the script necessary for the endian conversion.

There are additionally two files that I've used for the Pin Mapping. The [PDF](MA_Evaluation_Board.pdf) contains the layout of the Manuca Microcontroller (Air and Evaluation board) Boards. The [Excel file](PinMapping.xlsx) contains the condensed pin mapping for the SPI Connection.

Documentation of what had been done:

![Roadmap](https://github.com/melina2200/Research-Internship-NUS/blob/main/SPI-Connection-Manuca-FPGA/img/roadmap.png?raw=true)

**(1+2)** The Manuca board is able to output SPI data (MOSI, CLK, CS). As the endian is different to what the FPGA expects it was necessary to write a script to change the endianess of the MOSI output.

**(3+4)** Until now the FPGA has been controlled with a QSPI controller directly from a PC. In the next step the FPGA shall be conected to the Manuca Board instead of the PC and receive SPI input from the Manuca board. Currently the logic levels of the FPGA(2.5V) and the Manuca Board(5V or 3.3V) are different and therefore it is necessary to use a level converter. In the future the FPGA shall be changed to another model with a logic level of 3.3V which wouldn't require a seperate level shifter.

In the [main script](mbedOS-SPIconnectionSetup/main.cpp) it is specified which ports need to be used on the Manuca board and in the script to get a 5V or 3.3V output.

The binary can be build with [Mbed Studio](https://os.mbed.com/studio/). To combile the code for Manuca Air choose the target NUCLEO-L476RG. Make sure the PeripheralsPins.c file (within this folder: mbed-os/targets/TARGET_STM/TARGET_STM32L4/TARGET_STM32L476xG/TARGET_NUCLEO_L476RG) has been modified as follows to get the correct 3.3V output.

You will need to add the pin names to the PeripheralPins.c file in the same folder, in the arrays under their corresponding SPI function:
```
PinMap_SPI_MOSI
{PG_11, SPI_3, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF6_SPI3)},
PinMap_SPI_MISO
{PG_10, SPI_3, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF6_SPI3)},
PinMap_SPI_SCLK
{PG_9, SPI_3, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF6_SPI3)},
PinMap_SPI_SSEL
{PG_12, SPI_3, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF6_SPI3)},
```

Also make sure all pins are declared in the PinNames.h file in the same folder. (This should already be the case, if not the error message indicated pretty obvious that this file needs to be edited.)


The following picture shows the setup. The exact PinMapping can be found in the [Excel file](PinMapping.xlsx).

![Roadmap](https://github.com/melina2200/Research-Internship-NUS/blob/main/SPI-Connection-Manuca-FPGA/img/setup.png?raw=true)

**(5)** In the future the mbed OS setup shall be changed such that the endianess can be set directly in the code and would make the python conversion script superfluous.

**Further References:**

* [mbedOS SPI](https://os.mbed.com/docs/mbed-os/v6.15/apis/spi.html)
* [mbedOS QSPI](https://os.mbed.com/docs/mbed-os/v6.15/apis/spi-apis.html)
* [Datasheet KC705 Evaluation Board for the Kintex-7 FPGA](https://www.xilinx.com/support/documentation/boards_and_kits/kc705/ug810_KC705_Eval_Bd.pdf)
* [DatasheetFMC XM105 Debug Card](https://www.xilinx.com/support/documentation/boards_and_kits/ug537.pdf)
* [Datasheet QSPI Controller](https://www.micro-semiconductor.com/datasheet/f3-UMFT4222EV.pdf)
