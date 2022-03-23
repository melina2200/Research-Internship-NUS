# SPI connection between Manuca Microcontroller and FPGA

The [Hex Conversion Folder](pythonScript-hexConversion) contains the script necessary for the endian conversion.

There are additionally two files that I've used for the Pin Mapping. The [PDF](MA_Evaluation_Board.pdf) contains the layout of the Manuca Microcontroller (Air and Evaluation board) Boards. The [Excel file](PinMapping.xlsx) contains the condensed pin mapping for the SPI Connection.

Documentation of what had been done:

![Roadmap](https://github.com/melina2200/Research-Internship-NUS/blob/main/SPI-Connection-Manuca-FPGA/img/roadmap.png?raw=true)

**(1+2)** The Manuca board is able to output SPI data (MOSI, CLK, CS). As the endian is different to what the FPGA expects it was necessary to write a script to change the endianess of the MOSI output.

Until now the FPGA has been controlled with a QSPI controller directly from a PC. In the next step the FPGA shall be conected to the Manuca Board instead of the PC and receive SPI input from the Manuca board. Currently the logic levels of the FPGA(2.5V) and the Manuca Board(5V or 3.3V) are different and therefore it is necessary to use a level converter. In the future the FPGA shall be changed to another model with a logic level of 3.3V which wouldn't require a seperate level shifter.

In the main script it is specified which ports need to be used on the Manuca board and in the script to get a 5V or 3.3V output.

The following picture shows the setup. The exact PinMapping can be found in the [Excel file](PinMapping.xlsx).

![Roadmap](https://github.com/melina2200/Research-Internship-NUS/blob/main/SPI-Connection-Manuca-FPGA/img/setup.png?raw=true)

In the future the mbed OS setup shall be changed such that the endianess can be set directly in the code and would make the python conversion script superfluous.
