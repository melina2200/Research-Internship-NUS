# Hex Conversion for SPI connection

As the endian of the Manuca Microcontroller board and the mbed library used for this board and the FPGA aren't the same, it is necessary to changed the order of the bits that are being send via SPI and therefore convert the HEX values accordingly.

**Example:**

HEX:                0xFD        0x58

in BIT pattern:     1111 1101   0101 1000

change order:       1011 1111   0001 1010

converted to HEX:   0xBF        0x1A

The [python file](hexConversion.py) contains the conversion script itself. One needs to specifiy the file to read from and the file to write to within the script. The other files are example files that can be used/have been used for the conversion.
