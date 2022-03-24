#include "mbed.h"
#include "LSM9DS1.h"
#include "mbed_power_mgmt.h"

#include <math.h>
#include <time.h>
#include <chrono>
#include <deque>
#include <SPI.h>

//Pins for 3.3V output
/*
const PinName spi_MOSI = PinName::PG_11;/
const PinName spi_MISO = PinName::PG_10;
const PinName spi_CLK = PinName::PG_9;
const PinName spi_CS = PinName::PG_12;
*/

//pins for 5V output
const PinName spi_MOSI = PinName::PC_12;
const PinName spi_MISO = PinName::PC_11;
const PinName spi_CLK = PinName::PC_10;
const PinName spi_CS = PinName::PA_4;


SPI spi(spi_MOSI, spi_MISO, spi_CLK); 
DigitalOut cs(spi_CS);


short spi_addr;
short spi_data;


int main()
{
    cs = 0;
    spi.format(8,3);
    spi.frequency(10000000); //10MHz
    // spi.frequency(100000000); //100MHz
    wait_us(1000000);
    printf("SPI Master Initialized \n");

     //Unlock packet
    // org: 0xF0, 0x00, 0x00, 0x00
    // 1111 0000  0000 0000  0000 0000  0000 0000
    // 0000 1111  0000 0000  0000 0000  0000 0000
    // converted 0x0F, 0x00, 0x00, 0x00
    spi.write(0x0F);
    spi.write(0x00);
    spi.write(0x00);
    spi.write(0x00);

    while (1) {
        wait_us(100);
        cs = 1;   

                //Data packet write
        //original data: 0x18 0x00 0xa0 0x01 0x03 0x0a
        // 0001 1000  0000 0000  1010 0000  0000 0001  0000 0011  0000 1010
        // 0001 1000  0000 0000  0000 0101  1000 0000  1100 0000  0101 0000
        //converted data: 0x18 0x00 0x05 0x80 0xC0 0x90
            spi.write(0x18);
            spi.write(0x00);
            spi.write(0x05);
            spi.write(0x80);
            spi.write(0xC0);
            spi.write(0x50);

        //Data packet read
        //original data: 0x08 0x00 0xa0 0x01 0x03 0x0a
        // 0000 1000  0000 0000 1010 0000  0000 0001  0000 0011  0000 1010
        // 0001 0000  0000 0000 0000 0101  1000 0000  1100 0000  0101 0000
        //converted data: 0x10 0x00 0x05 0x80 0xC0 0x90 
            spi.write(0x10);
            spi.write(0x00);
            spi.write(0x05);
            spi.write(0x80);
            spi.write(0xC0);
            spi.write(0x50);
            cs = 0;

    }

    return 0;
}
