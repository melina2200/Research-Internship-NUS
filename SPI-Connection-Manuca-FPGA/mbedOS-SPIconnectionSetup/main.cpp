/*#include "mbed.h"
#include "LSM9DS1.h"
#include "mbed_power_mgmt.h"

#include <math.h>
#include <time.h>
#include <chrono>
#include <deque>
#include <SPI.h>

// #define PI 3.14159265358979323846

LSM9DS1 imu(PF_0,PF_1);
DigitalOut pwron2(PF_4);
DigitalOut pwron1(PG_15);
DigitalOut shutTof(PG_1);
DigitalOut pwronTof(PF_5);
float AccX, AccY, AccZ;
float GyroX, GyroY, GyroZ;
float AccSensitivty, GyroSensitivty;
float accAngleX, accAngleY, gyroAngleX, gyroAngleY, gyroAngleZ;
float elapsedTime, currentTime, previousTime;
float baseValue;
float peakTimeCounter, troughTimeCounter;
int repsCount;

bool peak, trough, update;

clock_t timer;

std::deque<float> gyroz_window;

float average();
void repsCounter();
float getTime();
bool checkStationary();
int getDirection();

const PinName spi_MOSI = PinName::PC_12;
const PinName spi_MISO = PinName::PC_11;
const PinName spi_CLK = PinName::PC_10;
const PinName spi_CS = PinName::PA_4;

//const PinName spi_MOSI = PinName::PB_14;
//const PinName spi_MISO = PinName::PB_13;
//const PinName spi_CLK = PinName::PB_12;
//const PinName spi_CS = PinName::PB_15;

const PinName spi_MOSI = PinName::PG_11;
const PinName spi_MISO = PinName::PG_10;
const PinName spi_CLK = PinName::PG_9;
const PinName spi_CS = PinName::PG_12;

//const PinName spi_MOSI = PinName::PA_7;
//const PinName spi_MISO = PinName::PG_3;
//const PinName spi_CLK = PinName::PA_5;
//const PinName spi_CS = PinName::PG_12;

const PinName chip_EN = PinName::PD_7;
const PinName spi_EN = PinName::PD_2;
const PinName clkExt_En = PinName::PG_4;
const PinName clk_Ext = PinName::PA_8;
const PinName clk_En = PinName::PF_2;
const PinName vco_En = PinName::PD_1;
const PinName clkSel_0 = PinName::PE_15;
const PinName clkSel_1 = PinName::PE_11;
const PinName clkSel_2 = PinName::PF_3;
const PinName clkSel_3 = PinName::PG_6;
const PinName clkSel_4 = PinName::PD_0;
const PinName clkSel_5 = PinName::PG_5;
const PinName divSel_0 = PinName::PD_14;
const PinName divSel_1 = PinName::PD_15;
const PinName divSel_2 = PinName::PF_14;
const PinName divSel_3 = PinName::PE_9;
const PinName fixDivSel_0 = PinName::PE_7;
const PinName fixDivSel_1 = PinName::PD_11;
const PinName dlIn_0 = PinName::PD_12;
const PinName dlIn_1 = PinName::PD_13;
const PinName dlIn_2 = PinName::PE_8;


SPI spi(spi_MOSI, spi_MISO, spi_CLK); 
DigitalOut cs(spi_CS);
DigitalOut chip_en(chip_EN);
DigitalOut spi_en(spi_EN);
DigitalOut clkext_en(clkExt_En);
PwmOut clk_ext(clk_Ext);
//DigitalOut clk_ext(clk_Ext);
DigitalOut clk_en(clk_En);
DigitalOut vco_en(vco_En);
DigitalOut clksel_0(clkSel_0);
DigitalOut clksel_1(clkSel_1);
DigitalOut clksel_2(clkSel_2);
DigitalOut clksel_3(clkSel_3);
DigitalOut clksel_4(clkSel_4);
DigitalOut clksel_5(clkSel_5);
DigitalOut divsel_0(divSel_0);
DigitalOut divsel_1(divSel_1);
DigitalOut divsel_2(divSel_2);
DigitalOut divsel_3(divSel_3);
DigitalOut fixdivsel_0(fixDivSel_0);
DigitalOut fixdivsel_1(fixDivSel_1);
DigitalOut dlin_0(dlIn_0);
DigitalOut dlin_1(dlIn_1);
DigitalOut dlin_2(dlIn_2);

short spi_addr;
short spi_data;


int main()
{
    
    chip_en = 1;
    spi_en = 1;
    clkext_en = 0;
    // clk_ext = 0;
    clk_en = 1;
    vco_en = 1;

    clksel_0 = 0;
    clksel_1 = 1;
    clksel_2 = 0;
    clksel_3 = 1;
    clksel_4 = 0;
    clksel_5 = 1;

    divsel_0 = 0;
    divsel_1 = 1;
    divsel_2 = 0;
    divsel_3 = 0;

    fixdivsel_0 = 0;
    fixdivsel_1 = 0;

    dlin_0 = 1;
    dlin_1 = 1;
    dlin_2 = 1;

    //SPI config.
    cs = 0;
    spi.format(8,3);
    spi.frequency(10000000); //10MHz
    wait_us(1000000); //1s
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


    

    int size=4, s;
    while (1) {
        wait_us(1000);     
        //RW loop
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
            //Randomized data (packs of 16-bits data)
            for (s=0; s<size*2; s++)
                spi.write(rand() % 256);

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


    }

    return 0;
}*/


#include "mbed.h"
#include "LSM9DS1.h"
#include "mbed_power_mgmt.h"

#include <math.h>
#include <time.h>
#include <chrono>
#include <deque>
#include <SPI.h>
/*
const PinName spi_MOSI = PinName::PG_11;//PC_12;
const PinName spi_MISO = PinName::PG_10;//PC_11;
const PinName spi_CLK = PinName::PG_9;//PC_10;
const PinName spi_CS = PinName::PG_12;//PA_4;
*/
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
    spi.frequency(1000); //10MHz
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