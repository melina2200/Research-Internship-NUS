// #include "LSM9DS1.h"
 
// LSM9DS1::LSM9DS1(PinName sda, PinName scl, uint8_t xgAddr, uint8_t mAddr) : i2c(sda, scl)
// {
//     // xgAddress and mAddress will store the 7-bit I2C address, if using I2C.
//     xgAddress = xgAddr;
//     mAddress = mAddr;
// }
 
// uint16_t LSM9DS1::begin(gyro_scale gScl, accel_scale aScl, mag_scale mScl, 
//                         gyro_odr gODR, accel_odr aODR, mag_odr mODR)
// {
//     // Store the given scales in class variables. These scale variables
//     // are used throughout to calculate the actual g's, DPS,and Gs's.
//     gScale = gScl;
//     aScale = aScl;
//     mScale = mScl;
    
//     // Once we have the scale values, we can calculate the resolution
//     // of each sensor. That's what these functions are for. One for each sensor
//     calcgRes(); // Calculate DPS / ADC tick, stored in gRes variable
//     calcmRes(); // Calculate Gs / ADC tick, stored in mRes variable
//     calcaRes(); // Calculate g / ADC tick, stored in aRes variable
    
    
//     // To verify communication, we can read from the WHO_AM_I register of
//     // each device. Store those in a variable so we can return them.
//     // The start of the addresses we want to read from
//     char cmd[2] = {
//         WHO_AM_I_XG,
//         0
//     };
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress<<1, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(xgAddress<<1, cmd+1, 1);
//     uint8_t xgTest = cmd[1];                    // Read the accel/gyro WHO_AM_I
    
//     // Reset to the address of the mag who am i
//     cmd[1] = WHO_AM_I_M;
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(mAddress<<1, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(mAddress<<1, cmd+1, 1);
//     uint8_t mTest = cmd[1];      // Read the mag WHO_AM_I
    
//     // Gyro initialization stuff:
//     initGyro(); // This will "turn on" the gyro. Setting up interrupts, etc.
//     setGyroODR(gODR); // Set the gyro output data rate and bandwidth.
//     setGyroScale(gScale); // Set the gyro range
    
//     // Accelerometer initialization stuff:
//     initAccel(); // "Turn on" all axes of the accel. Set up interrupts, etc.
//     setAccelODR(aODR); // Set the accel data rate.
//     setAccelScale(aScale); // Set the accel range.
    
//     // Magnetometer initialization stuff:
//     initMag(); // "Turn on" all axes of the mag. Set up interrupts, etc.
//     setMagODR(mODR); // Set the magnetometer output data rate.
//     setMagScale(mScale); // Set the magnetometer's range.
    
//     // Once everything is initialized, return the WHO_AM_I registers we read:
//     return (xgTest << 8) | mTest;
// }
 
// void LSM9DS1::initGyro()
// {
//     char cmd[4] = {
//         CTRL_REG1_G,
//         char(gScale | G_ODR_119_BW_14),
//         0,          // Default data out and int out
//         0           // Default power mode and high pass settings
//     };
 
//     // Write the data to the gyro control registers
//     i2c.write(xgAddress, cmd, 4);
// }
 
// void LSM9DS1::initAccel()
// {
//     char cmd[4] = {
//         CTRL_REG5_XL,
//         0x38,       // Enable all axis and don't decimate data in out Registers
//         char((A_ODR_119 << 5) | (aScale << 3) | (A_BW_AUTO_SCALE)),   // 119 Hz ODR, set scale, and auto BW
//         0           // Default resolution mode and filtering settings
//     };
 
//     // Write the data to the accel control registers
//     i2c.write(xgAddress, cmd, 4);
// }
 
// void LSM9DS1::initMag()
// {   
//     char cmd[4] = {
//         CTRL_REG1_M,
//         0x10,       // Default data rate, xy axes mode, and temp comp
//         char(mScale << 5), // Set mag scale
//         0           // Enable I2C, write only SPI, not LP mode, Continuous conversion mode
//     };
 
//     // Write the data to the mag control registers
//     i2c.write(mAddress, cmd, 4);
// }


// void LSM9DS1::readAccel()
// {
//     // The data we are going to read from the accel
//     char data[6];
 
//     // The start of the addresses we want to read from
//     char subAddress = OUT_X_L_XL;
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress, &subAddress, 1, true);
//     // Read in all 8 bit registers containing the axes data
//     i2c.read(xgAddress, data, 6);
 
//     // Reassemble the data and convert to g
//     ax_raw = data[0] | (data[1] << 8);
//     ay_raw = data[2] | (data[3] << 8);
//     az_raw = data[4] | (data[5] << 8);
//     ax = ax_raw * aRes;
//     ay = ay_raw * aRes;
//     az = az_raw * aRes;

//     printf("%f %f %f\n", ax, ay, az);
// }
 
// void LSM9DS1::readMag()
// {
//     // The data we are going to read from the mag
//     char data[6];
 
//     // The start of the addresses we want to read from
//     char subAddress = OUT_X_L_M;
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(mAddress, &subAddress, 1, true);
//     // Read in all 8 bit registers containing the axes data
//     i2c.read(mAddress, data, 6);
 
//     // Reassemble the data and convert to degrees
//     mx_raw = data[0] | (data[1] << 8);
//     my_raw = data[2] | (data[3] << 8);
//     mz_raw = data[4] | (data[5] << 8);
//     mx = mx_raw * mRes;
//     my = my_raw * mRes;
//     mz = mz_raw * mRes;
// }
 
// void LSM9DS1::readTemp()
// {
//     // The data we are going to read from the temp
//     char data[2];
 
//     // The start of the addresses we want to read from
//     char subAddress = OUT_TEMP_L;
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress, &subAddress, 1, true);
//     // Read in all 8 bit registers containing the axes data
//     i2c.read(xgAddress, data, 2);
 
//     // Temperature is a 12-bit signed integer   
//     temperature_raw = data[0] | (data[1] << 8);
 
//     temperature_c = (float)temperature_raw / 8.0 + 25;
//     temperature_f = temperature_c * 1.8 + 32;
// }
 
 
// void LSM9DS1::readGyro()
// {
//     // The data we are going to read from the gyro
//     char data[6];
 
//     // The start of the addresses we want to read from
//     char subAddress = OUT_X_L_G;
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress, &subAddress, 1, true);
//     // Read in all 8 bit registers containing the axes data
//     i2c.read(xgAddress, data, 6);
 
//     // Reassemble the data and convert to degrees/sec
//     gx_raw = data[0] | (data[1] << 8);
//     gy_raw = data[2] | (data[3] << 8);
//     gz_raw = data[4] | (data[5] << 8);
//     gx = gx_raw * gRes;
//     gy = gy_raw * gRes;
//     gz = gz_raw * gRes;
// }
 
// void LSM9DS1::setGyroScale(gyro_scale gScl)
// {
//     // The start of the addresses we want to read from
//     char cmd[2] = {
//         CTRL_REG1_G,
//         0
//     };
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(xgAddress, cmd+1, 1);
 
//     // Then mask out the gyro scale bits:
//     cmd[1] &= 0xFF^(0x3 << 3);
//     // Then shift in our new scale bits:
//     cmd[1] |= gScl << 3;
 
//     // Write the gyroscale out to the gyro
//     i2c.write(xgAddress, cmd, 2);
    
//     // We've updated the sensor, but we also need to update our class variables
//     // First update gScale:
//     gScale = gScl;
//     // Then calculate a new gRes, which relies on gScale being set correctly:
//     calcgRes();
// }
 
// void LSM9DS1::setAccelScale(accel_scale aScl)
// {
//     // The start of the addresses we want to read from
//     char cmd[2] = {
//         CTRL_REG6_XL,
//         0
//     };
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(xgAddress, cmd+1, 1);
 
//     // Then mask out the accel scale bits:
//     cmd[1] &= 0xFF^(0x3 << 3);
//     // Then shift in our new scale bits:
//     cmd[1] |= aScl << 3;
 
//     // Write the accelscale out to the accel
//     i2c.write(xgAddress, cmd, 2);
    
//     // We've updated the sensor, but we also need to update our class variables
//     // First update aScale:
//     aScale = aScl;
//     // Then calculate a new aRes, which relies on aScale being set correctly:
//     calcaRes();
// }
 
// void LSM9DS1::setMagScale(mag_scale mScl)
// {
//     // The start of the addresses we want to read from
//     char cmd[2] = {
//         CTRL_REG2_M,
//         0
//     };
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(mAddress, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(mAddress, cmd+1, 1);
 
//     // Then mask out the mag scale bits:
//     cmd[1] &= 0xFF^(0x3 << 5);
//     // Then shift in our new scale bits:
//     cmd[1] |= mScl << 5;
 
//     // Write the magscale out to the mag
//     i2c.write(mAddress, cmd, 2);
    
//     // We've updated the sensor, but we also need to update our class variables
//     // First update mScale:
//     mScale = mScl;
//     // Then calculate a new mRes, which relies on mScale being set correctly:
//     calcmRes();
// }
 
// void LSM9DS1::setGyroODR(gyro_odr gRate)
// {
//     // The start of the addresses we want to read from
//     char cmd[2] = {
//         CTRL_REG1_G,
//         0
//     };
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(xgAddress, cmd+1, 1);
 
//     // Then mask out the gyro odr bits:
//     cmd[1] &= (0x3 << 3);
//     // Then shift in our new odr bits:
//     cmd[1] |= gRate;
 
//     // Write the gyroodr out to the gyro
//     i2c.write(xgAddress, cmd, 2);
// }
 
// void LSM9DS1::setAccelODR(accel_odr aRate)
// {
//     // The start of the addresses we want to read from
//     char cmd[2] = {
//         CTRL_REG6_XL,
//         0
//     };
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(xgAddress, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(xgAddress, cmd+1, 1);
 
//     // Then mask out the accel odr bits:
//     cmd[1] &= 0xFF^(0x7 << 5);
//     // Then shift in our new odr bits:
//     cmd[1] |= aRate << 5;
 
//     // Write the accelodr out to the accel
//     i2c.write(xgAddress, cmd, 2);
// }
 
// void LSM9DS1::setMagODR(mag_odr mRate)
// {
//     // The start of the addresses we want to read from
//     char cmd[2] = {
//         CTRL_REG1_M,
//         0
//     };
 
//     // Write the address we are going to read from and don't end the transaction
//     i2c.write(mAddress, cmd, 1, true);
//     // Read in all the 8 bits of data
//     i2c.read(mAddress, cmd+1, 1);
 
//     // Then mask out the mag odr bits:
//     cmd[1] &= 0xFF^(0x7 << 2);
//     // Then shift in our new odr bits:
//     cmd[1] |= mRate << 2;
 
//     // Write the magodr out to the mag
//     i2c.write(mAddress, cmd, 2);
// }
 
// void LSM9DS1::calcgRes()
// {
//     // Possible gyro scales (and their register bit settings) are:
//     // 245 DPS (00), 500 DPS (01), 2000 DPS (10).
//     switch (gScale)
//     {
//         case G_SCALE_245DPS:
//             gRes = 245.0 / 32768.0;
//             break;
//         case G_SCALE_500DPS:
//             gRes = 500.0 / 32768.0;
//             break;
//         case G_SCALE_2000DPS:
//             gRes = 2000.0 / 32768.0;
//             break;
//     }
// }
 
// void LSM9DS1::calcaRes()
// {
//     // Possible accelerometer scales (and their register bit settings) are:
//     // 2 g (000), 4g (001), 6g (010) 8g (011), 16g (100).
//     switch (aScale)
//     {
//         case A_SCALE_2G:
//             aRes = 2.0 / 32768.0;
//             break;
//         case A_SCALE_4G:
//             aRes = 4.0 / 32768.0;
//             break;
//         case A_SCALE_8G:
//             aRes = 8.0 / 32768.0;
//             break;
//         case A_SCALE_16G:
//             aRes = 16.0 / 32768.0;
//             break;
//     }
// }
 
// void LSM9DS1::calcmRes()
// {
//     // Possible magnetometer scales (and their register bit settings) are:
//     // 2 Gs (00), 4 Gs (01), 8 Gs (10) 12 Gs (11). 
//     switch (mScale)
//     {
//         case M_SCALE_4GS:
//             mRes = 4.0 / 32768.0;
//             break;
//         case M_SCALE_8GS:
//             mRes = 8.0 / 32768.0;
//             break;
//         case M_SCALE_12GS:
//             mRes = 12.0 / 32768.0;
//             break;
//         case M_SCALE_16GS:
//             mRes = 16.0 / 32768.0;
//             break;
//     }
// }
 
// bool LSM9DS1::whoAmI(){
//     uint8_t resultXG,resultM ,whoAmIAddressXG=WHO_AM_I_XG,whoAmIAddressM=WHO_AM_I_M;
    
//     //acc gyro
//     i2c.start();
//     i2c.write(xgAddress);
//     i2c.write(whoAmIAddressXG);
//     i2c.write(xgAddress | 1);
//     resultXG = i2c.read(whoAmIAddressXG);
//     i2c.stop();
    
//     //magn
//     i2c.start();
//     i2c.write(xgAddress);
//     i2c.write(whoAmIAddressM);
//     i2c.write(xgAddress | 1);
//     resultM = i2c.read(whoAmIAddressM);
//     i2c.stop();
    
//     uint16_t combinedResult = (resultXG << 8) | resultM;
//     if(combinedResult!=((whoAmIAddressXG << 8) | whoAmIAddressM))
//         return true;
//     else
//         return false;
// }
            
// // Added 

#include "LSM9DS1.h"
#include "medianfilter.h"
#include "FFT.h"
#include "mbed.h"
#include "stm32l4xx_hal_sai.h"
#include <cstring>

//Added for Peak finding
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <unordered_map>
// #include <cmath>
// #include <iterator>
// #include <numeric>

// using namespace std;
// typedef long double ld;
// typedef unsigned int uint;
// typedef std::vector<ld>::iterator vec_iter_ld;
// /**
//  * Overriding the ostream operator for pretty printing vectors.
//  */
// template<typename T>
// std::ostream &operator<<(std::ostream &os, std::vector<T> vec) {
//     os << "[";
//     if (vec.size() != 0) {
//         std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(os, " "));
//         os << vec.back();
//     }
//     os << "]";
//     return os;
// }

// /**
//  * This class calculates mean and standard deviation of a subvector.
//  * This is basically stats computation of a subvector of a window size qual to "lag".
//  */
// class VectorStats {
// public:
//     /**
//      * Constructor for VectorStats class.
//      *
//      * @param start - This is the iterator position of the start of the window,
//      * @param end   - This is the iterator position of the end of the window,
//      */
//     VectorStats(vec_iter_ld start, vec_iter_ld end) {
//         this->start = start;
//         this->end = end;
//         this->compute();
//     }

//     /**
//      * This method calculates the mean and standard deviation using STL function.
//      * This is the Two-Pass implementation of the Mean & Variance calculation.
//      */
//     void compute() {
//         ld sum = std::accumulate(start, end, 0.0);
//         uint slice_size = std::distance(start, end);
//         ld mean = sum / slice_size;
//         std::vector<ld> diff(slice_size);
//         std::transform(start, end, diff.begin(), [mean](ld x) { return x - mean; });
//         ld sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//         ld std_dev = std::sqrt(sq_sum / slice_size);

//         this->m1 = mean;
//         this->m2 = std_dev;
//     }

//     ld mean() {
//         return m1;
//     }

//     ld standard_deviation() {
//         return m2;
//     }

// private:
//     vec_iter_ld start;
//     vec_iter_ld end;
//     ld m1;
//     ld m2;
// };

// /**
//  * This is the implementation of the Smoothed Z-Score Algorithm.
//  * This is direction translation of https://stackoverflow.com/a/22640362/1461896.
//  *
//  * @param input - input signal
//  * @param lag - the lag of the moving window
//  * @param threshold - the z-score at which the algorithm signals
//  * @param influence - the influence (between 0 and 1) of new signals on the mean and standard deviation
//  * @return a hashmap containing the filtered signal and corresponding mean and standard deviation.
//  */
// unordered_map<string, vector<ld>> z_score_thresholding(vector<ld> input, int lag, ld threshold, ld influence) {
//     unordered_map<string, vector<ld>> output;

//     uint n = (uint) input.size();
//     vector<ld> signals(input.size());
//     vector<ld> filtered_input(input.begin(), input.end());
//     vector<ld> filtered_mean(input.size());
//     vector<ld> filtered_stddev(input.size());

//     VectorStats lag_subvector_stats(input.begin(), input.begin() + lag);
//     filtered_mean[lag - 1] = lag_subvector_stats.mean();
//     filtered_stddev[lag - 1] = lag_subvector_stats.standard_deviation();

//     for (int i = lag; i < n; i++) {
//         if (abs(input[i] - filtered_mean[i - 1]) > threshold * filtered_stddev[i - 1]) {
//             signals[i] = (input[i] > filtered_mean[i - 1]) ? 1.0 : -1.0;
//             filtered_input[i] = influence * input[i] + (1 - influence) * filtered_input[i - 1];
//         } else {
//             signals[i] = 0.0;
//             filtered_input[i] = input[i];
//         }
//         VectorStats lag_subvector_stats(filtered_input.begin() + (i - lag), filtered_input.begin() + i);
//         filtered_mean[i] = lag_subvector_stats.mean();
//         filtered_stddev[i] = lag_subvector_stats.standard_deviation();
//     }

//     output["signals"] = signals;
//     output["filtered_mean"] = filtered_mean;
//     output["filtered_stddev"] = filtered_stddev;

//     return output;
// };

// // Should be in main function
// /*
// int lag = 30;
//     ld threshold = 5.0;
//     ld influence = 0.0;
//     unordered_map<string, vector<ld>> output = z_score_thresholding(input, lag, threshold, influence);
//     cout << output["signals"] << endl;
// */

#define G 9.81
 

int repCount = 0;

LSM9DS1::LSM9DS1(PinName sda, PinName scl, uint8_t xgAddr, uint8_t mAddr) : i2c(sda, scl)
{
    // xgAddress and mAddress will store the 7-bit I2C address, if using I2C.
    xgAddress = xgAddr;
    mAddress = mAddr;
}
 
bool LSM9DS1::begin(gyro_scale gScl, accel_scale aScl, mag_scale mScl,
                        gyro_odr gODR, accel_odr aODR, mag_odr mODR)
{
    // Store the given scales in class variables. These scale variables
    // are used throughout to calculate the actual g's, DPS,and Gs's.
    gScale = gScl;
    aScale = aScl;
    mScale = mScl;
 
    // Once we have the scale values, we can calculate the resolution
    // of each sensor. That's what these functions are for. One for each sensor
    calcgRes(); // Calculate DPS / ADC tick, stored in gRes variable
    calcmRes(); // Calculate Gs / ADC tick, stored in mRes variable
    calcaRes(); // Calculate g / ADC tick, stored in aRes variable
 
 
    // To verify communication, we can read from the WHO_AM_I register of
    // each device. Store those in a variable so we can return them.
    // The start of the addresses we want to read from
    char cmd[2] = {
        WHO_AM_I_XG,
        0
    };
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress<<1, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(xgAddress<<1, cmd+1, 1);
    uint8_t xgTest = cmd[1];                    // Read the accel/gyro WHO_AM_I
 
    // Reset to the address of the mag who am i
    cmd[0] = WHO_AM_I_M;
    // Write the address we are going to read from and don't end the transaction
    i2c.write(mAddress, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(mAddress, cmd+1, 1);
    uint8_t mTest = cmd[1];      // Read the mag WHO_AM_I
 
    // for(int ii = 0; ii < 3; ii++)
    // {
    //     gBiasRaw[ii] = 0;
    //     aBiasRaw[ii] = 0;
    //     gBias[ii] = 0;
    //     aBias[ii] = 0;
    // }
    // autoCalib = false;
 
    // Accelerometer initialization stuff:
    initAccel(); // "Turn on" all axes of the accel. Set up interrupts, etc.
    setAccelODR(aODR); // Set the accel data rate.
    setAccelScale(aScale); // Set the accel range.
 
    // Gyro initialization stuff:
    initGyro(); // This will "turn on" the gyro. Setting up interrupts, etc.
    setGyroODR(gODR); // Set the gyro output data rate and bandwidth.
    setGyroScale(gScale); // Set the gyro range
 
    // Magnetometer initialization stuff:
    initMag(); // "Turn on" all axes of the mag. Set up interrupts, etc.
    setMagODR(mODR); // Set the magnetometer output data rate.
    setMagScale(mScale); // Set the magnetometer's range.
 
    // Interrupt initialization stuff
    initIntr();
 
    // Once everything is initialized, return the WHO_AM_I registers we read:
    return ( ((xgTest << 8) | mTest) == (WHO_AM_I_AG_RSP << 8 | WHO_AM_I_M_RSP) );
}
 
void LSM9DS1::initGyro()
{
    /*
    char cmd[4] = {
        CTRL_REG1_G,
        gScale | G_ODR_119_BW_14,
        0,          // Default data out and int out
        0           // Default power mode and high pass settings
    };
    */
 
    char cmd[4] = {
        CTRL_REG1_G,
        static_cast<char>(gScale | G_ODR_119_BW_14),
        0x03,          // Data pass througn HPF and LPF2, default int out
        0x80           // Low-power mode, disable high-pass filter, default cut-off frequency
    };
 
    // Write the data to the gyro control registers
    i2c.write(xgAddress, cmd, 4);
}
 
void LSM9DS1::initAccel()
{
    char cmd[4] = {
        CTRL_REG5_XL,
        0x38,       // Enable all axis and don't decimate data in out Registers
        static_cast<char>((A_ODR_119 << 5) | (aScale << 3) | (A_BW_AUTO_SCALE)),   // 119 Hz ODR, set scale, and auto BW
        0           // Default resolution mode and filtering settings
    };
 
    // Write the data to the accel control registers
    i2c.write(xgAddress, cmd, 4);
}
 
void LSM9DS1::initMag()
{
    char cmd[4] = {
        CTRL_REG1_M,
        0x10,       // Default data rate, xy axes mode, and temp comp
        static_cast<char>(mScale << 5), // Set mag scale
        0           // Enable I2C, write only SPI, not LP mode, Continuous conversion mode
    };
 
    // Write the data to the mag control registers
    i2c.write(mAddress, cmd, 4);
}
 
void LSM9DS1::initIntr()
{
    char cmd[2];
    uint16_t thresholdG = 500;
    uint8_t durationG = 10;
    uint8_t thresholdX = 20;
    uint8_t durationX = 1;
    uint16_t thresholdM = 10000;
 
    // 1. Configure the gyro interrupt generator
    cmd[0] = INT_GEN_CFG_G;
    cmd[1] = (1 << 5);
    i2c.write(xgAddress, cmd, 2);
    // 2. Configure the gyro threshold
    cmd[0] = INT_GEN_THS_ZH_G;
    cmd[1] = (thresholdG & 0x7F00) >> 8;
    i2c.write(xgAddress, cmd, 2);
    cmd[0] = INT_GEN_THS_ZL_G;
    cmd[1] = (thresholdG & 0x00FF);
    i2c.write(xgAddress, cmd, 2);
    cmd[0] = INT_GEN_DUR_G;
    cmd[1] = (durationG & 0x7F) | 0x80;
    i2c.write(xgAddress, cmd, 2);
 
    // 3. Configure accelerometer interrupt generator
    cmd[0] = INT_GEN_CFG_XL;
    cmd[1] = (1 << 1);
    i2c.write(xgAddress, cmd, 2);
    // 4. Configure accelerometer threshold
    cmd[0] = INT_GEN_THS_X_XL;
    cmd[1] = thresholdX;
    i2c.write(xgAddress, cmd, 2);
    cmd[0] = INT_GEN_DUR_XL;
    cmd[1] = (durationX & 0x7F);
    i2c.write(xgAddress, cmd, 2);
 
    // 5. Configure INT1 - assign it to gyro interrupt
    cmd[0] = INT1_CTRL;
//    cmd[1] = 0xC0;
    cmd[1] = (1 << 7) | (1 << 6);
    i2c.write(xgAddress, cmd, 2);
    cmd[0] = CTRL_REG8;
//    cmd[1] = 0x04;
    cmd[1] = (1 << 2) | (1 << 5) | (1 << 4);
    i2c.write(xgAddress, cmd, 2);
 
    // Configure interrupt 2 to fire whenever new accelerometer
    // or gyroscope data is available.
    cmd[0] = INT2_CTRL;
    cmd[1] = (1 << 0) | (1 << 1);
    i2c.write(xgAddress, cmd, 2);
    cmd[0] = CTRL_REG8;
    cmd[1] = (1 << 2) | (1 << 5) | (1 << 4);
    i2c.write(xgAddress, cmd, 2);
 
    // Configure magnetometer interrupt
    cmd[0] = INT_CFG_M;
    cmd[1] = (1 << 7) | (1 << 0);
    i2c.write(xgAddress, cmd, 2);
 
    // Configure magnetometer threshold
    cmd[0] = INT_THS_H_M;
    cmd[1] = uint8_t((thresholdM & 0x7F00) >> 8);
    i2c.write(xgAddress, cmd, 2);
    cmd[0] = INT_THS_L_M;
    cmd[1] = uint8_t(thresholdM & 0x00FF);
    i2c.write(xgAddress, cmd, 2);
}
 
void LSM9DS1::calibration()
{
 
    uint16_t samples = 0;
    int32_t aBiasRaw[3] = {0, 0, 0};
    int32_t gBiasRaw[3] = {0, 0, 0};
    int32_t aFinal[3] = {0, 0, 0};
    int32_t gFinal[3] = {0, 0, 0};
 
    // Turn off the autoCalib first to get the raw data
    autoCalib = false;
 
    while(samples < 200)
    {    
        readAccel();
        
        //printf("Reading acc : %d\n", samples);

        //Accelerometer
        aBiasRaw[0] += ax_raw;
        aBiasRaw[1] += ay_raw;
        aBiasRaw[2] += az_raw;
        wait_us(1); // 1 ms

        // Gyroscope
        gBiasRaw[0] += gx_raw;
        gBiasRaw[1] += gy_raw;
        gBiasRaw[2] += gz_raw;
        wait_us(1); // 1 ms

        // Increase count
        samples++;
    }
 
    for(int ii = 0; ii < 3; ii++)
    {
        aFinal[ii] = aBiasRaw[ii] / samples;
        gFinal[ii] = gBiasRaw[ii] / samples;   
    }
    // Turn on the autoCalib
    autoCalib = true;

    
}
 
void LSM9DS1::readAccel()
{
    // The data we are going to read from the accel
    char data[6];
 
    // The start of the addresses we want to read from
    char subAddress = OUT_X_L_XL;
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress, &subAddress, 1, true);
    // Read in all 8 bit registers containing the axes data
    i2c.read(xgAddress, data, 6);
 
    // Reassemble the data and convert to g
    ax_raw = data[0] | (data[1] << 8);
    ay_raw = data[2] | (data[3] << 8);
    az_raw = data[4] | (data[5] << 8);
    //printf("%d %d %d\n", ax_raw, ay_raw, az_raw);
       
    //Put in main
    if(autoCalib)
    {
        ax_raw -= aBiasRaw[0];
        ay_raw -= aBiasRaw[1];
        az_raw -= aBiasRaw[2];
    }

    // Multiply with float gives erronous calibration
    // ax = ax_raw * aRes;
    // ay = ay_raw * aRes;
    // az = az_raw * aRes;

    // aFinal[0] = ax_raw / 8;
    // aFinal[1] = ay_raw / 8;
    // aFinal[2] = az_raw / 8;
    // printf("%d %d %d\n", aFinal[0], aFinal[1], aFinal[2]);
  
    // while(samples2 < 10){
    //     xfilteredinput[samples2++] = aFinal[0];
    //     yfilteredinput[samples2++] = aFinal[1];
    //     zfilteredinput[samples2++] = aFinal[2];
    //     printf("%d %d %d", xfilteredinput[asd++], yfilteredinput[asd++], zfilteredinput[asd++]);
    // }
    // samples2 = 0;
    // asd = 0;
    // memset(xfilteredoutput, 0, sizeof(element)*10);
    // medianfilter(xfilteredinput, xfilteredoutput, 10);
    // memset(yfilteredoutput, 0, sizeof(element)*10);
    // medianfilter(yfilteredinput, yfilteredoutput, 10);
    // memset(zfilteredoutput, 0, sizeof(element)*10);
    // medianfilter(zfilteredinput, zfilteredoutput, 10);

    // for(int i=0; i<10; i++){
    //     printf("%d %d %d\n", xfilteredoutput[i],yfilteredoutput[i], zfilteredoutput[i]);
    // }

    // Detect Pattern to count reps accurately
    // Determine, start position, backward motion and end of 1 rep

    // Condition to just test
    // if (ay_raw < -2800){
    //     repCount += 1;
    //     printf("Count: %d\n", repCount);
    // }

    
}
 
void LSM9DS1::readMag()
{
    // The data we are going to read from the mag
    char data[6];
 
    // The start of the addresses we want to read from
    char subAddress = OUT_X_L_M;
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(mAddress, &subAddress, 1, true);
    // Read in all 8 bit registers containing the axes data
    i2c.read(mAddress, data, 6);
 
    // Reassemble the data and convert to degrees
    mx_raw = data[0] | (data[1] << 8);
    my_raw = data[2] | (data[3] << 8);
    mz_raw = data[4] | (data[5] << 8);
    mx = mx_raw * mRes;
    my = my_raw * mRes;
    mz = mz_raw * mRes;
    //printf("%f %f %f\n", mx, my, mz);
}
 
void LSM9DS1::readIntr()
{
    char data[1];
    char subAddress = INT_GEN_SRC_G;
 
    i2c.write(xgAddress, &subAddress, 1, true);
    i2c.read(xgAddress, data, 1);
 
    intr = (float)data[0];
}
 
void LSM9DS1::readTemp()
{
    // The data we are going to read from the temp
    char data[2];
 
    // The start of the addresses we want to read from
    char subAddress = OUT_TEMP_L;
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress, &subAddress, 1, true);
    // Read in all 8 bit registers containing the axes data
    i2c.read(xgAddress, data, 2);
 
    // Temperature is a 12-bit signed integer
    temperature_raw = data[0] | (data[1] << 8);
 
    temperature_c = (float)temperature_raw / 8.0 + 25;
    temperature_f = temperature_c * 1.8 + 32;
}
 
void LSM9DS1::readGyro()
{
    // The data we are going to read from the gyro
    char data[6];
 
    // The start of the addresses we want to read from
    char subAddress = OUT_X_L_G;
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress, &subAddress, 1, true);
    // i2c.write(xgAddress, &subAddress, 1);
    // Read in all 8 bit registers containing the axes data
    i2c.read(xgAddress, data, 6);
 
    // Reassemble the data and convert to degrees/sec
    gx_raw = data[0] | (data[1] << 8);
    gy_raw = data[2] | (data[3] << 8);
    gz_raw = data[4] | (data[5] << 8);
 
    //
    if(autoCalib)
    {
        gx_raw -= gBiasRaw[0];
        gy_raw -= gBiasRaw[1];
        gz_raw -= gBiasRaw[2];
        // gx = gx_raw * gRes;
        // gy = gy_raw * gRes;
        // gz = gz_raw * gRes;
        //printf("%d %d %d\n", gx_raw, gy_raw, gz_raw);
        //printf("%d\n", gy_raw);

    }
}
 
void LSM9DS1::setGyroScale(gyro_scale gScl)
{
    // The start of the addresses we want to read from
    char cmd[2] = {
        CTRL_REG1_G,
        0
    };
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(xgAddress, cmd+1, 1);
 
    // Then mask out the gyro scale bits:
    cmd[1] &= 0xFF^(0x3 << 3);
    // Then shift in our new scale bits:
    cmd[1] |= gScl << 3;
 
    // Write the gyroscale out to the gyro
    i2c.write(xgAddress, cmd, 2);
 
    // We've updated the sensor, but we also need to update our class variables
    // First update gScale:
    gScale = gScl;
    // Then calculate a new gRes, which relies on gScale being set correctly:
    calcgRes();
}
 
void LSM9DS1::setAccelScale(accel_scale aScl)
{
    // The start of the addresses we want to read from
    char cmd[2] = {
        CTRL_REG6_XL,
        0
    };
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(xgAddress, cmd+1, 1);
 
    // Then mask out the accel scale bits:
    cmd[1] &= 0xFF^(0x3 << 3);
    // Then shift in our new scale bits:
    cmd[1] |= aScl << 3;
 
    // Write the accelscale out to the accel
    i2c.write(xgAddress, cmd, 2);
 
    // We've updated the sensor, but we also need to update our class variables
    // First update aScale:
    aScale = aScl;
    // Then calculate a new aRes, which relies on aScale being set correctly:
    calcaRes();
}
 
void LSM9DS1::setMagScale(mag_scale mScl)
{
    // The start of the addresses we want to read from
    char cmd[2] = {
        CTRL_REG2_M,
        0
    };
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(mAddress, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(mAddress, cmd+1, 1);
 
    // Then mask out the mag scale bits:
    cmd[1] &= 0xFF^(0x3 << 5);
    // Then shift in our new scale bits:
    cmd[1] |= mScl << 5;
 
    // Write the magscale out to the mag
    i2c.write(mAddress, cmd, 2);
 
    // We've updated the sensor, but we also need to update our class variables
    // First update mScale:
    mScale = mScl;
    // Then calculate a new mRes, which relies on mScale being set correctly:
    calcmRes();
}
 
void LSM9DS1::setGyroODR(gyro_odr gRate)
{
    char cmd[2];
    char cmdLow[2];
 
    // Enable the low-power mode
    if(gRate == G_ODR_15_BW_0 | gRate == G_ODR_60_BW_16 | gRate == G_ODR_119_BW_14 | gRate == G_ODR_119_BW_31) {
        cmdLow[0] = CTRL_REG3_G;
        cmdLow[1] = 1<<7; // LP_mode
 
        i2c.write(xgAddress, cmdLow, 2);
    }
 
    // The start of the addresses we want to read from
    cmd[0] = CTRL_REG1_G;
    cmd[1] = 0;
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(xgAddress, cmd+1, 1);
 
    // Then mask out the gyro odr bits:
    cmd[1] &= (0x3 << 3);
    // Then shift in our new odr bits:
    cmd[1] |= gRate;
 
    // Write the gyroodr out to the gyro
    i2c.write(xgAddress, cmd, 2);
}
 
void LSM9DS1::setAccelODR(accel_odr aRate)
{
    // The start of the addresses we want to read from
    char cmd[2] = {
        CTRL_REG6_XL,
        0
    };
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(xgAddress, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(xgAddress, cmd+1, 1);
 
    // Then mask out the accel odr bits:
    cmd[1] &= 0xFF^(0x7 << 5);
    // Then shift in our new odr bits:
    cmd[1] |= aRate << 5;
 
    // Write the accelodr out to the accel
    i2c.write(xgAddress, cmd, 2);
}
 
void LSM9DS1::setMagODR(mag_odr mRate)
{
    // The start of the addresses we want to read from
    char cmd[2] = {
        CTRL_REG1_M,
        0
    };
 
    // Write the address we are going to read from and don't end the transaction
    i2c.write(mAddress, cmd, 1, true);
    // Read in all the 8 bits of data
    i2c.read(mAddress, cmd+1, 1);
 
    // Then mask out the mag odr bits:
    cmd[1] &= 0xFF^(0x7 << 2);
    // Then shift in our new odr bits:
    cmd[1] |= mRate << 2;
 
    // Write the magodr out to the mag
    i2c.write(mAddress, cmd, 2);
}
 
void LSM9DS1::calcgRes()
{
    // Possible gyro scales (and their register bit settings) are:
    // 245 DPS (00), 500 DPS (01), 2000 DPS (10).
    switch (gScale)
    {
        case G_SCALE_245DPS:
            // gRes = 245.0 / 32768.0;
            gRes = 8.75*0.001; // deg./sec.
            break;
        case G_SCALE_500DPS:
            // gRes = 500.0 / 32768.0;
            gRes = 17.50*0.001; // deg./sec.
            break;
        case G_SCALE_2000DPS:
            // gRes = 2000.0 / 32768.0;
            gRes = 70.0*0.001; // deg./sec.
            break;
    }
}
 
void LSM9DS1::calcaRes()
{
    // Possible accelerometer scales (and their register bit settings) are:
    // 2 g (000), 4g (001), 6g (010) 8g (011), 16g (100).
    switch (aScale)
    {
        case A_SCALE_2G:
            aRes = 2.0 / 32768.0;
            printf("2G selected");
            break;
        case A_SCALE_4G:
            aRes = 4.0 / 32768.0;
            printf("4G selected");
            break;
        case A_SCALE_8G:
            aRes = 8.0 / 32768.0;
            printf("8G selected");
            break;
        case A_SCALE_16G:
            // aRes = 16.0 / 32768.0;
            aRes = 0.732*0.001; // g (gravity)
            printf("16G selected");
            break;
    }
}
 
void LSM9DS1::calcmRes()
{
    // Possible magnetometer scales (and their register bit settings) are:
    // 2 Gs (00), 4 Gs (01), 8 Gs (10) 12 Gs (11).
    switch (mScale)
    {
        case M_SCALE_4GS:
            // mRes = 4.0 / 32768.0;
            mRes = 0.14*0.001; // gauss
            break;
        case M_SCALE_8GS:
            // mRes = 8.0 / 32768.0;
            mRes = 0.29*0.001; // gauss
            break;
        case M_SCALE_12GS:
            // mRes = 12.0 / 32768.0;
            mRes = 0.43*0.001; // gauss
            break;
        case M_SCALE_16GS:
            // mRes = 16.0 / 32768.0;
            mRes = 0.58*0.001; // gauss
            break;
    }
}
 
 
/*
void LSM9DS1::enableXgFIFO(bool enable)
{
    char cmd[2] = {CTRL_REG9, 0};
 
    i2c.write(xgAddress, cmd, 1);
    cmd[1] = i2c.read(CTRL_REG9);
 
    if (enable) cmd[1] |= (1<<1);
    else cmd[1] &= ~(1<<1);
 
    i2c.write(xgAddress, cmd, 2);
}
 
void LSM9DS1::setXgFIFO(uint8_t fifoMode, uint8_t fifoThs)
{
    // Limit threshold - 0x1F (31) is the maximum. If more than that was asked
    // limit it to the maximum.
    char cmd[2] = {FIFO_CTRL, 0};
    uint8_t threshold = fifoThs <= 0x1F ? fifoThs : 0x1F;
    cmd[1] = ((fifoMode & 0x7) << 5) | (threshold & 0x1F);
    i2c.write(xgAddress, cmd, 2);
}
*/
 
            

