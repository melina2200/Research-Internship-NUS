
#include "bias.h"
#include "layer_functions.h"
#include "image.h"
#include "weights.h"
#include "im2col.h"
#include "quant_params.h"
#include <inttypes.h>

int main() {
    while(1){
        //matrices used to store the results of a layer after the convolution
        int32_t OUTPUT_MATRIX[36864];//48x48x16
        int8_t OUTPUT_MATRIX_int8[36864];//48x48x16
        //matrices used to store the flattend input after the im2col function (only for depthwise Conv Layers)
        int8_t INPUT_MATRIX[165888];//2304*9*8

        //int32_t test = 1234567;
        //printf("\r int32 TEST VALUE: %ld \n", test);
        //printf("\r IMAGE person 400: %d \n", IMAGE_PERSON[400]);
        im2col(IMAGE_PERSON,1,IMAGE_DIM,IMAGE_DIM,3,2,INPUT_MATRIX,0,1,0,1,-2); 
        //printf("\r INPUT_MATRIX 400: %d \n", INPUT_MATRIX[400]);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX1,CHANNELS1, WEIGHT_DIM1, 2304, OUTPUT_MATRIX, 1);
        //printf("\r OUTPUT_MATRIX 400: %ld \n", OUTPUT_MATRIX[400]);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX1,CHANNELS1, WEIGHT_DIM1, 2304,2);
        //printf("\r OUTPUT_MATRIX 400 after quant: %ld \n", OUTPUT_MATRIX[400]);
        add_bias(OUTPUT_MATRIX, bias1,2304, CHANNELS1);
        //printf("\r OUTPUT_MATRIX 400 after bias: %ld \n", OUTPUT_MATRIX[400]);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 2304, CHANNELS1, multiply1, add1, shift1, 0);
        //printf("\r OUTPUT_MATRIX_int8 400: %d \n", OUTPUT_MATRIX_int8[400]);
        printf("\r Layer 1 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS1,48,48, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX2,CHANNELS2, WEIGHT_DIM2, 2304, OUTPUT_MATRIX, CHANNELS1);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX2,CHANNELS2, WEIGHT_DIM2, 2304,128); 
        add_bias(OUTPUT_MATRIX, bias2,2304, CHANNELS2);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 2304, CHANNELS2, multiply2, add2, shift2, 0);
        printf("\r Layer 2 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX3,CHANNELS_IN3,CHANNELS_OUT3, 2304, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX3,CHANNELS_OUT3, CHANNELS_IN3, 2304,128); 
        add_bias(OUTPUT_MATRIX, bias3,2304, CHANNELS_OUT3);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 2304, CHANNELS_OUT3, multiply3, add3, shift3, 0);
        printf("\r Layer 3 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT3,48,48, 3, 2,INPUT_MATRIX, 0,1,0,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX4,CHANNELS4, WEIGHT_DIM4, 576, OUTPUT_MATRIX, CHANNELS_OUT3);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX4,CHANNELS4, WEIGHT_DIM4, 576,128); 
        add_bias(OUTPUT_MATRIX, bias4,576, CHANNELS4);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 576, CHANNELS4, multiply4, add4, shift4, 0);
        printf("\r Layer 4 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX5,CHANNELS_IN5,CHANNELS_OUT5, 576, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX5,CHANNELS_OUT5, CHANNELS_IN5, 576,128); 
        add_bias(OUTPUT_MATRIX, bias5,576, CHANNELS_OUT5);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 576, CHANNELS_OUT5, multiply5, add5, shift5, 0);
        printf("\r Layer 5 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT5,24,24, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX6,CHANNELS6, WEIGHT_DIM6, 576, OUTPUT_MATRIX, CHANNELS_OUT5);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX6,CHANNELS6, WEIGHT_DIM6, 576,128); 
        add_bias(OUTPUT_MATRIX, bias6,576, CHANNELS6);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 576, CHANNELS6, multiply6, add6, shift6, 0);
        printf("\r Layer 6 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX7,CHANNELS_IN7,CHANNELS_OUT7, 576, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX7,CHANNELS_OUT7, CHANNELS_IN7, 576,128); 
        add_bias(OUTPUT_MATRIX, bias7,576, CHANNELS_OUT7);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 576, CHANNELS_OUT7, multiply7, add7, shift7, 0);
        printf("\r Layer 7 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT7,24,24, 3, 2,INPUT_MATRIX, 0,1,0,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX8,CHANNELS8, WEIGHT_DIM8, 144, OUTPUT_MATRIX, CHANNELS_OUT7);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX8,CHANNELS8, WEIGHT_DIM8, 144,128); 
        add_bias(OUTPUT_MATRIX, bias8,144, CHANNELS8);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 144, CHANNELS8, multiply8, add8, shift8, 0);
        printf("\r Layer 8 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX9,CHANNELS_IN9,CHANNELS_OUT9, 144, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX9,CHANNELS_OUT9, CHANNELS_IN9, 144,128); 
        add_bias(OUTPUT_MATRIX, bias9,144, CHANNELS_OUT9);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 144, CHANNELS_OUT9, multiply9, add9, shift9, 0);
        printf("\r Layer 9 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT9,12,12, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX10,CHANNELS10, WEIGHT_DIM10, 144, OUTPUT_MATRIX, CHANNELS_OUT9);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX10,CHANNELS10, WEIGHT_DIM10, 144,128); 
        add_bias(OUTPUT_MATRIX, bias10,144, CHANNELS10);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 144, CHANNELS10, multiply10, add10, shift10, 0);
        printf("\r Layer 10 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX11,CHANNELS_IN11,CHANNELS_OUT11, 144, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX11,CHANNELS_OUT11, CHANNELS_IN11, 144,128); 
        add_bias(OUTPUT_MATRIX, bias11,144, CHANNELS_OUT11);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 144, CHANNELS_OUT11, multiply11, add11, shift11, 0);
        printf("\r Layer 11 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT11,12,12, 3, 2,INPUT_MATRIX, 0,1,0,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX12,CHANNELS12, WEIGHT_DIM12, 36, OUTPUT_MATRIX, CHANNELS_OUT11);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX12,CHANNELS12, WEIGHT_DIM12, 36,128); 
        add_bias(OUTPUT_MATRIX, bias12,36, CHANNELS12);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS12, multiply12, add12, shift12, 0);
        printf("\r Layer 12 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX13,CHANNELS_IN13,CHANNELS_OUT13, 36, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX13,CHANNELS_OUT13, CHANNELS_IN13, 36,128); 
        add_bias(OUTPUT_MATRIX, bias13,36, CHANNELS_OUT13);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS_OUT13, multiply13, add13, shift13, 0);
        printf("\r Layer 13 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT13,6,6, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX14,CHANNELS14, WEIGHT_DIM14, 36, OUTPUT_MATRIX, CHANNELS_OUT13);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX14,CHANNELS14, WEIGHT_DIM14, 36,128); 
        add_bias(OUTPUT_MATRIX, bias14,36, CHANNELS14);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS14, multiply14, add14, shift14, 0);
        printf("\r Layer 14 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX15,CHANNELS_IN15,CHANNELS_OUT15, 36, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX15,CHANNELS_OUT15, CHANNELS_IN15, 36,128); 
        add_bias(OUTPUT_MATRIX, bias15,36, CHANNELS_OUT15);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS_OUT15, multiply15, add15, shift15, 0);
        printf("\r Layer 15 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT15,6,6, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX16,CHANNELS16, WEIGHT_DIM16, 36, OUTPUT_MATRIX, CHANNELS_OUT15);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX16,CHANNELS16, WEIGHT_DIM16, 36,128); 
        add_bias(OUTPUT_MATRIX, bias16,36, CHANNELS16);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS16, multiply16, add16, shift16, 0);
        printf("\r Layer 16 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX17,CHANNELS_IN17,CHANNELS_OUT17, 36, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX17,CHANNELS_OUT17, CHANNELS_IN17, 36,128); 
        add_bias(OUTPUT_MATRIX, bias17,36, CHANNELS_OUT17);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS_OUT17, multiply17, add17, shift17, 0);
        printf("\r Layer 17 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT17,6,6, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX18,CHANNELS18, WEIGHT_DIM18, 36, OUTPUT_MATRIX, CHANNELS_OUT17);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX18,CHANNELS18, WEIGHT_DIM18, 36,128); 
        add_bias(OUTPUT_MATRIX, bias18,36, CHANNELS18);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS18, multiply18, add18, shift18, 0);
        printf("\r Layer 18 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX19,CHANNELS_IN19,CHANNELS_OUT19, 36, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX19,CHANNELS_OUT19, CHANNELS_IN19, 36,128); 
        add_bias(OUTPUT_MATRIX, bias19,36, CHANNELS_OUT19);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS_OUT19, multiply19, add19, shift19, 0);
        printf("\r Layer 19 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT19,6,6, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX20,CHANNELS20, WEIGHT_DIM20, 36, OUTPUT_MATRIX, CHANNELS_OUT19);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX20,CHANNELS20, WEIGHT_DIM20, 36,128); 
        add_bias(OUTPUT_MATRIX, bias20,36, CHANNELS20);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS20, multiply20, add20, shift20, 0);
        printf("\r Layer 20 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX21,CHANNELS_IN21,CHANNELS_OUT21, 36, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX21,CHANNELS_OUT21, CHANNELS_IN21, 36,128); 
        add_bias(OUTPUT_MATRIX, bias21,36, CHANNELS_OUT21);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS_OUT21, multiply21, add21, shift21, 0);
        printf("\r Layer 21 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT21,6,6, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX22,CHANNELS22, WEIGHT_DIM22, 36, OUTPUT_MATRIX, CHANNELS_OUT21);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX22,CHANNELS22, WEIGHT_DIM22, 36,128); 
        add_bias(OUTPUT_MATRIX, bias22,36, CHANNELS22);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS22, multiply22, add22, shift22, 0);
        printf("\r Layer 22 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX23,CHANNELS_IN23,CHANNELS_OUT23, 36, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX23,CHANNELS_OUT23, CHANNELS_IN23, 36,128); 
        add_bias(OUTPUT_MATRIX, bias23,36, CHANNELS_OUT23);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 36, CHANNELS_OUT23, multiply23, add23, shift23, 0);
        printf("\r Layer 23 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT23,6,6, 3, 2,INPUT_MATRIX, 0,1,0,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX24,CHANNELS24, WEIGHT_DIM24, 9, OUTPUT_MATRIX, CHANNELS_OUT23);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX24,CHANNELS24, WEIGHT_DIM24,9,128); 
        add_bias(OUTPUT_MATRIX, bias24,9, CHANNELS24);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 9, CHANNELS24, multiply24, add24, shift24, 0);
        printf("\r Layer 24 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX25,CHANNELS_IN25,CHANNELS_OUT25, 9, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX25,CHANNELS_OUT25, CHANNELS_IN25, 9,128); 
        add_bias(OUTPUT_MATRIX, bias25,9, CHANNELS_OUT25);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 9, CHANNELS_OUT25, multiply25, add25, shift25, 0);
        printf("\r Layer 25 Done\n");

        im2col(OUTPUT_MATRIX_int8,CHANNELS_OUT25,3,3, 3, 1,INPUT_MATRIX, 1,1,1,1,-128);
        conv_layer(INPUT_MATRIX, WEIGHT_MATRIX26,CHANNELS26, WEIGHT_DIM26, 9, OUTPUT_MATRIX, CHANNELS_OUT25);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX26,CHANNELS26, WEIGHT_DIM26,9,128); 
        add_bias(OUTPUT_MATRIX, bias26,9, CHANNELS26);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 9, CHANNELS26, multiply26, add26, shift26, 0);
        printf("\r Layer 26 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX27,CHANNELS_IN27,CHANNELS_OUT27, 9, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX27,CHANNELS_OUT27, CHANNELS_IN27, 9,128); 
        add_bias(OUTPUT_MATRIX, bias27,9, CHANNELS_OUT27);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 9, CHANNELS_OUT27, multiply27, add27, shift27, 0);
        printf("\r Layer 27 Done\n");

        avg_pool_layer(OUTPUT_MATRIX_int8, 9, 256, OUTPUT_MATRIX_int8);
        printf("\r Layer 28 Done\n");

        pointwise_conv_layer(OUTPUT_MATRIX_int8, WEIGHT_MATRIX29,CHANNELS_IN29,CHANNELS_OUT29, 1, OUTPUT_MATRIX);
        quantize_conv_layer(OUTPUT_MATRIX,WEIGHT_MATRIX29,CHANNELS_OUT29, CHANNELS_IN29, 1,128); 
        add_bias(OUTPUT_MATRIX, bias29,1, CHANNELS_OUT29);
        requantize_conv(OUTPUT_MATRIX,OUTPUT_MATRIX_int8, 1, CHANNELS_OUT29, multiply29, add29, shift29, 1);
        printf("\r Layer 29 Done\n");

        printf("output at 0: %d \n",OUTPUT_MATRIX_int8[0]);
        printf("output at 1: %d \n",OUTPUT_MATRIX_int8[1]);

        softmax_and_output(OUTPUT_MATRIX_int8,2);
    }
    return 0;
}

