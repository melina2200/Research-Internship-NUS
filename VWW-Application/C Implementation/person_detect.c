
#include "bias.h"
#include <math.h>
#include <inttypes.h>
#include "image.h"
#include "weights.h"
#include "im2col.h"
#include "quant_params.h"
#define C1 80 //weight dim x*y
#define R1 8 //channels
int32_t OUTPUT_MATRIX1[48*48*8]; //48*48*8
int8_t OUTPUT_MATRIX1_int8[48*48*8]; //48*48*8

int32_t OUTPUT_MATRIX2[48*48*16]; //48*48*6
int8_t OUTPUT_MATRIX2_int8[48*48*16]; //48*48*16

int32_t OUTPUT_MATRIX3[24*24*16];
int8_t OUTPUT_MATRIX3_int8[24*24*16]; 

int32_t OUTPUT_MATRIX4[24*24*32]; 
int8_t OUTPUT_MATRIX4_int8[24*24*32];

int32_t OUTPUT_MATRIX5[12*12*32]; 
int8_t OUTPUT_MATRIX5_int8[12*12*32];

int32_t OUTPUT_MATRIX6[12*12*64]; 
int8_t OUTPUT_MATRIX6_int8[12*12*64];

int32_t OUTPUT_MATRIX7[6*6*64]; 
int8_t OUTPUT_MATRIX7_int8[6*6*64];

int32_t OUTPUT_MATRIX8[6*6*128]; 
int8_t OUTPUT_MATRIX8_int8[6*6*128];

int32_t OUTPUT_MATRIX9[3*3*128]; 
int8_t OUTPUT_MATRIX9_int8[3*3*128];

int32_t OUTPUT_MATRIX10[3*3*256]; 
int8_t OUTPUT_MATRIX10_int8[3*3*256];

int8_t OUTPUT_MATRIX11_int8[1*1*256];

int32_t OUTPUT_MATRIX12[1*1*2]; 
int8_t OUTPUT_MATRIX12_int8[1*1*2];



int8_t INPUT_MATRIX1[48*48*WEIGHT_DIM1]; //48*48*9
int8_t INPUT_MATRIX2[48*48*WEIGHT_DIM2*CHANNELS1]; //48*48*9*8
int8_t INPUT_MATRIX3[24*24*WEIGHT_DIM4*CHANNELS_OUT3]; //24*24*9*16
int8_t INPUT_MATRIX4[24*24*WEIGHT_DIM6*CHANNELS_OUT5]; //24*24*9*32
int8_t INPUT_MATRIX5[12*12*WEIGHT_DIM8*CHANNELS_OUT7]; //12*12*9*32
int8_t INPUT_MATRIX6[12*12*WEIGHT_DIM10*CHANNELS_OUT9]; //12*12*9*64
int8_t INPUT_MATRIX7[6*6*WEIGHT_DIM12*CHANNELS_OUT11]; //6*6*9*64
int8_t INPUT_MATRIX8[6*6*WEIGHT_DIM14*CHANNELS_OUT13]; //6*6*9*128
int8_t INPUT_MATRIX9[3*3*WEIGHT_DIM24*CHANNELS_OUT23]; //3*3*9*128
int8_t INPUT_MATRIX10[3*3*WEIGHT_DIM26*CHANNELS_OUT25]; //3*3*9*256



void conv_layer(int8_t* input,int8_t* weights, int channels, int weight_dim, int output_dim, int32_t* output, int input_channels){
    for (int i=0;i<channels; i++) { //channels = 8
        for (int j=0;j<output_dim; j++) { //output_dim = 48*48= 2304
            int32_t temp=0;
            for (int k=0;k<weight_dim*input_channels; k++){ //weight_dim = 9
                int weight_ind = 0;
                if (input_channels == 1)
                    weight_ind = k*channels+i;
                else
                    weight_ind = k;
                temp += weights[weight_ind]*input[k*output_dim+j];
                /*if ((i==0)&&(j==0)){
                    printf("weight_ind: %d \n", weight_ind);
                    printf("k*output_dim+j: %d \n", k*output_dim+j);
                }*/
            }
            output[i*output_dim+j] = temp;
            //printf("i*output_dim+j: %d \n", i*output_dim+j);
        }
    }
    //printf("first entry output matrix: %d \n",output[0]);
    //printf("Conv Done\n");
}

void pointwise_conv_layer(int8_t* input,int8_t* weights, int channels_input, int channels_output, int output_dim, int32_t* output){
    for (int i=0;i<channels_output; i++) { //16
        for (int j=0;j<output_dim; j++) { //2304
            int32_t temp=0;
            for (int k=0;k<channels_input; k++){ //8
                temp += weights[k*channels_output+i]*input[k*output_dim+j];
                /*if ((i==0)&&(j==0)){
                    printf("weight_ind: %d \n", k*channels_output+i);
                    printf("input_ind: %d \n", k*output_dim+j);
                    printf("temp: %d \n", temp);
                }*/
            }
            output[i*output_dim+j] = temp;
            //printf("i*output_dim+j: %d \n", i*output_dim+j);
        }
    }
    //printf("first entry output matrix: %d \n",output[0]);
    //printf("Conv Pointwise Done\n");
}

void avg_pool_layer(int8_t* input, int pool_dim, int channels, int8_t* output){
    for (int i=0;i<channels; i++) { //channels = 256
        int32_t temp=0;
        for (int j=0;j<pool_dim; j++) { //output_dim = 3*3 = 9
                temp += input[i*pool_dim+j];
                /*if ((i==0)){
                    printf("i*pool_dim+j: %d \n", i*pool_dim+j);
                    printf("input[i*pool_dim+j]: %d \n", input[i*pool_dim+j]);
                }*/
        }
        output[i] = (int) temp/pool_dim;
        /*if (i==0)
            printf("output[0]: %d \n", output[i]);*/
    }
    //printf("first entry output matrix: %d \n",output[0]);
    //printf("AvgPool Done\n");
}


void quantize_conv_layer(int32_t* input,int8_t* weights, const int channels, int weight_dim, int output_dim, int multiplier) {
    int i, j;
    int32_t sum_weight[channels]; //channels
    for (int q=0; q<channels; q++) {
        sum_weight[q] = 0;
    }
    for(i=0; i<channels; i++) {
        for (j=0; j<weight_dim; j++) {
            sum_weight[i] += weights[j*channels+i];
            /*if (i==0)
            {
                printf("index: %d \n",j*channels+i);
                printf("entry: %d \n",weights[j*channels+i]);
                printf("first entry sum weights: %d \n",sum_weight[0]);
            }*/
        }
        //printf("Length of sum_weight: %d \n", (sizeof(sum_weight)/sizeof(*sum_weight)));
    }
    //printf("first entry sum weights: %d \n",sum_weight[0]);
    //printf("Done summing weights\n");

    for (i=0; i<output_dim*channels; i++) {
        int ind = (int) i/output_dim;
        //printf("ind: %d \n", ind);
        input[i] = input[i] + (multiplier*sum_weight[ind]);
    }
    //printf("first entry matrix: %d \n",input[0]);
    //printf("Quant Done\n");
}

void add_bias(int32_t* input, int32_t* bias, int output_dim, int channels) {
    for (int i=0; i<output_dim*channels; i++) {
        int indx = (int) i/output_dim;
        //printf("ind: %d \n", indx);
        //printf("i: %d \n", i);
        input[i] += bias[indx];
    }
    //printf("first entry matrix: %d \n",input[0]);
    //printf("Bias Done\n");
}

void requantize_conv(int32_t* input,int8_t* output, const int output_dim, const int channels, int64_t* multiply, int64_t* add, int64_t* shift, int last_layer) {

    int64_t OUTPUT64[output_dim*channels];

    for (int i=0; i<output_dim*channels; i++) {
        int ind = (int) i/output_dim;
        //printf("ind: %d \n", ind);
        //printf("i: %d \n", i);
        OUTPUT64[i] = input[i]*multiply[ind]+add[ind];
        /*if(i==0)
            printf("result after mult and add at 0 %" PRId64 "\n", OUTPUT64[i]);*/
        input[i] = OUTPUT64[i]>>shift[ind];
        if(last_layer == 1){
            input[i] +=3;
        }
        /*if(i==0)
            printf("result after shift at 0: %d \n",input[i]);*/
        input[i] += -128;
        if (input[i] < -128) {
            //printf("Too small %d, %d \n", i, OUTPUT64[i]);
            output[i] = -128;
        }
        else if (input[i] > 127) {
            //printf("Too large %d \n", i);
            output[i] = 127;
        }
        else{
            output[i] = input[i];
        }
    }
    //printf("Requant Done\n");
}

void softmax_and_output(int8_t* input, const int input_dim) {

    float OUTPUT32[input_dim];
    float SOFTMAX[input_dim];
    float softmax_sum = 0;
    for (int i=0; i<input_dim; i++) {
        OUTPUT32[i] = input[i];
        OUTPUT32[i] -= 3;
        OUTPUT32[i] *= 0.038815176;
        SOFTMAX[i] = expf(OUTPUT32[i]);
        softmax_sum += SOFTMAX[i];
        //printf("SOFTMAX[i] %.6f \n", SOFTMAX[i]);
    }
    //printf("softmax_sum %.6f \n", softmax_sum);
    for (int i=0; i<input_dim; i++) {
        OUTPUT32[i] = SOFTMAX[i]/softmax_sum;
        if(i==0)
            printf("Probability NO Person %.6f \n", OUTPUT32[i]);
        if(i==1)
            printf("Probability Person %.6f \n", OUTPUT32[i]);
        OUTPUT32[i] = OUTPUT32[i]/0.00390625;
        OUTPUT32[i] -= (int) 128;
        if (OUTPUT32[i] < -128) {
            //printf("Too small %d, %d \n", i, OUTPUT64[i]);
            input[i] = -128;
        }
        else if (OUTPUT32[i] > 127) {
            //printf("Too large %d \n", i);
            input[i] = 127;
        }
        else{
            input[i] = OUTPUT32[i];
        }
    }
    printf("Output 0 %.6f \n", OUTPUT32[0]);
    printf("Output 1 %.6f \n", OUTPUT32[1]);
    printf("Result 0 %d \n", input[0]);
    printf("Result 1 %d \n", input[1]);
    printf("Softmax and Output Done\n");
}




int main() {
    im2col(IMAGE_NOPERSON,1,IMAGE_DIM,IMAGE_DIM,3,2,INPUT_MATRIX1,0,1,0,1,-2); 
    conv_layer(INPUT_MATRIX1, WEIGHT_MATRIX1,CHANNELS1, WEIGHT_DIM1, 48*48, OUTPUT_MATRIX1, 1);
    quantize_conv_layer(OUTPUT_MATRIX1,WEIGHT_MATRIX1,CHANNELS1, WEIGHT_DIM1, 48*48,2);
    add_bias(OUTPUT_MATRIX1, bias1,48*48, CHANNELS1);
    requantize_conv(OUTPUT_MATRIX1,OUTPUT_MATRIX1_int8, 48*48, CHANNELS1, multiply1, add1, shift1, 0);
    printf("Layer 1 Done\n");

    im2col(OUTPUT_MATRIX1_int8,CHANNELS1,48,48, 3, 1,INPUT_MATRIX2, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX2, WEIGHT_MATRIX2,CHANNELS2, WEIGHT_DIM2, 48*48, OUTPUT_MATRIX1, CHANNELS1);
    quantize_conv_layer(OUTPUT_MATRIX1,WEIGHT_MATRIX2,CHANNELS2, WEIGHT_DIM2, 48*48,128); 
    add_bias(OUTPUT_MATRIX1, bias2,48*48, CHANNELS2);
    requantize_conv(OUTPUT_MATRIX1,OUTPUT_MATRIX1_int8, 48*48, CHANNELS2, multiply2, add2, shift2, 0);
    printf("Layer 2 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX1_int8, WEIGHT_MATRIX3,CHANNELS_IN3,CHANNELS_OUT3, 48*48, OUTPUT_MATRIX2);
    quantize_conv_layer(OUTPUT_MATRIX2,WEIGHT_MATRIX3,CHANNELS_OUT3, CHANNELS_IN3, 48*48,128); 
    add_bias(OUTPUT_MATRIX2, bias3,48*48, CHANNELS_OUT3);
    requantize_conv(OUTPUT_MATRIX2,OUTPUT_MATRIX2_int8, 48*48, CHANNELS_OUT3, multiply3, add3, shift3, 0);
    printf("Layer 3 Done\n");

    im2col(OUTPUT_MATRIX2_int8,CHANNELS_OUT3,48,48, 3, 2,INPUT_MATRIX3, 0,1,0,1,-128);
    conv_layer(INPUT_MATRIX3, WEIGHT_MATRIX4,CHANNELS4, WEIGHT_DIM4, 24*24, OUTPUT_MATRIX3, CHANNELS_OUT3);
    quantize_conv_layer(OUTPUT_MATRIX3,WEIGHT_MATRIX4,CHANNELS4, WEIGHT_DIM4, 24*24,128); 
    add_bias(OUTPUT_MATRIX3, bias4,24*24, CHANNELS4);
    requantize_conv(OUTPUT_MATRIX3,OUTPUT_MATRIX3_int8, 24*24, CHANNELS4, multiply4, add4, shift4, 0);
    printf("Layer 4 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX3_int8, WEIGHT_MATRIX5,CHANNELS_IN5,CHANNELS_OUT5, 24*24, OUTPUT_MATRIX4);
    quantize_conv_layer(OUTPUT_MATRIX4,WEIGHT_MATRIX5,CHANNELS_OUT5, CHANNELS_IN5, 24*24,128); 
    add_bias(OUTPUT_MATRIX4, bias5,24*24, CHANNELS_OUT5);
    requantize_conv(OUTPUT_MATRIX4,OUTPUT_MATRIX4_int8, 24*24, CHANNELS_OUT5, multiply5, add5, shift5, 0);
    printf("Layer 5 Done\n");

    im2col(OUTPUT_MATRIX4_int8,CHANNELS_OUT5,24,24, 3, 1,INPUT_MATRIX4, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX4, WEIGHT_MATRIX6,CHANNELS6, WEIGHT_DIM6, 24*24, OUTPUT_MATRIX4, CHANNELS_OUT5);
    quantize_conv_layer(OUTPUT_MATRIX4,WEIGHT_MATRIX6,CHANNELS6, WEIGHT_DIM6, 24*24,128); 
    add_bias(OUTPUT_MATRIX4, bias6,24*24, CHANNELS6);
    requantize_conv(OUTPUT_MATRIX4,OUTPUT_MATRIX4_int8, 24*24, CHANNELS6, multiply6, add6, shift6, 0);
    printf("Layer 6 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX4_int8, WEIGHT_MATRIX7,CHANNELS_IN7,CHANNELS_OUT7, 24*24, OUTPUT_MATRIX4);
    quantize_conv_layer(OUTPUT_MATRIX4,WEIGHT_MATRIX7,CHANNELS_OUT7, CHANNELS_IN7, 24*24,128); 
    add_bias(OUTPUT_MATRIX4, bias7,24*24, CHANNELS_OUT7);
    requantize_conv(OUTPUT_MATRIX4,OUTPUT_MATRIX4_int8, 24*24, CHANNELS_OUT7, multiply7, add7, shift7, 0);
    printf("Layer 7 Done\n");

    im2col(OUTPUT_MATRIX4_int8,CHANNELS_OUT7,24,24, 3, 2,INPUT_MATRIX5, 0,1,0,1,-128);
    conv_layer(INPUT_MATRIX5, WEIGHT_MATRIX8,CHANNELS8, WEIGHT_DIM8, 12*12, OUTPUT_MATRIX5, CHANNELS_OUT7);
    quantize_conv_layer(OUTPUT_MATRIX5,WEIGHT_MATRIX8,CHANNELS8, WEIGHT_DIM8, 12*12,128); 
    add_bias(OUTPUT_MATRIX5, bias8,12*12, CHANNELS8);
    requantize_conv(OUTPUT_MATRIX5,OUTPUT_MATRIX5_int8, 12*12, CHANNELS8, multiply8, add8, shift8, 0);
    printf("Layer 8 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX5_int8, WEIGHT_MATRIX9,CHANNELS_IN9,CHANNELS_OUT9, 12*12, OUTPUT_MATRIX6);
    quantize_conv_layer(OUTPUT_MATRIX6,WEIGHT_MATRIX9,CHANNELS_OUT9, CHANNELS_IN9, 12*12,128); 
    add_bias(OUTPUT_MATRIX6, bias9,12*12, CHANNELS_OUT9);
    requantize_conv(OUTPUT_MATRIX6,OUTPUT_MATRIX6_int8, 12*12, CHANNELS_OUT9, multiply9, add9, shift9, 0);
    printf("Layer 9 Done\n");

    im2col(OUTPUT_MATRIX6_int8,CHANNELS_OUT9,12,12, 3, 1,INPUT_MATRIX6, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX6, WEIGHT_MATRIX10,CHANNELS10, WEIGHT_DIM10, 12*12, OUTPUT_MATRIX6, CHANNELS_OUT9);
    quantize_conv_layer(OUTPUT_MATRIX6,WEIGHT_MATRIX10,CHANNELS10, WEIGHT_DIM10, 12*12,128); 
    add_bias(OUTPUT_MATRIX6, bias10,12*12, CHANNELS10);
    requantize_conv(OUTPUT_MATRIX6,OUTPUT_MATRIX6_int8, 12*12, CHANNELS10, multiply10, add10, shift10, 0);
    printf("Layer 10 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX6_int8, WEIGHT_MATRIX11,CHANNELS_IN11,CHANNELS_OUT11, 12*12, OUTPUT_MATRIX6);
    quantize_conv_layer(OUTPUT_MATRIX6,WEIGHT_MATRIX11,CHANNELS_OUT11, CHANNELS_IN11, 12*12,128); 
    add_bias(OUTPUT_MATRIX6, bias11,12*12, CHANNELS_OUT11);
    requantize_conv(OUTPUT_MATRIX6,OUTPUT_MATRIX6_int8, 12*12, CHANNELS_OUT11, multiply11, add11, shift11, 0);
    printf("Layer 11 Done\n");

    im2col(OUTPUT_MATRIX6_int8,CHANNELS_OUT11,12,12, 3, 2,INPUT_MATRIX7, 0,1,0,1,-128);
    conv_layer(INPUT_MATRIX7, WEIGHT_MATRIX12,CHANNELS12, WEIGHT_DIM12, 36, OUTPUT_MATRIX7, CHANNELS_OUT11);
    quantize_conv_layer(OUTPUT_MATRIX7,WEIGHT_MATRIX12,CHANNELS12, WEIGHT_DIM12, 36,128); 
    add_bias(OUTPUT_MATRIX7, bias12,36, CHANNELS12);
    requantize_conv(OUTPUT_MATRIX7,OUTPUT_MATRIX7_int8, 36, CHANNELS12, multiply12, add12, shift12, 0);
    printf("Layer 12 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX7_int8, WEIGHT_MATRIX13,CHANNELS_IN13,CHANNELS_OUT13, 36, OUTPUT_MATRIX8);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX13,CHANNELS_OUT13, CHANNELS_IN13, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias13,36, CHANNELS_OUT13);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS_OUT13, multiply13, add13, shift13, 0);
    printf("Layer 13 Done\n");

    im2col(OUTPUT_MATRIX8_int8,CHANNELS_OUT13,6,6, 3, 1,INPUT_MATRIX8, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX8, WEIGHT_MATRIX14,CHANNELS14, WEIGHT_DIM14, 36, OUTPUT_MATRIX8, CHANNELS_OUT13);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX14,CHANNELS14, WEIGHT_DIM14, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias14,36, CHANNELS14);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS14, multiply14, add14, shift14, 0);
    printf("Layer 14 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX8_int8, WEIGHT_MATRIX15,CHANNELS_IN15,CHANNELS_OUT15, 36, OUTPUT_MATRIX8);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX15,CHANNELS_OUT15, CHANNELS_IN15, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias15,36, CHANNELS_OUT15);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS_OUT15, multiply15, add15, shift15, 0);
    printf("Layer 15 Done\n");

    im2col(OUTPUT_MATRIX8_int8,CHANNELS_OUT15,6,6, 3, 1,INPUT_MATRIX8, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX8, WEIGHT_MATRIX16,CHANNELS16, WEIGHT_DIM16, 36, OUTPUT_MATRIX8, CHANNELS_OUT15);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX16,CHANNELS16, WEIGHT_DIM16, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias16,36, CHANNELS16);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS16, multiply16, add16, shift16, 0);
    printf("Layer 16 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX8_int8, WEIGHT_MATRIX17,CHANNELS_IN17,CHANNELS_OUT17, 36, OUTPUT_MATRIX8);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX17,CHANNELS_OUT17, CHANNELS_IN17, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias17,36, CHANNELS_OUT17);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS_OUT17, multiply17, add17, shift17, 0);
    printf("Layer 17 Done\n");

    im2col(OUTPUT_MATRIX8_int8,CHANNELS_OUT17,6,6, 3, 1,INPUT_MATRIX8, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX8, WEIGHT_MATRIX18,CHANNELS18, WEIGHT_DIM18, 36, OUTPUT_MATRIX8, CHANNELS_OUT17);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX18,CHANNELS18, WEIGHT_DIM18, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias18,36, CHANNELS18);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS18, multiply18, add18, shift18, 0);
    printf("Layer 18 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX8_int8, WEIGHT_MATRIX19,CHANNELS_IN19,CHANNELS_OUT19, 36, OUTPUT_MATRIX8);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX19,CHANNELS_OUT19, CHANNELS_IN19, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias19,36, CHANNELS_OUT19);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS_OUT19, multiply19, add19, shift19, 0);
    printf("Layer 19 Done\n");

    im2col(OUTPUT_MATRIX8_int8,CHANNELS_OUT19,6,6, 3, 1,INPUT_MATRIX8, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX8, WEIGHT_MATRIX20,CHANNELS20, WEIGHT_DIM20, 36, OUTPUT_MATRIX8, CHANNELS_OUT19);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX20,CHANNELS20, WEIGHT_DIM20, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias20,36, CHANNELS20);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS20, multiply20, add20, shift20, 0);
    printf("Layer 20 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX8_int8, WEIGHT_MATRIX21,CHANNELS_IN21,CHANNELS_OUT21, 36, OUTPUT_MATRIX8);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX21,CHANNELS_OUT21, CHANNELS_IN21, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias21,36, CHANNELS_OUT21);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS_OUT21, multiply21, add21, shift21, 0);
    printf("Layer 21 Done\n");

    im2col(OUTPUT_MATRIX8_int8,CHANNELS_OUT21,6,6, 3, 1,INPUT_MATRIX8, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX8, WEIGHT_MATRIX22,CHANNELS22, WEIGHT_DIM22, 36, OUTPUT_MATRIX8, CHANNELS_OUT21);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX22,CHANNELS22, WEIGHT_DIM22, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias22,36, CHANNELS22);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS22, multiply22, add22, shift22, 0);
    printf("Layer 22 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX8_int8, WEIGHT_MATRIX23,CHANNELS_IN23,CHANNELS_OUT23, 36, OUTPUT_MATRIX8);
    quantize_conv_layer(OUTPUT_MATRIX8,WEIGHT_MATRIX23,CHANNELS_OUT23, CHANNELS_IN23, 36,128); 
    add_bias(OUTPUT_MATRIX8, bias23,36, CHANNELS_OUT23);
    requantize_conv(OUTPUT_MATRIX8,OUTPUT_MATRIX8_int8, 36, CHANNELS_OUT23, multiply23, add23, shift23, 0);
    printf("Layer 23 Done\n");

    im2col(OUTPUT_MATRIX8_int8,CHANNELS_OUT23,6,6, 3, 2,INPUT_MATRIX9, 0,1,0,1,-128);
    conv_layer(INPUT_MATRIX9, WEIGHT_MATRIX24,CHANNELS24, WEIGHT_DIM24, 9, OUTPUT_MATRIX9, CHANNELS_OUT23);
    quantize_conv_layer(OUTPUT_MATRIX9,WEIGHT_MATRIX24,CHANNELS24, WEIGHT_DIM24,9,128); 
    add_bias(OUTPUT_MATRIX9, bias24,9, CHANNELS24);
    requantize_conv(OUTPUT_MATRIX9,OUTPUT_MATRIX9_int8, 9, CHANNELS24, multiply24, add24, shift24, 0);
    printf("Layer 24 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX9_int8, WEIGHT_MATRIX25,CHANNELS_IN25,CHANNELS_OUT25, 9, OUTPUT_MATRIX10);
    quantize_conv_layer(OUTPUT_MATRIX10,WEIGHT_MATRIX25,CHANNELS_OUT25, CHANNELS_IN25, 9,128); 
    add_bias(OUTPUT_MATRIX10, bias25,9, CHANNELS_OUT25);
    requantize_conv(OUTPUT_MATRIX10,OUTPUT_MATRIX10_int8, 9, CHANNELS_OUT25, multiply25, add25, shift25, 0);
    printf("Layer 25 Done\n");

    im2col(OUTPUT_MATRIX10_int8,CHANNELS_OUT25,3,3, 3, 1,INPUT_MATRIX10, 1,1,1,1,-128);
    conv_layer(INPUT_MATRIX10, WEIGHT_MATRIX26,CHANNELS26, WEIGHT_DIM26, 9, OUTPUT_MATRIX10, CHANNELS_OUT25);
    quantize_conv_layer(OUTPUT_MATRIX10,WEIGHT_MATRIX26,CHANNELS26, WEIGHT_DIM26,9,128); 
    add_bias(OUTPUT_MATRIX10, bias26,9, CHANNELS26);
    requantize_conv(OUTPUT_MATRIX10,OUTPUT_MATRIX10_int8, 9, CHANNELS26, multiply26, add26, shift26, 0);
    printf("Layer 26 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX10_int8, WEIGHT_MATRIX27,CHANNELS_IN27,CHANNELS_OUT27, 9, OUTPUT_MATRIX10);
    quantize_conv_layer(OUTPUT_MATRIX10,WEIGHT_MATRIX27,CHANNELS_OUT27, CHANNELS_IN27, 9,128); 
    add_bias(OUTPUT_MATRIX10, bias27,9, CHANNELS_OUT27);
    requantize_conv(OUTPUT_MATRIX10,OUTPUT_MATRIX10_int8, 9, CHANNELS_OUT27, multiply27, add27, shift27, 0);
    printf("Layer 27 Done\n");

    avg_pool_layer(OUTPUT_MATRIX10_int8, 9, 256, OUTPUT_MATRIX11_int8);
    printf("Layer 28 Done\n");

    pointwise_conv_layer(OUTPUT_MATRIX11_int8, WEIGHT_MATRIX29,CHANNELS_IN29,CHANNELS_OUT29, 1, OUTPUT_MATRIX12);
    quantize_conv_layer(OUTPUT_MATRIX12,WEIGHT_MATRIX29,CHANNELS_OUT29, CHANNELS_IN29, 1,128); 
    add_bias(OUTPUT_MATRIX12, bias29,1, CHANNELS_OUT29);
    requantize_conv(OUTPUT_MATRIX12,OUTPUT_MATRIX12_int8, 1, CHANNELS_OUT29, multiply29, add29, shift29, 1);
    printf("Layer 29 Done\n");

    printf("output at 0: %d \n",OUTPUT_MATRIX12_int8[0]);
    printf("output at 1: %d \n",OUTPUT_MATRIX12_int8[1]);

    softmax_and_output(OUTPUT_MATRIX12_int8,2);

    /*
    int obtained_label = 0;
    for(int i=0;i<100000;i++){

        microspeech_conv_layer();
        quantize_conv_layer();
        microspeech_bias_ReLu();
        requantize_conv();
        reshape_conv_output();
        microspeech_fc_layer();
        quantize_fc_layer();    
        obtained_label = requantize_fc();
        printf("Output Label:%d \n", obtained_label);
    }
    printf("Output Label:%d \n", obtained_label);*/
    return 0;
}

