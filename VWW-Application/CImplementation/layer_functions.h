//Depthwise Convolution, input and weight matrix must already be flattened
void conv_layer(int8_t* input,int8_t* weights, int channels, int weight_dim, int output_dim, int32_t* output, int groups){
    printf("channels: %d \n", channels);
    printf("groups: %d \n", groups);
    //if groups is 1 each filter gets multiplied with each input channel
    if (groups == 1)
    {
        for (int i=0;i<channels; i++) { 
            for (int j=0;j<output_dim; j++) {
                int32_t temp=0;
                for (int k=0;k<weight_dim; k++){ 
                    int weight_ind = k*channels +i;
                    temp += weights[weight_ind]*input[k*output_dim+j];
                    /*if ((i==1)&&(j==0)){
                        printf("weight_ind: %d \n", weight_ind);
                        printf("k*output_dim+j: %d \n", k*output_dim+j);
                    }*/
                }
                output[i*output_dim+j] = temp;
                //printf("i*output_dim+j: %d \n", i*output_dim+j);
            }
        }
    }
    //if groups == channels each filter just gets multiplied with one channel
    else if (groups == channels)
    {
        for (int i=0;i<channels; i++) { 
            for (int j=0;j<output_dim; j++) { 
                int32_t temp=0;
                for (int k=0;k<weight_dim; k++){ 
                    int weight_ind = k*channels +i;
                    int input_ind = i*output_dim*weight_dim+k*output_dim+j;
                    temp += weights[weight_ind]*input[input_ind];
                    /*if ((i==1)&&(j==0)){
                        printf("weight_ind: %d \n", weight_ind);
                        printf("input_ind: %d \n", input_ind);
                    }*/
                }
                output[i*output_dim+j] = temp;
                //printf("i*output_dim+j: %d \n", i*output_dim+j);
            }
        }
    }
    //
    else
        printf("This case is not implemented/ not feasible. Make sure that groups == channels or groups == 1 ");
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