/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/person_detection/main_functions.h"

// #include "tensorflow/lite/micro/examples/person_detection/detection_responder.h"
// #include "tensorflow/lite/micro/examples/person_detection/image_provider.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"
#include "tensorflow/lite/micro/examples/person_detection/no_person_image_data.h"
// #include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "mbed.h"
#include <cstdint>

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
// tflite::MicroProfiler* profiler = nullptr; // added by yujie
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 135 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop() {

//NO PERSON IMAGE DATA
  Timer t1;
  t1.start();
  // Get sample image from saved in header file (already preprocessed)
  for(int j=0; j< kMaxImageSize; j++){
    input->data.int8[j] = g_no_person_data[j];
  }
  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status;
  // for(int i=0; i<1000; i++){
  invoke_status = interpreter->Invoke();
  // }
  

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    //return;
  }

  TfLiteTensor* output = interpreter->output(0);

  // ### edited by yujie for sample input data
  // Process the inference results.
  // int8_t person_score = output->data.uint8[kPersonIndex];
  // int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  // RespondToDetection(error_reporter, person_score, no_person_score);
  // ### edited by yujie for sample input data
  TF_LITE_REPORT_ERROR(error_reporter, "Person_score: %d, No_person_score: %d\n", 
        static_cast<int>(output->data.uint8[kPersonIndex]), static_cast<int>(output->data.uint8[kNotAPersonIndex]));
  const int input_dim =2;
  int input[input_dim]; 
  input[0] = output->data.uint8[kPersonIndex];
  input[1] = output->data.uint8[kNotAPersonIndex];
  float OUTPUT32[input_dim];
  float SOFTMAX[input_dim];
  float softmax_sum = 0;
  for (int i=0; i<input_dim; i++) {
    if (input[i] < -128) {
      input[i] = -128;
    }
    else if (input[i] > 127) {
      input[i] = 127;
    }
    OUTPUT32[i] = input[i];
    OUTPUT32[i] -= 3;
    OUTPUT32[i] *= 0.038815176;
    SOFTMAX[i] = exp(OUTPUT32[i]);
    softmax_sum += SOFTMAX[i];
    //printf("SOFTMAX[i] %.6f \n", SOFTMAX[i]);
  }
    //printf("softmax_sum %.6f \n", softmax_sum);
  for (int i=0; i<input_dim; i++) {
    OUTPUT32[i] = SOFTMAX[i]/softmax_sum;
    if(i==0)
      printf("\rProbability NO Person %.6f \n", OUTPUT32[i]);
    if(i==1)
      printf("\rProbability Person %.6f \n", OUTPUT32[i]);
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
  printf("\rOutput 0 %.6f \n", OUTPUT32[0]);
  printf("\rOutput 1 %.6f \n", OUTPUT32[1]);
  t1.stop();
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke time: %d", static_cast<u_long>(t1.read_high_resolution_us())); 
 
}
