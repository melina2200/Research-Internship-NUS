/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_TIME_H_
#define TENSORFLOW_LITE_MICRO_MICRO_TIME_H_

#include <cstdint>

namespace tflite {

// These functions should be implemented by each target platform, and provide an
// accurate tick count along with how many ticks there are per second.
int32_t ticks_per_second();

// Return time in ticks.  The meaning of a tick varies per platform.
int32_t GetCurrentTimeTicks();

int32_t TicksToUs(int32_t ticks);


}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_TIME_H_
