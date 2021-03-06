#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import partial
import argparse
import numpy as np
import time
import sys

import tritongrpcclient.core as grpcclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()
    try:
        TRTISClient = grpcclient.InferenceServerClient(FLAGS.url)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = 'simple'

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('INPUT0'))
    inputs.append(grpcclient.InferInput('INPUT1'))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.ones(shape=(1, 16), dtype=np.int32)

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs.append(grpcclient.InferOutput('OUTPUT0'))
    outputs.append(grpcclient.InferOutput('OUTPUT1'))

    # Define the callback function. Note the last parameter should be
    # result InferenceServerClient would povide the results of an
    # inference in the same parameter.
    def callback(user_data, result):
        user_data.append(result)

    # list to hold the results of inference.
    user_data = []

    # Inference call
    TRTISClient.async_infer(partial(callback, user_data), inputs, outputs,
                            model_name)

    # Wait until the results are available in user_data
    time_out = 10
    while ((len(user_data) == 0) and time_out > 0):
        time_out = time_out - 1
        time.sleep(1)

    # Display and validate the available results
    if ((len(user_data) == 1)):
        # Validate the values by matching with already computed expected
        # values.
        output0_data = user_data[0].as_numpy('OUTPUT0')
        output1_data = user_data[0].as_numpy('OUTPUT1')
        for i in range(16):
            print(str(input0_data[0][i]) + " + " + str(input1_data[0][i]) +
                  " = " + str(output0_data[0][i]))
            print(str(input0_data[0][i]) + " - " + str(input1_data[0][i]) +
                  " = " + str(output1_data[0][i]))
            if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
                print("sync infer error: incorrect sum")
                sys.exit(1)
            if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
                print("sync infer error: incorrect difference")
                sys.exit(1)
        print("PASS: Async infer")

    TRTISClient.close()
