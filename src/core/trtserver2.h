// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

/// \file

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "src/core/trtserver.h"

#ifdef __cplusplus
extern "C" {
#endif

/// TRTSERVER2_InferenceRequest
///
/// Object representing an inference request. The inference request
/// provides the meta-data and input tensor values needed for an
/// inference and returns the inference result meta-data and output
/// tensors. An inference request object can be modified and reused
/// multiple times.
///

/// Create a new inference request object.
/// \param inference_request Returns the new request object.
/// \param server the inference server object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestNew(
    TRTSERVER2_InferenceRequest** inference_request, TRTSERVER_Server* server);

/// Delete an inference request object.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestDelete(
    TRTSERVER2_InferenceRequest* inference_request);

/// Set the ID for a request.
/// \param inference_request The request object.
/// \param id The ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetId(
    TRTSERVER2_InferenceRequest* inference_request, const char* id);

/// Set the flag(s) associated with a request. 'flags'
/// should holds a bitwise-or of all flag values, see
/// TRTSERVER_Request_Options_Flag for available flags.
/// \param inference_request The request object.
/// \param flags The flags.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetFlags(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t flags);

/// The correlation ID of the inference request. Default is 0, which
/// indictes that the request has no correlation ID. The correlation ID
/// is used to indicate two or more inference request are related to
/// each other. How this relationship is handled by the inference
/// server is determined by the model's scheduling policy.
/// \param inference_request The request object.
/// \param correlation_id The correlation ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetCorrelationId(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t correlation_id);

/// Set the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
/// \param inference_request The request object.
/// \param priority The priority level.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetPriority(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t priority);

/// Set the timeout for a request, in microseconds. The default is 0
/// which indicates that the request has no timeout.
/// \param inference_request The request object.
/// \param timeout_us The timeout, in microseconds.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetTimeoutMicroseconds(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t timeout_us);

/// Add an input to a request.
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param shape The shape of the input.
/// \param shape_count The number of dimensions of 'shape'.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestAddInput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const int64_t* shape, uint64_t shape_count);

/// Remove an input from a request.
/// \param inference_request The request object.
/// \param name The name of the input.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestRemoveInput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Remove all inputs from a request.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestClearInputs(
    TRTSERVER2_InferenceRequest* inference_request);

/// Assign a buffer of data to an input. The buffer will be appended
/// to any existing buffers for that input. The 'inference_request'
/// object takes ownership of the buffer and so the caller should not
/// modify or free the buffer until that ownership is released by
/// 'inference_request' being deleted or by the input being removed
/// from 'inference_request'.
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param base The base address of the input data.
/// \param byte_size The size, in bytes, of the input data.
/// \param memory_type The memory type of the input data.
/// \param memory_type_id The memory type id of the input data.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestAppendInputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRTSERVER_Memory_Type memory_type,
    int64_t memory_type_id);

/// Clear all input data from an input, releasing ownership of the
/// buffer(s) that were appended to the input with
/// TRTSERVER2_InferenceRequestAppendInputData.
/// \param inference_request The request object.
/// \param name The name of the input.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestClearInputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Add an output request to a request.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestAddRequestedOutput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Remove an output request from a request.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveRequestedOutput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Remove all output requests from a request.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestClearRequestedOutputs(
    TRTSERVER2_InferenceRequest* inference_request);

/// Set that an output should be returned as a tensor of
/// classification strings instead of as the tensor defined by the model.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param count Indicates how many classification values should be
/// returned for the output. The 'count' highest priority values are
/// returned. The default is 0, indicating that the output tensor
/// should not be returned as a classification.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetRequestedOutputClassificationCount(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    uint32_t count);





/// Return the success or failure status of the inference
/// request. Return a TRTSERVER_Error object on failure, return nullptr
/// on success.
/// \param response The response object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseStatus(
    TRTSERVER_InferenceResponse* response);



/// Get the results data for a named output. The result data is
/// returned as the base pointer to the data and the size, in bytes, of
/// the data. The caller does not own the returned data and must not
/// modify or delete it. The lifetime of the returned data extends only
/// as long as 'response' and must not be accessed once 'response' is
/// deleted.
/// \param response The response object.
/// \param name The name of the output.
/// \param base Returns the result data for the named output.
/// \param byte_size Returns the size, in bytes, of the output data.
/// \param memory_type Returns the memory type of the output data.
/// \param memory_type_id Returns the memory type id of the output data.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseOutputData(
    TRTSERVER_InferenceResponse* response, const char* name, const void** base,
    size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
    int64_t* memory_type_id);


/// Type for inference completion callback function. If non-nullptr,
/// the 'trace_manager' object is the trace manager associated with
/// the request that is completing. The callback function takes
/// ownership of the TRTSERVER_TraceManager object and must call
/// TRTSERVER_TraceManagerDelete to release the object. The callback
/// function takes ownership of the TRTSERVER2_InferenceRequest object
/// and must call TRTSERVER2_InferenceRequestDelete to release the
/// object. The 'userp' data is the same as what is supplied in the
/// call to TRTSERVER2_ServerInferAsync.
typedef void (*TRTSERVER2_InferenceCompleteFn_t)(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER2_InferenceRequest* request, void* userp);

/// Perform inference using the meta-data and inputs supplied by the
/// 'inference_request'. The caller releases ownership of
/// 'inference_request' and 'trace_manager' and must not access them
/// in any way after this call, until ownership is returned via the
/// completion function.
/// \param server The inference server object.
/// \param trace_manager The trace manager object for this request, or
/// nullptr if no tracing.
/// \param inference_request The request object.
/// \param response_allocator The TRTSERVER_ResponseAllocator to use
/// to allocate buffers to hold inference results.
/// \param response_allocator_userp User-provided pointer that is
/// delivered to the response allocator's allocation function.
/// \param complete_fn The function called when the inference
/// completes.
/// \param complete_userp User-provided pointer that is delivered to
/// the completion function.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_ServerInferAsync(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER2_InferenceRequest* inference_request,
    TRTSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRTSERVER2_InferenceCompleteFn_t complete_fn, void* complete_userp);

#ifdef __cplusplus
}
#endif
