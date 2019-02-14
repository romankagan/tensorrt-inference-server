# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import sys
sys.path.append("../common")

from builtins import range
from future.utils import iteritems
import os
import time
import threading
import traceback
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status

_no_batching = (int(os.environ['NO_BATCHING']) == 1)
_model_instances = int(os.environ['MODEL_INSTANCES'])

if _no_batching:
    _trials = ("savedmodel_nobatch", "graphdef_nobatch")
elif os.environ['BATCHER_TYPE'] == "VARIABLE":
    _trials = []
else:
    _trials = ("custom", "savedmodel", "graphdef")

_protocols = ("http", "grpc")
_max_queue_delay_ms = 0
_check_exception = None

class SequenceBatcherTest(unittest.TestCase):
    def setUp(self):
        global _check_exception
        _check_exception = None

    def check_deferred_exception(self):
        if _check_exception is not None:
            raise _check_exception

    def check_sequence(self, trial, model_name, input_dtype, correlation_id,
                       sequence_thresholds, values, expected_result,
                       protocol, batch_size=1, sequence_name="<unknown>"):
        """Perform sequence of inferences. The 'values' holds a list of
        tuples, one for each inference with format:

        (flag_str, value, (ls_ms, gt_ms), (pre_delay_ms, post_delay_ms)

        """
        global _check_exception

        if (("savedmodel" in trial) or ("graphdef" in trial) or
            ("netdef" in trial) or ("custom" in trial)):
            tensor_shape = (1,)
        elif "plan" in trial:
            tensor_shape = (1,1,1)
        else:
            self.assertFalse(True, "unknown trial type: " + trial)

        # Can only send the request exactly once since it is a
        # sequence model with state
        configs = []
        if protocol == "http":
            configs.append(("localhost:8000", ProtocolType.HTTP, False))
        if protocol == "grpc":
            configs.append(("localhost:8001", ProtocolType.GRPC, False))
        if protocol == "streaming":
            configs.append(("localhost:8001", ProtocolType.GRPC, True))

        self.assertEqual(len(configs), 1)

        for config in configs:
            ctx = InferContext(config[0], config[1], model_name,
                               correlation_id=correlation_id, streaming=config[2],
                               verbose=True)
            # Execute the sequence of inference...
            try:
                seq_start_ms = int(round(time.time() * 1000))

                for flag_str, value, thresholds, delay_ms in values:
                    if delay_ms is not None:
                        time.sleep(delay_ms[0] / 1000.0)

                    flags = InferRequestHeader.FLAG_NONE
                    if flag_str is not None:
                        if "start" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
                        if "end" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

                    input_list = list()
                    for b in range(batch_size):
                        input_list.append(np.full(tensor_shape, value, dtype=input_dtype))

                    start_ms = int(round(time.time() * 1000))
                    results = ctx.run(
                        { "INPUT" : input_list }, { "OUTPUT" : InferContext.ResultFormat.RAW},
                        batch_size=batch_size, flags=flags)
                    end_ms = int(round(time.time() * 1000))

                    self.assertEqual(len(results), 1)
                    self.assertTrue("OUTPUT" in results)
                    result = results["OUTPUT"][0][0]
                    print("{}: {}".format(sequence_name, result))

                    if thresholds is not None:
                        lt_ms = thresholds[0]
                        gt_ms = thresholds[1]
                        if lt_ms is not None:
                            self.assertTrue((end_ms - start_ms) < lt_ms,
                                            "expected less than " + str(lt_ms) +
                                            "ms response time, got " + str(end_ms - start_ms) + " ms")
                        if gt_ms is not None:
                            self.assertTrue((end_ms - start_ms) > gt_ms,
                                            "expected greater than " + str(gt_ms) +
                                            "ms response time, got " + str(end_ms - start_ms) + " ms")
                    if delay_ms is not None:
                        time.sleep(delay_ms[1] / 1000.0)

                seq_end_ms = int(round(time.time() * 1000))
                self.assertEqual(result, expected_result)

                if sequence_thresholds is not None:
                    lt_ms = sequence_thresholds[0]
                    gt_ms = sequence_thresholds[1]
                    if lt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                        "sequence expected less than " + str(lt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
                    if gt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                        "sequence expected greater than " + str(gt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
            except Exception as ex:
                _check_exception = ex

    def check_sequence_async(self, trial, model_name, input_dtype, correlation_id,
                             sequence_thresholds, values, expected_result,
                             protocol, batch_size=1, sequence_name="<unknown>"):
        """Perform sequence of inferences using async run. The 'values' holds
        a list of tuples, one for each inference with format:

        (flag_str, value, pre_delay_ms)

        """
        global _check_exception

        if (("savedmodel" in trial) or ("graphdef" in trial) or
            ("netdef" in trial) or ("custom" in trial)):
            tensor_shape = (1,)
        elif "plan" in trial:
            tensor_shape = (1,1,1)
        else:
            self.assertFalse(True, "unknown trial type: " + trial)

        # Can only send the request exactly once since it is a
        # sequence model with state
        configs = []
        if protocol == "http":
            configs.append(("localhost:8000", ProtocolType.HTTP, False))
        if protocol == "grpc":
            configs.append(("localhost:8001", ProtocolType.GRPC, False))
        if protocol == "streaming":
            configs.append(("localhost:8001", ProtocolType.GRPC, True))

        self.assertEqual(len(configs), 1)

        for config in configs:
            ctx = InferContext(config[0], config[1], model_name,
                               correlation_id=correlation_id, streaming=config[2],
                               verbose=True)
            # Execute the sequence of inference...
            try:
                seq_start_ms = int(round(time.time() * 1000))
                result_ids = list()

                for flag_str, value, pre_delay_ms in values:
                    flags = InferRequestHeader.FLAG_NONE
                    if flag_str is not None:
                        if "start" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
                        if "end" in flag_str:
                            flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

                    input_list = list()
                    for b in range(batch_size):
                        input_list.append(np.full(tensor_shape, value, dtype=input_dtype))

                    if pre_delay_ms is not None:
                        time.sleep(pre_delay_ms / 1000.0)

                    result_ids.append(ctx.async_run(
                        { "INPUT" : input_list }, { "OUTPUT" : InferContext.ResultFormat.RAW},
                        batch_size=batch_size, flags=flags))

                seq_end_ms = int(round(time.time() * 1000))

                # Wait for the results in the order sent
                result = None
                for id in result_ids:
                    results = ctx.get_async_run_results(id, True)
                    self.assertEqual(len(results), 1)
                    self.assertTrue("OUTPUT" in results)
                    result = results["OUTPUT"][0][0]
                    print("{}: {}".format(sequence_name, result))

                self.assertEqual(result, expected_result)

                if sequence_thresholds is not None:
                    lt_ms = sequence_thresholds[0]
                    gt_ms = sequence_thresholds[1]
                    if lt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) < lt_ms,
                                        "sequence expected less than " + str(lt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
                    if gt_ms is not None:
                        self.assertTrue((seq_end_ms - seq_start_ms) > gt_ms,
                                        "sequence expected greater than " + str(gt_ms) +
                                        "ms response time, got " + str(seq_end_ms - seq_start_ms) + " ms")
            except Exception as ex:
                _check_exception = ex

    def check_setup(self, model_name):
        # Make sure test.sh set up the correct batcher settings
        ctx = ServerStatusContext("localhost:8000", ProtocolType.HTTP, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
        bconfig = ss.model_status[model_name].config.sequence_batching
        self.assertEqual(bconfig.max_queue_delay_microseconds, _max_queue_delay_ms * 1000) # 10 secs

    def check_status(self, model_name, static_bs, exec_cnt, infer_cnt):
        ctx = ServerStatusContext("localhost:8000", ProtocolType.HTTP, model_name, True)
        ss = ctx.get_server_status()
        self.assertEqual(len(ss.model_status), 1)
        self.assertTrue(model_name in ss.model_status,
                        "expected status for model " + model_name)
        vs = ss.model_status[model_name].version_status
        self.assertEqual(len(vs), 1)
        self.assertTrue(1 in vs, "expected status for version 1")
        infer = vs[1].infer_stats
        self.assertEqual(len(infer), len(static_bs),
                         "expected batch-sizes (" + ",".join(str(b) for b in static_bs) +
                         "), got " + str(vs[1]))
        for b in static_bs:
            self.assertTrue(b in infer,
                            "expected batch-size " + str(b) + ", got " + str(vs[1]))
        self.assertEqual(vs[1].model_execution_count, exec_cnt,
                        "expected model-execution-count " + str(exec_cnt) + ", got " +
                        str(vs[1].model_execution_count))
        self.assertEqual(vs[1].model_inference_count, infer_cnt,
                        "expected model-inference-count " + str(infer_cnt) + ", got " +
                        str(vs[1].model_inference_count))

    def get_expected_result(self, expected_result, value, trial, flag_str=None):
        # Adjust the expected_result for models that
        # couldn't implement the full accumulator. See
        # qa/common/gen_qa_sequence_models.py for more
        # information.
        if (not _no_batching and ("custom" not in trial)) or ("graphdef" in trial):
            expected_result = value
            if (flag_str is not None) and ("start" in flag_str):
                expected_result += 1
        return expected_result

    def test_simple_sequence(self):
        # Send one sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                try:
                    model_name = tu.get_sequence_model_name(trial, np.int32)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    self.check_sequence(trial, model_name, np.int32, 5,
                                        (3000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start", 1, None, None),
                                         (None, 2, None, None),
                                         (None, 3, None, None),
                                         (None, 4, None, None),
                                         (None, 5, None, None),
                                         (None, 6, None, None),
                                         (None, 7, None, None),
                                         (None, 8, None, None),
                                         ("end", 9, None, None)),
                                        self.get_expected_result(45, 9, trial, "end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, (1,), 9 * (idx + 1), 9 * (idx + 1))
                except InferenceServerException as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_length1_sequence(self):
        # Send a length-1 sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                try:
                    model_name = tu.get_sequence_model_name(trial, np.int32)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    self.check_sequence(trial, model_name, np.int32, 99,
                                        (3000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start,end", 42, None, None),),
                                        self.get_expected_result(42, 42, trial, "start,end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, (1,), (idx + 1), (idx + 1))
                except InferenceServerException as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_batch_size(self):
        # Send sequence with a batch-size > 1 and check for error.

        # When 4 model instances the max-batch-size is 1 so can't test
        # since that gives a different error: "batch-size 2 exceeds
        # maximum batch size"
        if (_model_instances == 4) or _no_batching:
            return

        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                try:
                    model_name = tu.get_sequence_model_name(trial, np.int32)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    self.check_sequence(trial, model_name, np.int32, 27,
                                        (3000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start", 1, None, None),
                                         ("end", 9, None, None)),
                                        self.get_expected_result(10, 9, trial, "end"),
                                        protocol, batch_size=2,
                                        sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.assertTrue(False, "expected error")
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith(
                            str("inference request to model '{}' must specify " +
                                "batch-size 1 due to requirements of sequence " +
                                "batcher").format(model_name)))

    def test_no_correlation_id(self):
        # Send sequence without correlation ID and check for error.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                try:
                    model_name = tu.get_sequence_model_name(trial, np.int32)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    self.check_sequence(trial, model_name, np.int32, 0, # correlation_id = 0
                                        (3000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start", 1, None, None),
                                         ("end", 9, None, None)),
                                        self.get_expected_result(10, 9, trial, "end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.assertTrue(False, "expected error")
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith(
                            str("inference request to model '{}' must specify a " +
                                "non-zero correlation ID").format(model_name)))

    def test_no_sequence_start(self):
        # Send sequence without start flag for never before seen
        # correlation ID. Expect failure.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                try:
                    model_name = tu.get_sequence_model_name(trial, np.int32)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    self.check_sequence(trial, model_name, np.int32, 37469245,
                                        (3000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        ((None, 1, None, None),
                                         (None, 2, None, None),
                                         ("end", 3, None, None)),
                                        self.get_expected_result(6, 3, trial, "end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.assertTrue(False, "expected error")
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith(
                            str("inference request for sequence 37469245 to " +
                                "model '{}' must specify the START flag on the first " +
                                "request of the sequence").format(model_name)))

    def test_no_sequence_start2(self):
        # Send sequence without start flag after sending a valid
        # sequence with the same correlation ID. Expect failure for
        # the second sequence.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                try:
                    model_name = tu.get_sequence_model_name(trial, np.int32)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    self.check_sequence(trial, model_name, np.int32, 3,
                                        (3000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start", 1, None, None),
                                         (None, 2, None, None),
                                         ("end", 3, None, None),
                                         (None, 55, None, None)),
                                        self.get_expected_result(6, 3, trial, None),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_status(model_name, (1,), 3 * (idx + 1), 3 * (idx + 1))
                    self.check_deferred_exception()
                    self.assertTrue(False, "expected error")
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith(
                            str("inference request for sequence 3 to model '{}' must " +
                                "specify the START flag on the first request of " +
                                "the sequence").format(model_name)))

    def test_no_sequence_end(self):
        # Send sequence without end flag. Use same correlation ID to
        # send another sequence. The first sequence will be ended
        # automatically but the second should complete successfully.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                try:
                    model_name = tu.get_sequence_model_name(trial, np.int32)

                    self.check_setup(model_name)
                    self.assertFalse("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                    self.assertFalse("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)

                    self.check_sequence(trial, model_name, np.int32, 4566,
                                        (3000, None),
                                        # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                                        (("start", 1, None, None),
                                         (None, 2, None, None),
                                         ("start", 42, None, None),
                                         ("end", 9, None, None)),
                                        self.get_expected_result(51, 9, trial, "end"),
                                        protocol, sequence_name="{}_{}".format(
                                            self._testMethodName, protocol))

                    self.check_deferred_exception()
                    self.check_status(model_name, (1,), 4 * (idx + 1), 4 * (idx + 1))
                except InferenceServerException as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_half_batch(self):
        # Test model instances together are configured with
        # total-batch-size 4.  Send two equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 2 inferences.
        for trial in _trials:
            try:
                model_name = tu.get_sequence_model_name(trial, np.int32)
                protocol = "streaming"

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 8)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 987,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           (None, 2, None),
                           (None, 3, None),
                           ("end", 4, None)),
                          self.get_expected_result(10, 4, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 988,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 0, None),
                           (None, 9, None),
                           (None, 5, None),
                           ("end", 13, None)),
                          self.get_expected_result(27, 13, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), 4 * min(2, _model_instances), 8)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_skip_batch(self):
        # Test model instances together are configured with
        # total-batch-size 4. Send four sequences in parallel where
        # two sequences have shorter length so that padding must be
        # applied correctly for the longer sequences.
        for trial in _trials:
            try:
                model_name = tu.get_sequence_model_name(trial, np.int32)
                protocol = "streaming"

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 12)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           ("end", 3, None)),
                          self.get_expected_result(4, 3, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           (None, 12, None),
                           (None, 13, None),
                           ("end", 14, None)),
                          self.get_expected_result(50, 14, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1003,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           ("end", 113, None)),
                          self.get_expected_result(224, 113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1004,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           (None, 1113, None),
                           ("end", 1114, None)),
                          self.get_expected_result(4450, 1114, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                threads[1].start()
                threads[3].start()
                time.sleep(1)
                threads[0].start()
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                if _model_instances == 1:
                    self.check_status(model_name, (1,), 4, 12)
                elif _model_instances == 2:
                    self.check_status(model_name, (1,), 8, 12)
                elif _model_instances == 4:
                    self.check_status(model_name, (1,), 12, 12)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_full_batch(self):
        # Test model instances together are configured with
        # total-batch-size 4. Send four equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 4 inferences.
        for trial in _trials:
            try:
                model_name = tu.get_sequence_model_name(trial, np.int32)
                protocol = "streaming"

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 12)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           (None, 2, None),
                           ("end", 3, None)),
                          self.get_expected_result(6, 3, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           (None, 12, None),
                           ("end", 13, None)),
                          self.get_expected_result(36, 13, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1003,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           (None, 112, None),
                           ("end", 113, None)),
                          self.get_expected_result(336, 113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1004,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(3336, 1113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), 3 * _model_instances, 12)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_backlog(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 5 equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 4 inferences plus the 5th should go in the
        # backlog and then get handled once there is a free slot.
        for trial in _trials:
            try:
                protocol = "streaming"
                model_name = tu.get_sequence_model_name(trial, np.int32)

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 12)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           (None, 2, None),
                           ("end", 3, None)),
                          self.get_expected_result(6, 3, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           (None, 12, None),
                           ("end", 13, None)),
                          self.get_expected_result(36, 13, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1003,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           (None, 112, None),
                           ("end", 113, None)),
                          self.get_expected_result(336, 113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1004,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(3336, 1113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1005,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11111, None),
                           (None, 11112, None),
                           ("end", 11113, None)),
                          self.get_expected_result(33336, 11113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), (3 * _model_instances) + 3, 15)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_backlog_fill(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 sequences in parallel, two of
        # which are shorter. Send 2 additional sequences that should
        # go into backlog but should immediately fill into the short
        # sequences.

        # Only works with 1 model instance since otherwise an instance
        # can run ahead and handle more work than expected (leads to
        # intermittent failures)
        if _model_instances != 1:
            return

        for trial in _trials:
            try:
                protocol = "streaming"
                model_name = tu.get_sequence_model_name(trial, np.int32)

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 10)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 2)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           (None, 2, None),
                           ("end", 3, None)),
                          self.get_expected_result(6, 3, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           ("end", 13, None)),
                          self.get_expected_result(24, 13, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1003,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           ("end", 113, None)),
                          self.get_expected_result(224, 113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1004,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(3336, 1113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1005,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start,end", 11111, None),),
                          self.get_expected_result(11111, 11111, trial, "start,end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1006,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start,end", 22222, None),),
                          self.get_expected_result(22222, 22222, trial, "start,end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                threads[0].start()
                threads[1].start()
                threads[2].start()
                threads[3].start()
                time.sleep(2)
                threads[4].start()
                threads[5].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), (3 * _model_instances), 12)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_backlog_fill_no_end(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 sequences in parallel, two of
        # which are shorter. Send 2 additional sequences that should
        # go into backlog but should immediately fill into the short
        # sequences. One of those sequences is filled before it gets
        # its end request.

        # Only works with 1 model instance since otherwise an instance
        # can run ahead and handle more work than expected (leads to
        # intermittent failures)
        if _model_instances != 1:
            return

        for trial in _trials:
            try:
                protocol = "streaming"
                model_name = tu.get_sequence_model_name(trial, np.int32)

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 10)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           (None, 2, None),
                           ("end", 3, None)),
                          self.get_expected_result(6, 3, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           ("end", 13, None)),
                          self.get_expected_result(24, 13, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1003,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           ("end", 113, None)),
                          self.get_expected_result(224, 113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1004,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(3336, 1113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1005,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start,end", 11111, None),),
                          self.get_expected_result(11111, 11111, trial, "start,end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1006,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 22222, None),
                           (None, 22223, None),
                           ("end", 22224, 2000),),
                          self.get_expected_result(66669, 22224, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                threads[0].start()
                threads[1].start()
                threads[2].start()
                threads[3].start()
                time.sleep(2)
                threads[4].start()
                threads[5].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), (3 * _model_instances) + 2, 14)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_backlog_same_correlation_id(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 4 inferences. Send a 5th with the same
        # correlation ID as one of the first four.
        for trial in _trials:
            try:
                protocol = "streaming"
                model_name = tu.get_sequence_model_name(trial, np.int32)

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 12)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 2)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           (None, 2, None),
                           ("end", 3, None)),
                          self.get_expected_result(6, 3, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           (None, 12, None),
                           ("end", 13, None)),
                          self.get_expected_result(36, 13, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1003,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           (None, 112, None),
                           ("end", 113, None)),
                          self.get_expected_result(336, 113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1004,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(3336, 1113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11111, None),
                           ("end", 11113, None)),
                          self.get_expected_result(22224, 11113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                threads[0].start()
                threads[1].start()
                threads[2].start()
                threads[3].start()
                time.sleep(2)
                threads[4].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), (3 * _model_instances) + 2, 14)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_backlog_same_correlation_id_no_end(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 sequences in parallel and
        # make sure they get completely batched into batch-size 4
        # inferences. One of the sequences is shorter and does not
        # have an end marker but has same correlation ID as the 5th
        # sequence. We expect that short sequence to get ended early
        # (because of the same correlation ID) and make room for the
        # 5th sequence.

        # Only works with 1 model instance since otherwise an instance
        # can run ahead and handle more work than expected (leads to
        # intermittent failures)
        if _model_instances != 1:
            return

        for trial in _trials:
            try:
                protocol = "streaming"
                model_name = tu.get_sequence_model_name(trial, np.int32)

                self.check_setup(model_name)

                # Need scheduler to wait for queue to contain all
                # inferences for both sequences.
                self.assertTrue("TRTSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_DELAY_SCHEDULER"]), 16)
                self.assertTrue("TRTSERVER_BACKLOG_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRTSERVER_BACKLOG_DELAY_SCHEDULER"]), 0)

                threads = []
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1, None),
                           (None, 3, None)),
                          self.get_expected_result(4, 3, trial, None),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1002,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11, None),
                           (None, 12, None),
                           (None, 12, None),
                           ("end", 13, None)),
                          self.get_expected_result(48, 13, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1003,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 111, None),
                           (None, 112, None),
                           (None, 112, None),
                           ("end", 113, None)),
                          self.get_expected_result(448, 113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1004,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 1111, None),
                           (None, 1112, None),
                           (None, 1112, None),
                           ("end", 1113, None)),
                          self.get_expected_result(4448, 1113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))
                threads.append(threading.Thread(
                    target=self.check_sequence_async,
                    args=(trial, model_name, np.int32, 1001,
                          (3000, None),
                          # (flag_str, value, pre_delay_ms)
                          (("start", 11111, None),
                           ("end", 11113, None)),
                          self.get_expected_result(22224, 11113, trial, "end"),
                          protocol),
                    kwargs={'sequence_name' : "{}_{}".format(self._testMethodName, protocol)}))

                threads[0].start()
                threads[1].start()
                threads[2].start()
                threads[3].start()
                time.sleep(2)
                threads[4].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, (1,), 4 * _model_instances, 16)
            except InferenceServerException as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

if __name__ == '__main__':
    unittest.main()