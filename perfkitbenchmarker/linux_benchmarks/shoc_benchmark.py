# Copyright 2017 PerfKitBenchmarker Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Runs SHOC benchmark"""

import numpy
import re
import os
from perfkitbenchmarker import configs
from perfkitbenchmarker import flags
from perfkitbenchmarker import sample
from perfkitbenchmarker import regex_util
from perfkitbenchmarker.linux_packages import shoc_benchmark_suite
from perfkitbenchmarker.linux_packages import cuda_toolkit_8


flags.DEFINE_integer('shoc_iterations', 1,
                     'number of iterations to run',
                     lower_bound=1)


FLAGS = flags.FLAGS

BENCHMARK_NAME = 'shoc'
BENCHMARK_VERSION = '0.2'
# Note on the config: gce_migrate_on_maintenance must be false,
# because GCE does not support migrating the user's GPU state.
BENCHMARK_CONFIG = """
shoc:
  description: Runs SHOC Benchmark Suite.
  flags:
    gce_migrate_on_maintenance: False
  vm_groups:
    default:
      vm_spec:
        GCP:
          image: ubuntu1604-cuda-hpl
          machine_type: n1-standard-4-k80x1
          zone: us-east1-d
          boot_disk_size: 200
        AWS:
          image: ami-a9d276c9
          machine_type: p2.xlarge
          zone: us-west-2b
          boot_disk_size: 200
        Azure:
          image: Canonical:UbuntuServer:16.04.0-LTS:latest
          machine_type: Standard_NC6
          zone: eastus
"""


def GetConfig(user_config):
  config = configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)
  return config


def CheckPrerequisites(benchmark_config):
  """Verifies that the required resources are present.

  Raises:
    perfkitbenchmarker.data.ResourceNotFound: On missing resource.
  """
  #cuda_toolkit_8.CheckPrerequisites()


def Prepare(benchmark_spec):
  """Install SHOC.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
  """
  vm = benchmark_spec.vms[0]
  vm.Install('shoc_benchmark_suite')


def _ExtractResult(shoc_output, result_name):
  result_line = [x for x in shoc_output.splitlines() if x.find(result_name) != -1][0].split() #TODO: ew
  result_value = float(result_line[-2])
  result_units = result_line[-1]
  return (result_value, result_units)

def _MakeSamplesFromOutput(stdout, metadata):
  results = []
  for metric in ('stencil:', 'stencil_dp:'):
    (value, unit) = _ExtractResult(stdout, metric) 
    results.append(sample.Sample(
        metric[:-1], # strip trailing colon
        value,
        unit,
        metadata))
  return results
  

def Run(benchmark_spec):
  """Sets the GPU clock speed and runs the SHOC benchmark.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.

  Returns:
    A list of sample.Sample objects.
  """
  vm = benchmark_spec.vms[0]
  # Note:  The clock speed is set in this function rather than Prepare()
  # so that the user can perform multiple runs with a specified
  # clock speed without having to re-prepare the VM.
  cuda_toolkit_8.SetAndConfirmGpuClocks(vm)
  num_iterations = FLAGS.shoc_iterations
  stencil2d_path = os.path.join(shoc_benchmark_suite.SHOC_BIN_DIR,
                                 'TP', 'CUDA', 'Stencil2D'
  num_gpus = cuda_toolkit_8.QueryNumberOfGpus(vm)
  metadata = {}
  results = []
  metadata['benchmark_version'] = BENCHMARK_VERSION
  metadata['num_iterations'] = num_iterations
  metadata['num_gpus'] = num_gpus
  metadata['memory_clock_MHz'] = FLAGS.gpu_clock_speeds[0]
  metadata['graphics_clock_MHz'] = FLAGS.gpu_clock_speeds[1]
  run_command = 'mpirun -np %s %s --customSize 20480,20480' %
    (num_gpus, stencil2d_path)
  metadata['command'] = run_cmd
  stdout, _ = vm.RemoteCommand(run_command, should_log=True)
  results.extend(_MakeSamplesFromOutput(stdout, metadata))
  return results
  #run_command = ('%s/extras/demo_suite/bandwidthTest --device=all'
  #               % cuda_toolkit_8.CUDA_TOOLKIT_INSTALL_DIR)
  #for i in range(num_iterations):
  #  stdout, _ = vm.RemoteCommand(run_command, should_log=True)
  #  raw_results.append(_ParseOutputFromSingleIteration(stdout))
  #  if 'device_info' not in metadata:
  #    metadata['device_info'] = _ParseDeviceInfo(stdout)
  #return _CalculateMetricsOverAllIterations(raw_results, metadata)


def Cleanup(benchmark_spec):
  pass

