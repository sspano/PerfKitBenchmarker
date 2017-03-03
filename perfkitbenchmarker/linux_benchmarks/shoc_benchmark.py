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
BENCHMARK_VERSION = '0.22'
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
          image: ubuntu-1604-xenial-v20170302
          image_project: ubuntu-os-cloud
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
  
def _MakeSamplesFromStencilOutput(stdout, metadata):
  dp_mean_results = [x for x in stdout.splitlines() if x.find('DP_Sten2D(mean)') != -1][0].split()
  dp_units = dp_mean_results[2]
  dp_median = float(dp_mean_results[3])
  dp_mean = float(dp_mean_results[4])
  dp_stddev = float(dp_mean_results[5])
  dp_min = float(dp_mean_results[6])
  dp_max = float(dp_mean_results[7])
   
  sp_mean_results = [x for x in stdout.splitlines() if x.find('SP_Sten2D(mean)') != -1][0].split()
  sp_units = sp_mean_results[2]
  sp_median = float(sp_mean_results[3])
  sp_mean = float(sp_mean_results[4])
  sp_stddev = float(sp_mean_results[5])
  sp_min = float(sp_mean_results[6])
  sp_max = float(sp_mean_results[7])

  results = []
  results.append(sample.Sample(
      'Stencil2D DP mean',
      dp_mean,
      dp_units,
      metadata))

  results.append(sample.Sample(
      'Stencil2D SP mean',
      sp_mean,
      sp_units,
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
                                'TP', 'CUDA', 'Stencil2D')
  num_gpus = cuda_toolkit_8.QueryNumberOfGpus(vm)
  metadata = {}
  results = []
  metadata['benchmark_version'] = BENCHMARK_VERSION
  metadata['num_iterations'] = num_iterations
  metadata['num_gpus'] = num_gpus
  metadata['memory_clock_MHz'] = FLAGS.gpu_clock_speeds[0]
  metadata['graphics_clock_MHz'] = FLAGS.gpu_clock_speeds[1]
  run_command = ('mpirun -np %s %s --customSize 19456,19456' %
                 (num_gpus, stencil2d_path))
  metadata['run_command'] = run_command
  stdout, _ = vm.RemoteCommand(run_command, should_log=True)
  results.extend(_MakeSamplesFromStencilOutput(stdout, metadata))
  return results


def Cleanup(benchmark_spec):
  pass

