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

import re
import os
from perfkitbenchmarker import abort
from perfkitbenchmarker import configs
from perfkitbenchmarker import flags
from perfkitbenchmarker import sample
from perfkitbenchmarker import regex_util
from perfkitbenchmarker.linux_packages import cuda_toolkit_8
from perfkitbenchmarker import num_gpus_map_util

FLAGS = flags.FLAGS

BENCHMARK_NAME = 'assert_num_gpus'
# Note on the config: gce_migrate_on_maintenance must be false,
# because GCE does not support migrating the user's GPU state.
BENCHMARK_CONFIG = """
assert_num_gpus:
  description: Assert correct number of gpus on VMs
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
  cuda_toolkit_8.CheckPrerequisites()


def Prepare(benchmark_spec):
  abort.AbortWithoutCleanup('shit')
  vm = benchmark_spec.vms[0]
  vm.Install('cuda_toolkit_8')
  

def Run(benchmark_spec):
  vm = benchmark_spec.vms[0]
  cuda_toolkit_8.SetAndConfirmGpuClocks(vm)

  for i in range(100): 
    num_gpus = cuda_toolkit_8.QueryNumberOfGpus(vm)
    expected_num_gpus = num_gpus_map_util.num_gpus_map[vm.machine_type]
    if num_gpus != expected_num_gpus: 
      abort.AbortWithoutCleanup(
          'got incorrect number of gpus. Expected %s, recieved %s'
          % (expected_num_gpus, num_gpus)) 
  return results


def Cleanup(benchmark_spec):
  pass

