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
from perfkitbenchmarker import configs
from perfkitbenchmarker import flags
from perfkitbenchmarker import sample
from perfkitbenchmarker import regex_util
from perfkitbenchmarker.linux_packages import cuda_toolkit_8


FLAGS = flags.FLAGS

BENCHMARK_NAME = 'shoc'
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
  cuda_toolkit_8.CheckPrerequisites()


def Prepare(benchmark_spec):
  """Install CUDA toolkit 8.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
  """
  vm = benchmark_spec.vms[0]
  vm.Install('shoc_benchmark_suite')


def Run(benchmark_spec):
  pass


def Cleanup(benchmark_spec):
  pass

