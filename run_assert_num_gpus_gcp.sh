#!/bin/bash

while ./pkb.py --benchmarks=assert_num_gpus --cloud=GCP --benchmark_config_file=shoc_config.yml --flag_matrix=GCP --stop_after_benchmark_failure --gce_network_name=default 2>&1 | tee gcp_num_gpus_test.log; do :; done
