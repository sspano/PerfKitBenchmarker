
import logging
from datetime import datetime
from perfkitbenchmarker import configs
from perfkitbenchmarker import flags
from perfkitbenchmarker import timing_util
from perfkitbenchmarker import stages
from perfkitbenchmarker import sample
import logging


flags.DEFINE_string('parallel_prefix', 'parallel',
                    'Prefix to add before all benchmark test names')

FLAGS = flags.FLAGS

SAMPLE_PREFIX = 'parallel_'

BENCHMARK_NAME = 'parallel'
BENCHMARK_CONFIG = """
parallel:
  description: Run multiple benchmarks at the same time
"""

def GetConfig(user_config):
  return configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)


def GetUniqueSubBenchmarks(benchmark_spec):
  seen = set()
  for benchmark_round in benchmark_spec.pkb['subbenchmarks']:
    for bm in benchmark_round:
      if bm.name not in seen:
        yield bm
        seen.add(bm.name)


def Prepare(benchmark_spec):
  timer = timing_util.IntervalTimer()
  for bm in GetUniqueSubBenchmarks(benchmark_spec):
    bm.sample_name = '%s_%s' % ('parallel', bm.name)
    bm.vms = benchmark_spec.vms
    benchmark_spec.pkb['prepare'](bm, timer)


def _KeepOnlyRunStageFlag():
  orig_run_stage = FLAGS.run_stage[:]
  for st in (stages.PROVISION, stages.PREPARE, stages.CLEANUP, stages.TEARDOWN):
    try:
      FLAGS.run_stage.remove(st)
    except:
      pass
  logging.info('Original run stages: %s reduced to %s', orig_run_stage, FLAGS.run_stage)
  return orig_run_stage


def Run(benchmark_spec):
  results = list()
  orig_run_stage = _KeepOnlyRunStageFlag()
  benchmarks = benchmark_spec.pkb['subbenchmarks']
  benchmark_names = [[_.name for _ in bm] for bm in benchmarks]
  results.append(sample.Sample('overall_start', 0, 's', dict(sorties=benchmark_names)))
  for pos, bm_round in enumerate(benchmarks):
    results.append(sample.Sample('sortie_start', pos, 'index', dict(sorties=[_.name for _ in bm])))
    ret_code = benchmark_spec.pkb['run'](bm_round)
    results.append(sample.Sample('sortie_end', pos, 'index'))
    if ret_code != 0:
      # probably should indicate failure
      return results
  results.append(sample.Sample('overall_end', 0, 's'))
  flags.run_stage = orig_run_stage
  return results


def Cleanup(benchmark_spec):
  timer = timing_util.IntervalTimer()
  for bm in GetUniqueSubBenchmarks(benchmark_spec):
    if stages.CLEANUP in flags.run_stage:
      benchmark_spec.pkb['cleanup'](bm, timer)
