
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

SUBBENCHMARK_KEY = 'subbenchmarks'

def GetConfig(user_config):
  return configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)


def GetSubBenchmarks(benchmark_spec, want_unique=True):
  """Return the benchmarks to run from the benchmark_spec.

  Args:
     benchmark_spec: the benchmark spec, will look to the dictionary
       field "pkb" and the "subbenchmarks" key
     want_unique: True if you only want the unique benchmarks based on
       their name

  Yields:
    Benchmarks from the benchmark_spec
  """
  seen = set()
  for benchmark_round in benchmark_spec.pkb[SUBBENCHMARK_KEY]:
    for bm in benchmark_round:
      if want_unique and bm.name in seen:
        continue
      yield bm
      seen.add(bm.name)


def Prepare(benchmark_spec):
  timer = timing_util.IntervalTimer()
  # ALL the benchmarks should get the sample_name and vms set on them
  for bm in GetSubBenchmarks(benchmark_spec, False):
    bm.sample_name = '%s_%s' % ('parallel', bm.name)
    bm.vms = benchmark_spec.vms
  # but ONLY run the prepare phase on unique benchmarks
  for bm in GetSubBenchmarks(benchmark_spec):
    benchmark_spec.pkb['prepare'](bm, timer)


def _KeepOnlyRunStageFlag():
  """Sets the run_stage flag to only have the "Run" phase.

  Returns:
    the original run_stage flag
  """
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
  benchmarks = benchmark_spec.pkb[SUBBENCHMARK_KEY]
  benchmark_names = [[_.name for _ in bm] for bm in benchmarks]
  results.append(sample.Sample('overall_start', 0, 's', dict(sorties=benchmark_names)))
  for pos, bm_round in enumerate(benchmarks):
    names = [_.name for _ in bm_round]
    logging.info('Running tests in parallel: %s', names)
    results.append(sample.Sample('sortie_start', pos, 'index', dict(sorties=names)))
    ret_code = benchmark_spec.pkb['run'](bm_round)
    results.append(sample.Sample('sortie_end', pos, 'index'))
    if ret_code != 0:
      logging.warn('Failed when running %s', names)
      # probably should indicate failure
      return results
  results.append(sample.Sample('overall_end', 0, 's'))
  # set the run_stage flag back to normal
  flags.run_stage = orig_run_stage
  return results


def Cleanup(benchmark_spec):
  timer = timing_util.IntervalTimer()
  # only do the cleanup on unique benchmarks
  for bm in GetSubBenchmarks(benchmark_spec):
    if stages.CLEANUP in flags.run_stage:
      benchmark_spec.pkb['cleanup'](bm, timer)
