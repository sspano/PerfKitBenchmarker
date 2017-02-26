import sys
import logging


class __State():
  pass


__m = __State()


def IsAborted():
  return __m.__ABORT_CALLED


def AbortWithoutCleanup(message):
  __m.__ABORT_CALLED = True
  logging.error('Abort called - exiting without cleanup: %s' %  message)
  sys.exit(1)

