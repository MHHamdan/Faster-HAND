"""
HAND V2 -- Efficient Long-Document HAND.

Stage 0 scope: engineering only. Nothing in this package changes the model. It provides

  hand_v2.data       a faster input pipeline that yields the *same* samples as hand/
  hand_v2.profiling  the measurements that justify it
  hand_v2.train      a V2 entry point that reuses tools/train_hand.py's flags and
                     records every run into experiments/benchmark_suite/

The frozen baseline in hand/ and tools/ is not imported-from-and-modified; V2 subclasses
and wraps it. See hand_v2/README.md.
"""
__version__ = "2.0.0.dev0"
