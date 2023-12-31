import os

# Use real tensors locally rather than fake tensors and
# then check that each remote call is computing the same result
# as the local tensor
check_correctness_per_operator = False
# instead of sending compiled functions to the worker to run
# run them locally istead with DTensors. Makes it easier to debug
# local
simulate_function_calls = False

# worker will print what commands it is running before running them
verbose_worker = False

# keep a cache of fake tensors to speed things up
do_fake_mode_caching = True

if check_correctness_per_operator:
    simulate_function_calls = True

# these are just for the nanoGPT example, putting
# them here to so they are easy to find
nanogpt_use_single_controller = True
nanogpt_compile = True

assert nanogpt_compile or not do_fake_mode_caching, "somewhere in there there are problems with view operators"

avoid_fake_mode_optimization = True
