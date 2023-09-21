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
verbose_worker =  False


if check_correctness_per_operator:
    simulate_function_calls = True

# these are just for the nanoGPT example, putting
# them here to so they are easy to find
nanogpt_use_single_controller = True
nanogpt_compile = True
