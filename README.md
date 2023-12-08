Getting Started With Examples
-----------------------------

Install the package:

    pip install -e .

For supervisor
--------------

Run the example_train application, which simulates training.
There is some stuff commented out which can be enabled to inject failures.

    python example_train/launch.py

The other files in the folder are the different parts of the job (health_check, train, supervisor script).

For single_controller tensors
-----------------------------

Run the tests (kinda noisy at this point)

    python test/simple.py

Prepare data for nanoGPT example:

    cd nanoGPT
    python data/shakespeare_char/prepare.py

Run the nanoGPT example:

    cd nanoGPT
    python train.py config/train_shakespeare_char.py

See flags in `single_controller/config.py` for changing behaviors.
