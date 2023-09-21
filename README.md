Getting Started With Examples
-----------------------------

Install the package:

    python setup.py develop

Run the tests (kinda noisy at this point)

    python test/simple.py

Prepare data for nanoGPT example:

    cd nanoGPT
    python data/shakespeare_char/prepare.py

Run the nanoGPT example:

    cd nanoGPT
    python train.py config/train_shakespeare_char.py

See flags in `single_controller/config.py` for changing behaviors.
