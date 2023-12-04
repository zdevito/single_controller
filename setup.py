from setuptools import setup, find_packages

setup(
    name='single_controller',
    version='1.0',
    packages=find_packages(),
    install_requires=[],
    author='Your Name',
    description='Single controller',
    entry_points = {
        'console_scripts': [
            'example_train = example_train.supervise:main'
        ]
    }
)
