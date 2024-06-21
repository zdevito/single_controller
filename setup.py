from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
      CppExtension,
      BuildExtension
)
from subprocess import run
import os
from glob import glob

controller_C = CppExtension(
      'controller._gradient_generator',
      ['controller/_gradient_generator.cpp'],
      extra_compile_args=['-g'],
      include_dirs = [os.path.dirname(os.path.abspath(__file__))],
)


setup(
    name='single_controller',
    version='1.0',
    packages=find_packages(),
    install_requires=[],
    author='Your Name',
    description='Single controller',
    ext_modules=[controller_C],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
)

with open('compile_commands.json', 'w') as cc:
      run(['ninja', '-C', glob('build/temp.*')[0], '-t', 'compdb'], stdout=cc)
