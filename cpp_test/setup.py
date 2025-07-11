from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cpp_add_extension',
    ext_modules=[
        CppExtension('cpp_add_extension', ['main.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })