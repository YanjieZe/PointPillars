from platform import version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pillars_form',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('pillars_form.cuda', [
            'src/pillars_form.cpp',
            'src/pillars_form_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })