from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import mpi4py
from mpi4py import get_include as mpi_get_include

# You can use mpi4py's get_config() to get MPI compiler and linker flags
mpi_cfg = mpi4py.get_config()

extensions = [
    Extension(
        "bigfile.pyxbigfile",
        sources=[
            "bigfile/pyxbigfile.pyx",
            "src/bigfile.c",
            "src/bigfile-record.c",
            "src/bigfile-mpi.c",        # Include the MPI C code
        ],
        depends=[
            "src/bigfile.h",
            "src/bigfile-internal.h",
            "src/bigfile-mpi.h",        # Include the MPI header
        ],
        include_dirs=[
            "src/",
            numpy.get_include(),
            mpi_get_include(),          # Include mpi4py's include directory
        ] + mpi_cfg.get('include_dirs', []),  # Add MPI include dirs from mpi4py config
        libraries=mpi_cfg.get('libraries', []) + ['m'],  # Add MPI libraries from mpi4py config
        library_dirs=mpi_cfg.get('library_dirs', []),    # Add MPI library dirs from mpi4py config
        extra_compile_args=mpi_cfg.get('extra_compile_args', []),
        extra_link_args=mpi_cfg.get('extra_link_args', []),
    )
]

def find_version(path):
    import re
    # path shall be a plain ASCII text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(
    name="bigfile",
    version=find_version("bigfile/version.py"),
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    url="http://github.com/rainwoodman/bigfile",
    description="Python binding of BigFile, a peta-scale IO format",
    zip_safe=False,
    package_dir={'bigfile': 'bigfile'},
    install_requires=['cython', 'numpy', 'mpi4py'],  # Ensure mpi4py is installed
    scripts=['scripts/bigfile-convert', 'scripts/hdf2bigfile'],
    packages=['bigfile', 'bigfile.tests'],
    license='BSD-2-Clause',
    ext_modules=cythonize(extensions),
)
