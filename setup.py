""" :noindex:
Setup.py file with generic info
"""
import os
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as testcommand


def read(fname) -> str:
    """From Wenjie Lei 2019:
    Utility function to read the README.md file.
    Used for the long_description.  It's nice, because now 1) we have a top levelx
    README.md file and 2) it's easier to type in the README.md file than to put a raw
    string in below ...
    """

    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception as e:
        return "Can't open %s" % fname


class PyTest(testcommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.tests")]

    def initialize_options(self):
        testcommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        import sys
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name="lwsspy",
    description="LWSS Collection of Python Function",
    long_description="%s" % read("README.md"),
    version="0.0.1",
    author="Lucas Sawade",
    author_email="lsawade@princeton.edu",
    license='GNU Lesser General Public License, Version 3',
    keywords="collection, functions",
    url='https://github.com/lsawade/GCMT3D',
    packages=find_packages(exclude=['*.Notebooks', '*.notebooks.*',
                                    'notebooks.*', 'notebooks',
                                    'paraview_tools']),
    package_dir={"": "."},
    include_package_data=True,
    # exclude_package_data={'lwsspy': ['download_cache']},
    package_data={'lwsspy': [
        'download_cache/*',
        'plot_util/fonts/*.ttc',
        'plot_util/fonts/*.ttf',
        'constant_data/gcmt/*.csv',
        'constant_data/ttc.mat'
    ]},
    install_requires=['numpy', 'matplotlib',  'obspy',
                      'PyYAML', 'h5py', 'mpi4py', 'matplotlib',
                      'pyasdf', 'autopep8', 'xarray', 'beautifulsoup4',
                      #   'pyvista'
                      ],
    tests_require=['pytest'],
    cmdclass={'tests': PyTest},
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        ("License :: OSI Approved "
         ":: GNU General Public License v3 or later (GPLv3+)"),
    ],
    # install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme", "numpydoc"],
        "tests": ["pytest", "py"]
    },
    entry_points={
        'console_scripts': [
            'invert-cmt = lwsspy.seismo.invertcmt.GCMT3DInversion:bin',
            'download-data = lwsspy.seismo.download_waveforms_to_storage:bin',
            'plot_csv_depth_slice = lwsspy.seismo.specfem.plot_csv_depth_slice:bin',
            'plot_specfem_xsec_depth = lwsspy.seismo.specfem.plot_specfem_xsec_depth:bin',
            'pick_data_from_image = lwsspy.plot_util.pick_data_from_image:bin'
        ]
    }
)
