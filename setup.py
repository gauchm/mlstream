from setuptools import setup
import os


def readme():
    with open(os.path.dirname(os.path.realpath(__file__)) + '/README.md') as f:
        return f.read()

requires = [
    'pandas',
    'numpy',
    'h5py',
    'tqdm',
    'netCDF4',
    'numba',
    'scipy',
]

# to save build resources, we mock torch and xgboost while building the docs
if not os.getenv('READTHEDOCS'):
  requires.append('torch')

setup(name='mlstream',
      version='0.1.2',
      description='Machine learning for streamflow prediction',
      long_description=readme(),
      long_description_content_type='text/markdown',
      keywords='ml hydrology streamflow machine learning',
      url='http://github.com/gauchm/mlstream',
      author='Martin Gauch',
      author_email='martin.gauch@uwaterloo.ca',
      license='Apache-2.0',
      packages=['mlstream', 'mlstream.models'],
      install_requires=requires,
      include_package_data=True,
      zip_safe=False)
