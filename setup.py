from setuptools import setup
from os import path


def readme():
    with open(path.dirname(path.realpath(__file__)) + '/README.md') as f:
        return f.read()


setup(name='mlstream',
      version='0.1',
      description='Machine learning for streamflow prediction',
      long_description=readme(),
      long_description_content_type='text/markdown',
      keywords='ml hydrology streamflow machine learning',
      url='http://github.com/gauchm/mlstream',
      author='Martin Gauch',
      author_email='martin.gauch@uwaterloo.ca',
      license='Apache-2.0',
      packages=['mlstream', 'mlstream.models'],
      install_requires=[
          'torch',
          'scikit-learn',
          'pandas',
          'numpy',
          'h5py',
          'tqdm',
          'netCDF4',
          'numba',
          'scipy',
          'xgboost'
      ],
      include_package_data=True,
      zip_safe=False)
