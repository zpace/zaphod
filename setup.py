from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='zaphodIFU',
      version='0.1',
      description='Fake IFU generator for MaNGA spectral fitting diagnosis',
      url='http://github.com/zpace/zaphod',
      author='Zach Pace',
      author_email='zpace@astro.wisc.edu',
      license='MIT',
      packages=['fake'],
      install_requires=[
          'numpy'],
      include_package_data=True,
      zip_safe=False)
