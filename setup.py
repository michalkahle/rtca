from setuptools import setup

setup(name='rtca',
      version='0.2',
      description='Load and analyze RTCA data.',
      url='http://github.com/michalkahle',
      author='Michal Kahle',
      author_email='michalkahle@gmail.com',
      license='MIT',
      py_modules=['rtca', 'drc'],
      install_requires=['pandas_access'],
      zip_safe=False)
