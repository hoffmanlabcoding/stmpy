from setuptools import setup, find_packages

setup(
  name = 'hoffmanstmpy',
  version = '1.0.0',
  packages = find_packages(),
  package_data = {
    "":["*.txt", "*.mat"]
    },
  include_package_data=True,
  license='MIT',
  description = 'Scanning tunneling microscopy data analysis suite',
  author = 'Harris Pirie',
  author_email = 'hoffmanlabcoding@gmail.com',
  url = 'https://github.com/hoffmanlabcoding/stmpy',
  download_url = 'https://github.com/hoffmanlabcoding/stmpy.git',
  keywords = ['STM', 'Python', 'Data Analysis'],
  install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'opencv-python',
          'scikit-image'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
