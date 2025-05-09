# stmpy

Stmpy is a python-based data analysis package for scanning tunneling microscopy data.

  - Load all default Nanonis filetypes
  - Analysis tools: curve fitting, linecuts, drift correction and more.
  - Easily create movies and high-quality figures

Get started with the stmpy 101 tutorial in `stmpy/doc/`

### Installation

For developers

1. **Clone** or **download** `stmpy` to your local drive.
2. Navigate to within local copy of stmpy.
```sh
$ python -m pip install --editable .
```
- *[deprecated](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/):* ~~`python setup.py develop`~~

For users (in progress)

```sh
$ pip install hoffmanstmpy
```

### Usage
In the stmpy-doc folder, there are three tutorial notebooks and one coding template.
1. **Stmpy 101 - getting started.ipynb**: basic usage of stmpy to analyze topography maps and DOS maps
2. **Stmpy 102 - dfc tutorial.ipynb**: drift correction tutorial, including theory behind drift correction and usage of this module
3. **piezo calibration.ipynb**: describe how to calibrate your piezo with two topos
4. **stmpy notebook template -- topos and dos maps.ipynb**: template notebook that has example codes for different usage of stmpy (kinda of serving as table of content for this library), including loop through all the topos in a folder, drift correct and take linecuts on a DOS map, still growing...

License
----

MIT
