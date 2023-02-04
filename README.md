![Build Status](https://github.com/dsavransky/MeanStars/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/dsavransky/MeanStars/badge.svg?branch=main)](https://coveralls.io/github/dsavransky/MeanStars?branch=main)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
[![PyPI version](https://badge.fury.io/py/MeanStars.svg)](https://badge.fury.io/py/MeanStars)

# MeanStars

This code provides functionality for automating calculations based on Eric Mamajek's "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence" (http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt).

The purpose of the code is to allow users to automatically interpolate the property columns of the table, and to calcualte colors based on any combination of the photometry columns.

Also provides utilities for matching a wide variety of spectral type strings.

## Installation and Requirements
MeanStars requires the following python modules:
* numpy
* scipy
* astropy

To install, simply run:

    pip install MeanStars
    
Alternatively, grab a copy of the repository and run:

    pip install .
   
To install in developer (editable) mode, grab a copy of the repository and run:

    pip install -e .
    
This last option is useful if you want to update the data file.

## Usage
To use MeanStars, you must first create a MeanStars object:

```python
from MeanStars import MeanStars
ms = MeanStars()
```
This object contains a number of useful attributes, including:
* `ms.data`: The full dataset in astropy table format
* `ms.bands`: The names of all unique bands from the dataset (in a string array)
* `ms.colors`: All of the original colors from the dataset (in an nx2 string array, where n is the length of `ms.bands` and the color is the first column minus the second).  The same information is also encoded in `ms.colorstr`, which is the original color name from the dataset
* `ms.noncolors`: All data attributes not related to color (these include stellar masses, radii, etc.)
* `ms.SpecTypes`: All of the major spectral types from the dataset (in a string array - nominally this will always be O,B,A,F,G,K,M,L,T,Y)
* `ms.colorgraph`: A directed graph (encoded as a dictionary) mapping all relationships between bands established by the colors in the original dataset

### Interpolating Colors
MeanStars provides two methods for interpolating colors:  `TeffColor` and `SpTColor`, where the former interpolates by effective temperature, and the latter by spectral type.  In general, it is unlikely that you will want to query the data by any spectral type not explicilty listed, so the `SpTColor` most frequently acts as a simple lookup table of the data.  In each case, the methods are called by providing the two bands defining the color (called `start` and `end` in the code such that the color is `start` - `end`), and the temperature or spectral type.

So, to find the 'U-B' color of a 29000 K star, you would execute:

```python
ms.TeffColor('U','B',29000)
```
This particular instance corresponds to an exact entry in the data table, and the value returned should exactly match the entry.  A more interesting case is the 'U-H' color of a 6000 K star:
```python
ms.TeffColor('U','H',6000)
```
Here, you are requesting a color not found in the table at a temperature not found in the table (but bracketed by other temperature values). You can query to find the specific sets of color combined to give this result by running:
```python
ms.searchgraph('U','H')
```
which will return `['U', 'B', 'V', 'Ks', 'H']`, meaning that the 'U-B','B-V','V-Ks', and 'Ks-H' colors were added to get the result. 

Interpolating by spectral type works exactly the same way, excpet that the type is defined by two input variables representing the major and minor subtype.  So, to find the 'U-H' color of a G2V dwarf, you would execute:
```python
ms.SpTColor('U','H','G',2)
```

Each time one of these routines is called on a new color (for a given object instance), the generated interpolant is saved in `ms.Teffinterps` and `ms.SpTinterps` (as appropriate to the method call).  This means that the interpolant is generated just once per object instance, speeding up subsequent computations. 

### Interpolating Other Properties

Just as with colors, any other property in the original data set can be interpolated as a function of effective temperature or spectral type, via methods `TeffOther` and `SpTOther`, respectively. The methods use the same syntax as their color counterparts, save that the property is defined by a single string input.  

So, to find the mean solar mass of a 5500 K star, you would run:
```python
ms.TeffOther('Msun',5500)
```
and to find the mean radius of a K9V star, you would run:
```python
ms.SpTOther('R_Rsun','K',9)
```

### Looking Up Nearest Data Entries

For any data columns with monotonically varying data (such that a nearest-neighbor interpolant can be defined), the nearest data entry can be identified via the `tableLookup` method.  For example, to see which spectral type has closest log(solar luminosity) to 3, we can execute:
```python
ind = ms.tableLookup('logL', 3.0)
print(ms.data[ind]['SpT','logL'])
```
which returns:
```
SpT logL
--- ----
B3V 2.99
```

### Matching Spectral Types

MeanStars provides a regular expression and a utility method for parsing spectral type strings.  These are intended to work with Morgan-Keenan (MK, MKK, or Yerkes) style spectral type strings containing a Spectral Class (and Subclass) and (optionally) a Luminosity Class. The default matching behavior (provided by attribute `specregex`) is to Letter|number|roman numeral where the letter is the spectral class (one of OBAFKGMLTY), the number is the subclass (nominally between 0, inclusive, and 10, exclusive), and the roman numeral is the luminosity type (nominally I through VII). The number can be an integer or have a decimal, there can be parentheses around the number and/or roman numeral, and there can be spaces in between any or all of the values. The luminosity class (roman numeral) and subtype (number) can be two values separated by a slash or dash. Examples of valid strings that can be matched by this regular expression include: G0V, G(0)V, G(0)(V), G0.5V, G5/6V, G(5/6)(IV/V), G 0.5 (V), G 0.5V. Both the subclass and luminosity class are treated as optional, and so we can also match things like G and G0. 

The second supported case (using the same attribute) is where the type string contains multiple spectral types, spearated by a slash or dash. All of the same formatting is supported as in the previous case, and examples of valid matches include things like  G8/K0IV.  In these cases, as many elements as possible are extracted from both classes. That is, the regular expression always returns 6 matching groups in cases where any match is possible. In cases where only one class string is present, the second three groups are all None.

This regular expressions is wrapped by class method `matchSpecType`, which will attempt to return a single value for the spectral class, subclass, and luminosity class for a given string. It will also look for leading white dwarf ('D', 'WD', or 'wd') and subdwarf ('sd') patterns and set the luminosity class accordingly. 

The method takes a single string input:
```python
ms.matchSpecType("G0V")
```
will return: `('G', 0.0, 'V')`.



