import os.path
import numpy as np
import astropy.io.ascii  # type: ignore
import re
import scipy.interpolate  # type: ignore
import pkg_resources
from typing import Tuple, Optional, List, Dict
import warnings
import numpy.typing as npt


class MeanStars:
    """MeanStars implements an automated lookup and interpolation
        functionality over the data from: "A Modern Mean Dwarf Stellar Color
        and Effective Temperature Sequence"
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        Eric Mamajek (JPL/Caltech, University of Rochester)

    Args:
        datapath (str, optional):
            Full path to data file.  If None (default) use internal file distributed
            with the package.

    Attributes:
        bands (numpy.ndarray):
            Array of all available band strings.
        colorgraph (dict):
            Graph of all available colors (band 1 - band 2) encoded as a dictionary.
            Keys are each of the bands in attribute ``bands`` and values are lists of
            all other bands for which a color is known.  For example, entry
            ``'B': ['V', 'U']`` means that there exist B-V and B-U (or their negative)
            colors.
        colors (numpy.ndarray):
            2D string array of all colors. First column - second column.
        colorstr (numpy.ndarray):
            Color strings matching the information in attribute ``colors``.
        data (astropy.table.table.Table):
            The original data, read from the selected file on disk.
        data_version (str):
            Version string, extracted from the file.  If a version string cannot be
            identified, this attribute is set to 'unknown'.
        MK (numpy.ndarray):
            Array of spectral class letters matching all rows in data.
        MKn (numpy.ndarray):
            Array of spectral subtype numbers matching all rows in data.
        noncolors (numpy.ndarray):
            Data columns for quantities other than colors.
        nondec (re.Pattern):
            Regular expression matching non numeric values.
        romandict (dict):
            Dictionary of roman numerals and their numerical values.
        specdict (dict):
            Dictionary mapping spectral classes to numerical values, with O = 0, and
            Y = 9
        specregex (re.Pattern):
            Regular expression matching a string with a spectral class, subclass and
            luminosity class.
        specregex_mixedtype (re.Pattern):
            Regular expression matching a string with multiple spectral classes and a
            luminosity class.
        specregex_nolum (re.Pattern):
            Regular expression matching string with a spectral class.
        spectral_classes (str):
            All valid spectral class letters.
        SpecTypes (numpy.ndarray):
            Array of unique entires in attribute ``MK``.
        SpTinterps (dict):
            Dictionary of spectral type interpolants by color.
        Teff (numpy.ndarray):
            Effective temperature values from data.
        Teffinterps (dict):
            Dictionary of effective temperature interpolants by color.
    """

    def __init__(self, datapath: Optional[str] = None) -> None:

        if datapath is None:
            filename = "EEM_dwarf_UBVIJHK_colors_Teff.txt"
            datapath = pkg_resources.resource_filename("MeanStars", filename)
        assert os.path.isfile(datapath), "Could not locate %s." % datapath

        self.data = astropy.io.ascii.read(
            datapath, fill_values=[("...", np.nan), ("....", np.nan), (".....", np.nan)]
        )

        # attempt to get version
        verregex = re.compile(r"Version \S+")
        verstr = list(filter(verregex.match, self.data.meta["comments"]))
        if len(verstr) == 1:
            self.data_verstion = verstr[0]
        else:
            self.data_version = "unknown"

        # Some definitions
        # Roman Numerals:
        self.romandict = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}
        # Spectral Classes:
        self.spectral_classes = "OBAFGKMLTY"
        self.specdict = {}
        for j, s in enumerate(self.spectral_classes):
            self.specdict[s] = j

        # Define standard pass-bands. lambda is effective wavelength in nm.  BW is the
        # FWHM in nm. UBVRI are from Bessell (2005). YJHKLMN from Binney (1998).
        self.bands = {
            "U": {"lambda": 366.3, "BW": 65},
            "B": {"lambda": 436.1, "BW": 89},
            "V": {"lambda": 544.8, "BW": 84},
            "R": {"lambda": 640.7, "BW": 158},
            "I": {"lambda": 798.0, "BW": 154},
            "Y": {"lambda": 1020, "BW": 120},
            "J": {"lambda": 1220, "BW": 213},
            "H": {"lambda": 1630, "BW": 307},
            "K": {"lambda": 2190, "BW": 390},
            "L": {"lambda": 3450, "BW": 472},
            "M": {"lambda": 4750, "BW": 460},
            "N": {"lambda": 10500, "BW": 2500},
        }

        # Spectral Type regexs
        # Default spectral type string is Letter|number|roman numeral
        # First regex assumes all three are present, that the number can be an integer
        # or have a decimal, that there can be parentheses around the number and/or
        # roman numeral. The luminosity class (roman numeral) and subtype (number) can
        # be two values separated by a slash or dash. Also allows for spaces in between.
        # Examples of thing this matches:
        # G0V, G(0)V, G(0)(V), G0.5V, G5/6V, G(5/6)(IV/V), G 0.5 (V), G 0.5V
        numstr = r"\d*\.?\d*"  # generic regex for any valid number
        self.specregex = re.compile(
            (
                rf"([{self.spectral_classes}])\s*\(*({numstr}[/-]?{numstr})\)*"
                r"\s*\(*([IV]+[/-]{0,1}[IV]*)"
            )
        )

        # Alternatively, you will sometimes have mixed types of the form:
        # G8/K0IV.  This one supports all the same basic options as the previous one.
        # In these cases, we are only extracting the leading class.
        self.specregex_mixedtype = re.compile(
            (
                rf"([{self.spectral_classes}])\s*\(*({numstr}[/-]?{numstr})\)*[/-]"
                rf"[{self.spectral_classes}]\s*\(*{numstr}[/-]?{numstr}\)*"
                r"\s*\(*([IV]+\/{0,1}[IV]*)"
            )
        )

        # As fallback options, we also consider the cases where the luminosity type
        # or luminosity type AND subtype are missing. Note that the default specregex
        # will match cases where luminosity type is present but subtype is missing,
        # but will NOT match a string without a luminosity type.
        # For a missing subtype, the second group (both here and in the default) will
        # be a blank string
        self.specregex_nolum = re.compile(
            rf"([{self.spectral_classes}])\s*\(*({numstr}[/-]?{numstr})\)*"
        )

        # for identifying non-numeric values:
        self.nondec = re.compile(r"[^\d.-]+")

        # get all the spectral types
        MK = []
        MKn = []
        for s in self.data["SpT"].data:
            m = self.specregex.match(s)
            MK.append(m.groups()[0])  # type: ignore
            MKn.append(m.groups()[1])  # type: ignore
        self.MK = np.array(MK)
        self.MKn = np.array(MKn).astype(float)
        self.SpecTypes = np.unique(self.MK)

        # find all the colors and everything else
        keys = self.data.keys()
        colorregex = re.compile(r"(\w{1,2})-(\w{1,2})")
        colors = np.array([])
        noncolors = []
        dontwant = ["SpT", "#SpT", "Teff"]
        for k in keys:
            m = colorregex.match(k)
            if m:
                if colors.size == 0:
                    colors = np.array(m.groups())
                else:
                    colors = np.vstack((colors, np.array(m.groups())))
            else:
                if k not in dontwant:
                    noncolors.append(k)

        # all the bands
        bands = np.unique(colors)

        # build a directed (bi-directional) graph of colors
        colorgraph: Dict[str, List[str]] = {}
        for b in bands:
            colorgraph[b] = []

        for r in colors:
            colorgraph[r[0]].append(r[1])
            colorgraph[r[1]].append(r[0])

        # attributes
        self.colors = colors
        self.bands = bands
        self.colorgraph = colorgraph
        self.colorstr = np.array(["-".join(c) for c in self.colors])
        self.noncolors = np.array(noncolors)
        self.Teff = self.getFloatData("Teff")

        # storage dicts
        self.Teffinterps: Dict[str, scipy.interpolate.interp1d] = {}
        self.SpTinterps: Dict[str, scipy.interpolate.interp1d] = {}
        self.lookupinterps: Dict[str, scipy.interpolate.interp1d] = {}

    def searchgraph(
        self, start: str, end: str, path: List[str] = []
    ) -> Optional[List[str]]:
        """Find the shortest path between any two bands in the color graph

        Args:
            start (str):
                Starting band
            end (str):
                Ending band

        Returns:
            list(str) or None:
                Shortest path from start to end.  None if no path exists
        """
        assert start in self.bands, "%s is not a known band" % start
        assert end in self.bands, "%s is not a known band" % end

        path = path + [start]
        if start == end:
            return path
        bestpath = None
        for node in self.colorgraph[start]:
            if node not in path:
                newpath = self.searchgraph(node, end, path)
                if newpath:
                    if not bestpath or len(newpath) < len(bestpath):
                        bestpath = newpath
        return bestpath

    def translatepath(self, path: List[str]) -> npt.NDArray[np.float_]:
        """Translate a path between bands to additions/subtractions of colors

        Args:
            path (list(str)):
                path as returned by search graph

        Returns:
            ~numpy.ndarray:
                nx2 ndarray where n is len(path)
                The first column is the index of the color (into self.colorstr)
                and the second column is -1 for subtraction and +1 for addition.
        """

        assert np.all(
            [p in self.bands for p in path]
        ), "All path elements must be known bands"
        res = np.zeros((len(path) - 1, 2))
        for j in range(len(path) - 1):
            tmp = np.where(self.colorstr == "-".join(path[j : j + 2]))[0]
            if tmp.size > 0:
                res[j] = np.array([tmp[0], 1])
            else:
                tmp = np.where(self.colorstr == "-".join(path[j : j + 2][::-1]))[0]
                if tmp.size == 0:
                    raise LookupError
                res[j] = np.array([tmp[0], -1])
        return res

    def getFloatData(self, key: str) -> npt.NDArray[np.float_]:
        """Grab a numeric data column from the table and strip any non-numeric
        characters as needed.

        Args:
            key (str):
                Name of column to grab

        Returns:
            ~numpy.ndarray(float):
                Numerical values from columns

        """
        assert key in self.data.keys(), "%s not found in data table." % key

        tmp = self.data[key].data
        if isinstance(tmp, np.ma.core.MaskedArray):
            tmp = tmp.data
        if np.issubdtype(tmp.dtype, np.number):
            return np.array(tmp).astype(float)
        else:
            return np.array(
                [self.nondec.sub("", v) if v != "nan" else v for v in tmp]
            ).astype(float)

    def interpTeff(self, start: str, end: str) -> None:
        """Create an interpolant as a function of effective temprature for the
        start-end color and add it to the self.Teffinterps dict

        Args:
            start (str):
                Starting band
            end (str):
                Ending band

        """

        name = "-".join([start, end])

        if name in self.Teffinterps:
            return

        vals = self.getDataForColorInterp(start, end)

        self.Teffinterps[name] = scipy.interpolate.interp1d(
            self.Teff[~np.isnan(vals)], vals[~np.isnan(vals)], bounds_error=False
        )

    def getDataForColorInterp(self, start: str, end: str) -> npt.NDArray[np.float_]:
        """Grab all data for start-end color

        Args:
            start (str):
                Starting band
            end (str):
                Ending band
        Returns:
            ~numpy.ndarray(float):
                color values

        """

        assert start in self.bands, "%s is not a known band" % start
        assert end in self.bands, "%s is not a known band" % end

        path = self.searchgraph(start, end)
        assert path, "No connection between %s and %s" % (start, end)

        res = self.translatepath(path)

        vals = np.zeros(len(self.data))
        for r in res:
            vals += r[1] * self.getFloatData(self.colorstr[r[0].astype(int)])

        return vals

    def TeffColor(
        self, start: str, end: str, Teff: npt.ArrayLike
    ) -> npt.NDArray[np.float_]:
        """Calculate the start-end color at a given effective temperature

        Args:
            start (str):
                Starting band
            end (str):
                Ending band
            Teff (float or array-like of floats):
                Effective Temperature in K

        Returns:
            ~numpy.ndarray(float):
                start-end color at Teff (float, or array of floats)
        """

        self.interpTeff(start, end)

        return np.array(self.Teffinterps["-".join([start, end])](Teff))

    def interpSpT(self, start: str, end: str) -> None:
        """Create an interpolant as a function of spectral type for the
        start-end color and add it to the self.SpTinterps dict

        Args:
            start (str):
                Starting band
            end (str):
                Ending band

        """

        name = "-".join([start, end])

        if name in self.SpTinterps:
            return

        vals = self.getDataForColorInterp(start, end)

        self.SpTinterps[name] = {}
        for ll in self.SpecTypes:
            tmp = vals[self.MK == ll]
            if np.all(np.isnan(tmp)):
                self.SpTinterps[name][ll] = lambda x: np.array(
                    [np.nan] * len(np.array([x]).flatten())
                )
            elif len(np.where(np.isfinite(tmp))[0]) == 1:
                arg = float(self.MKn[self.MK == ll][np.isfinite(tmp)][0])
                tmp = tmp[np.isfinite(tmp)][0]
                self.SpTinterps[name][ll] = lambda x, tmp=tmp, arg=arg: np.array(
                    [tmp if y == arg else np.nan for y in np.array([x]).flatten()]
                )
            else:
                self.SpTinterps[name][ll] = scipy.interpolate.interp1d(
                    self.MKn[self.MK == ll][np.isfinite(tmp)].astype(float),
                    tmp[np.isfinite(tmp)],
                    bounds_error=False,
                )

    def SpTColor(
        self, start: str, end: str, MK: str, MKn: npt.ArrayLike
    ) -> npt.NDArray[np.float_]:
        """Calculate the start-end color for a given spectral type

        Args:
            start (str):
                Starting band
            end (str):
                Ending band
            MK (str):
                Spectral type (OBAFGKMLTY)
            MKn (float, array-like of floats):
                Spectral sub-type

        Returns:
            ~numpy.ndarray(float):
                start-end color at MKn
        """

        assert MK in self.MK, "%s is not a known spectral type" % MK
        self.interpSpT(start, end)

        return np.array(self.SpTinterps["-".join([start, end])][MK](MKn))

    def getDataForOtherInterp(self, key: str) -> npt.NDArray[np.float_]:
        """Grab all data for the given key

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)

        Returns:
            ~numpy.ndarray(float):
                Interpolated values

        """

        assert key in self.noncolors, "%s is not a known property" % key

        vals = self.getFloatData(key)

        return vals

    def interpOtherTeff(self, key: str) -> None:
        """Create an interpolant as a function of effective temprature for the
        given key and add it to the self.Teffinterps dict

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)

        """

        if key in self.Teffinterps:
            return

        vals = self.getDataForOtherInterp(key)

        self.Teffinterps[key] = scipy.interpolate.interp1d(
            self.Teff[~np.isnan(vals)], vals[~np.isnan(vals)], bounds_error=False
        )

    def TeffOther(self, key: str, Teff: npt.ArrayLike) -> npt.NDArray[np.float_]:
        """Calculate the given property at a given effective temperature

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)
            Teff (float or array-like of floats):
                Effective Temperature in K

        Returns:
            ~numpy.ndarray(float):
                property at Teff (float, or array of floats)
        """

        self.interpOtherTeff(key)

        return np.array(self.Teffinterps[key](Teff))

    def interpOtherSpT(self, key: str) -> None:
        """Create an interpolant as a function of spectral type for the
        given key and add it to the self.SpTinterps dict

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)

        """

        if key in self.SpTinterps:
            return

        vals = self.getDataForOtherInterp(key)

        self.SpTinterps[key] = {}
        for ll in self.SpecTypes:
            tmp = vals[self.MK == ll]
            if np.all(np.isnan(tmp)):
                self.SpTinterps[key][ll] = lambda x: np.array(
                    [np.nan] * len(np.array([x]).flatten())
                )
            elif len(np.where(np.isfinite(tmp))[0]) == 1:
                arg = float(self.MKn[self.MK == ll][np.isfinite(tmp)][0])
                tmp = tmp[np.isfinite(tmp)][0]
                self.SpTinterps[key][ll] = lambda x, tmp=tmp, arg=arg: np.array(
                    [tmp if y == arg else np.nan for y in np.array([x]).flatten()]
                )
            else:
                self.SpTinterps[key][ll] = scipy.interpolate.interp1d(
                    self.MKn[self.MK == ll][np.isfinite(tmp)].astype(float),
                    tmp[np.isfinite(tmp)],
                    bounds_error=False,
                )

    def SpTOther(self, key: str, MK: str, MKn: npt.ArrayLike) -> npt.NDArray[np.float_]:
        """Calculate the property color for a given spectral type

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)
            MK (str):
                Spectral type (OBAFGKMLTY)
            MKn (float, array-like of floats):
                Spectral sub-type

        Returns:
            ~numpy.ndarray(float):
                key value at MKn
        """

        assert MK in self.MK, "%s is not a known spectral type" % MK
        self.interpOtherSpT(key)

        return np.array(self.SpTinterps[key][MK](MKn))

    def interpTableLookup(self, key: str) -> None:
        """Create a lookup interpolant for row number for the
        given key and add it to the self.lookupinterps dict

        Args:
            key (str):
                Property to interpolate

        """

        if key in self.lookupinterps:
            return

        vals = self.getFloatData(key)
        inds = np.arange(len(self.data))

        self.lookupinterps[key] = scipy.interpolate.interp1d(
            vals[~np.isnan(vals)],
            inds[~np.isnan(vals)],
            kind="nearest",
            bounds_error=False,
        )

    def tableLookup(self, key: str, val: float) -> int:
        """Return index of nearest row to given value for given key

        Args:
            key (str):
                Property to look up.
            val (float):
                Value in key's column to match

        Returns:
            int:
                Index of data row closest to given value in the given key
        """

        self.interpTableLookup(key)

        return int(self.lookupinterps[key](val))

    def matchSpecType(self, spec: str) -> Optional[Tuple[str, float, str]]:
        """Match as much spectral type information as possible from type string

        Args:
            spec (str):
                Input string.

        Returns:
            tuple:
                Spectral Class (str):
                    OBAFGKMLTY or D
                Spectral sub-class (float):
                    [0, 10)
                Luminosity Class (str):
                    Roman numeral I - VII

        .. note::

            Preferentially matches dwarfs.  If multiple luminosity classes are present
            but one of them is V, then that's what will be returned.  Otherwise, it will
            be the first class listed. For multiple spectral subclasses, and average
            will be returned.

        .. warning::

            For any missing spectral subclasses, 5 will be returned.

        """

        # If this is a white dwarf, can return right away
        if spec.startswith("D"):
            return "D", 0, "VII"

        # check for subdwarf prefix
        if spec.startswith("sd"):
            subdwarf = True
            spec = spec.strip("sd")
            lumClass = "VI"
        else:
            subdwarf = False

        # First try for a full set of values:
        tmp = self.specregex.match(spec)
        # If default did not work, look for mixed types
        if not (tmp):
            tmp = self.specregex_mixedtype.match(spec)
        # If that didn't work, try just matching spectral type
        if not (tmp):
            tmp = self.specregex_nolum.match(spec)
            if tmp:
                warnings.warn(f"Missing luminosity class for {spec}. Assigning V.")
        if not (tmp):
            warnings.warn(f"Unable to match spectral type {spec}.")
            return None

        # At this point, should have at least the spectral class
        specClass = tmp.groups()[0]
        if tmp.groups()[1] in ["", ".."]:
            warnings.warn(f"Missing subclass for {spec}. Assigning 5.")
            specSubClass = 5
        else:
            if "/" in tmp.groups()[1]:
                tmp2 = tmp.groups()[1].split("/")
            elif "-" in tmp.groups()[1]:
                tmp2 = tmp.groups()[1].split("-")
            else:
                tmp2 = [tmp.groups()[1]]

            # handle outlier case of repeated .. in value
            tmp2 = [t.replace("..", ".") for t in tmp2]

            # evaluate subclass value
            specSubClass = np.array(tmp2).astype(float).mean()

        # Finally, deal with luminosity class
        if not (subdwarf):
            if len(tmp.groups()) == 3:
                if "/" in tmp.groups()[2]:
                    tmp2 = tmp.groups()[2].split("/")
                elif "-" in tmp.groups()[2]:
                    tmp2 = tmp.groups()[2].split("-")
                else:
                    tmp2 = [tmp.groups()[2]]

                # preferentially select dwarfs
                if "V" in tmp2:
                    lumClass = "V"
                else:
                    lumClass = tmp2[0]
            else:
                lumClass = "V"

        return specClass, specSubClass, lumClass
