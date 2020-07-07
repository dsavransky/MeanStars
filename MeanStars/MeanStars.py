import os.path
import numpy as np
import astropy.io.ascii
import re
import scipy.interpolate
import pkg_resources


class MeanStars:
    def __init__(self, datapath=None):
        """MeanStars implements an automated lookup and interpolation
        functionality over th data from: "A Modern Mean Dwarf Stellar Color
        and Effective Temperature Sequence"
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        Eric Mamajek (JPL/Caltech, University of Rochester)
        """

        if datapath is None:
            filename = "EEM_dwarf_UBVIJHK_colors_Teff.txt"
            datapath = pkg_resources.resource_filename("MeanStars", filename)
        assert os.path.isfile(datapath), "Could not locate %s." % datapath

        self.data = astropy.io.ascii.read(
            datapath, fill_values=[("...", np.nan), ("....", np.nan), (".....", np.nan)]
        )

        # spectral type regexp
        specregex = re.compile(r"([OBAFGKMLTY])(\d*\.\d+|\d+)V")

        # get all the spectral types
        MK = []
        MKn = []
        for s in self.data["SpT"].data:
            m = specregex.match(s)
            MK.append(m.groups()[0])
            MKn.append(m.groups()[1])
        self.MK = np.array(MK)
        self.MKn = np.array(MKn)
        self.SpecTypes = np.unique(self.MK)

        # find all the colors and everything else
        keys = self.data.keys()
        colorregex = re.compile(r"(\w{1,2})-(\w{1,2})")
        colors = None
        noncolors = []
        dontwant = ["SpT", "#SpT", "Teff"]
        for k in keys:
            m = colorregex.match(k)
            if m:
                if colors is None:
                    colors = np.array(m.groups())
                else:
                    colors = np.vstack((colors, np.array(m.groups())))
            else:
                if k not in dontwant:
                    noncolors.append(k)

        # all the bands
        bands = np.unique(colors)

        # build a directed (bi-directional) graph of colors
        colorgraph = {}
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
        self.Teffinterps = {}
        self.SpTinterps = {}

        # useful regexs
        self.specregex = re.compile(r"([OBAFGKMLTY])(\d*\.\d+|\d+).*")
        self.nondec = re.compile(r"[^\d.-]+")

    def searchgraph(self, start, end, path=[]):
        """Find the shortest path between any two bands in the color graph

        Args:
            start (str):
                Starting band
            end (str):
                Ending band

        Returns:
            path (list of str):
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

    def translatepath(self, path):
        """Translate a path between bands to additions/subtractions of colors

        Args:
            path (list str):
                path as returned by search graph

        Returns:
            res (nx2 ndarray where n is len(path)):
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

    def getFloatData(self, key):
        """"Grab a numeric data column from the table and strip any non-numeric
        characters as needed.

        Args:
            key (str):
                Name of column to grab

        Returns:
            vals (float ndarray):
                Numerical values from columns

        """
        assert key in self.data.keys(), "%s not found in data table." % key

        tmp = self.data[key].data
        if isinstance(tmp, np.ma.core.MaskedArray):
            tmp = tmp.data
        if np.issubdtype(tmp.dtype, np.number):
            return tmp.astype(float)
        else:
            return np.array(
                [self.nondec.sub("", v) if v != "nan" else v for v in tmp]
            ).astype(float)

    def interpTeff(self, start, end):
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

    def getDataForColorInterp(self, start, end):
        """Grab all data for start-end color

        Args:
            start (str):
                Starting band
            end (str):
                Ending band
        Returns:
            vals (float ndarray):
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

    def TeffColor(self, start, end, Teff):
        """Calculate the start-end color at a given effective temperature

        Args:
            start (str):
                Starting band
            end (str):
                Ending band
            Teff (float or array-like of floats):
                Effective Temperature in K

        Returns:
            start-end color at Teff (float, or array of floats)
        """

        self.interpTeff(start, end)

        return self.Teffinterps["-".join([start, end])](Teff)

    def interpSpT(self, start, end):
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

    def SpTColor(self, start, end, MK, MKn):
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
            start-end color at MKn (float, or array of floats)
        """

        assert MK in self.MK, "%s is not a known spectral type" % MK
        self.interpSpT(start, end)

        return self.SpTinterps["-".join([start, end])][MK](MKn)

    def getDataForOtherInterp(self, key):
        """Grab all data for the given key

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)

        Returns:
            vals (float ndarray):
                color values

        """

        assert key in self.noncolors, "%s is not a known property" % key

        vals = self.getFloatData(key)

        return vals

    def interpOtherTeff(self, key):
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

    def TeffOther(self, key, Teff):
        """Calculate the given property at a given effective temperature

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)
            Teff (float or array-like of floats):
                Effective Temperature in K

        Returns:
            property at Teff (float, or array of floats)
        """

        self.interpOtherTeff(key)

        return self.Teffinterps[key](Teff)

    def interpOtherSpT(self, key):
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

    def SpTOther(self, key, MK, MKn):
        """Calculate the property color for a given spectral type

        Args:
            key (str):
                Property to interpolate (must be in MeanStars.noncolors)
            MK (str):
                Spectral type (OBAFGKMLTY)
            MKn (float, array-like of floats):
                Spectral sub-type

        Returns:
            key value at MKn (float, or array of floats)
        """

        assert MK in self.MK, "%s is not a known spectral type" % MK
        self.interpOtherSpT(key)

        return self.SpTinterps[key][MK](MKn)
