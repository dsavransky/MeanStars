import unittest
from MeanStars import MeanStars
import numpy as np
import warnings


class TestMeanStars(unittest.TestCase):
    """

    Test of basic MeanStars functionality.

    """

    def setUp(self):
        self.ms = MeanStars()

    def tearDown(self):
        delattr(self, "ms")

    def test_init(self):
        """
        Test of initialization and __init__.

        Ensure that all expected attributes are populated and self-consistent
        """

        expected_atts = [
            "MK",
            "MKn",
            "SpTinterps",
            "SpecTypes",
            "Teffinterps",
            "bands",
            "colorgraph",
            "colors",
            "colorstr",
            "data",
            "noncolors",
        ]

        for att in expected_atts:
            self.assertTrue(hasattr(self.ms, att))

        for c in self.ms.colors.flatten():
            self.assertTrue(c in self.ms.bands, "Color component not found in bands.")

        self.assertTrue(
            len(self.ms.colors) == len(self.ms.colorstr),
            "Colors array doesn't match colorstr array.",
        )
        self.assertTrue(
            len(self.ms.MK) == len(self.ms.MKn), "MK array doesn't match MKn array."
        )

    def test_searchgraph(self):
        """
        Ensure that the graph search always returns the original color
        """

        for c in self.ms.colors:
            self.assertTrue(
                np.all(self.ms.searchgraph(c[0], c[1]) == c),
                "searchgraph doesn't recover original color.",
            )

    def test_TeffColor(self):

        # do all the colors
        for cind in np.arange(len(self.ms.colors)):
            vals = self.ms.getFloatData(self.ms.colorstr[cind])
            goodinds = np.isfinite(vals)

            self.assertTrue(
                np.all(
                    self.ms.TeffColor(
                        self.ms.colors[cind][0],
                        self.ms.colors[cind][1],
                        self.ms.Teff[goodinds],
                    )
                    == vals[goodinds]
                ),
                "Cannot reproduce colors from interpolant for %s"
                % self.ms.colorstr[cind],
            )

    def test_SpTColor(self):

        SpT = self.ms.data["SpT"].data

        # do all the colors
        for cind in np.arange(len(self.ms.colors)):
            vals = self.ms.getFloatData(self.ms.colorstr[cind])
            goodinds = np.isfinite(vals)

            for v, s in zip(vals[goodinds], SpT[goodinds]):
                m = self.ms.specregex.match(s)
                self.assertTrue(m, "Couldn't decompose spectral type from data.")
                self.assertTrue(
                    self.ms.SpTColor(
                        self.ms.colors[cind][0],
                        self.ms.colors[cind][1],
                        m.groups()[0],
                        float(m.groups()[1]),
                    )
                    == v,
                    "Cannot reproduce colors from interpolant for %s"
                    % self.ms.colorstr[cind],
                )

    def test_TeffOther(self):

        # do all the properties
        for key in self.ms.noncolors:
            vals = self.ms.getFloatData(key)
            goodinds = np.isfinite(vals)

            self.assertTrue(
                np.all(
                    self.ms.TeffOther(key, self.ms.Teff[goodinds]) == vals[goodinds]
                ),
                "Cannot reproduce values from interpolant for %s" % key,
            )

    def test_SpTOther(self):

        SpT = self.ms.data["SpT"].data

        # do all the properties
        for key in self.ms.noncolors:
            vals = self.ms.getFloatData(key)
            goodinds = np.isfinite(vals)

            for v, s in zip(vals[goodinds], SpT[goodinds]):
                m = self.ms.specregex.match(s)
                self.assertTrue(m, "Couldn't decompose spectral type from data.")
                self.assertTrue(
                    self.ms.SpTOther(key, m.groups()[0], float(m.groups()[1])) == v,
                    "Cannot reproduce values from interpolant for %s" % key,
                )

    def test_tableLookup(self):

        # check for logL and B-V as representative
        keys = ["logL", "B-V"]
        for key in keys:
            vals = self.ms.getFloatData(key)
            goodinds = np.where(np.isfinite(vals))[0]

            for ind in goodinds:
                self.assertTrue(
                    ind == self.ms.tableLookup(key, vals[ind]),
                    f"Table lookup failed for key {key} at index {ind}",
                )

    def test_cnumregex(self):
        """Test compound number regular expression"""

        # all of these should be matchable
        nummatches = ["10", "0001", "100.", "10.5", "0.124", ".235802300"]
        nummatches += ["10/10.", "10-1", ".1/.1234", ".1/10.000", "100.00-100.00"]
        for nm in nummatches:
            assert self.ms.cnumre.match(nm).group() == nm, f"cnumre failed for {nm}"

        # none of these should be matchable:
        numnonmatches = [".", "asdf", "X.X/Y.Y", "/", "-"]
        for nm in numnonmatches:
            self.assertTrue(
                self.ms.cnumre.match(nm) is None,
                "cnumre should not have matched {}".format(nm),
            )

        # the first part of these should be matched
        numpartmatches = ["10-", "0001/", "100.asdf", "10.5/-", "0.124-/", ".2358023/"]

        for nm in numpartmatches:
            assert nm.startswith(
                self.ms.cnumre.match(nm).group()
            ), f"cnumre failed for {nm}"

    def test_specregex(self):

        # check single types
        valid_spectypes_single = [
            "O0I",
            "B(1)II",
            "A(2)(III)",
            "F3.5IV",
            "F9IV-V",
            "G4/5V",
            "G(5-6)(IV/V)",
            "G8+V",
            "G8-V",
            "K 7 (V)",
            "M 0.5V",
            "L (5) VI",
            "T (8) (VII)",
            "Y (0/1) (V-VI)",
        ]
        for s in valid_spectypes_single:
            tmp = self.ms.specregex.match(s)
            assert tmp is not None, f"specregex failed for {s}"
            assert tmp.group() == s, f"specregex didn't fully match {s}"
            assert (
                None not in tmp.groups()[:3]
            ), f"specregex didn't fully extract from {s}"

        # check mixed types
        valid_spectypes_mixed = [
            "O9/B0I",
            "B(9)/A(1)II",
            "A(9.5)/F(0.5)(III)",
            "F 9.5/G0 (IV/V)",
            "A8-9/F(0/1)(III)",
            "M2.5V/M3V",
        ]
        for s in valid_spectypes_mixed:
            tmp = self.ms.specregex.match(s)
            assert tmp is not None, f"specregex failed for {s}"
            assert tmp.group() == s, f"specregex didn't fully match {s}"
            assert (None not in tmp.groups()[3:]) and (
                None not in (tmp.groups()[:2])
            ), f"specregex didn't fully extract from {s}"

        # check missing luminosity classes
        valid_spectypes_nolum = [
            "O0",
            "B(1)",
            "A(2)",
            "F3.5",
            "G4/5",
            "G(5/6)",
            "K 7",
            "M 0.5",
            "L (5)",
            "T (8)",
            "Y (0/1)",
        ]
        for s in valid_spectypes_nolum:
            tmp = self.ms.specregex.match(s)
            assert tmp is not None, f"specregex failed for {s}"
            assert tmp.group() == s, f"specregex didn't fully match {s}"
            assert (None not in tmp.groups()[:2]) and (
                tmp.groups()[2] == ""
            ), f"specregex didn't fully extract from {s}"

        # check missing subtype
        valid_spectypes_nosub = [
            "AIII",
            "MV",
        ]
        for s in valid_spectypes_nosub:
            tmp = self.ms.specregex.match(s)
            assert tmp is not None, f"specregex failed for {s}"
            assert tmp.group() == s, f"specregex didn't fully match {s}"
            assert (
                (tmp.groups()[0] is not None)
                and (tmp.groups()[2] is not None)
                and (tmp.groups()[1] == "")
            ), f"specregex didn't fully extract from {s}"

        # finally check for unmatchable stuff
        invalid_spectypes = ["D", "sD", "thisisnotasubtype", "0V/IV", "123.5", "NaCL"]
        for s in invalid_spectypes:
            self.assertTrue(
                self.ms.specregex.match(s) is None,
                "specregex should not have matched {}".format(s),
            )

    def test_matchSpecType(self):
        """Test expected returns for a variety of sample spectypes"""

        specdict = {
            "A": ("A", 5, "V"),
            "A0": ("A", 0.0, "V"),
            "A0/1IV": ("A", 0.5, "IV"),
            "A0V": ("A", 0.0, "V"),
            "A0Vvar": ("A", 0.0, "V"),
            "A1.0V": ("A", 1.0, "V"),
            "A2 V": ("A", 2.0, "V"),
            "A2.5VA": ("A", 2.5, "V"),
            "A3III/V": ("A", 3.0, "V"),
            "A3MF0(IV)": ("A", 3.0, "V"),
            "A3MF0-F2": ("A", 3.0, "V"),
            "A4(M)A5-A7": ("A", 4.0, "V"),
            "A4IVs": ("A", 4.0, "IV"),
            "A5III": ("A", 5.0, "III"),
            "A5IIevar": ("A", 5.0, "II"),
            "A5IV/V": ("A", 5.0, "V"),
            "A5V...": ("A", 5.0, "V"),
            "A7III/IV": ("A", 7.0, "III"),
            "A7IIIvar": ("A", 7.0, "III"),
            "A7IV-V": ("A", 7.0, "V"),
            "A8 III": ("A", 8.0, "III"),
            "A9Ib/II": ("A", 9.0, "I"),
            "A9V": ("A", 9.0, "V"),
            "Am(kA2hA5mA7V)": ("A", 5, "V"),
            "B": ("B", 5, "V"),
            "B7V": ("B", 7.0, "V"),
            "B8": ("B", 8.0, "V"),
            "B9 IV": ("B", 9.0, "IV"),
            "B9 Vne": ("B", 9.0, "V"),
            "B9.5-A0": ("B", 9.5, "V"),
            "B9.5V": ("B", 9.5, "V"),
            "B9p": ("B", 9.0, "V"),
            "DA": ("D", 0.0, "VII"),
            "DA3": ("D", 0.0, "VII"),
            "DA:": ("D", 0.0, "VII"),
            "DAn": ("D", 0.0, "VII"),
            "DAs": ("D", 0.0, "VII"),
            "DAw": ("D", 0.0, "VII"),
            "DAw...": ("D", 0.0, "VII"),
            "DQ6": ("D", 0.0, "VII"),
            "DZ7": ("D", 0.0, "VII"),
            "F": ("F", 5, "V"),
            "F V": ("F", 5, "V"),
            "F0": ("F", 0.0, "V"),
            "F0 IV": ("F", 0.0, "IV"),
            "F0+ V ({lambda} Boo)...": ("F", 0.0, "V"),
            "F0III/IV": ("F", 0.0, "III"),
            "F0IV": ("F", 0.0, "IV"),
            "F0IV...": ("F", 0.0, "IV"),
            "F0IV/V": ("F", 0.0, "V"),
            "F0Ib-II": ("F", 0.0, "I"),
            "F0V": ("F", 0.0, "V"),
            "F0V...": ("F", 0.0, "V"),
            "F0Vp": ("F", 0.0, "V"),
            "F1 V": ("F", 1.0, "V"),
            "F1III-IV": ("F", 1.0, "III"),
            "F1IV": ("F", 1.0, "IV"),
            "F2/3IV/V": ("F", 2.5, "V"),
            "F2/3V": ("F", 2.5, "V"),
            "F2III-IV": ("F", 2.0, "III"),
            "F2III-IVvar": ("F", 2.0, "III"),
            "F2IV": ("F", 2.0, "IV"),
            "F2IV/V": ("F", 2.0, "V"),
            "F2V:var": ("F", 2.0, "V"),
            "F3 V": ("F", 3.0, "V"),
            "F3/5": ("F", 4.0, "V"),
            "F3/5V": ("F", 4.0, "V"),
            "F3III-IV": ("F", 3.0, "III"),
            "F4VkF2mF1": ("F", 4.0, "V"),
            "F5+...": ("F", 5.0, "V"),
            "F5.5IV-V": ("F", 5.5, "V"),
            "F5/6V": ("F", 5.5, "V"),
            "F5/8": ("F", 6.5, "V"),
            "F5IV": ("F", 5.0, "IV"),
            "F5IV-V": ("F", 5.0, "V"),
            "F5V": ("F", 5.0, "V"),
            "F5V+...": ("F", 5.0, "V"),
            "F5V_Fe-0.5": ("F", 5.0, "V"),
            "F6.5V": ("F", 6.5, "V"),
            "F6/7V": ("F", 6.5, "V"),
            "F6/8V": ("F", 7.0, "V"),
            "F6III": ("F", 6.0, "III"),
            "F6IV": ("F", 6.0, "IV"),
            "F6IV-V": ("F", 6.0, "V"),
            "F6IVwvar": ("F", 6.0, "IV"),
            "F7": ("F", 7.0, "V"),
            "F7 V": ("F", 7.0, "V"),
            "F7(W)F3V": ("F", 7.0, "V"),
            "F7.7V": ("F", 7.7, "V"),
            "F7/8IV/V": ("F", 7.5, "V"),
            "F7/8V": ("F", 7.5, "V"),
            "F7IV": ("F", 7.0, "IV"),
            "F7IV-V": ("F", 7.0, "V"),
            "F8": ("F", 8.0, "V"),
            "F8 IV": ("F", 8.0, "IV"),
            "F8 IV/V": ("F", 8.0, "V"),
            "F8 V": ("F", 8.0, "V"),
            "F8+...": ("F", 8.0, "V"),
            "F8/G0 V": ("F", 8.0, "V"),
            "F8/G0V": ("F", 8.0, "V"),
            "F8V": ("F", 8.0, "V"),
            "F8V-VI": ("F", 8.0, "V"),
            "F8V_Fe-0.8_CH-0.5": ("F", 8.0, "V"),
            "F8Vw": ("F", 8.0, "V"),
            "F9": ("F", 9.0, "V"),
            "F9 IV": ("F", 9.0, "IV"),
            "F9 IV/V": ("F", 9.0, "V"),
            "F9 V": ("F", 9.0, "V"),
            "F9.5V": ("F", 9.5, "V"),
            "F9IV": ("F", 9.0, "IV"),
            "F9IV-V": ("F", 9.0, "V"),
            "F9V": ("F", 9.0, "V"),
            "F9V_Fe+0.3": ("F", 9.0, "V"),
            "F9V_Fe-1.5_CH-0.7": ("F", 9.0, "V"),
            "G": ("G", 5, "V"),
            "G V": ("G", 5, "V"),
            "G0": ("G", 0.0, "V"),
            "G0 IV": ("G", 0.0, "IV"),
            "G0 IV-V": ("G", 0.0, "V"),
            "G0 V": ("G", 0.0, "V"),
            "G0 VI": ("G", 0.0, "VI"),
            "G0(V)": ("G", 0.0, "V"),
            "G0-V(k)": ("G", 0.0, "V"),
            "G0...": ("G", 0.0, "V"),
            "G0.5V_Fe-0.5": ("G", 0.5, "V"),
            "G0.5Vb": ("G", 0.5, "V"),
            "G0/1 V": ("G", 0.5, "V"),
            "G0/1V": ("G", 0.5, "V"),
            "G0/2 V": ("G", 1.0, "V"),
            "G0/2V": ("G", 1.0, "V"),
            "G0/G1IV/V": ("G", 0.5, "V"),
            "G0IV": ("G", 0.0, "IV"),
            "G0IV-V": ("G", 0.0, "V"),
            "G0IV...": ("G", 0.0, "IV"),
            "G0V": ("G", 0.0, "V"),
            "G0V_Fe+0.4": ("G", 0.0, "V"),
            "G0VmF2": ("G", 0.0, "V"),
            "G0Vs": ("G", 0.0, "V"),
            "G0Vvar": ("G", 0.0, "V"),
            "G1": ("G", 1.0, "V"),
            "G1 IV": ("G", 1.0, "IV"),
            "G1 IV/V": ("G", 1.0, "V"),
            "G1 V": ("G", 1.0, "V"),
            "G1-1.5 V": ("G", 1.25, "V"),
            "G1.5 V": ("G", 1.5, "V"),
            "G1.5IV-V_Fe-1": ("G", 1.5, "V"),
            "G1.5V": ("G", 1.5, "V"),
            "G1.5V(n)": ("G", 1.5, "V"),
            "G1/2V": ("G", 1.5, "V"),
            "G1/G2V": ("G", 1.5, "V"),
            "G1IV": ("G", 1.0, "IV"),
            "G1V": ("G", 1.0, "V"),
            "G1V_CH-0.4(k)": ("G", 1.0, "V"),
            "G2": ("G", 2.0, "V"),
            "G2 IV": ("G", 2.0, "IV"),
            "G2 V": ("G", 2.0, "V"),
            "G2.0V": ("G", 2.0, "V"),
            "G2.5 IV": ("G", 2.5, "IV"),
            "G2.5 V": ("G", 2.5, "V"),
            "G2.5V_Hdel1": ("G", 2.5, "V"),
            "G2/3 IV": ("G", 2.5, "IV"),
            "G2/3IV": ("G", 2.5, "IV"),
            "G2/3V": ("G", 2.5, "V"),
            "G2/G3": ("G", 2.5, "V"),
            "G2/G3 IV/V": ("G", 2.5, "V"),
            "G2/G3 V": ("G", 2.5, "V"),
            "G2III": ("G", 2.0, "III"),
            "G2V+G2V": ("G", 2.0, "V"),
            "G3/5 V": ("G", 4.0, "V"),
            "G3/5III": ("G", 4.0, "III"),
            "G3/5V": ("G", 4.0, "V"),
            "G3/G5 V": ("G", 4.0, "V"),
            "G3/G5V": ("G", 4.0, "V"),
            "G3IV": ("G", 3.0, "IV"),
            "G3IV/V": ("G", 3.0, "V"),
            "G3V": ("G", 3.0, "V"),
            "G5 III/IV": ("G", 5.0, "III"),
            "G5 IV": ("G", 5.0, "IV"),
            "G5 IV/V": ("G", 5.0, "V"),
            "G5/6V": ("G", 5.5, "V"),
            "G5/8V+(F)": ("G", 6.5, "V"),
            "G5IV": ("G", 5.0, "IV"),
            "G5WF8V": ("G", 5.0, "V"),
            "G6/8III/IV": ("G", 7.0, "III"),
            "G6/8V": ("G", 7.0, "V"),
            "G6/G8IV": ("G", 7.0, "IV"),
            "G7IV-V": ("G", 7.0, "V"),
            "G8+V": ("G", 8.0, "V"),
            "G8-V": ("G", 8.0, "V"),
            "G8/9 IV": ("G", 8.5, "IV"),
            "G8/K0": ("G", 8.0, "V"),
            "G8/K0 IV": ("G", 8.0, "IV"),
            "G8/K0 V": ("G", 8.0, "V"),
            "G8/K0(IV)": ("G", 8.0, "IV"),
            "G8/K0IV": ("G", 8.0, "IV"),
            "G8/K0V": ("G", 8.0, "V"),
            "G8/K0V(W)": ("G", 8.0, "V"),
            "G8/K1(III+F/G)": ("G", 8.0, "III"),
            "G8/K1(V)": ("G", 8.0, "V"),
            "G8III": ("G", 8.0, "III"),
            "G8III-IV": ("G", 8.0, "III"),
            "G8IIIBCNIV": ("G", 8.0, "III"),
            "G8IV-V": ("G", 8.0, "V"),
            "G8IV-V+...": ("G", 8.0, "V"),
            "G8IV/V": ("G", 8.0, "V"),
            "G8IVvar": ("G", 8.0, "IV"),
            "G9-IV-V_Hdel1": ("G", 9.0, "V"),
            "G9.0V": ("G", 9.0, "V"),
            "G9/K0": ("G", 9.0, "V"),
            "G9IV-V": ("G", 9.0, "V"),
            "G9IVa": ("G", 9.0, "IV"),
            "G9V": ("G", 9.0, "V"),
            "G:+...": ("G", 5, "V"),
            "K": ("K", 5, "V"),
            "K V": ("K", 5, "V"),
            "K(2)V": ("K", 2.0, "V"),
            "K/MV": ("K", 5, "V"),
            "K0": ("K", 0.0, "V"),
            "K0 III": ("K", 0.0, "III"),
            "K0 III-IV": ("K", 0.0, "III"),
            "K0 III/IV": ("K", 0.0, "III"),
            "K0 IV": ("K", 0.0, "IV"),
            "K0 IV-V": ("K", 0.0, "V"),
            "K0 IV/V": ("K", 0.0, "V"),
            "K0 V": ("K", 0.0, "V"),
            "K0+F/G": ("K", 0.0, "V"),
            "K0+IV": ("K", 0.0, "IV"),
            "K0-V": ("K", 0.0, "V"),
            "K0.5 V": ("K", 0.5, "V"),
            "K0.5V": ("K", 0.5, "V"),
            "K0/1 V + G (III)": ("K", 0.5, "V"),
            "K0/1V": ("K", 0.5, "V"),
            "K0/1V(+G)": ("K", 0.5, "V"),
            "K0/1V+(G)": ("K", 0.5, "V"),
            "K0/2V": ("K", 1.0, "V"),
            "K0/4": ("K", 2.0, "V"),
            "K0III": ("K", 0.0, "III"),
            "K0III-IV": ("K", 0.0, "III"),
            "K0IIIB": ("K", 0.0, "III"),
            "K0IIIvar": ("K", 0.0, "III"),
            "K0IV": ("K", 0.0, "IV"),
            "K0IV SB": ("K", 0.0, "IV"),
            "K0IV-V": ("K", 0.0, "V"),
            "K0IV/V": ("K", 0.0, "V"),
            "K0V(k)": ("K", 0.0, "V"),
            "K1 IIIb": ("K", 1.0, "III"),
            "K1 IIIb Fe-0.5": ("K", 1.0, "III"),
            "K1 IV-V": ("K", 1.0, "V"),
            "K1.5 III": ("K", 1.5, "III"),
            "K1.5III": ("K", 1.5, "III"),
            "K1.5V": ("K", 1.5, "V"),
            "K1.5V(k)": ("K", 1.5, "V"),
            "K1/2 III": ("K", 1.5, "III"),
            "K1/2 V": ("K", 1.5, "V"),
            "K1/2III": ("K", 1.5, "III"),
            "K1/2V": ("K", 1.5, "V"),
            "K1/K2 V": ("K", 1.5, "V"),
            "K2 II": ("K", 2.0, "II"),
            "K2 III": ("K", 2.0, "III"),
            "K2 III/IV": ("K", 2.0, "III"),
            "K2/3(V)": ("K", 2.5, "V"),
            "K2/3V": ("K", 2.5, "V"),
            "K2/4": ("K", 3.0, "V"),
            "K2IV/V": ("K", 2.0, "V"),
            "K3/5V": ("K", 4.0, "V"),
            "K3/K4V": ("K", 3.5, "V"),
            "K3III": ("K", 3.0, "III"),
            "K3V(P)": ("K", 3.0, "V"),
            "K3Vk:": ("K", 3.0, "V"),
            "K5/M0V": ("K", 5.0, "V"),
            "K5/M1V+K(5)V": ("K", 5.0, "V"),
            "K7.0V": ("K", 7.0, "V"),
            "K7.5 V": ("K", 7.5, "V"),
            "K7/M0 V": ("K", 7.0, "V"),
            "L1.5": ("L", 1.5, "V"),
            "M": ("M", 5, "V"),
            "M V": ("M", 5, "V"),
            "M...": ("M", 5, "V"),
            "M0": ("M", 0.0, "V"),
            "M0 V": ("M", 0.0, "V"),
            "M0...": ("M", 0.0, "V"),
            "M0.0": ("M", 0.0, "V"),
            "M0.0 V": ("M", 0.0, "V"),
            "M0.0V": ("M", 0.0, "V"),
            "M0.5": ("M", 0.5, "V"),
            "M0.5 V": ("M", 0.5, "V"),
            "M0.5V:": ("M", 0.5, "V"),
            "M0.5V:e": ("M", 0.5, "V"),
            "M0.5Ve": ("M", 0.5, "V"),
            "M0.5Vk": ("M", 0.5, "V"),
            "M0/1V": ("M", 0.5, "V"),
            "M1 III": ("M", 1.0, "III"),
            "M1.9": ("M", 1.9, "V"),
            "M1/2V": ("M", 1.5, "V"),
            "M1:": ("M", 1.0, "V"),
            "M2+V": ("M", 2.0, "V"),
            "M2.5V/M3V": ("M", 2.75, "V"),
            "M2.5Ve": ("M", 2.5, "V"),
            "M2.5Vk": ("M", 2.5, "V"),
            "M2.5e": ("M", 2.5, "V"),
            "M2/3V": ("M", 2.5, "V"),
            "M5.5 V": ("M", 5.5, "V"),
            "M7.25": ("M", 7.25, "V"),
            "M7.5": ("M", 7.5, "V"),
            "M8": ("M", 8.0, "V"),
            "M8.5": ("M", 8.5, "V"),
            "M9": ("M", 9.0, "V"),
            "T8.5": ("T", 8.5, "V"),
            "sdBV": ("B", 5, "VI"),
            "sdG2": ("G", 2.0, "VI"),
            "sdM:": ("M", 5, "VI"),
            "WD": ("D", 0.0, "VII"),
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for s in specdict:
                self.assertTrue(
                    self.ms.matchSpecType(s) == specdict[s],
                    "Spectral type matching failed for {}".format(s),
                )
