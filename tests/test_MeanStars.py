import unittest
from MeanStars import MeanStars
import numpy as np


class TestMeanStars(unittest.TestCase):
    """

    Test of basic MeanStars functionality.

    """

    def setUp(self):
        self.ms = MeanStars()

        self.valid_spectypes_single = [
            "O0I",
            "B(1)II",
            "A(2)(III)",
            "F3.5IV",
            "G4/5V",
            "G(5-6)(IV/V)",
            "K 7 (V)",
            "M 0.5V",
            "L (5) VI",
            "T (8) (VII)",
            "Y (0/1) (V-VI)",
        ]

        self.valid_spectypes_mixed = [
            "O9/B0I",
            "B(9)/A(1)II",
            "A(9.5)/F(0.5)(III)",
            "F 9.5/G0 (IV/V)",
            "A8-9/F(0/1)(III)",
        ]

        self.valid_spectypes_nolum = [
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

    def test_specregex(self):
        for s in self.valid_spectypes_single:
            self.assertTrue(
                self.ms.specregex.match(s) is not None,
                "specregex failed for {}".format(s),
            )

        for s in self.valid_spectypes_mixed:
            self.assertTrue(
                self.ms.specregex.match(s) is None,
                "specregex should not have matched {}".format(s),
            )

        for s in self.valid_spectypes_nolum:
            self.assertTrue(
                self.ms.specregex.match(s) is None,
                "specregex should not have matched {}".format(s),
            )

    def test_specregex_mixedtype(self):
        for s in self.valid_spectypes_mixed:
            self.assertTrue(
                self.ms.specregex_mixedtype.match(s) is not None,
                "specregex_mixedtype failed for {}".format(s),
            )

        for s in self.valid_spectypes_single:
            self.assertTrue(
                self.ms.specregex_mixedtype.match(s) is None,
                "specregex_mixedtype should not have matched {}".format(s),
            )

        for s in self.valid_spectypes_nolum:
            self.assertTrue(
                self.ms.specregex_mixedtype.match(s) is None,
                "specregex_mixedtype should not have matched {}".format(s),
            )

    def test_specregex_nolum(self):
        for s in self.valid_spectypes_nolum:
            self.assertTrue(
                self.ms.specregex_nolum.match(s) is not None,
                "specregex_nolum failed for {}".format(s),
            )

    def test_matchSpecType(self):
        random_types = [
            "A0",
            "A0/1IV",
            "A0V",
            "A0VSB",
            "A0Vn",
            "A0Vvar",
            "A0p",
            "A1.0V",
            "A1V",
            "A2",
            "A2.5VA",
            "A2V",
            "A2VANN",
            "A2Va",
            "A2Vm",
            "A3III/V",
            "A3IV",
            "A3IVn",
            "A3IVvSB",
            "A3MF0(IV)",
            "A3MF0-F2",
            "A3V",
            "A3Vvar",
            "A3m",
            "A4(M)A5-A7",
            "A4IVs",
            "A4V",
            "A5",
            "A5III",
            "A5IIevar",
            "A5IV/V",
            "A5V",
            "A5V...",
            "A5VSB",
            "A5mF2 (IV)",
            "A5n",
            "A6VN",
            "A7III/IV",
            "A7IIIvar",
            "A7IV",
            "A7IV-V",
            "A7V",
            "A9Ib/II",
            "A9V",
            "APSREU(CR)",
            "Am",
            "Am(kA2hA5mA7V)",
            "B7V",
            "B8",
            "B8V",
            "B9.5V",
            "B9p",
            "DA",
            "DA3",
            "DA:",
            "DAn",
            "DAs",
            "DAw",
            "DAw...",
            "DQ6",
            "DZ7",
            "F0",
            "F0III/IV",
            "F0IV",
            "F0IV...",
            "F0IV/V",
            "F0Ib-II",
            "F0V",
            "F0V...",
            "F0Vp",
            "F1III-IV",
            "F1IV",
            "F1V",
            "F2",
            "F2/3IV/V",
            "F2/3V",
            "F2III-IV",
            "F2III-IVvar",
            "F2IV",
            "F2IV/V",
            "F2IVSB",
            "F2MF5IIP",
            "F2V",
            "F2V:var",
            "F3/5",
            "F3/5V",
            "F3III-IV",
            "F3IV",
            "F3IV-V",
            "F3IV/V",
            "F3V",
            "F3Vwvar",
            "F4V",
            "F5",
            "F5+...",
            "F5.5V",
            "F5/6V",
            "F5/8",
            "F5IV",
            "F5IV-V",
            "F5V",
            "F5V+...",
            "F6/7V",
            "F6/8V",
            "F6III",
            "F6IV",
            "F6IV-V",
            "F6IVwvar",
            "F6V",
            "F6V:",
            "F6Vs",
            "F6Vvar",
            "F7(W)F3V",
            "F7.7V",
            "F7/8IV/V",
            "F7/8V",
            "F7IV",
            "F7IV-V",
            "F7V",
            "F7Vn",
            "F7Vvar",
            "F8",
            "F8+...",
            "F8/G0V",
            "F8:",
            "F8IV-V",
            "F8V",
            "F8V-VI",
            "F8V...",
            "F8VSB",
            "F8Vw",
            "F9IV",
            "F9V",
            "Fp",
            "G",
            "G0",
            "G0...",
            "G0.5Vb",
            "G0/1V",
            "G0/2V",
            "G0/G1IV/V",
            "G0IV",
            "G0IV...",
            "G0V",
            "G0Vs",
            "G0Vvar",
            "G1",
            "G1.5Vb",
            "G1/2V",
            "G1/G2V",
            "G1IV",
            "G1V",
            "G2.0V",
            "G2/3IV",
            "G2/3V",
            "G2/8+(F)",
            "G2III",
            "G2IV",
            "G2V",
            "G2V+G2V",
            "G2V...",
            "G3/5III",
            "G3/5V",
            "G3/G5V",
            "G3IV",
            "G3IV/V",
            "G3V",
            "G4V",
            "G4V+...",
            "G5",
            "G5/6V",
            "G5/8V+(F)",
            "G5IV",
            "G5V",
            "G5Vp",
            "G5WF8V",
            "G6/8III/IV",
            "G6/8V",
            "G6/G8IV",
            "G6IV",
            "G6IV+...",
            "G6V",
            "G7.0V",
            "G7V",
            "G8",
            "G8.0IV",
            "G8.0V",
            "G8.5V",
            "G8/K0(IV)",
            "G8/K0IV",
            "G8/K0V",
            "G8/K0V(W)",
            "G8/K1(III+F/G)",
            "G8/K1(V)",
            "G8III",
            "G8III-IV",
            "G8IIIBCNIV",
            "G8IV",
            "G8IV+(F)",
            "G8IV-V",
            "G8IV-V+...",
            "G8IV/V",
            "G8IVvar",
            "G8V",
            "G8VSB",
            "G8Vp",
            "G8Vvar",
            "G9.0V",
            "G9III",
            "G9IV-V",
            "G9IVa",
            "G9V",
            "G:+...",
            "K",
            "K(2)V",
            "K/MV",
            "K0",
            "K0+F/G",
            "K0...",
            "K0.5V",
            "K0/1V",
            "K0/1V(+G)",
            "K0/1V+(G)",
            "K0/2V",
            "K0/4",
            "K0III",
            "K0III-IV",
            "K0IIIB",
            "K0IIIvar",
            "K0IV",
            "K0IV SB",
            "K0IV-V",
            "K0IV/V",
            "K0V",
            "K0Ve",
            "K0Vvar",
            "K1(V)",
            "K1+...",
            "K1.5III",
            "K1.5V",
            "K1/2III",
            "K1/2V",
            "K1III",
            "K1III(+M)",
            "K1III(P)",
            "K1III/IV",
            "K1IIIB",
            "K1IV",
            "K1IV/V",
            "K1V",
            "K1V+G",
            "K1V...",
            "K1Vp",
            "K2",
            "K2(V)",
            "K2.0V",
            "K2.5V",
            "K2/3(V)",
            "K2/3V",
            "K2/4",
            "K2III",
            "K2IIIB",
            "K2IIIp",
            "K2IIIvar",
            "K2IV/V",
            "K2V",
            "K2V:",
            "K3.0V",
            "K3/4(III)",
            "K3/4V",
            "K3/5V",
            "K3/K4V",
            "K3III",
            "K3V",
            "K3V(P)",
            "K3Vk:",
            "K4",
            "K4.0V",
            "K4.5",
            "K4/5(V)",
            "K4/5V",
            "K4:",
            "K4III",
            "K4V",
            "K4V(P)",
            "K4V...",
            "K4VP",
            "K5",
            "K5.0V",
            "K5/M0V",
            "K5/M1V+K(5)V",
            "K5:",
            "K5III",
            "K5IV",
            "K5V",
            "K5V+G/KIII",
            "K5V...",
            "K5V:",
            "K5p",
            "K6",
            "K6:",
            "K6V:",
            "K6Ve",
            "K7",
            "K7.0V",
            "K7V",
            "K7V:",
            "K7Vk",
            "K7Vvar",
            "K8",
            "K8V",
            "K8V:",
            "K9.0V",
            "K9Vk:",
            "K:",
            "K:...",
            "Kp",
            "M",
            "M...",
            "M0",
            "M0...",
            "M0.0V",
            "M0.5",
            "M0.5V:",
            "M0.5V:e",
            "M0.5Ve",
            "M0.5Vk",
            "M0/1V",
            "M0:",
            "M0:V:",
            "M0V",
            "M0V...",
            "M0V:",
            "M0V:p",
            "M0V:p...",
            "M0VEP",
            "M0VP",
            "M0Ve",
            "M0Vk",
            "M0Vk:",
            "M0Vvar",
            "M0p",
            "M1",
            "M1+...",
            "M1.0V",
            "M1.5+V",
            "M1.5V",
            "M1.5Ve",
            "M1/2V",
            "M1:",
            "M1:comp",
            "M1V",
            "M1V...",
            "M1V:",
            "M1V:e...",
            "M1VE",
            "M1Ve+...",
            "M1Vk",
            "M1Vvar",
            "M2",
            "M2+V",
            "M2...",
            "M2.0V",
            "M2.5",
            "M2.5V",
            "M2.5V/M3V",
            "M2.5Ve",
            "M2.5Vk",
            "M2/3V",
            "M2:",
            "M2V",
            "M2V:",
            "M2Ve",
            "M2Vk:",
            "M3",
            "M3.0V",
            "M3.5",
            "M3.5 V",
            "M3.5V",
            "M3.5Ve",
            "M3.5e",
            "M3:",
            "M3V",
            "M3V:",
            "M3Ve",
            "M3Ve+...",
            "M3Vkee",
            "M4",
            "M4+...",
            "M4.0",
            "M4.0V",
            "M4.5V",
            "M4III",
            "M4V",
            "M4V:",
            "M4Ve",
            "M4e...",
            "M5",
            "M5.0V",
            "M5Ve",
            "M6",
            "M9",
            "M:",
            "MPE",
            "Ma",
            "Me",
            "Mp",
            "sdG2",
            "sdM:",
        ]

        for s in random_types:
            self.assertTrue(
                self.ms.matchSpecType(s) is not None,
                "Spectral type matching failed for {}".format(s),
            )
