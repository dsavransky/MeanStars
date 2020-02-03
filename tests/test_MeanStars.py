import unittest
from MeanStars import MeanStars
import numpy as np

class TestMeanStars(unittest.TestCase):
    """ 

    Test of basic MeanStars functionality.

    """

    
    def setUp(self):
        self.ms = MeanStars()

    def tearDown(self):
        delattr(self,'ms')

    def test_init(self):
        """
        Test of initialization and __init__.

        Ensure that all expected attributes are populated and self-consistent 
        """

        expected_atts = ['MK',
                         'MKn',
                         'SpTinterps',
                         'SpecTypes',
                         'Teffinterps',
                         'bands',
                         'colorgraph',
                         'colors',
                         'colorstr',
                         'data',
                         'noncolors']

        for att in expected_atts:
            self.assertTrue(hasattr(self.ms,att))

        for c in self.ms.colors.flatten():
            self.assertTrue(c in self.ms.bands,"Color component not found in bands.")

        self.assertTrue(len(self.ms.colors) == len(self.ms.colorstr),"Colors array doesn't match colorstr array.")
        self.assertTrue(len(self.ms.MK) == len(self.ms.MKn),"MK array doesn't match MKn array.")


    def test_searchgraph(self):
        """
        Ensure that the graph search always returns the original color
        """

        for c in self.ms.colors:
            self.assertTrue(np.all(self.ms.searchgraph(c[0],c[1]) == c), "searchgraph doesn't recover original color.")

    def test_TeffColor(self):

        #do all the colors
        for cind in np.arange(len(self.ms.colors)):
            vals = self.ms.getFloatData(self.ms.colorstr[cind])
            goodinds = np.isfinite(vals)
        
            self.assertTrue(np.all(self.ms.TeffColor(self.ms.colors[cind][0],self.ms.colors[cind][1],self.ms.Teff[goodinds]) == vals[goodinds]),"Cannot reproduce colors from interpolant for %s"%self.ms.colorstr[cind])


    def test_SpTColor(self):

        SpT = self.ms.data['SpT'].data
        
        #do all the colors
        for cind in np.arange(len(self.ms.colors)):
            vals = self.ms.getFloatData(self.ms.colorstr[cind])
            goodinds = np.isfinite(vals)
        
            for v,s in zip(vals[goodinds],SpT[goodinds]):
                m = self.ms.specregex.match(s)
                self.assertTrue(m,"Couldn't decompose spectral type from data.")
                self.assertTrue(self.ms.SpTColor(self.ms.colors[cind][0],self.ms.colors[cind][1],m.groups()[0],float(m.groups()[1])) == v,"Cannot reproduce colors from interpolant for %s"%self.ms.colorstr[cind])


    def test_TeffOther(self):

        #do all the properties
        for key in self.ms.noncolors:
            vals = self.ms.getFloatData(key)
            goodinds = np.isfinite(vals)
        
            self.assertTrue(np.all(self.ms.TeffOther(key,self.ms.Teff[goodinds]) == vals[goodinds]),"Cannot reproduce values from interpolant for %s"%key)

    def test_SpTOther(self):

        SpT = self.ms.data['SpT'].data
        
        #do all the properties
        for key in self.ms.noncolors:
            vals = self.ms.getFloatData(key)
            goodinds = np.isfinite(vals)
        
            for v,s in zip(vals[goodinds],SpT[goodinds]):
                m = self.ms.specregex.match(s)
                self.assertTrue(m,"Couldn't decompose spectral type from data.")
                self.assertTrue(self.ms.SpTOther(key,m.groups()[0],float(m.groups()[1])) == v,"Cannot reproduce values from interpolant for %s"%key)

