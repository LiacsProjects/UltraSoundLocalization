# Run with 'python test_localization.py or ./test_localisation.py'
from locationdirectmethod import Localization
from locationdirectmethod import LocalizationPlot
import unittest
import numpy as np
from numpy import genfromtxt
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

class TestLocalization(unittest.TestCase):

    def test_determine_coordinates(self, data: np.ndarray, data2: np.ndarray, data3: np.ndarray):
        sample = Localization()
        #determine location in determine_coordinates()
        positions = sample.determine_coordinates(data, data2, data3, [[0.0,0.0,0.0],[0.0,2.00,0.0],[2.80,2.00,0.0]])
        LocalizationPlot.positions3D(sample)
        return sample.positions

if __name__ == '__main__':
    data = np.array(genfromtxt('file-1.csv', delimiter=','))
    data2 = np.array(genfromtxt('file-2.csv', delimiter=','))
    data3 = np.array(genfromtxt('file-3.csv', delimiter=','))
    #make pandas dataframe to put results in .csv file
    df = pd.DataFrame(columns=['time'])
    df.loc[:,"time"] = np.arange(0,30,1) 
    df.insert(1, 'd1', data) 
    df.insert(2, 'd2', data2) 
    df.insert(3, 'd3', data3) 
    test = TestLocalization()
    positions = test.test_determine_coordinates(data, data2, data3)
    df2 = pd.DataFrame((positions), columns=['time','mx', 'my', 'mz'])
    result = pd.concat([df, df2], axis=1)
    print(result)
    result.to_csv('fileresults.csv')  