# Run with 'python test_localization.py or ./test_localisation.py'
from locationdirectmethod import Localization3D
from locationdirectmethod import LocalizationPlot3D
from locationreflectivemethod import Localization2D
from locationreflectivemethod import LocalizationPlot2D
import unittest
import numpy as np
from numpy import genfromtxt
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

class TestLocalization(unittest.TestCase):

    def test_determine_coordinates_2D(self, data: np.ndarray, data2: np.ndarray, data3: np.ndarray, data4: np.ndarray, data5: np.ndarray, data6: np.ndarray, data7: np.ndarray, data8: np.ndarray):
        sample = Localization2D()
        #determine location in determine_coordinates()
        positions = sample.determine_coordinates(data, data2, data3, data4, data5, data6, data7, data8 [[0.39,0.55,0.0],[0.55,0.39,0.0],[0.39,0.55,0.0],[0.55,0.39,0.0],[0.39,0.55,0.0],[0.55,0.39,0.0],[0.39,0.55,0.0],[0.55,0.39,0.0]])
        # print(positions)
        LocalizationPlot2D.positions(sample)
        return sample.positions

    def test_determine_coordinates_3D(self, data: np.ndarray, data2: np.ndarray, data3: np.ndarray):
        sample = Localization3D()
        #determine location in determine_coordinates()
        positions = sample.determine_coordinates(data, data2, data3, [[0.0,0.0,0.0],[0.0,2.00,0.0],[2.80,2.00,0.0]])
        # print(positions)
        LocalizationPlot3D.positions3D(sample)
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
    #positions = test.test_determine_coordinates_2D(data, data2, data3, data4, data5, data6, data7, data8)
    positions = test.test_determine_coordinates_3D(data, data2, data3)
    df2 = pd.DataFrame((positions), columns=['time','mx', 'my', 'mz'])
    result = pd.concat([df, df2], axis=1)
    print(result)
    result.to_csv('fileresults.csv')  