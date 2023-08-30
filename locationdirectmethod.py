# 3D localisation using trilateration
import matplotlib.pyplot as plt
import numpy as np

class Localization3D:

    def __init__(self):
        self.state = {'pos':False}
        self.positions = None

    def determine_coordinates(self, data: np.ndarray, data2: np.ndarray, data3: np.ndarray, beacon_cords: 'list[list[float,float]]') -> np.ndarray:
        '''Calculates the exact position at all points in time, using the distances to all beacons and trilateration'''

        x0, y0 = beacon_cords[0][0], beacon_cords[0][1] # As defined by 'τ' in the notes by Richard van Dijk
        x1, y1 = beacon_cords[1][0], beacon_cords[1][1]
        x2, y2 = beacon_cords[2][0], beacon_cords[2][1]

        r = np.array([0.0,0.0,0.0,0.0])
        coordinates = []
        a = 0.39
        c = 0.55
        for time in range(30):  # For all time points in the data
            r[0] = data[time] #r1
            r[1] = data2[time] #r3
            r[2] = data3[time] #r4        
          
            my = (r[0] ** 2 - r[1] ** 2 + y1 ** 2) / (2 * y1)
            mx = (r[0] ** 2 - r[2] ** 2 + x2 ** 2 + y2 ** 2 - 2 * y2 * my) / (2 * x2)
            mz = np.sqrt(r[1] ** 2 - mx ** 2 - my ** 2)
                
            coordinates.append([time, mx, my, mz])  

        positions = (np.array(coordinates))
        self.positions = positions
        self.state['pos'] = True
        return positions
    
class LocalizationPlot3D():
    #plotting a 3D scatterplot
    def positions3D(*samples: Localization) -> None:
        #plot all points using the sample.positions data
        for sample in samples:
            if not sample.state['pos']:
                raise RuntimeError("No positions found")
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(sample.positions[:,3], sample.positions[:,1], sample.positions[:,2])     
        ax.set_xlabel('Z-axis')
        ax.set_ylabel('X-axis')
        ax.set_zlabel('Y-axis')
        plt.show()
