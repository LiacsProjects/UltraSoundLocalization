# 2D Localisation using trilateration
import matplotlib.pyplot as plt
import numpy as np

class Localization:

    def __init__(self):
        self.state = {'pos':False}
        self.positions = None

    def determine_coordinates(self, data: np.ndarray, data2: np.ndarray, data3: np.ndarray, data4: np.ndarray, data5: np.ndarray, data6: np.ndarray, data7: np.ndarray, data8: np.ndarray, beacon_cords: 'list[list[float,float]]') -> np.ndarray:
        '''Calculates the exact position at all points in time, using the distances to all beacons and trilateration'''

        x0, y0 = beacon_cords[0][0], beacon_cords[0][1] # As defined by 'τ' in the notes by Richard van Dijk
        x1, y1 = beacon_cords[1][0], beacon_cords[1][1]
        x2, y2 = beacon_cords[2][0], beacon_cords[2][1]
        x3, y3 = beacon_cords[3][0], beacon_cords[3][1]
        x4, y4 = beacon_cords[4][0], beacon_cords[4][1]
        x5, y5 = beacon_cords[5][0], beacon_cords[5][1]
        x6, y6 = beacon_cords[6][0], beacon_cords[6][1]
        x7, y7 = beacon_cords[7][0], beacon_cords[7][1]

        r = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        initialvalue = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        coordinates = []
        a = 0.39
        c = 0.55
        for time in range(30):  # For all time points in the data
            # Distances to each beacon
            r[0] = data[time]
            r[1] = data2[time] 
            r[2] = data3[time]
            r[3] = data4[time]
            r[4] = data5[time]
            r[5] = data6[time]
            r[6] = data7[time]
            r[7] = data8[time] 
            
            if (time == 0):
                for j in range(8):
                    initialvalue[j] = r[j]

            usable = np.array([False, False, False, False, False, False, False, False], dtype=bool)
            gebruikt = np.array([False, False, False, False, False, False, False, False], dtype=bool)
            if (time != 0):
                for i in range(8):
                    if (r[i] != initialvalue[i]):
                        usable[i] = True

            for i in range(8):
                for j in range(8):
                    for k in range(8):
                            #if 3 sensors are usable, calculate the position using trilateration for those sensors
                            if (usable[i] == True and usable[j] == True and usable[k] == True):
                                if (i == 1 and j == 2 and k == 4 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[1]
                                    r2 = r[2]
                                    r3 = r[4]
                                    my = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                                    mx = (r1 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2 + (2 * a + c) ** 2 + my * (2 * a + 2 * c)) / (2 * a + 2 * c)
                                    coordinates.append([time, mx, my, 0.0]) 
                                    print("2, 3, 5")  
                                    amount += 1
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break
                                elif (i == 1 and j == 2 and k == 7 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[1]
                                    r2 = r[2]
                                    r3 = r[7]
                                    my = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                                    mx = (r1 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2 + (2 * a) * my) / (2 * a + 2 * c)
                                    coordinates.append([time, mx, my, 0.0]) 
                                    print("2, 3, 8") 
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break 
                                elif (i == 1 and j == 3 and k == 4 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[1]
                                    r2 = r[3]
                                    r3 = r[4]
                                    mx = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                                    my = (r1 ** 2 - r2 ** 2 + (2 * a + c) ** 2 - (2 * a) * mx) / (2 * a + 2 * c)
                                    coordinates.append([time, mx, my, 0.0])  
                                    print("2, 4, 5") 
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break 
                                elif (i == 0 and j == 7 and k == 2 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[0]
                                    r2 = r[7]
                                    r3 = r[2]
                                    mx = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                                    my = (r2 ** 2 - r3 ** 2 + mx * (2 * a + 2 * c)) / ((2 * a) + (2 * c))
                                    coordinates.append([time, mx, my, 0.0]) 
                                    print("1, 3, 8")   
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break 
                                elif (i == 3 and j == 4 and k == 6 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[3]
                                    r2 = r[4]
                                    r3 = r[6]
                                    mx = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                                    my = (r1 ** 2 - r2 ** 2 + mx * (2 * a + 2 * c)) / (-2 * a - 2 * c)
                                    coordinates.append([time, mx, my, 0.0])   
                                    print("4, 5, 7")
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break 
                                elif (i == 3 and j == 6 and k == 5 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[3]
                                    r2 = r[6]
                                    r3 = r[5]
                                    my = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c) 
                                    mx = (r1 ** 2 - r2 ** 2 + my * (-2 * a - 2 * c)) / (2 * a + 2 * c)
                                    coordinates.append([time, mx, my, 0.0]) 
                                    print("4, 6, 7")
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break
                                elif (i == 0 and j == 6 and k == 5 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[0]
                                    r2 = r[6]
                                    r3 = r[5]
                                    my = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                                    mx = (r1 ** 2 - r2 ** 2 + (2 * a + c) ** 2 - (2 * a) * my) / (2 * a + 2 * c)
                                    coordinates.append([time, mx, my, 0.0])   
                                    print("1, 6, 7")
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break
                                elif (i == 0 and j == 7 and k == 5 and gebruikt[i] == False and gebruikt[j] == False and gebruikt[k] == False):
                                    r1 = r[0]
                                    r2 = r[7]
                                    r3 = r[5]
                                    mx = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                                    my = (r1 ** 2 - r3 ** 2 - a ** 2 + (2 * a + c) ** 2 + mx * (2 * a + 2 * c)) / (2 * a + 2 * c)
                                    coordinates.append([time, mx, my, 0.0])   
                                    print("1, 6, 8")  
                                    gebruikt[i] = True
                                    gebruikt[j] = True
                                    gebruikt[k] = True
                                    break
            
            #check whether the middle sensor is on the x or y axis
            # first sensor is always on the y-axis
            # third sensor is always on the x-axis
           
            for k in range(8):
                usable[k] = False
                gebruikt[k] = False
            amount = 0

        positions = (np.array(coordinates))
        self.positions = positions
        self.state['pos'] = True
        return positions
    
class LocalizationPlot():
    #plotting a 2D scatterplot
    def positions(*samples: Localization) -> None:
        #plot all points using the sample.positions data
        for sample in samples:
            if not sample.state['pos']:
                raise RuntimeError("No positions found")
            plt.scatter(sample.positions[:,1], sample.positions[:,2])
            for t in range(len(sample.positions[:,0])):
                time = sample.positions[t][0] # annotate time
                plt.annotate(time, (sample.positions[t][1], sample.positions[t][2]))
        plt.show()
