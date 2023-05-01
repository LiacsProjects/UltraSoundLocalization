import math
import numpy as np

class Localization:
     
     def __init__(self):
        self.data = None
        self.wav_samplerate = None
        self.data_duration = None
        self.state = {'import':False, 'fft':False, 'pos':False}
        self.samples_per_second = 0
        self.fft_window_size = 0
        self.frequencies = None
        self.distances = None
        self.positions = None   

    def determine_coordinates(self, distances: np.ndarray, beacon_cords: 'list[list[float,float]]') -> np.ndarray:
        '''Calculates the exact position at all points in time, using the distances to all beacons and trilateration'''

        tx, ty = beacon_cords[1][0], beacon_cords[2][1] # As defined by 'τ' in the notes by Richard van Dijk
        r4 = None

        coordinates = []
        for time in distances:  # For all time points in the data
            r1, r2, r3 = time[1], time[2], time[3]  # Distances to each beacon
            if len(time) == 5:
                r4 = time[4]

            if tx != 0 and ty != 0:
                mx = (r1 ** 2 - r2 ** 2 + tx ** 2) / (2 * tx)
                my = (r1 ** 2 - r3 ** 2 + ty ** 2) / (2 * ty)
                mz = math.sqrt(abs(r1 ** 2 - mx ** 2 - my ** 2))
                coordinates.append([time[0], mx, my, mz])
            else:
                raise ValueError("tx and ty cannot be equal to 0")

            # TODO Refine the data using the fourth beacon, instead of returning an error if its not accurate
            if r4:
                if ((mx ** mx) + (my ** my) + (mz ** mz) == (r2 ** r2) + (r3 ** r3) - (r4 ** r4)):
                    pass
                else:
                    raise ValueError("No refinement of measurement")

        positions = (np.array(coordinates))
        self.positions = positions
        self.state['pos'] = True
        return positions
