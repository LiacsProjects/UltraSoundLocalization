# Here the code for indoor localization based on ultrasound
from scipy.io import wavfile
from scipy.signal import blackman
from sklearn import linear_model
import matplotlib.pyplot as plt
from plotly.offline import plot
from plotly.graph_objs import Scatter
from decimal import Decimal
import random
import numpy as np
import math

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
    
    def locate(self, filepath: str, f_domains: 'list[list[int,int]]', sweep_duration: float, pos_beacons: 'list[list[float]]', pos_start: 'list[float]', temperature: float=20.0, samples_per_second: int=5, fft_window_size: int=8192, noise_level: int=7, surrounding_freqs: int=2, peak_tolerance: float=0.1, reg_sweep_perc: float=0.8) -> np.ndarray:
        '''Determines the position of a recorder through time based on perceived changes in ultrasound signals.
        The signals come from a number of beacons, of which the positions and frequency domains are known. The starting position is also known.
        
        Parameters
        ----------
        filepath: str
            The path to the file, only .wav files supported.
        f_domains: list[list[int,int]]
            The frequency domains in Hz of the sweeps.
        sweep_duration: float
            The duration of one sweep (min-max) in seconds.
        pos_beacons: list[list[float]]
            The x,y,z positions for all beacons in meters.
            The beacons should be in the same plane. 
            The positions should be provided in the same order as f_domains. 
        pos_start: tuple[float]
            The starting x,y.z position of the recorder in meters.
        temperature: float, optional
            The temperature in degrees Celsius during the experiment (default is 20.0)
        samples_per_second: int, optional
            The number of sample windows taken each second in data reduction. 
            The data size is reduced with a factor of 1/(fft_window_size/samples_per_second) (default is 5).
        fft_window_size: int, optional
            The size of the sample window for each sample (default is 8192).
        noise_level: int, optional
            The number of frequencies per timestep that may be considered as sweeps.
        surrounding_freqs: int, optional
            The number of frequencies on each side of a peak in FFT to consider in the weighted average (default is 2).
        peak_tolerane: float, optional
            The degree in the frequency peaks can diverge from the min/max of the frequency domain (default is 0.1).
        reg_sweep_perc: float optional
            The percentage of the sweep that is used in linear regression,
            to determine the coordinates of the peaks as the intersection points of two sweep lines (default is 0.8).
        '''

        # Check if the number of beacons is the same in f_domains and pos_beacons
        n_beacons = len(f_domains)
        if n_beacons != len(pos_beacons):
            raise ValueError("The number of beacons in f_domains doesn't match with pos_beacons.")
        # Check if all coordinates are defined in x,y,z format
        if len(pos_start) != 3:
            raise ValueError("Please provide starting position in format x-y-z.")
        for pos in pos_beacons:
            if len(pos) != 3:
                raise ValueError("Please provide all beacon position in format x-y-z.")

        # Make sure all domains are of format lowerbound, upperbound
        for i in range(len(f_domains)):
            f_domains[i].sort()

        if not self.check_orthogonal(pos_beacons):
            raise ValueError("Please place the beacons in orthogonal positions.")

        self.import_audio(filepath) # Read the file
        if not self.check_sound():
            raise ValueError("This file is completely emtpy, please try another file.")
        self.compress(samples_per_second, fft_window_size) # Perform data reduction
        f_array = self.get_frequencies(f_domains, sweep_duration, surrounding_freqs, noise_level) # Retrieve the sweep frequency data
        if not self.check_sweep_presence(f_array):
            raise ValueError("Unable to identify a sweep in this recording with these settings.")

        distances = []
        # extremes = np.ndarray((0,2)) # Uncomment for visualization
        for beacon in range(n_beacons):
            starting_distance = self.distance(pos_beacons[beacon],pos_start) # Calculate the starting distance to the beacon
            extreme_values = self.identify_extremes(f_array[:,beacon], f_domains[beacon], sweep_duration, peak_tolerance, reg_sweep_perc) # Identify the peak coordinates
            offsets = self.time_offsets(extreme_values,sweep_duration) # Calculate the time offsets of the coordinates
            distances.append(self.offsets_to_distance(starting_distance, offsets, sweep_duration, temperature)) # Convert those offsets to distances
            # extremes = np.concatenate((extremes,extreme_values)) # Uncomment for visualization
        
        # self.plot_spectrum(show_datapoints=extremes) # Uncomment for visualization
        
        distances = self.distances_per_timestep(distances, sweep_duration, n_beacons)
        positions = self.determine_coordinates(distances, pos_beacons)

        return positions

    def import_audio(self, filepath: str) -> None:
        '''Imports & reads .wav file, sets object's attribute values'''

        if self.state['import']:
            raise RuntimeWarning("Overwriting imported audio data with new file.")
        
        self.wav_samplerate, self.data = wavfile.read(filepath)
        
        # Ignore 2nd, 3rd, etc. channels for multi-channel files
        if self.data.ndim > 1:
            self.data = self.data[:,0]
        
        self.data_duration = self.data.shape[0] / self.wav_samplerate

        self.state['import'] = True

    def check_import(self) -> None:
        '''Checks whether .wav file has been imported yet'''

        if not self.state['import']:
            raise RuntimeError("No wav file has been imported yet.")

    def check_sound(self) -> None:
        '''Checks if the .wav file has sound'''

        if max(self.data) == 0:
            return False
        else:
            return True

    def generate_sweep(self, start_frequency: int, stop_frequency: int, samplerate: int, sweep_duration: float, total_duration: float, asynchronous: bool=True) -> 'tuple[int, np.ndarray]':
        '''Generates a pure sweep .wav file, returns sweep samplerate & data'''

        phase = 0
        frequency_increment = (stop_frequency - start_frequency)/(sweep_duration*samplerate)
        if asynchronous: # Simulate asynchronous beacons
            frequency = random.randint(min(start_frequency, stop_frequency), max(start_frequency, stop_frequency))
            frequency_increment *= random.choice([1,-1])
        else:
            frequency = start_frequency
        datapoints = int(total_duration*samplerate)
        data = np.ndarray(datapoints, 'int16')

        # Generate sound waves
        for i in range(datapoints):
            phase_increment = (2*math.pi*frequency)/samplerate
            data[i] = 15000 * math.sin(phase)
            phase += phase_increment
            frequency += frequency_increment
            if frequency >= max(stop_frequency, start_frequency) or frequency <= min(stop_frequency, start_frequency):
                frequency_increment *= -1
        
        wavfile.write("sweep_1beacon_"+str(start_frequency)+"-"+str(stop_frequency)+"Hz"+".wav", samplerate, data)

        # Returning the data prevents having to open the .wav file first before reading it
        return samplerate, data

    def mix_sweeps(self, name: str, samplerate: int, total_duration: float, *argv) -> str:
        '''Mixes sweep data into one wav. file, normalizing volume'''

        datapoints = int(total_duration*samplerate)
        data = np.zeros(datapoints, 'int32')
        for sweep in argv:
            if sweep[0] != samplerate:
                raise ValueError("The specified sample rate doesn't match to that from all of your sweeps")
            if len(sweep[1]) < datapoints:
                raise ValueError("The specified total duration exceeds that of some of your sweeps exceed")
            for i in range(datapoints): #add sweep data
                data[i] += sweep[1][i]

        # Normalize volume level
        max_value = np.amax(abs(data))
        data = (data*(15000/max_value)).astype('int16')

        filename = "sweep_" + name + ".wav"
        wavfile.write(filename, samplerate, data)

        return filename

    def compress(self, samples_per_second: int=5, fft_window_size: int=8192) -> None:
        '''Reduces data size by taking n samples at k times per second,
        where k = samples_per_second and n = fft_window_size.'''

        self.check_import()
        if self.samples_per_second != 0:
            raise RuntimeError('''The audio data has already been compressed.
            If you want to set a different compression factor, import the audio file again.''')
        elif samples_per_second == 0: # Note: this is the function parameter, not the object attribute
            print('Compression factor 0 equals no compression.')
        else:
            data_points_distance = int(self.wav_samplerate/samples_per_second)

            if data_points_distance < fft_window_size:
                raise RuntimeWarning('fft_window_size cannot be this high with the specified samples_per_second and the .wav files samplerate')
            if data_points_distance == fft_window_size:
                raise RuntimeWarning('There will be no data reduction with the specified value for fft_window_size.')
                
            new_size = int(len(self.data)/int(self.wav_samplerate/samples_per_second))*fft_window_size
            cutoff_size = (int(len(self.data)/data_points_distance)*data_points_distance) - fft_window_size
            new_data = np.ndarray(new_size, type(self.data))

            k = 0
            for i in range(0,cutoff_size,data_points_distance):
                for j in range(fft_window_size):
                    new_data[k] = self.data[i+j]
                    k += 1

            self.data = new_data
            self.samples_per_second = samples_per_second
            self.fft_window_size = fft_window_size

    def fast_fourier_transform(self, sample: np.ndarray, Fs: int, N: int) -> 'tuple[np.ndarray, np.ndarray]':
        '''Performs fast fourier transformation, returns non-imaginary, single-sided spectrum.
        N = sample size, Fs = samplerate'''

        f = Fs*np.arange((N/2))/N # frequency vector
        Y_k = np.fft.fft(sample)[0:int(N/2)]/N # FFT function from numpy
        Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
        Pxx = np.abs(Y_k) # be sure to get rid of imaginary part

        return Pxx, f
    
    def fft_full_file(self):
        '''Performs a fourier transformation over the whole file, returning its spectrum.
        Audio has to be imported but not yet compressed.'''

        self.check_import()
        if self.samples_per_second != 0:
            raise RuntimeError("""Impossible to perform FFT over whole file if it has been compressed.
            Try get_frequencies instead.""")
        
        N = self.wav_samplerate*self.data_duration # total points in signal

        return self.fast_fourier_transform(self.data, self.wav_samplerate, N)

    def discard_frequencies(self, frequencies: np.ndarray, bounds: 'list[int]' = [17500,23000]) -> np.ndarray:
        '''Discards all magnitudes for frequencies below lowerbound and above upperbound'''

        bounds.sort()
        lowerbound, upperbound = bounds[0], bounds[1]

        not_discarded = []
        for magnitude, frequency in frequencies:
            if frequency >= lowerbound and frequency <= upperbound:
                not_discarded.append([magnitude,frequency])
        
        return np.array(not_discarded)

    def find_peak_indices(self, fft_output: np.ndarray, n: int) -> list:
        '''Find the indices of the n highest peaks in an array'''

        sorted_indices = fft_output[:, 0].argsort() # Indices sorted from lowest to highest Pxx

        # Loop through the sorted_indices to find the frequencies that are peaks
        j = len(sorted_indices)-1 # Start at highest index
        peaks = []
        while len(peaks) != n and j >= 0: # Stop if we run out of data or if the number of peaks to be found is reached
            index = sorted_indices[j] # Retrieve the index
            # A peak is found if the magnitude is higher than the peak's neighbours
            # Use min/max to prevent out of bounds
            if fft_output[index][0] > fft_output[max(index-1,0)][0] and fft_output[index][0] > fft_output[min(index+1,len(fft_output)-1)][0]:
                peaks.append(index)
            j -= 1
        
        return peaks

    def calc_weighted_frequencies(self, frequency_array: np.ndarray, peak_indices: list, surrounding_freqs: int = 2) -> list:
        '''Refines a fft output by using a number of surrounding frequencies in a weighted average'''

        weighted_frequencies = []
        surrounding_data = np.zeros(((2*surrounding_freqs)+1,2)) # Will contain data surrounding the peak
        for index in peak_indices:
            n = 0 # Adds surrounding data to the array, max and min prevent out of bounds
            for a in range(max(index-surrounding_freqs,0), min(index+surrounding_freqs+1,len(frequency_array))):
                surrounding_data[n] = frequency_array[a,1], frequency_array[a,0]
                n += 1
            total_ampl = np.sum(surrounding_data[:,1]) # Sums all the magnitudes
            weighted_f = 0
            for frequency in surrounding_data: # Surrounding frequencies are weighted by their relative magnitude
                weighted_f += (frequency[1]/total_ampl) * frequency[0]
            weighted_frequencies.append(weighted_f)

        return weighted_frequencies

    def similarity(self, x: float, y: float, tolerance: float) -> float:
        '''Returns a similarity score of x and y, negative if x and y are not almost equal with specified tolerance percentage'''

        if tolerance > 1.0 or tolerance < 0.0:
            raise ValueError("Tolerance percentage should be in interval [0.0,1.0]")

        similarity = abs(x-y)
        if not (x < (1+tolerance)*y and x > (1-tolerance)*y):
            similarity = -1

        return similarity

    def freqs_per_domain(self, frequencies: np.ndarray, f_domains: 'list[list[int,int]]') -> 'list[list[list[int]]]':
        ''''Selects per timestep the frequencies that fit into each domain in f_domains'''
        
        candidates_per_domain = [] # Will contain frequencies per beacon per timestep
        n_beacons = len(f_domains)
        
        # Loop through the timesteps
        for timestep in frequencies:
            freqs_beacons = [] # Will contain frequencies per domain for this timestep
            for n in range(n_beacons): # Loop through the domains
                freqs = [] # Will contain frequencies for this domain
                minimum, maximum = f_domains[n][0] - 50, f_domains[n][1] + 50 # Predefined frequency domain
                # Only append the frequencies that are within the predefined frequency domains
                for frequency in timestep:
                    if frequency > minimum and frequency < maximum:
                        freqs.append(frequency)
                freqs_beacons.append(freqs)
            candidates_per_domain.append(freqs_beacons)

        return candidates_per_domain

    def add_sweeplike(self, x: int, y: int, sign: int, min: int, max: int) -> 'tuple[int, bool]':
        '''Adds sign*y to x as if x is a value of a sweep, flipping over if min/max is passed.
        Returns the answer and whether a sign change (flipping over) has occured.'''

        sign_change = False
        if sign == 1: # Adding
            room_to_add = max - x # Room to the max of the domain
        elif sign == -1: # Subtracting
            room_to_add = x - min # Room to the min of the domain
        else:
            raise ValueError("Sign has to be either 1 or -1.")
        if room_to_add >= y: # If there is enough room, simply add/subtract
            answer = x + (sign * y)
        else: # If not, add/subtract the room_to_add and subtract/add the other part
            answer = x + (sign * room_to_add) + (-1 * sign * (y-room_to_add))
            sign_change = True # Sign change has occurred

        return answer, sign_change  
    
    def likely_sweeps(self, candidate_sweeps: np.ndarray, f_domains: 'list[list[int,int]]', sweep_duration: float) -> np.ndarray:
        '''Determines which frequencies are most likely to be part of one of the sweeps''' 
        
        n_beacons = len(f_domains)
        sweeps = np.ndarray((len(candidate_sweeps),n_beacons)) # Will contain frequencies per distinct beacons

        # Select for each beacon & for each timesteps the frequencies that lie within the domain of that beacon
        candidates_per_beacon = self.freqs_per_domain(candidate_sweeps, f_domains)

        # For each beacon, determine whether/which candidates fit to the sweep
        for n in range(n_beacons):
            
            gradient = (f_domains[n][1] - f_domains[n][0])/(sweep_duration*self.samples_per_second) # Expected gradient for that sweep
            previous_f = None # None indicates start of file or unknown previous point
            
            i = 0 # For each timestep, determine whether/which candidates fit to the sweep
            while i < len(candidates_per_beacon):
                
                freqs_timestep = candidates_per_beacon[i][n] # The candidate frequencies from this timestep
                sweeps[i][n] = None # If we don't find a frequency that fits in the sweep, the value is None
                scores = [] # Will contain 'good' candidates, to select the best from later

                for frequency in freqs_timestep: # For each candidate frequency, determine whether/to what extent it fits to the sweep
                    
                    #If we have no information about the previous point, try to form a gradient with all next possible points
                    if not previous_f:
                        # Retrieve the candiates from the next timestep
                        next_timestep = candidates_per_beacon[min(i+1, len(candidates_per_beacon)-1)][n] 
                        for next_frequency in next_timestep: # NOTE For future work: we might wanna look into more than one timestep
                            # Compare the formed gradient with frequency and next frequency to the expected gradient
                            gradient_similarity = self.similarity(abs(next_frequency-frequency), gradient, 0.1)
                            # Add the frequency candidate with its score if the gradient is similar enough
                            # similarity() returns -1 if not similar at all, thus check if gradient_similarity >= 0
                            # Filter out gradients that are formed at the end of the domains, those are likely to be faulty
                            if gradient_similarity >= 0 and next_frequency > f_domains[n][0]+0.5*gradient and next_frequency < f_domains[n][1]-0.5*gradient:
                                sign = -1 if next_frequency - frequency < 0 else 1 # Keep track of the sign of the identified gradient
                                scores.append([frequency,gradient_similarity,sign])
                    
                    # If we know the previous point, compare the candidate value to the value we expect based upon that previous point and the gradient
                    else:
                        # Calculated expected frequency, gradient should flip if end of domain is reached (sweep) 
                        expected_f, sign_change = self.add_sweeplike(previous_f, gradient, sign, f_domains[n][0], f_domains[n][1])
                        expectation_difference = abs(expected_f-frequency) # Difference between the frequency candidate and what we expect
                        if expectation_difference < 0.5*gradient: # Threshold, points with high expectation_difference aren't considered
                            scores.append([frequency,expectation_difference,sign_change])
                
                # Select the candidate with the best score and process all necessary values
                # If score contains items (if at least one point was acceptable)
                if scores != []:
                    best = []
                    # Find lowest score
                    for score in scores:
                        if best == []:
                            best = score
                        else:
                            if score[1] < best[1]:
                                best = score
                    # Set the sign of that best score
                    if not previous_f:
                        sign = best[2]
                    else:
                        if best[2]:
                            sign *= -1
                    # Remember that best score
                    sweeps[i][n] = best[0]
                    previous_f = best[0]
                # Reset previous_f if no score was acceptable
                else:
                    previous_f = None
                
                i += 1

        return sweeps
    
    def get_frequencies(self, f_domains: 'list[list[int,int]]', sweep_duration: float, surrounding_freqs: int = 2, noise_level: int = 15) -> np.ndarray:
        '''Gets frequencies per timestamp for each beacon using FFT and a evaluation algorithm.'''

        # First some checks
        self.check_import() # Audio should be imported
        if self.samples_per_second == 0: # Audio should be compressed
            raise RuntimeError("Audio not compressed. Use compress() method.")

        # Some constants
        N = self.fft_window_size # Total points per signal
        w = blackman(N) # Window applied to datapoints (for improved FFT)
        domains_array = np.array(f_domains) # Turn into array to easily retrieve min/max
        min_f, max_f = np.min(domains_array), np.max(domains_array)

        i = 0
        candidate_sweeps = [] # Frequencies that may be part of a sweep
        plotted = True # Uncomment for visualization
        # Loops over all samples, retrieves frequencies, selects the strongest
        while i < len(self.data):
            # Perform FFT over sample
            sample = self.data[i:i+self.fft_window_size] # Current sample
            Pxx, f = self.fast_fourier_transform(w*sample, self.wav_samplerate, N) # FFT over sample, using the blackman window to reduce spectral leakage
            frequencies = np.column_stack((Pxx, f)) # Array with frequencies and their magnitudes
            if plotted: # Uncomment for visualization
                 plt.plot(f,Pxx)
                 plt.show()
                 plotted = False
            # Filter the derived frequencies
            frequencies = self.discard_frequencies(frequencies, [min_f-250, max_f+250]) # Get rid of hearable noise
            peak_indices = self.find_peak_indices(frequencies, noise_level) #Find noise_level amount of peaks within the array of frequencies
             # Refine the values of the just identified peaks with a weighted average using surrounding frequencies
            weighted_frequencies = self.calc_weighted_frequencies(frequencies, peak_indices, surrounding_freqs)
            candidate_sweeps.append(weighted_frequencies) # And append them to the list
            i += self.fft_window_size # Next sample 

        # Of the candidate sweeps, add the frequencies that are most likely to be sweeps to the array
        candidate_sweeps = np.array(candidate_sweeps) # For easier manipulation
        frequency_array = self.likely_sweeps(candidate_sweeps, f_domains, sweep_duration) # Algorithm that matches frequencies to beacons   

        self.frequencies = frequency_array
        self.state ["fft"] = True

        return frequency_array

    def mock_ultrasound_frequency_array(self, n_beacons: int=3, separate_domains: bool=True) -> np.ndarray:
        '''Sets up the class instance in such a way that FFT can be skipped, returning an array of mock frequencies.
        An n_beacons amount of ultrasound sweeps are generated, starting at random phases. 
        The audio is imported and the necessary object attributes are set to support further manipulations. 
        The sweeps don't undergo Fourier transformation, instead their frequencies are set directly.'''

        if self.state['import']: 
            raise RuntimeWarning("Overwriting imported audio data with new mock data.")
        if n_beacons == 0:
            raise ValueError("Cannot generate mock frequency array for 0 beacons.")
        
        self.wav_samplerate = 96000
        self.data_duration = 8.0
        self.state['import'] = True
        self.samples_per_second = 10
        self.fft_window_size = 512
        self.data = np.ndarray(int(self.fft_window_size*self.samples_per_second*self.data_duration))

        frequency_limits = [(19000,21000) for n in range(n_beacons)]
        if separate_domains: 
            # Divide ultrasound range 19000-21000 equally over n_beacons
            domain_distance = 50 # 50 Hz between the limits of the separate sweeps
            domain_size = int((21000-19000-(50*(n_beacons-1)))/n_beacons)
            frequency = 19000
            for n in range(n_beacons):
                frequency_limits[n] = (frequency, frequency + domain_size)
                frequency += domain_size + domain_distance

        datapoints = int(self.data_duration*self.wav_samplerate) # Total number of datapoints
        frequency_array = np.ndarray((int(self.samples_per_second*self.data_duration),n_beacons), int) # Size of to-be-filled array, with reduced number of datapoints

        for n in range(n_beacons):
            start_frequency, stop_frequency = frequency_limits[n]
            frequency_increment = (stop_frequency - start_frequency)/(1.0*self.wav_samplerate) # Using sweep duration 1.0s
            frequency = random.randint(min(start_frequency, stop_frequency), max(start_frequency, stop_frequency))
            frequency_increment *= random.choice([1,-1])

            # Generate frequencies 
            j = 0
            for i in range(datapoints):
                # Only append those that would remain after data reduction
                if i % (self.wav_samplerate/self.samples_per_second) == self.fft_window_size/2:
                    frequency_array[j,n] = frequency
                    j += 1
                frequency += frequency_increment
                if frequency >= max(stop_frequency, start_frequency) or frequency <= min(stop_frequency, start_frequency):
                    frequency_increment *= -1
        
        self.frequencies = frequency_array
        self.state['fft'] = True
        return frequency_array

    def find_next_not_nan(self, data: np.ndarray, start_index: int) -> 'tuple[float, float]':
        '''Finds the coordinates of the next defined datapoint in data. Returns None if no such point is found.'''

        i = start_index + 1
        while i < len(data):
            if data[i] == data[i]: # If the datapoint is not NaN
                return (i, data[i])
            i += 1
        
        return None

    def find_sweep_points(self, data: np.ndarray, start_index: int, search_direction: int, sweep_duration: float, reg_sweep_perc: float = 0.8) -> 'tuple[float,float]':
        '''Finds the coordinates of all points that make up a percentage of the sweep.
        The search is started at start_index in direction search_direction, the percentage is given by reg_sweep_perc'''

        points = []
        range = int(reg_sweep_perc*sweep_duration*self.samples_per_second) # Number of points that we will search for

        i = start_index
        while i < min(len(data),start_index+range) and i > max(-1, start_index-range):
            if data[i] == data[i]: # If the datapoint is not NaN
                # Time coordinate is set as midpoint of the window
                t = i/self.samples_per_second + (0.5*self.fft_window_size/self.wav_samplerate)
                points.append([t, data[i]])
            i = i + search_direction*1

        return points

    def regression_input(self, frequency_array: np.ndarray, previous_frequencies: 'list[list[int]]', sign: int, sign_change_index: int, sweep_duration: float, reg_sweep_perc: float=0.8) -> 'list[list[float, int]]':
        '''Returns two lists of points, corresponding to both sides of a peak,
        Which point are returned is based on the nature of the sign change and its surrounding data.'''

        # Will retrun true if sign change happened after the peak
        if ((sign == -1 and previous_frequencies[-2][1] >= frequency_array[sign_change_index]) or
        (sign == 1 and previous_frequencies[-2][1] < frequency_array[sign_change_index])):
            # In this case, the second previous point is the starting point of the first line
            points_1 = self.find_sweep_points(frequency_array, previous_frequencies[-2][0], -1, sweep_duration, reg_sweep_perc)
            # The second line starts at the first previous point
            points_2 = self.find_sweep_points(frequency_array, previous_frequencies[-1][0], 1, sweep_duration, reg_sweep_perc)
        # If the sign change happened within the peak
        else:
            # In this case, the first previous point is the starting point of the first line
            points_1 = self.find_sweep_points(frequency_array, previous_frequencies[-1][0], -1, sweep_duration, reg_sweep_perc)
            # The second line starts at the point of the sign change
            points_2 = self.find_sweep_points(frequency_array, sign_change_index, 1, sweep_duration, reg_sweep_perc)
            
        return points_1, points_2

    def identify_extremes(self, frequency_array: np.ndarray, f_domain: list, sweep_duration: float, tolerance: float=0.1, reg_sweep_perc: float=0.9) -> np.ndarray:
        '''Find the coordinates of the extreme values of a frequency array corresponding to a single beacon.'''

        if len(frequency_array.shape) != 1:
            raise ValueError("Unallowed dimension of frequency_array. Please let its data correspond to a single beacon")

        # Any extremes that are not within these bounds will considered as noise
        lowerlowerbound = f_domain[0] - tolerance*(f_domain[1]-f_domain[0])
        upperlowerbound = f_domain[0] + tolerance*(f_domain[1]-f_domain[0])
        lowerupperbound = f_domain[1] - tolerance*(f_domain[1]-f_domain[0])
        upperupperbound = f_domain[1] + tolerance*(f_domain[1]-f_domain[0])

        i = 0
        previous_sign = None # Not yet determined
        previous_frequencies = []
        extreme_values = []
        while i < len(frequency_array): # Loop through the frequency array
            if frequency_array[i] == frequency_array[i]: # Returns false if frequency_array[i] is NaN (undefined datapoint)
                frequency = frequency_array[i]
                if len(previous_frequencies) > 0: # If a previous frequency is known 
                    sign = 1 if frequency >= previous_frequencies[-1][1] else -1 # Determine the gradient sign
                    if previous_sign and sign != previous_sign: # If previous sign is known (not None) and it's different
                        points_1, points_2 = self.regression_input(frequency_array, previous_frequencies, sign, i, sweep_duration, reg_sweep_perc) # Determine which two lines we need to calculation the intersection for
                        if len(points_1) > 1 and len(points_2) > 1: # If we have enough points to form a line
                            t,f = self.regression_intersection(points_1 ,points_2) # Calculate the intersection
                            if lowerlowerbound <= f <= upperlowerbound or lowerupperbound <= f <= upperupperbound:
                                extreme_values.append([t,f]) # Append the line intersection if its within the bounds
                    previous_sign = sign # Remember the sign
                previous_frequencies.append([i,frequency]) # Remember the last two frequencies
                previous_frequencies = previous_frequencies[-2:]
            i += 1

        return np.array(extreme_values)

    def regression_intersection(self, points_1: 'list[list[float, int]]', points_2: 'list[list[float, int]]') -> 'tuple[float]':
        '''Forms linear regression lines for points_1 and points_2, 
        then calculates and returns their intersection point.'''

        # Turn all data into 2D arrays
        points_1_t = np.array(points_1)[:,0].reshape(-1,1)
        points_1_f = np.array(points_1)[:,1].reshape(-1,1)
        points_2_t = np.array(points_2)[:,0].reshape(-1,1)
        points_2_f = np.array(points_2)[:,1].reshape(-1,1)

        # Make the lineair models and remember the gradients/bases
        reg_1 = linear_model.LinearRegression()
        reg_1.fit(points_1_t, points_1_f)
        gradient_1, base_1 = reg_1.coef_[0], reg_1.intercept_
        reg_2 = linear_model.LinearRegression()
        reg_2.fit(points_2_t, points_2_f)
        gradient_2, base_2 = reg_2.coef_[0], reg_2.intercept_

        # Use ax + b = cx + d -> x = (d-c)/(a-b)
        intersect_t = (base_2 - base_1)/(gradient_1 - gradient_2)
        intersect_f = reg_2.predict(np.array([intersect_t]))

        # Uncomment for visualization
        # prediction_1 = reg_1.predict(points_1_t)
        # plt.scatter(points_1_t,points_1_f)
        # plt.plot(points_1_t, prediction_1)
        # prediction_2 = reg_2.predict(points_2_t)
        # plt.scatter(points_2_t,points_2_f)
        # plt.scatter(intersect_t, intersect_f)
        # plt.plot(points_2_t, prediction_2)
        # plt.show()

        return intersect_t[0], intersect_f[0,0] 

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
        r4 = None

        twox, twoy = False, False
        draai1, draai2, draai3 = False, False, False
        r = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        initialvalue = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        coordinates = []
        a = 0.39
        c = 0.55
        for time in range(30):  # For all time points in the data
            r[0] = data[time]
            r[1] = data2[time] 
            r[2] = data3[time]
            r[3] = data4[time]
            r[4] = data5[time]
            r[5] = data6[time]
            r[6] = data7[time]
            r[7] = data8[time]  # Distances to each beacon
            
            if (time == 0):
                for j in range(8):
                    initialvalue[j] = r[j]

            usable = np.array([False, False, False, False, False, False, False, False], dtype=bool)
            if (time != 0):
                for i in range(8):
                    if (r[i] != initialvalue[i]):
                        #interessante waarde die we kunnen gebruiken die meer dan 2 cm verschilt van initiele beginwaarde
                        #kijk welke configuraties mogelijk zijn voor de berekening
                        usable[i] = True


            if (usable[1] == True and usable[2] == True and usable[4] == True): 
                # b4 on y, b1,b2 on x, draai
                r1 = r[1]
                r2 = r[2]
                r3 = r[4]
                print("hello2")
                my = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                mx = (r1 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2 + (2 * a + c) ** 2 + my * (2 * a + 2 * c)) / (2 * a + 2 * c)
                coordinates.append([time, mx, my, 0])   

            if (usable[1] == True and usable[2] == True and usable[7] == True):
                # b1,b2 on y, b7 on x, already on true origin
                r1 = r[1]
                r2 = r[2]
                r3 = r[7]
                print("hello")
                my = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                mx = (r1 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2 + (2 * a) * my) / (2 * a + 2 * c)
                coordinates.append([time, mx, my, 0])   

            elif (usable[1] == True and usable[3] == True and usable[4] == True):
                # b3,b4 on y, b1 on x, draai
                r1 = r[3]
                r2 = r[4]
                r3 = r[1]
                mx = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                my = (r1 ** 2 - r2 ** 2 + (2 * a + c) ** 2 - (2 * a) * mx) / (2 * a + 2 * c)
                coordinates.append([time, mx, my, 0])   

            elif (usable[2] == True and usable[7] == True and usable[0] == True):
                # b2 on y, b7,b0 on x, already on true origin
                r1 = r[2]
                r2 = r[0]
                r3 = r[7]
                mx = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                my = (r2 ** 2 - r3 ** 2 + mx * (2 * a + 2 * c)) / ((2 * a) + (2 * c))
                coordinates.append([time, mx, my, 0])   

            elif (usable[3] == True and usable[4] == True and usable[6] == True):
                # b6 on y, b3,b4 on x, draai2
                r1 = r[6]
                r2 = r[3]
                r3 = r[4]
                a = y6
                b = x3
                c = x4
                mx = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                my = (r1 ** 2 - r2 ** 2 + mx * (2 * a + 2 * c)) / (-2 * a - 2 * c)
                coordinates.append([time, mx, my, 0])   

            elif (usable[3] == True and usable[5] == True and usable[6] == True):
                # b5,b6 on y, b3 on x, draai2
                r1 = r[5]
                r2 = r[6]
                r3 = r[3]
                my = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c) 
                mx = (r1 ** 2 - r2 ** 2 + my * (-2 * a - 2 * c)) / (2 * a + 2 * c)
                coordinates.append([time, mx, my, 0])   

            elif (usable[5] == True and usable[6] == True and usable[0] == True):
                # b0 on y, b5,b6 on x, draai3
                r1 = r[0]
                r2 = r[5]
                r3 = r[6]
                my = (r2 ** 2 - r3 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                mx = (r1 ** 2 - r2 ** 2 + (2 * a + c) ** 2 - (2 * a) * my) / (2 * a + 2 * c)
                coordinates.append([time, mx, my, 0])    

            elif (usable[5] == True and usable[7] == True and usable[0] == True):
                # b7,b0 on y, b5 on x, draai3
                r1 = r[0]
                r2 = r[7]
                r3 = r[5]
                mx = (r1 ** 2 - r2 ** 2 - a ** 2 + (a + c) ** 2) / (2 * c)
                my = (r1 ** 2 - r3 ** 2 - a ** 2 + (2 * a + c) ** 2 + mx * (2 * a + 2 * c)) / (2 * a + 2 * c)
                coordinates.append([time, mx, my, 0])    
            
            #check whether the middle sensor is on the x or y axis
            # first sensor is always on the y-axis
            # third sensor is always on the x-axis
            else:
                #estimated guess based on previous information?
                coordinates.append([time, 0.0, 0.0, 0.0])
            
            for k in range(8):
                usable[k] = False
            twox, twoy = False, False
            draai1, draai2, draai3 = False, False, False

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

    def check_orthogonal(self, positions: 'list[list[float]]') -> bool:
        '''Checks whether all coordinates in positions are orthogonal to each other'''

        # For all combinations of vectors, their dot product should be 0
        for i in range(len(positions)):
            vector_1 = np.array(positions[i])
            for j in range(i+1,len(positions)):
                vector_2 = np.array(positions[j])
                if np.sum(vector_1 * vector_2) != 0:
                    return False

        return True

    def distance(self, vector_1: 'list[float]', vector_2: 'list[float]'):
        '''Calculates the Euclidian distance between points in space'''
        
        vector_1, vector_2 = np.array(vector_1), np.array(vector_2)
        return np.linalg.norm(vector_1-vector_2)
    
    def time_offsets(self, extreme_values: np.ndarray, sweep_duration: float) -> np.ndarray:
        'Calculates time offsets of a given array of extreme value coordinates, based on the sweep duration'
        # NOTE if the first peak isn't identified for some reason, all the values shift. This might be problematic for combining the distances.

        offsets = [[0.0,0.0]] # First offset is 0
        n = 1 # Start at second entry
        time_0 = extreme_values[0,0] # First peak/valley as reference
        while n < len(extreme_values): # Loop through extremes
            time_n = extreme_values[n,0] # Remember their time
            sweeps = np.round((time_n - time_0)/sweep_duration) # How many sweeps have passed
            offset = time_n - time_0 - sweeps*sweep_duration # Time delay between expected and actual time
            offsets.append([sweeps*sweep_duration, offset]) # Append it to the list
            n += 1

        return np.array(offsets)

    def offsets_to_distance(self, starting_distance: int, time_offsets: np.ndarray, sweep_duration: float, temperature: int=20):
        '''Calculates distance per time offset, using temperature to determine sound velocity.'''

        v_sound = 331+(0.61*temperature)

        for i in range(len(time_offsets)):
            time_offsets[i,1] = time_offsets[i,1]*v_sound + starting_distance

        distances = time_offsets

        return distances
    
    def distances_per_timestep(self, distances: 'list[np.ndarray]', sweep_duration: float, n_beacons: int) -> np.ndarray:
        '''Combines all data in distances to one consistent array with the known distances per timestep.'''

        self.data_duration = 30.0
        t = 0
        output = []
        data_left = True if max([len(distances[i]) for i in range(n_beacons)]) > 0 else False
        while data_left and t/100 < self.data_duration/sweep_duration:
            data_row = [t/100] + [float("nan") for i in range(n_beacons)]
            for i in range(n_beacons):
                if len(distances[i]) > 0 and int(distances[i][0,0]*100) == t:
                    if distances[i][0,1] == distances[i][0,1]:
                        data_row[i+1] = distances[i][0,1]
                    distances[i] = np.delete(distances[i], 0, 0)
            output.append(data_row)
            data_left = True if max([len(distances[i]) for i in range(n_beacons)]) > 0 else False
            t += int(sweep_duration*100)

        distances = np.array(output)
        self.distances = distances
        return distances

    def check_sweep_presence(self, frequencies):
        #Function to check whether a sweep is actually present in the audio file

        #Keeps track of the number of NaNs
        counter = 0
        #Keeps track of the total amount of frequencies
        total_length = 0

        #Now we will track if any frequencies are NaN
        for i in frequencies:
            for j in i:
                total_length = total_length + 1
                if math.isnan(j) == True:
                    counter = counter + 1
                else:
                    continue
        NaNpercentage = (counter / total_length) * 100

        if NaNpercentage > 90:
            return False
        else:
            return True

class LocalizationPlot():
    '''Some relevant plotting methods to accompany the Localization class'''
        
    def waveform(sample: Localization, test=False) -> None:
        '''Plots wav file. Assuming sample.data has already been transformed with FFT if fast_fourier == true.''' 

        sample.check_import()
        if sample.samples_per_second == 0:
            time = [i/sample.wav_samplerate for i in range(int(sample.wav_samplerate*sample.data_duration))]
        else:
            data_points_distance = int(sample.wav_samplerate/sample.samples_per_second)
            time = np.ndarray(sample.data.shape[0], float)
            k = 0
            for i in range(0,int(sample.data_duration*sample.wav_samplerate),data_points_distance):
                for j in range(0,sample.fft_window_size):
                    time[k] = (i + j)/sample.wav_samplerate
                    k += 1
        plt.plot(time, sample.data)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        if not test:
            plt.show()

    def spectrum(sample: Localization, test=False, show_datapoints: np.ndarray=np.ndarray((0))) -> None:
        '''Plots frequency data in sample.frequencies.
        show_datapoints can e.g. be used to plot extreme values.'''

        if not sample.state['fft']:
            raise RuntimeError("No frequencies have been calculated yet. Please use FFT or use mock frequencies.")
        time = [(i/sample.samples_per_second) + (0.5*sample.fft_window_size/sample.wav_samplerate) for i in range(int(sample.samples_per_second*sample.data_duration))]
        plt.plot(time, sample.frequencies)
        for datapoint in show_datapoints:
            plt.scatter(datapoint[0], datapoint[1],c='black',marker='x')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency (Hz)')
        if not test:
            plt.show()

    def positions(*samples: Localization) -> None:
        '''Plots positions data for all samples using the positions attribute'''

        for sample in samples:
            if not sample.state['pos']:
                raise RuntimeError("No positions have been determined yet for at least one of your samples.")
            plt.scatter(sample.positions[:,1], sample.positions[:,2])
            for t in range(len(sample.positions[:,0])):
                time = sample.positions[t][0] # annotate time
                plt.annotate(time, (sample.positions[t][1], sample.positions[t][2]))
        plt.show()


# An example from our experiments
# sample = Localization()
# positions = sample.locate('Audio/Experiments/Meting11 02 192kHz.wav', [[19000,19500],[20000,20500],[21000,21500]], 0.2, [[0.0,0.0,0.0],[3.15,0.0,0.0],[0.0,4.0,0.0]],[0.06,0.78,0.55],fft_window_size=8192, noise_level=15, samples_per_second=22)
# LocalizationPlot.spectrum(sample)
# LocalizationPlot.positions(sample)