# Run using an IDE extension 
# or calling 'python test_localization.py'
from SE_site.localization.Localization.localization import Localization
from SE_site.localization.Localization.localization import LocalizationPlot
import unittest
import numpy as np
from numpy import genfromtxt
import os
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

class TestLocalization(unittest.TestCase):

    def test_import_audio(self):
        for mono_or_stereo in ['mono.wav','stereo.wav']:
            sample = Localization()
            sample.import_audio('Audio/Samples/440hz_' + mono_or_stereo)
            self.assertEqual(type(sample.data), np.ndarray)
            self.assertEqual(sample.data.ndim, 1) # Should reduce dimensions to 1
            self.assertEqual(type(sample.wav_samplerate), int)
            self.assertEqual(type(sample.data_duration), float)
            self.assertEqual(sample.state['import'], True)

    def test_plot_waveform(self):
        sample = Localization()
        with self.assertRaises(RuntimeError):
            LocalizationPlot.waveform(sample, test=True)
        sample.import_audio('Audio/Samples/440hz_mono.wav')
        LocalizationPlot.waveform(sample, test=True)
    
    def test_plot_spectrum(self):
        sample = Localization()
        with self.assertRaises(RuntimeError):
            LocalizationPlot.spectrum(sample,test=True)
        sample.mock_ultrasound_frequency_array()
        LocalizationPlot.spectrum(sample, test=True)

    def test_generate_sweep(self):
        sample = Localization()
        # Total duration shorter than sweep duration
        sample.generate_sweep(440,880,44100,1.0,1.0)
        # Check data duration & swapping stop and start frequency
        sample.generate_sweep(880,440,44100,1.0,2.0)
        sample.import_audio("sweep_1beacon_880-440Hz.wav")
        self.assertEqual(sample.data_duration, 2.0)
        os.remove("sweep_1beacon_880-440Hz.wav")

    def test_mix_sweeps(self):
        sample = Localization()
        sweep_1 = sample.generate_sweep(261,329,44100,1.0,8.0)
        sweep_2 = sample.generate_sweep(329,392,44100,1.0,10.0)
        sweep_3 = sample.generate_sweep(392,523,88200,1.0,8.0)
        # Correct file format, only one sweep
        self.assertEqual('sweep_test.wav', sample.mix_sweeps('test', 44100, 8.0, sweep_1))
        # ValueError when mixed file duration is longer than sweep duration
        with self.assertRaises(ValueError):
            sample.mix_sweeps('test', 44100, 10.0, sweep_1, sweep_2)
        # ValueError when sweep samplerates don't match
        with self.assertRaises(ValueError):
            sample.mix_sweeps('test', 44100, 10.0, sweep_1, sweep_2, sweep_3)
        # Remove test files, save chord file for manual test
        os.remove('sweep_test.wav') 
        sweep_3 = sample.generate_sweep(392,523,44100,1.0,8.0)
        sample.import_audio(sample.mix_sweeps('chord', 44100, 8.0, sweep_1, sweep_2, sweep_3))

    def test_mix_sweeps_new(self):
        sample = Localization()
        sweep_1 = sample.generate_sweep(261,329,44100,1.0,8.0)
        sweep_2 = sample.generate_sweep(329,392,44100,1.0,10.0)
        sweep_3 = sample.generate_sweep(392,523,88200,1.0,8.0)
        # Correct file format, only one sweep
        self.assertEqual('sweep_test.wav', sample.mix_sweeps('test', 44100, 8.0, sweep_1))
        # ValueError when mixed file duration is longer than sweep duration
        with self.assertRaises(ValueError):
            sample.mix_sweeps('test', 44100, 10.0, sweep_1, sweep_2)
        # ValueError when sweep samplerates don't match
     
        sample.mix_sweeps('test', 44100, 10.0, sweep_1, sweep_2, sweep_3)
       # sample.export_audio()

    def test_compress(self):
        sample = Localization()
        # Error if no audio has been imported yet
        with self.assertRaises(RuntimeError):
            sample.compress()
        sample.import_audio("Audio/Samples/440hz_mono.wav")
        data_before = sample.data.copy()
        # No difference if samples_per_second = 0
        sample.compress(samples_per_second=0)
        self.assertEqual(len(data_before),len(sample.data))
        with self.assertRaises(RuntimeWarning): # Too many samples
            sample.compress(fft_window_size=1000000)
        with self.assertRaises(RuntimeWarning): # No compression (samplerate/10 = 4410)
            sample.compress(fft_window_size=8820)
        sample.compress() # Test size reduction
        self.assertEqual(8192*5*sample.data_duration, len(sample.data))
        with self.assertRaises(RuntimeError):
            # Cannot compress if already compressed
            sample.compress()
        LocalizationPlot.waveform(sample, test=True) # Test no errors with plotting

    def test_mock_frequency_array(self):
        sample = Localization()
        sample.import_audio("Audio/Samples/440hz_mono.wav")
        with self.assertRaises(RuntimeWarning): # Audio has already been imported
            sample.mock_ultrasound_frequency_array()
        sample = Localization()
        with self.assertRaises(ValueError): # Zero beacons should error
            sample.mock_ultrasound_frequency_array(n_beacons=0)
        for n in range(1,5): # Test up to 5 beacons
            sample = Localization()
            sample.mock_ultrasound_frequency_array(n_beacons=n)
            self.assertEqual(n,sample.frequencies.shape[1])
        sample = Localization()
        f_array = sample.mock_ultrasound_frequency_array(separate_domains=False)
        for n in range(0,3): # Test max/min if domains are not separate
            max, min = np.amax(f_array[:,n]), np.amin(f_array[:,n])
            self.assertAlmostEqual(max,21000,delta=100)
            self.assertAlmostEqual(min,19000,delta=100)
        sample = Localization()
        f_array = sample.mock_ultrasound_frequency_array()
        # Test separate domains
        self.assertAlmostEqual(np.amin(f_array[:,0]),19000,delta=50)
        self.assertGreater(np.amin(f_array[:,1]),np.amax(f_array[:,0]))
        self.assertGreater(np.amin(f_array[:,2]),np.amax(f_array[:,1]))
        self.assertAlmostEqual(np.amax(f_array[:,2]),21000,delta=50)
    
    def test_find_next_not_nan(self):
        sample = Localization()
        self.assertEqual(sample.find_next_not_nan(np.array([1,float("nan"),2]),0),(2,2)) # Value 2 at index 2 is the first non-nan value after 1
        self.assertEqual(sample.find_next_not_nan(np.array([1,float("nan"),float("nan")]),0),None) # No value left, so should be None
    
    def test_identify_extremes(self):
        sample = Localization()
        with self.assertRaises(ValueError):
            sample.identify_extremes(sample.mock_ultrasound_frequency_array(),[19000,21000], 1.0)
        sample = Localization()
        extremes = sample.identify_extremes((sample.mock_ultrasound_frequency_array(n_beacons=1))[:,0],[19000,21000], 1.0)
        self.assertAlmostEqual(abs(extremes[0,0]-extremes[1,0]),1.0,5) # Difference between mock extremes should be 1.00000

    def test_time_offsets(self):
        sample = Localization()
        f_array = sample.mock_ultrasound_frequency_array() # Perfect mock array, time offset should be 0
        extremes = sample.identify_extremes(f_array[:,0],[19000,19700],1.0)
        time_offsets = sample.time_offsets(extremes,1.0)
        for time_offset in time_offsets:
            self.assertAlmostEqual(time_offset[1], 0.0, 3)
        sample = Localization()     
        f_array = sample.mock_ultrasound_frequency_array() # Perfect mock array
        extremes = sample.identify_extremes(f_array[:,0],[19000,19700],1.0)
        extremes = np.concatenate((extremes[:2],extremes[3:])) # Simulate missing peak
        time_offsets_missing_peak = sample.time_offsets(extremes,1.0)
        for time_offset in time_offsets_missing_peak:
            self.assertAlmostEqual(time_offset[1], 0.0, 3)

    def test_distance(self):
        sample = Localization()
        self.assertEqual(sample.distance([1,1,1],[1,1,1]),0) # Zero if point is the same
        self.assertEqual(sample.distance([0,0,0],[0,2,0]),2) # Difference in 1D
        self.assertEqual(sample.distance([0,0,0],[0,-2,0]),2) # Only positive values
        self.assertEqual(sample.distance([1,1,1],[5,4,1]),5) # Difference in 2D (5-4-3)-rule

    def test_check_orthogonal(self):
        sample = Localization()
        # Some example cases
        self.assertEqual(sample.check_orthogonal([[1,2,3],[1,2,3]]),False)
        self.assertEqual(sample.check_orthogonal([[0,0,0],[0,0,0]]),True)
        self.assertEqual(sample.check_orthogonal([[1,0,0],[0,1,0],[0,0,0]]),True)
        self.assertEqual(sample.check_orthogonal([[1,2,0],[2,-1,10]]),True)
    
    def test_offsets_to_distances(self):
        sample = Localization()
        # Simple test case with starting distance is 1 meter
        test = np.array([[0.0,0.0],[1.0,0.1],[2.0,0.25],[3.0,0.30],[4.0,0.30],[5.0,0.2]])
        test = sample.offsets_to_distance(1,test,1.0)
        answer = np.array([[0.0,1.0],[1.0,35.32],[2.0,86.8],[3.0,103.96],[4.0,103.96],[5.0,69.64]])
        self.assertEqual(len(test),len(answer))
        i = 0
        while i < len(test):
            self.assertAlmostEqual(test[i,1],answer[i,1],5)
            i += 1

    def test_get_frequencies(self):
        # Basic example of perfect sweep
        sample = Localization()
        sweep_1 = sample.generate_sweep(19000,19700,96000,1.0,8.0,asynchronous=False)
        sweep_2 = sample.generate_sweep(19750,20450,96000,1.0,8.0,asynchronous=False)
        sweep_3 = sample.generate_sweep(20500,21200,96000,1.0,8.0,asynchronous=False)
        sample.import_audio(sample.mix_sweeps('sweepmix', 96000, 8.0, sweep_1, sweep_2, sweep_3))
        sample.compress(samples_per_second=10,fft_window_size=4096)
        f_array = sample.get_frequencies([[19000,19700],[19750,20450],[20500,21200]],15.0)
        self.assertAlmostEqual(np.amin(f_array[:,0]),19000,delta=100)
        self.assertGreater(np.amin(f_array[:,1]),np.amax(f_array[:,0]))
        self.assertGreater(np.amin(f_array[:,2]),np.amax(f_array[:,1]))
        self.assertAlmostEqual(np.amax(f_array[:,2]),21200,delta=100)
        os.remove('sweep_sweepmix.wav')

    def test_fast_fourier_transform(self):
        sample = Localization()
        sample.import_audio('Audio/Samples/string.wav')
        N = sample.wav_samplerate*sample.data_duration # total points in signal
        Pxx, f = sample.fast_fourier_transform(sample.data, sample.wav_samplerate, N)
        for freq in f: # All frequencies should be positive
            self.assertEqual(freq - abs(freq),0)
        self.assertEqual(len(Pxx),len(f)) # Length of Pxx and f should be equal

    def test_discard_frequencies(self):
        sample = Localization()
        test_array = np.array([[0,1],[0,2],[0,3],[0,4],[0,5]])
        # Nothing discarded (bounds are inclusive)
        self.assertEqual(len(sample.discard_frequencies(test_array,[1,5])),len(test_array))
        # One thing discarded
        self.assertEqual(len(sample.discard_frequencies(test_array,[2,5])),4)
        # One thing discarded, order of bounds doesn't matter
        self.assertEqual(len(sample.discard_frequencies(test_array,[5,2])),4)

    def test_find_peak_indices(self):
        sample = Localization()
        # Test case, 4 and 2.5 are the peaks
        test_1 = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[2,0],[1,0],[2.5,0],[1,0],[0,0]])
        indices = sample.find_peak_indices(test_1,2)
        self.assertEqual(test_1[indices[0]][0],4)
        self.assertEqual(test_1[indices[1]][0],2.5)
        # No error if peaks are on the side
        test_2 = np.array([[2,0],[1,0],[0,0],[0,0],[0,0],[0,0],[1,0],[2,0]])
        indices = sample.find_peak_indices(test_2,2)
        # Empty array if no peaks are found
        self.assertEqual(len(indices),0)

    def test_calc_weighted_frequencies(self):
        sample = Localization()
        test_1 = np.array([[1,2],[1,2],[1,2]])
        # All surrounding frequencies are 2 -> average = 2
        self.assertEqual(sample.calc_weighted_frequencies(test_1,[1],1)[0],2)
        # No error if peaks are on the sides 
        self.assertEqual(sample.calc_weighted_frequencies(test_1,[0],1)[0],2)
        # No error if surrounding freqs is out of bounds
        self.assertEqual(sample.calc_weighted_frequencies(test_1,[0],100)[0],2)
        # Proper average calculation (1/4)*2 + (2/4)*2 + (1/4)*1 = 1.75
        test_2 = np.array([[1,2],[2,2],[1,1]])
        self.assertEqual(sample.calc_weighted_frequencies(test_2,[1],1)[0],1.75)

    def test_freqs_per_domain(self):
        sample = Localization()
        test = np.array([[19200,19300,19400], # All from one beacon
                         [19200,20200,21200], # All from separate beacons
                         [0,0,0], # All from none of the beacons
                         [21900,19200,19300,20940]]) # Two from one beacon, domain boundaries
        outcome = sample.freqs_per_domain(test,[[19000,19900],[20000,20900],[21000,21900]])
        self.assertEqual(len(outcome[0][0]),3)
        self.assertEqual(len(outcome[1][0]),1)
        self.assertEqual(outcome[1][0][0],19200)
        self.assertEqual(len(outcome[2][0]),0)
        self.assertEqual(len(outcome[3][0]),2)
        self.assertEqual(outcome[3][1][0],20940)

    def test_similarity(self):
        sample = Localization()
        self.assertEqual(sample.similarity(10,10,0.1),0) # Numbers are entirely similar
        self.assertEqual(sample.similarity(10,9.5,0.1),0.5) # Known difference
        self.assertEqual(sample.similarity(9.5,10,0.1),0.5) # Always absolute (>=0) similarity
        self.assertEqual(sample.similarity(-9.5,10,0.1),-1) # -1 if not similar at all
        with self.assertRaises(ValueError):
            sample.similarity(1,1,-1.0) # Unallowed tolerance value

    def test_add_sweeplike(self):
        sample = Localization()
        self.assertEqual(sample.add_sweeplike(8,1,1,0,10),(9,False)) # Simple addition
        self.assertEqual(sample.add_sweeplike(9,1,1,0,10),(10,False)) # Domain boundaries
        self.assertEqual(sample.add_sweeplike(9,2,1,0,10),(9,True)) # Sign change ocurred
        self.assertEqual(sample.add_sweeplike(4,1,-1,0,10),(3,False)) # Simple subtraction
        self.assertEqual(sample.add_sweeplike(1,2,-1,0,10),(1,True)) # Sign change lowerbound
        with self.assertRaises(ValueError):
            self.assertEqual(sample.add_sweeplike(1,2,20,10,0))

    def test_distances_per_timestep(self):
        sample = Localization()
        sample.data_duration = 4.0
        for i in [3,4]: # Try for three and four beacons
            # Create example with distance = 1 for all timesteps for all beacons
            example = [np.array([[t, 1.0] for t in range(5)]) for n in range(i)]
            # Set some of those to 'nan'
            example[0][0,1] = float("nan")
            example[0][3,1] = float("nan")
            example[1][4,1] = float("nan")
            np.delete(example[2], 0, 0) # Remove an entire row
            # Now test that it doesn't error and has the right format
            answer = sample.distances_per_timestep(example, 1.0, i)
            self.assertEqual(len(answer), sample.data_duration)
            self.assertEqual(len(answer[0]), i+1)
        
    def test_locate(self):
        sample = Localization()
        with self.assertRaises(ValueError):
            # Lengths of f_domains and pos_beacons dont match
            sample.locate('Audio/Samples/string.wav', [[0,0],[0,0],[0,0]], 0.0, [[0.0,0.0,0.0],[0.0,0.0,0.0]],[0.0,0.0,0.0])
        sample = Localization()
        with self.assertRaises(ValueError):
            # Only x and y beacon coordinates are given
            sample.locate('Audio/Samples/string.wav', [[0,0],[0,0],[0,0]], 0.0, [[0.0,0.0],[0.0,0.0],[0.0,0.0]],[0.0,0.0,0.0])
        sample = Localization()
        with self.assertRaises(ValueError):
            # Only x and y starting coordinates are given
            sample.locate('Audio/Samples/string.wav', [[0,0],[0,0],[0,0]], 0.0, [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],[0.0,0.0])
        # No errors if all parameters are ok
        positions = sample.locate('Audio/Experiments/Meting11 02 192kHz.wav', [[19000,19500],[20000,20500],[21000,21500]], 0.2, [[0.0,0.0,0.0],[3.15,0.0,0.0],[0.0,4.0,0.0]],[0.06,0.78,0.55],fft_window_size=8192, noise_level=15, samples_per_second=22)
        # LocalizationPlot.waveform(sample)
        LocalizationPlot.spectrum(sample)
        LocalizationPlot.positions(sample)
        return sample.positions

    def test_check_sweep_presence(self):
        sample = Localization()
        sweep_1 = sample.generate_sweep(19000,19700,96000,1.0,8.0,asynchronous=False)
        sweep_2 = sample.generate_sweep(19750,20450,96000,1.0,8.0,asynchronous=False)
        sweep_3 = sample.generate_sweep(20500,21200,96000,1.0,8.0,asynchronous=False)
        sample.import_audio(sample.mix_sweeps('sweepmix', 96000, 8.0, sweep_1, sweep_2, sweep_3))
        sample.compress(samples_per_second=10,fft_window_size=4096)
        f_array = sample.get_frequencies([[19000,19700],[19750,20450],[20500,21200]],1.0)
        self.assertEqual(sample.check_sweep_presence(f_array), True)

    def test_determine_coordinates(self, data: np.ndarray, data2: np.ndarray, data3: np.ndarray, data4: np.ndarray, data5: np.ndarray, data6: np.ndarray, data7: np.ndarray, data8: np.ndarray):
        sample = Localization()
        #which sensors are used for trilateration
        positions = sample.determine_coordinates(data, data2, data3, data4, data5, data6, data7, data8, [[0.39,0.94,0.0],[0.94,0.39,0.0],[0.39,0.94,0.0],[0.94,0.39,0.0],[0.39,0.94,0.0],[0.94,0.39,0.0],[0.39,0.94,0.0],[0.94,0.39,0.0]])
        #rotate points to true origin
        LocalizationPlot.positions(sample)
        return sample.positions

if __name__ == '__main__':
    # unittest.main()
    #switched 1 and 2 to make sensor in the 90 degree angle sensor 1
    data = np.array(genfromtxt('experiment8sensor1object.csv', delimiter=','))
    data2 = np.array(genfromtxt('experiment8sensor1object-2.csv', delimiter=','))
    data3 = np.array(genfromtxt('experiment8sensor1object-3.csv', delimiter=','))
    data4 = np.array(genfromtxt('experiment8sensor1object-4.csv', delimiter=','))
    data5 = np.array(genfromtxt('experiment8sensor1object-5.csv', delimiter=','))
    data6 = np.array(genfromtxt('experiment8sensor1object-6.csv', delimiter=','))
    data7 = np.array(genfromtxt('experiment8sensor1object-7.csv', delimiter=','))
    data8 = np.array(genfromtxt('experiment8sensor1object-8.csv', delimiter=','))
    # distances = np.array([data,data2,data3])
    # print(distances)
    df = pd.DataFrame(columns=['time'])
    df.loc[:,"time"] = np.arange(0,30,1) 
    df.insert(1, 'd1', data) 
    df.insert(2, 'd2', data2) 
    df.insert(3, 'd3', data3) 
    df.insert(4, 'd4', data4) 
    df.insert(5, 'd5', data5) 
    df.insert(6, 'd6', data6) 
    df.insert(7, 'd7', data7) 
    df.insert(8, 'd8', data8) 
    test = TestLocalization()
    positions = test.test_determine_coordinates(data, data2, data3, data4, data5, data6, data7, data8)
    df2 = pd.DataFrame((positions), columns=['time','mx', 'my', 'mz'])
    result = pd.concat([df, df2], axis=1)
    print(result)
    result.to_csv('experiment8sensor1objectresult.csv')  