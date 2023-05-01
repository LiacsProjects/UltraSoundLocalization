import serial
import time
import matplotlib.pyplot as plt
import numpy as np

# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('COM4', 9800, timeout=1)
time.sleep(2)

data = []
data2 = []
data3 = []
for i in range(30):
    line = ser.readline()   # read a byte string
    if line:
        line = line.strip()
        string = line.decode()  # convert the byte string to a unicode string
        num = int(string) # convert the unicode string to an int
        print("sensor 1: ")
        print(num)
        data.append(num) # add int to data list
    line2 = ser.readline()   # read a byte string
    if line2:
        line2 = line2.strip()
        string = line2.decode()  # convert the byte string to a unicode string
        num = int(string) # convert the unicode string to an int
        print("sensor 2: ")
        print(num)
        data2.append(num) # add int to data list    
    line3 = ser.readline()   # read a byte string
    if line3:
        line3 = line3.strip()
        string = line3.decode()  # convert the byte string to a unicode string
        num = int(string) # convert the unicode string to an int
        print("sensor 3: ")
        print(num)
        data3.append(num) # add int to data list    
ser.close()

# build the plot
plt.plot(data)
plt.plot(data2)
plt.plot(data3)
plt.xlabel('Time')
plt.ylabel('Distance of object')
plt.title('Distance to object over time')
plt.show()
np.savetxt('data.csv', data, delimiter=',')
np.savetxt('data2.csv', data2, delimiter=',')
np.savetxt('data3.csv', data3, delimiter=',')