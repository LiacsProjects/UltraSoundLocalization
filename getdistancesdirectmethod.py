import serial
import time
import matplotlib.pyplot as plt
import numpy as np

# make sure the 'COM#' matches the Arduino
ser = serial.Serial('COM7', baudrate=19200, timeout=1)
ser2 = serial.Serial('COM5', baudrate=19200, timeout=1)
ser3 = serial.Serial('COM6', baudrate=19200, timeout=1)
ser4 = serial.Serial('COM4', baudrate=19200, timeout=1)
time.sleep(5)

data, data2, data3 = [], [], []
for i in range(30):
    #send start signal to serial port
    ser.write(b'S')
    ser4.write(b'S')
    line = ser.readline()  # read a byte string
    if line:
        line = line.strip()
        string = line.decode()  # convert the byte string to a unicode string
        num = int(string) # convert the unicode string to an int
    line2 = ser4.readline()
    if line2:
        line2 = line2.strip()
        string2 = line2.decode()  
        num2 = int(string2) 
        print("sensor 1: ", num2 * 2)
        data.append(num2 * 2 / 100) # add int to data list
    ser2.write(b'S')
    ser4.write(b'S')
    line3 = ser2.readline() 
    if line3:
        line3 = line3.strip()
        string3 = line3.decode() 
        num3 = int(string3) 
    line4 = ser4.readline()   
    if line4:
        line4 = line4.strip()
        string4 = line4.decode()  
        num4 = int(string4) 
        print("sensor 1: ", num4 * 2)
        data2.append(num4 * 2 / 100) 
    ser3.write(b'S')
    ser4.write(b'S')
    line5 = ser3.readline()   
    if line5:
        line5 = line5.strip()
        string5 = line5.decode()  
        num5 = int(string5) 
    line6 = ser4.readline()   
    if line6:
        line6 = line6.strip()
        string6 = line6.decode() 
        num6 = int(string6) 
        print("sensor 3: ", num6 * 2, "\n")
        data3.append(num6 * 2 / 100)  
ser.close()
ser2.close()
ser3.close()
ser4.close()

# build the plot
plt.plot(data)
plt.plot(data2)
plt.plot(data3)
plt.xlabel('Time (s)')
plt.ylabel('Distance of object (m)')
plt.title('Distance to object over time')
plt.legend(['Sensor 1', 'Sensor 2','Sensor 3'])
plt.show()
# save data to .csv file
np.savetxt('file-1.csv', data, delimiter=',')
np.savetxt('file-2.csv', data2, delimiter=',')
np.savetxt('file-3.csv', data3, delimiter=',')
