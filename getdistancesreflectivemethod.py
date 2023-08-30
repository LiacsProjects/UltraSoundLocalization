import serial
import time
import matplotlib.pyplot as plt
import numpy as np

# make sure the 'COM#' matches the Arduino
ser = serial.Serial('COM5', 9800, timeout=1)
ser2 = serial.Serial('COM4', 9800, timeout=1)
ser3 = serial.Serial('COM7', 9800, timeout=1)
ser4 = serial.Serial('COM6', 9800, timeout=1)
time.sleep(5)

data, data2, data3, data4, data5, data6, data7, data8 = [], [], [], [], [], [], [], []
for i in range(30):
    line = ser.readline()   # read a byte string
    if line:
        line = line.strip()
        string = line.decode() # convert the byte string to a unicode string
        num = int(string) # convert the unicode string to an int
        print("sensor 1: ", num)
        data.append(num / 100) # add int to data list
    line2 = ser.readline()  
    if line2:
        line2 = line2.strip()
        string = line2.decode()  
        num = int(string) 
        print("sensor 2: ", num)
        data2.append(num / 100)
    line3 = ser2.readline()  
    if line3:
        line3 = line3.strip()
        string = line3.decode() 
        num = int(string)
        print("sensor 3: ", num)
        data3.append(num / 100)  
    line4 = ser2.readline()  
    if line4:
        line4 = line4.strip()
        string = line4.decode() 
        num = int(string)
        print("sensor 4: ", num)
        data4.append(num / 100)    
    line5 = ser3.readline()  
    if line5:
        line5 = line5.strip()
        string = line5.decode() 
        num = int(string) 
        print("sensor 5: ", num)
        data5.append(num / 100)   
    line6 = ser3.readline()  
    if line6:
        line6 = line6.strip()
        string = line6.decode()  
        num = int(string) 
        print("sensor 6: ", num)
        data6.append(num / 100)    
    line7 = ser4.readline()   
    if line7:
        line7 = line7.strip()
        string = line7.decode()  
        num = int(string) 
        print("sensor 7: ", num)
        data7.append(num / 100)     
    line8 = ser4.readline()   
    if line8:
        line8 = line8.strip()
        string = line8.decode()  
        num = int(string) 
        print("sensor 8: ", num, "\n")
        data8.append(num / 100)    
ser.close()
ser2.close()
ser3.close()
ser4.close()

# build the plot
plt.plot(data)
plt.plot(data2)
plt.plot(data3)
plt.plot(data4)
plt.plot(data5)
plt.plot(data6)
plt.plot(data7)
plt.plot(data8)
plt.xlabel('Time')
plt.ylabel('Distance of object')
plt.title('Distance to object over time')
plt.show()
# save data to .csv file
np.savetxt('file-1.csv', data, delimiter=',')
np.savetxt('file-2.csv', data2, delimiter=',')
np.savetxt('file-3.csv', data3, delimiter=',')
np.savetxt('file-4.csv', data4, delimiter=',')
np.savetxt('file-5.csv', data5, delimiter=',')
np.savetxt('file-6.csv', data6, delimiter=',')
np.savetxt('file-7.csv', data7, delimiter=',')
np.savetxt('file-8.csv', data8, delimiter=',')