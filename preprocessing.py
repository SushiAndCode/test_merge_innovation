import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import math
import time
import platform
from datetime import datetime
from scipy.signal import butter, filtfilt, lfilter
import warnings

#To suppress warnings in curvature computation
warnings.filterwarnings("ignore")


def truncate(number, digits):
    #Function to truncate number
    #Parameters
    #number: the number to truncate
    #digit: how many digit u want to keep
    nbDecimals = len(str(number).split('.')[1])
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return float(math.trunc(stepper * number) / stepper)

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#Function to convert string csv values of time in integer values
def time_convert(val):
    h, m, s_ms = val.split(':')
    s, ms = s_ms.split('.')
    return int(h) * 3600 * 1000 + int(m) * 60 * 1000 + int(s) * 1000 + int(ms)

#Convert from GPS coordinates to meters
def GPS_to_m_converter(lon1, lat1, lon2, lat2):
    #Radius of the earth
    R = 6372795.477598
    #Conversion from latitude and longitude to distance
    lon1 = lon1*math.pi/180
    lat1 = lat1*math.pi/180
    lon2 = lon2*math.pi/180
    lat2 = lat2*math.pi/180
    #Take the average of the 2 values
    latm = (lat1+lat2)/2
    lonm = (lon1+lon2)/2
    # deltax is the distance in meters along the x axis, 
    # we calculate the distance between the 2 point keeping the 
    # latitude and changing the longitude
    deltax = R*math.acos(truncate(math.sin(latm)*math.sin(latm) + math.cos(latm)*math.cos(latm)*math.cos(lon1-lon2), 15))
    deltay = R*math.acos(truncate(math.sin(lat1)*math.sin(lat2) +math.cos(lat1)*math.cos(lat2)*math.cos(lonm-lonm),15))
    dist = math.sqrt(pow(deltax,2) + pow(deltay,2))
    return dist

def computeDist(df):
    ''' Compute the distance (curviliear abscissa) covered by the trajectory of the bike.
        For every sampled point convert the GPS coordinates to points in plane and compute the 
        distance. Apply also some filling for missing values (due to different T_sampling)

        INPUT   dataframe with at least Latitude and Longitude columns
        OUTPUT  input dataframe with column Distance added

        IMPROVEMENT Interpolate distance for missing values linearly instead of previous/post values,
                allows for better precision in measures
    '''
    # total distance covered from the bike
    dist = 0
    #Iterate over all the rows of the df
    for row in df.index:
        #Check that the GPS is not recording just 0
        if df['Latitude'][row] > 30 and df['Longitude'][row] > 1:
            #Check that it is not the last row or that the next row has the same coordinates due to low sampling
            if(row < len(df.index)-1 and df['Longitude'][row] != df['Longitude'][row+1]):
                #Compute the distance between current sampled point and the next one in the df
                dist = dist + GPS_to_m_converter(df['Longitude'][row],df['Latitude'][row],df['Longitude'][row+1],df['Latitude'][row+1])
                #Update cumulated distance
                df.loc[row,'Distance'] = dist
    print('\nDISTANCE OF THE OVERALL SESSION [m]:')
    print(dist)
    #Fill missing values for the distance
    df['Distance'].interpolate(method='linear', inplace=True)
    #df['Distance'].fillna(method='bfill', inplace=True)
    #df['Distance'].fillna(method='ffill', inplace=True)

def lapsDetector(data):

    # Computing distance of the bike along the track
    computeDist(data)
    #Variable support to avoid to use data1
    all_data_bike = data
    #List representing the headers of teh dataframe
    headers = all_data_bike.columns.to_list()
    #I choose one row every 8 to avoid wasting time and be coherent with positions_moto
    all_data_bike = all_data_bike.iloc[::8,:]
    #Conversion from dataframe to list of all_data_bike after the grouping of the rows
    all_data_bike = all_data_bike.values.tolist()
    #Set of data with only time and GPS coordinates
    positions_moto = data[["Latitude","Longitude","time"]]
    #I choose one row every 8 to avoid wasting time
    positions_moto = positions_moto.iloc[::8,:]
    #Conversion from dataframe to list of positions_moto
    positions_moto = positions_moto.values.tolist()

    #Counter of the laps
    laps_counter = 0
    #Flag to show when a laps starts
    primo = 1
    #Flag to save the time when a lap starts 
    tempo_primo = 0
    #Flag to sign up that the lap is finished
    finished_lap = 0
    #List to save inside all the couples of indexes of start_position and finish_position(of the bike) of a laps according to the dataframe positions_moto 
    list_of_laps = []
    #List of dataframes corresponding to the lap
    dataframes_laps = dict()

    #Pass trough all the positions of the bike in the telemetry
    for position in positions_moto:
        #condition to understand when the bike pass through the start for the first time
        if(primo == 1 and float(position[1]) >= 10.31243007387261 and float(position[1]) <= 10.31251768450424 and float(position[0]) >= 45.085652665199974 and float(position[0]) <= 45.08587982499387):
            #Set the flag to zero to show the bike has already passed from the start
            primo = 0
            #Set the flag to the sum between the time when the bike pass on the start plus 20 seconds 
            #Is done like this in order to let the bike to exit from the area which stand for the beginnign of the track
            tempo_primo = time_convert(position[2]) + time_convert("00:00:20.000")
            #Save the corresponding index of the dataframe equal to the start of the lap
            start_lap = positions_moto.index(position)
        #Check when the bike ends the laps
        if(time_convert(position[2]) > tempo_primo and primo == 0 and float(position[1]) >= 10.31243007387261 and float(position[1]) <= 10.31251768450424 and float(position[0]) >= 45.085652665199974 and float(position[0]) <= 45.08587982499387):
            #I rise the laps counter of 1
            laps_counter = laps_counter + 1
            #Reset all the flag to restart a new lap
            primo = 1
            tempo_primo = 0
            #Save the corresponding index of the dataframe equal to the end of the lap
            end_lap = positions_moto.index(position)
            #Set the variable to 1 to point out that the laps is finished
            finished_lap = 1
        if(finished_lap == 1):
            #Insert inside the list the index of the beginning and of the end of the lap
            list_of_laps.append([start_lap,end_lap]) 
            #Set the flag to 0 
            finished_lap = 0
            #Create the key value for the dictionary
            stringa_lap = "lap" + str(laps_counter)
            #Save into the dictionary an array containing rows of the dataframe which stands for a frame of the laps
            #All the rows inside togheter represent the lap
            dataframes_laps.update({stringa_lap : all_data_bike[start_lap:end_lap]})


    '''45.08537982499387, 10.31251768450424
    45.085252665199974, 10.31243007387261

    y1 = np.linspace(10.31243007387261, 10.31251768450424, 150)
    x1 = np.linspace(45.085652665199974, 45.08587982499387, 150)
    #y2 = np.linspace(10.312380203902705, 10.312443949974858, 150)
    #x2 = np.linspace(45.085246788575844, 45.0852604857657, 150)
    print("ADESSO PRINTO IL TRACCIATO")
    plt.plot(data["Latitude"],data["Longitude"], label='Series 1')

    plt.plot(x1,y1, label='Series 2', color = 'orange')
    #plt.plot(x2,y2,label='Series 3',color = 'red')
    #plt.plot(x3,y3,label='Series 4',color = 'red')
    #plt.plot(x4,y4,label='Series 5',color = 'red')
    
    plt.show()
    print("FINITO PRINT DEL TRACCIATO")'''

    #df with all the telemetry data for all the laps
    df_laps = dict().fromkeys(dataframes_laps.keys())
    #df with laptimes 
    laptimes = dict().fromkeys(dataframes_laps.keys())
    #Bets laptime
    best_lap = 0
    best_lap_idx = 0
    i = 1
    #Iterate over all the laps
    for lap in dataframes_laps.keys():
        #Conversion from list into dataframe and assigned to final df
        df_laps[lap] = pd.DataFrame(dataframes_laps[lap], columns = headers)
        df_laps[lap]['Abscissa'] = df_laps[lap]['Distance'] - df_laps[lap]['Distance'][0]
        #Curvature computation 
        curvatureCal(df_laps[lap])
        #Compute laptime of the current lap
        curr_laptime= datetime.strptime(df_laps[lap]['time'][len(df_laps[lap].index)-1], "%H:%M:%S.%f") - datetime.strptime(df_laps[lap]['time'][0], "%H:%M:%S.%f")
        #Convert to float the timedelta object
        curr_laptime = curr_laptime.total_seconds()
        laptimes[lap] =  curr_laptime
        #Update best time value if still 0 or current is faster
        if best_lap == 0 or curr_laptime < best_lap:
            best_lap = curr_laptime  
            best_lap_idx = i  
        i+=1

    #Print the number of laps
    print('\nNUMBER OF LAPS IN THE SESSION:')
    print(len(dataframes_laps.keys()))
    print('\nBEST LAP TIME:')
    print(f'Lap {best_lap_idx}: {best_lap} [s]')

    return df_laps, laptimes, best_lap_idx

def curvatureCal(df):
    """
    Calculates cumulative travel, curvature of each segment and components of curvature vectors.
    :param x: x coordinates list
    :param y: y coordinates list
    :return: Cumulative travel, curvatures with sign and k_urvature vectors
    """
    #Sampling only different values for lat and long is needed, due to low sampling rate most of them are equal

    #Empty arrays to store lat, long, indexes to puth back in original df
    x = np.empty(0)
    y = np.empty(0)
    idx = np.empty(0)
    #Create a copy of the df to store different values 
    for i in df.index:
        if i == 0:
            x = np.append(x, df.loc[i,'Latitude'])
            y = np.append(y, df.loc[i,'Longitude'])
            idx = np.append(idx, i)
        else:
            if df.loc[i,'Latitude'] != x[-1]:
                x = np.append(x, df.loc[i,'Latitude'])
                y = np.append(y, df.loc[i,'Longitude'])
                idx = np.append(idx, i)

    z = np.zeros(len(x))

    track = np.transpose(np.array([x, y, z]))
    L = [0]
    curvature = [0]
    k = np.array([[0, 0, 0]])
    for i in range(len(x)-2):
        R, k_i = circumcenter(track[i,:], track[i+1,:], track[i+2,:])
        k = np.append(k, [np.transpose(k_i)], axis=0)
        L.append(L[i]+np.linalg.norm(track[i, :]-track[i-1, :]))
        sign = np.sign(np.cross(track[i+1, :]-track[i, :], k_i)[2])
        curvature.append(R**(-1)*sign*(-1))
    
    curvature = abs(np.asarray(curvature))
    #Save curvature in the original df at indexes in column Curv
    df['Curv'] = np.nan
    j = 0
    for i in idx:
        #Curvature is shorter due to following values reuired for computation
        if j == len(curvature)-1:
            break
        df.loc[i,'Curv'] = curvature[j]
        j += 1
    
    #Fill missing values of curvature with linear interp
    df['Curv'].interpolate(method='linear', inplace=True)
    '''#Lowpass filter curvature signal
    cutoff = 1
    fs = 5
    order = 2
    df['Curv'] = butter_lowpass_filter(df['Curv'], cutoff, fs, order)
    '''
    #lfilter scipy
    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    df['Curv'] = lfilter(b, a, df['Curv'])
    #return L, curvature, k

def circumcenter (B, A, C):
    """
    Calculates curvature vectors and radius of route from three points in xy plane
    :param B: coordinates (x,y) of current point
    :param A: coordinates (x,y) of point before
    :param C: coordinates (x,y) of point after
    :return: R_adius of each section and components of k_urvature vector
    """
    A = A.T
    B = B.T
    C = C.T

    D = np.cross(B-A, C-A)
    b = np.linalg.norm(A-C)
    c = np.linalg.norm(A-B)
    E = np.cross(D, B-A)
    F = np.cross(D, C-A)
    G = (b**2*E-c**2*F)/np.linalg.norm(D)**2/2
    R = np.linalg.norm(G)

    if R == 0:
        k = G
    else:
        k = np.transpose(G)/R**2

    return R, k


def load_log(path):
    # Given forlder path of the logs, check files in directory and load csv
    # After loading the logs check if it is a real log or comes from OpenLap simulation
    # If it is a real log, uses lapDetector function to compute laps and export the best one
    # If it is an OpenLap log, export directly since it is just one single lap

    #Check the OS, Unix uses '/' between folders, Windows uses '\'
    if platform.system() == "Windows":
        append = "\\"
    else:
        append = "/"

    #Open the directories
    dir = os.listdir(path)
    #Check every file inside the directory
    for file in dir:
        #If a file has the csv extension 
        if os.path.splitext(file)[1] == ".csv":
            # Helper variable to remeber filename of the log and later check if coming from OpenLap
            filename = file
            tel = pd.read_csv(path + append + file, low_memory = False)

     ### SINGLE LAPS SEGMENTATION ###
    
    #Check if is an OpenLap log
    if 'OpenLap' in filename:
        tel['Abscissa'] = tel.index
        lap_time = round(tel['time'][len(tel['time'])-1], 3)
        print('\nLAP SIMULATED FROM MODEL')
        print('\nNUMBER OF LAPS IN THE SESSION:')
        print(1)
        print('\nBEST LAP TIME:')
        print(lap_time)
        return tel
    else:
        #Get df divided for every lap from total session
        df_laps, laptimes, best_lap_idx = lapsDetector(tel)
        #Saving just the two single laps to be compared
        tel = df_laps[f'lap{best_lap_idx}']
        return tel, df_laps
