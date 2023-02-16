import pandas as pd
import os 
import matplotlib.pyplot as plt
import platform
import math
import numpy as np
import preprocessing as p
from scipy.signal import lfilter
import statistics as st
import seaborn as sns

#   ---------- CONSTANT DECLARATIONS ----------
#Threshold to calculate mean and standard deviaton for Brake
THRESHOLD = 1
#Value of the interval where the function checks if there's no brake
BRAKE_INTERVAL_OFFSET = 50

TPS_THRESHOLD = 90

#   ---------- FUNCTIONS ----------
#Function to check when the bike is out the pit: done by checking suspensions
def starting_index(dataframe):
    for i in range(len(dataframe)):
        #Check all the susp. values, when front is 50 or more and post is 10 or more I assume bike is out the pit
        if(dataframe["PotAnt"][i] >= 50 and dataframe["PotPost"][i] >= 10):
            #Return index of the row 
            return i

#Function to check when the bike is no more out the pit: done by checking suspensions
def ending_index(dataframe, index):
    #Starting from the end of the dataframe, going backwards
    for i in range(len(dataframe)-1, index, -1):
        #Check all the susp. values, when front is 50 or more and post is 10 or more I assume bike is out the pit
        if(dataframe["PotAnt"][i] >= 50 and dataframe["PotPost"][i] >= 10):
            #Return index of the row
            return i+1

#Function to calculate mean and standard deviation on the Brake
#Modify THRESHOLD values to increase/decrease roughness of the filter
def brake_mean_std(dataframe, start, end, threshold):
    sum = 0
    count = 0
    for i in range(start, end):
        #We want to calculate mean and std only on the values that are not zero or noise
        if dataframe["Brake"][i] >= threshold:
            sum += dataframe["Brake"][i]
            count += 1
    #Mean calc.
    mean = sum/count
    sum = 0 
    for i in range(start, end):
        if dataframe["Brake"][i] >= threshold:
            sum += pow(dataframe["Brake"][i] - mean, 2)
    #Standard deviation calc.
    std = math.sqrt(sum/count)
    return mean, std

#Function to apply on the brake signal: all the values below mean - std. deviation are setted to 0
def brake_filter(x, mean, std):
    if(x < mean-std or x > mean+ 4*std):
        return 0
    else:
        return x

#Function to convert string csv values of time in integer values
def time_convert(val):
    h, m, s_ms = val.split(':')
    s, ms = s_ms.split('.')
    return int(h) * 3600 * 1000 + int(m) * 60 * 1000 + int(s) * 1000 + int(ms)

#Function to check if in a specific interval there's no brake
def no_value_interval(dataframe, col, index, offset, max_offset):
    #Calc. of the range's end
    end_range = index + offset
    #Check if the value calculated is admittable, else set it to the end of the dataframe
    if end_range > max_offset:
        end_range = max_offset
    for i in range(index, end_range):
        if dataframe[col][i] > 0:
            return False
    #True if only brake was always at 0
    return True

#Function to analyze a Brake channel, return the dictionary with all the info
def analyze_brake(start_index, end_index, dataframe):
    braking = 0
    count = 0
    brake_zones = {}
    peak_pressure = 0
    tps_zero = None
    brake_latitude = ""
    brake_longitude = ""
    delay_tps_brake = 0
    for i in range(start_index, end_index):
        #Start braking
        if(dataframe["Brake"][i] > 0 and braking == 0):
            braking = 1
            #Count the braking zone
            count += 1
            #Save latitude, longitude of the brake zone and the braking start time
            brake_start_time = dataframe["time"][i]
            brake_latitude = dataframe["Latitude"][i]
            brake_longitude = dataframe["Longitude"][i]
            brake_abscissa = dataframe["Abscissa"][i]
            #Set preak pressure to the first val
            peak_pressure = dataframe["Brake"][i]
            #Calc. of the delay tps ==> brake
            if tps_zero != None:
                if isinstance(dataframe['time'][0], float):
                    delay_tps_brake = brake_start_time - tps_zero
                else:
                    delay_tps_brake = time_convert(brake_start_time) - time_convert(tps_zero)
        #Check if i've stopped braking and there will be a no brake zone for a while
        #(done because the filter set some brake values to 0 in the braking zone)
        elif(dataframe["Brake"][i] == 0 and braking == 1 and no_value_interval(dataframe, "Brake", i, BRAKE_INTERVAL_OFFSET, end_index)):
            braking = 0
            #Save the braking end time
            brake_end_time = dataframe["time"][i]
            #Calc. of the braking zone duration
            if isinstance(dataframe['time'][0], float):
                brake_time = brake_end_time - brake_start_time
            else:
                brake_time = time_convert(brake_end_time) - time_convert(brake_start_time)
            #Add values to the dictionary
            brake_zones["Brake_" + str(count)] = [brake_time, brake_latitude, brake_longitude, brake_abscissa, peak_pressure, delay_tps_brake]
        #Check for greater values of pressure
        elif(dataframe["Brake"][i] > peak_pressure and braking == 1):
            #Update peak pressure val
            peak_pressure = dataframe["Brake"][i]
        #Check when tps goes to 0 val.
        elif(dataframe["tps"][i] <= 0.5 and braking == 0):
            tps_zero = dataframe["time"][i]
    return brake_zones

def tps_over_threshold_perc(dataframe, start, end):
    count = 0
    for i in range(start, end):
        if dataframe["tps"][i] > TPS_THRESHOLD:
            count += 1
    return count/(end-start)*100.0

def avg_delta_null_full(dataframe, start, end):
    started = False
    time_one = 0
    deltas = []
    for i in range(start, end):
        if(dataframe["tps"][i] < 5 and not started):
            time_one = time_convert(dataframe["time"][i])
            started = True
        elif(dataframe["tps"][i] > 95 and started):
            deltas.append(time_convert(dataframe["time"][i]) - time_one)
            started = False
    sum = 0
    for el in deltas:
        sum += el
    return sum/len(deltas)

def analyze_tps(dataframe, start, end):
    res = tps_over_threshold_perc(dataframe, start, end)
    print("Percentage of TPS over " + str(TPS_THRESHOLD) + ": " + str(round(res, 2)) + "%")
    res = avg_delta_null_full(dataframe, start, end)
    print("Delta of tps 10/90: " + str(res) + " ms")

def plot_tps_80_100(target_t, comp_t):
    temp = target_t.plot(x="Abscissa", y="tps", ylim=[95, 100], label="Target")
    comp_t.plot(ax=temp, x="Abscissa", y="tps", ylim=[95, 100], label="Rookie")
    plt.grid()
    plt.show()

def plot_tps_slowest_turn(target_t, comp_t):
    delta = 0
    delta_max = 0
    start_max = 0
    end_max = 0
    started = False
    for i in range(0, len(comp_t)-1):
        if(comp_t["tps"][i] < 5 and not started):
            time_one = time_convert(comp_t["time"][i])
            start = comp_t["Abscissa"][i]
            started = True
        elif(comp_t["tps"][i] > 95 and started):
            delta = time_convert(comp_t["time"][i]) - time_one
            if delta > delta_max:
                delta_max = delta
                start_max = start
                end_max = comp_t["Abscissa"][i]
            started = False
    temp = comp_t.plot(x="Abscissa", y="tps", xlim=[start_max, end_max], label="Rookie")
    target_t.plot(ax=temp, x="Abscissa", y="tps", xlim=[start_max, end_max], label="Target")
    plt.grid()
    plt.show()

#Function to print all the values stored in the dictionary produced by analisys
def print_brake_dict(dictionary):
    #Foreach key in the dictionary print the values
    for k in dictionary:
        print(k)
        print("Brake time: " + str(dictionary[k][0]) + " ms")
        print("Latitude: " + str(dictionary[k][1]))
        print("Longitude: " + str(dictionary[k][2]))
        print("Abscissa: " + str(dictionary[k][3]) + " m")
        print("Peak pressure: " + str(dictionary[k][4]) + " bar")
        print("Dealy TPS-Brake: " + str(dictionary[k][5]) + " ms\n")

def filter(df, channel, n=3):
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    df[channel] = lfilter(b, a, df[channel])

#Function to create and plot the main page of the telemetry analisys
def telemetry_main_graph(tel1, tel2, channel, y_label, unit):
    figure, axes = plt.subplots(3)
    tel1.plot(ax=axes[0], x="Abscissa", y=channel, ylabel=y_label, label=unit, xlabel="Abscissa")
    axes[0].grid()
    axes[0].set_ylim([0, 17.5])
    tel2.plot(ax=axes[1], x="Abscissa", y=channel, ylabel=y_label, label=unit, xlabel="Abscissa")
    # xlabel="Time (h:m:s:ms)"
    axes[1].grid()
    '''axes[1].set_ylim([0, 17.5])
    axes[2] = plt.plot( tel1["time"], tel1[channel]-tel2[channel])
    axes[2].grid()'''
    axes[2].set_ylim([0, 5])
    plt.show()

#Function to print the values stored in two dictionaries compared
def print_brake_dict_comparison(dic1, dic2):
    #Check for the smallest dictionary
    minDic = {}
    if len(dic1) <= len(dic2):
        minDic = dic1
    else:
        minDic = dic2
    #Foreach key in the smallest dictionary print the values of both (they use the same keys)
    for k in minDic:
        print("\n\t-------" + k + "-------")
        print("Brake time: " + str(dic1[k][0]) + " ms - " + str(dic2[k][0]) + " ms")
        #print("Latitude: " + str(dic1[k][1]) + " - " + str(dic2[k][1]))
        #print("Longitude: " + str(dic1[k][2]) + " - " + str(dic2[k][2]))
        print("Abscissa: " + str(dic1[k][3]) + " m - " + str(dic2[k][3]) + " m")
        print("Peak pressure: " + str(dic1[k][4]) + " bar - " + str(dic2[k][4]) + " bar")
        print("Dealy TPS-Brake: " + str(dic1[k][5]) + " ms - " + str(dic2[k][5]) + " ms")

def slipEstimation(df):
    ''' Estimation of the slip coefficient using V_GPS and hall sensor measurement in the rear wheel

        INPUT  df that contain at least two columns: V_GPS and VEHICLE_VehSpeedRear

        OUTPUT same df with slipCoeff colums 
    '''

    #Slip estimation for every sampled point in the log with standard formulation 
    #slipCoeff = (vlong-wR)/vcenter
    #It assumes that the bike is always proceeding in plane, so that vlong = vcenter
    df['slipCoeff'] = ( df['V_GPS'] - df['VEHICLE_VehSpeedRear'] ) / df['V_GPS']

    figure, axes = plt.subplots(2)
    df.plot(ax=axes[0], x='Abscissa', y='slipCoeff')
    axes[0].grid()
    df.plot(ax=axes[1], x="Abscissa", y='tps')
    # xlabel="Time (h:m:s:ms)"
    axes[1].grid()
    plt.show()



#   ---------- SCRIPT'S MAIN PART ----------
def telemetry_compare(target_tel, comp_tel):

    #Calc. the index of start and end of target and comparison telemetry
    target_start_index = 0 #starting_index(target_tel)
    comp_start_index = 0 #starting_index(comp_tel)
    target_end_index = len(target_tel.index)-1 #ending_index(target_tel, target_start_index)
    comp_end_index = len(comp_tel.index)-1 #ending_index(comp_tel, comp_start_index)


    ### BRAKING ###

    plt.plot(target_tel['Curv'])

    #Calculate mean and standard deviation of both Brake values 
    target_brake_mean, target_brake_std = brake_mean_std(target_tel, target_start_index, target_end_index, THRESHOLD)
    comp_brake_mean, comp_brake_std = brake_mean_std(comp_tel, comp_start_index, comp_end_index, THRESHOLD)

    #   ---------- DEBUG INSTRUCTIONS ----------
    #Prepare pre-filter brake graph
    #temp = comp_tel.plot(x="time", y="Brake", label="Pre-filter brake")

    #Apply filter for noise on the Brake channels
    target_tel["Brake"] = target_tel["Brake"].apply(brake_filter, args=(target_brake_mean, target_brake_std))
    comp_tel["Brake"] = comp_tel["Brake"].apply(brake_filter, args=(comp_brake_mean, comp_brake_std))

    #Analyze the two telemetries
    target_brake_analyzed = analyze_brake(target_start_index, target_end_index, target_tel)
    comp_brake_analyzed = analyze_brake(comp_start_index, comp_end_index, comp_tel)

    #Print the dictionary values
    #print_brake_dict(target_brake_analyzed)
    #print_brake_dict(comp_brake_analyzed)
    print_brake_dict_comparison(target_brake_analyzed, comp_brake_analyzed)

    #Create the graph and show
    telemetry_main_graph(target_tel, comp_tel, "Brake", "BRAKE", "bar")

    #   ---------- DEBUG INSTRUCTIONS ----------
    #Post-filter brake graph
    #comp_tel.plot(ax=temp, x="time", y="Brake", label="Post-filter brake")

    #Other graphs useful during the debug
    #fig1, axes1 = plt.subplots(3)
    #target_tel.plot(ax=axes1[2], x="time", y="tps", xlabel="", ylabel="Target TPS", label="%")
    temp = comp_tel.plot(x="Abscissa", y="Brake", label="bar")
    comp_tel.plot(ax=temp, x="Abscissa", y="tps", xlabel="", ylabel="Rookie TPS", label="%")
    plt.grid()
    plt.show()
    #comp_brake = comp_tel.plot(ax=axes1[2], x="time", y="Brake", ylabel="Rookie BRAKE", label="bar")
    #target_tel.plot(x="time", y="tps")

    figure, axes = plt.subplots(2)
    comp_tel.plot(ax=axes[0], x='Abscissa', y='Brake')
    axes[0].grid()
    comp_tel.plot(ax=axes[1], x="Abscissa", y='Curv')
    # xlabel="Time (h:m:s:ms)"
    axes[1].grid()
    plt.show()

    ### TPS ###

    print("\n\n--- TPS ANALYSIS ---\n\n")
    print("\n\nTARGET telemetry:\n")
    analyze_tps(target_tel, target_start_index, target_end_index)
    print("\n\nROOKIE telemetry:\n")
    analyze_tps(comp_tel, comp_start_index, comp_end_index)
    plot_tps_80_100(target_tel, comp_tel)
    plot_tps_slowest_turn(target_tel, comp_tel)

    ### TIRES ###

    #Slip estimation
    '''    plt.plot(target_tel['theta'])
    plt.show()
    plt.figure()
    plt.scatter(range(len(target_tel['Gy'])),target_tel['Gy'])
    plt.plot(range(len(target_tel['Gy'])),target_tel['thetaDot'], 'r--')
    plt.show()
    slipEstimation(target_tel)
    slipEstimation(comp_tel)
    '''

    ### ENGINE ###

    #Create the graph and show Lambda signal
    #telemetry_main_graph(target_tel, comp_tel, "afr1_old", "LAMBDA", "bar")

    # LISTA SENSORI:
    #
    # Brake     ==> freno
    # tps       ==> farfalla
    # rpm       ==> giri/min motore
    # PotAnt    ==> sospensione anteriore
    # PotPost   ==> sospensione posteriore

    # APPUNTI:
    #
    #   X I LOG DI EDO:
    # Tra una riga e l'altra del dataframe passano 21 ms
    # 2 minuti di log ==> 5714 righe circa
    #
    #   X I LOG DI CESARE:
    # Tra una riga e l'altra del dataframe passano 6 ms
    # 2 minuti di log ==> 20000 righe circa

    # PROBLEMATICHE:
    #
    # 1) Una telemetria logga ogni 21 ms, mentre l'altra 6 ==> impossibile fare la differenza punto-punto con tempi diversi di log
    # 2) I tempi di delay tra freno e acceleratore sono piccolissimi, difficili da calcolare (infatti solitamente mi da sempre o
    #       6 o 21 ms come valore, ma è il delay tra un log e l'altro) ==> se > 1 s togli (filtra)
    # 3) Non avendo telemetrie allineate sul giro momentaneamente la funzione per visualizzare le analisi affiancate di
    #       entrambe le telemetrie si ferma a quella con meno entries


def compute_consist(df_laps):
    #Dictionary to store braking info for every lap
    brakings = dict.fromkeys(df_laps.keys())
    #Iterate over all laps to analyze brakings of every lap
    for i in df_laps.keys():
        #Calc. the index of start and end of telemetry
        start_index = 0 
        end_index = len(df_laps[i].index)-1
        #Calculate mean and standard deviation of both Brake values 
        brake_mean, brake_std = brake_mean_std(df_laps[i], start_index, end_index, THRESHOLD)
        #Apply filter for noise on the Brake channels
        df_laps[i]["Brake"] = df_laps[i]["Brake"].apply(brake_filter, args=(brake_mean, brake_std))
        #Analyze the telemetry
        brakings[i] = analyze_brake(start_index, end_index, df_laps[i])
    #Store the average and std for every brake zone
    #For every braking zone metrics will contain mean values and standar deviation
    metrics = dict.fromkeys(brakings[i].keys())
    #Save all values to create box plots for every braking
    values = dict.fromkeys(brakings[i].keys())
    #Iterate over all braking zones to compare the maneuver among all laps
    for i in metrics.keys():
        #Store how many braking zones found to divide for getting the avergage
        #Sometimes one more zone is identified so need to check in how many laps the corresponding braking is identified
        n_braking_zones = 0
        #Preallocate mean and std dict for every braking zone
        metrics[i] = dict.fromkeys(['Mean', 'Std'])
        metrics[i]['Mean'] = np.zeros(len(brakings['lap1'][i]))
        #Preallocate t_brake, abscissa, peak pressure and delay for every braking zone
        values[i] = dict.fromkeys(['t_brake', 'abscissa', 'pressure', 'delay'])
        values[i]['t_brake']    = []
        values[i]['abscissa']   = []
        values[i]['pressure']   = []
        values[i]['delay']      = []
        #Temporary vars for stdev computation
        std_t        = []
        std_abscissa = []
        std_pressure = []
        std_delay    = []
        #Iterate over all laps to get the corresponding braking 
        for j in brakings.keys():
            #Check if the braking zone is present in the lap, otherwise proceed
            try:
                metrics[i]['Mean'] = np.add(np.asarray(metrics[i]['Mean']), np.asarray(brakings[j][i]))
                std_t.append(brakings[j][i][0])
                std_abscissa.append(brakings[j][i][3])
                std_pressure.append(brakings[j][i][4])                
                std_delay.append(brakings[j][i][5])
                n_braking_zones += 1
                values[i]['t_brake'].append(brakings[j][i][0])
                values[i]['abscissa'].append(brakings[j][i][3])
                values[i]['pressure'].append(brakings[j][i][4])  
                values[i]['delay'].append(brakings[j][i][5])
            except:
                pass
        #Divide by the total number of braking zones to get the average
        metrics[i]['Mean'] = metrics[i]['Mean'] / n_braking_zones
        #Computing std from temporary variables and saving
        metrics[i]['Std'] = [
            st.stdev(std_t), 
            0, #Std of gps coordinates is not computed and set to zero
            0, 
            st.stdev(std_abscissa),
            st.stdev(std_pressure),
            st.stdev(std_delay)]
        
    return metrics, values

def plot_consist(values, metrics, type='Line'):
    #Saving data of t_brake, abscissa and peak pressure for every braking zone 
    fig, (ax1, ax2) = plt.subplots(2)
    df_val = pd.DataFrame(values)

    #Based on string type plot lines or boxplot
    if type == 'Box':
        sns.boxplot(data=df_val.loc['pressure',:], ax=ax1)
        ax1.grid()
        plt.xlabel('Braking zone')
        plt.ylabel('Peak pressure standard deviation [bar]')
        sns.boxplot(data=df_val.loc['abscissa',:], ax=ax2)
        ax2.grid()
        plt.xlabel('Braking zone')
        plt.ylabel('Abscissa standard deviation [m]')
        plt.show()
    if type == 'Line':
        #Saving in a list variance of peak pressure of the braking point
        var_pressure = [metrics[i]['Std'][4] for i in metrics.keys()]
        ax1.grid()
        ax1.plot(var_pressure)
        ax1.set(xlabel='Braking zone', ylabel='Peak pressure stdev [bar]')
        #Saving in a list variance of abscissa of the braking point
        var_abscissa = [metrics[i]['Std'][3] for i in metrics.keys()]
        ax2.grid()
        ax2.plot(var_abscissa)
        ax2.set(xlabel='Braking zone', ylabel='Abscissa stdev [m]', ylim=[0, 100])
        plt.show()

    '''x = [i for i in range(1,len(values.keys())+1)]
    data_t = []
    data_abscissa = []
    data_pressure = []
    for i in values.keys():
        data_t.append(values[i]['t_brake'])
        data_abscissa.append(values[i]['abscissa'])
        data_pressure.append(values[i]['pressure'])

    #Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Statistics of braking zones')
    ax1.boxplot(x, data_t)
    ax2.boxplot(x, data_abscissa)
    ax3.boxplot(x, data_pressure)'''
    
            



