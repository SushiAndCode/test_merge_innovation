import pandas as pd
import os 
import matplotlib.pyplot as plt
import platform
import math
import preprocessing as p
import telemetry_analysis as a


def main():
    ### IMPORT FILES ###

    #Folders paths


    # SIMONE BRUNELLO
    #target_tel_directory_string = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/test_telemetrie/CREMONA_23-10-2021_Cesare/Turno 3 run 2 - 12.50"
    #comparison_tel_directory_string = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/test_telemetrie/CREMONA_23-10-2021_Cesare/Turno 1 - 10.05"

    # SIMONE DEIDIER
    #target_tel_directory_string = "/Users/simonedeidier/Desktop/Varie/Politecnico di Milano/PoliMi Motorcycle Factory/Innovation/2021-10-23_CREMONA_CESARE/Turno 3 run 2 - 12.50"
    #comparison_tel_directory_string = "/Users/simonedeidier/Desktop/Varie/Politecnico di Milano/PoliMi Motorcycle Factory/Innovation/2021-10-23_CREMONA_CESARE/Turno 1 - 10.05"
    
    # LEO
    target_tel_directory_string = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/test_telemetrie/CREMONA_23-10-2021_Cesare/Turno 3 run 2 - 12.50"
    comparison_tel_directory_string = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/test_telemetrie/CREMONA_23-10-2021_Cesare/Turno 1 - 10.05"
    # Directory for Open lap log 
    #target_tel_directory_string = "C:/Users/leona/Documents/PMF/21-23/Dynamics/Repo/OpenLap_PMF"
    
    # Loading laps dataframe
    print('\n\t-----TARGET DATA-----')
    target_tel, target_df_laps = p.load_log(target_tel_directory_string)
    print('\n\t-----COMPARISON DATA-----')
    comp_tel, comp_df_laps = p.load_log(comparison_tel_directory_string)

    # Consistency along the session
    target_consist_metrics, target_consist_values = a.compute_consist(target_df_laps)
    #comp_consist_metrics = a.compute_consist(comp_df_laps)

    #Plotting consistency analysis results
    a.plot_consist(target_consist_values, target_consist_metrics)

    # Best lap copmarison
    a.telemetry_compare(target_tel, comp_tel)


if __name__ == "__main__":
    main()
