import math
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})
import scipy as sc 
import numpy as np
from numpy.linalg import lstsq


""" Droop Function DE """
def FCR_DK_POP(freq_in,SOC_init,battery_max_cap,max_charge, charging_max):
    #POP specific linear functions
    POP0_droop_pos_coords = [(49.8,1),(49.98,0)]
    POP0_droop_neg_coords = [(50.02,0),(50.2,-1)]
    
    POP2_droop_pos_coords = [(49.8,0.5),(49.98,-0.5)]
    POP2_droop_neg_coords = [(50.02,-0.5),(50.1,-1)]
    
    POP1_droop_pos_coords = [(49.9,1),(49.98,0.5)]
    POP1_droop_neg_coords = [(50.02,0.5),(50.2,-0.5)]    
    
    power_ratio = []
  
    """ SOC variables """
    SOC = []
    
    power_kWs = [SOC_init * battery_max_cap * 3600] #63000
    power_kWs_app = [] # emppty list needed for more accuracy
    SOC_max = battery_max_cap * 3600
    
    
    #Never allowing charge/discharge to reach 100%
    ideal_cap_max = (battery_max_cap*max_charge) * 3600
    ideal_cap_min = (battery_max_cap - battery_max_cap*max_charge) * 3600 
     
    
    """ POP thresholds and Initialization  """
    POP_min = 0.2
    POP_max = 0.8    
    
    POP_min_threshold = POP_min + 0.01
    POP_max_threshold = POP_max - 0.01
    
    if SOC_init >= POP_max:
        POP = 1
    elif SOC_init <= POP_min:
        POP = 2
    else:
        POP = 0
        
    
        
    #Internal counters
    POP0_counter = 0
    POP1_counter = 0
    POP2_counter = 0
    idx_counter = 0
    max_counter = 0
    min_counter = 0
    
    
    first_idx = freq_in.first_valid_index()
    last_idx = freq_in.last_valid_index()
    
    
    for i in range(first_idx,last_idx+1):    
        if POP == 0:
            if freq_in[i] <= 49.8:
                POP0_counter += 1
                idx_counter += 1
                power_ratio.append(1)
                if i == first_idx:
                    power_kWs_app.append(power_kWs[0] + power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
                
                if power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                    POP = 2
                    if power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1]  = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                        
            elif freq_in[i] >= 49.98 and freq_in[i] <=50.02:
                POP0_counter += 1
                idx_counter += 1
                power_ratio.append(0)
                if i == first_idx:
                    power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
                
                if power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                    POP = 2
                    if power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1]  = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
            elif freq_in[i] >= 50.2:  
                POP0_counter += 1
                idx_counter += 1
                power_ratio.append(-1)
                if i == first_idx:
                    power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
                if power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                    POP = 2
                    if power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1]  = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
            elif freq_in[i] > 49.8 and freq_in[i] < 49.98:
                POP0_counter += 1
                idx_counter += 1
                x_coords, y_coords = zip(*POP0_droop_pos_coords)
                A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords)[0]
                power_ratio.append(m*freq_in[i] + c)
                if i == first_idx:
                    power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
               
                if power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                    POP = 2
                    if power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1]  = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
            else:  #freq_in[i] > 50.02 and freq_in[i] < 50.2:
                 POP0_counter += 1
                 idx_counter += 1
                 x_coords, y_coords = zip(*POP0_droop_neg_coords)
                 A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                 m, c = lstsq(A, y_coords)[0]
                 power_ratio.append(m*freq_in[i] + c)
                 if i == first_idx:
                    power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                 else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                                                                                  
                 if power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                 elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                    POP = 2
                    if power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1]  = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                 SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
    
    
        elif POP == 1:
            if freq_in[i] <= 49.9:
                POP1_counter += 1
                idx_counter += 1
                power_ratio.append(1) 
                if i == first_idx:
                   power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                   power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                                                                                  
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                     POP = 2
                     if power_kWs_app[idx_counter-1] < ideal_cap_min:
                         power_kWs_app[idx_counter-1]  = ideal_cap_min
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         min_counter += 1
                elif power_kWs_app[idx_counter-1] > ideal_cap_max:
                         power_kWs_app[idx_counter-1] = ideal_cap_max
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         max_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
                        
            elif freq_in[i] >= 49.98 and freq_in[i] <=50.02:
                POP1_counter += 1
                idx_counter += 1
                power_ratio.append(0.5)
                if i == first_idx:
                   power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                   power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                     POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                     POP = 2
                     if power_kWs_app[idx_counter-1] < ideal_cap_min:
                         power_kWs_app[idx_counter-1]  = ideal_cap_min
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         min_counter += 1
                elif power_kWs_app[idx_counter-1] > ideal_cap_max:
                         power_kWs_app[idx_counter-1] = ideal_cap_max
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         max_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
                    
            elif freq_in[i] >= 50.2:  
                POP1_counter += 1
                idx_counter += 1
                power_ratio.append(-0.5)
                if i == first_idx:
                   power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                   power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                     POP = 2
                     if power_kWs_app[idx_counter-1] < ideal_cap_min:
                         power_kWs_app[idx_counter-1]  = ideal_cap_min
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         min_counter += 1
                elif power_kWs_app[idx_counter-1] > ideal_cap_max:
                         power_kWs_app[idx_counter-1] = ideal_cap_max
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         max_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
                    
            elif freq_in[i] > 49.9 and freq_in[i] < 49.98:
                POP1_counter += 1
                idx_counter += 1
                x_coords, y_coords = zip(*POP1_droop_pos_coords)
                A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords)[0]
                power_ratio.append(m*freq_in[i] + c)
                if i == first_idx:
                   power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                   power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                    POP = 2
                    if power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1]  = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                elif power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1] = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                
                
            else: #freq_in[i] > 50.02 and freq_in[i] < 50.2:
                 POP1_counter += 1
                 idx_counter += 1
                 x_coords, y_coords = zip(*POP1_droop_neg_coords)
                 A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                 m, c = lstsq(A, y_coords)[0]
                 power_ratio.append(m*freq_in[i] + c)
                 if i == first_idx:
                     power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                 else:
                     power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                 if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                 elif power_kWs_app[idx_counter-1]/SOC_max <= POP_min:
                     POP = 2
                     if power_kWs_app[idx_counter-1] < ideal_cap_min:
                         power_kWs_app[idx_counter-1]  = ideal_cap_min
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         min_counter += 1
                 elif power_kWs_app[idx_counter-1] > ideal_cap_max:
                         power_kWs_app[idx_counter-1] = ideal_cap_max
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         max_counter += 1
                 SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                                                                             
                                                                                                     
                                                         
        elif POP == 2:
            if freq_in[i] <= 49.8:
                POP2_counter += 1
                idx_counter += 1
                power_ratio.append(0.5)
                if i == first_idx:
                     power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1] = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                        
                
            elif freq_in[i] >= 49.98 and freq_in[i] <=50.02:
                POP2_counter += 1
                idx_counter += 1
                power_ratio.append(-0.5)
                if i == first_idx:
                     power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1] = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                    
            elif freq_in[i] >= 50.1:  
                POP2_counter += 1
                idx_counter += 1
                power_ratio.append(-1)
                if i == first_idx:
                     power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1] = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                    
            elif freq_in[i] > 49.8 and freq_in[i] < 49.98:
                POP2_counter += 1
                idx_counter += 1
                x_coords, y_coords = zip(*POP2_droop_pos_coords)
                A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords)[0]
                power_ratio.append(m*freq_in[i] + c)
                if i == first_idx:
                     power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                
                if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                elif power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                    POP = 1
                    if power_kWs_app[idx_counter-1] > ideal_cap_max:
                        power_kWs_app[idx_counter-1]  = ideal_cap_max
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        max_counter += 1
                elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                        power_kWs_app[idx_counter-1] = ideal_cap_min
                        power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                        min_counter += 1
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                    
            else:  #freq_in[i] > 50.02 and freq_in[i] < 50.1:
                 POP2_counter += 1
                 idx_counter += 1
                 x_coords, y_coords = zip(*POP2_droop_neg_coords)
                 A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                 m, c = lstsq(A, y_coords)[0]
                 power_ratio.append(m*freq_in[i] + c)
                 if i == first_idx:
                      power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                 else:
                     power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                 if power_kWs_app[idx_counter-1]/SOC_max < POP_max_threshold and power_kWs_app[idx_counter-1]/SOC_max > POP_min_threshold:
                    POP = 0
                 elif power_kWs_app[idx_counter-1]/SOC_max >= POP_max:
                     POP = 1
                     if power_kWs_app[idx_counter-1] > ideal_cap_max:
                         power_kWs_app[idx_counter-1]  = ideal_cap_max
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         max_counter += 1
                 elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                         power_kWs_app[idx_counter-1] = ideal_cap_min
                         power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         min_counter += 1
                 SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
    
    
    return power_ratio , power_kWs_app, SOC , POP0_counter, POP1_counter , POP2_counter, max_counter, min_counter












