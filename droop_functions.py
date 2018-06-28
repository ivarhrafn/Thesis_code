
import numpy as np
from numpy.linalg import lstsq
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('darkgrid')
plt.style.use('seaborn-deep')

#Droop Function DE
def FCR_DE(freq_in,SOC_init,battery_max_cap,max_charge, charging_max):
    #POP specific linear functions
    POP0_droop_pos_coords = [(49.8,1),(49.99,0)]
    POP0_droop_neg_coords = [(50.01,0),(50.2,-1)]

    power_ratio = []
   
    """ SOC variables """
    SOC = []
    
    
    power_kWs = [SOC_init * battery_max_cap * 3600] #63000
    power_kWs_app = [] # emppty list needed for more accuracy
    SOC_max = battery_max_cap * 3600
    
    
    #Setting maximum and minimum allowed charge
    ideal_cap_max = (battery_max_cap*max_charge) * 3600
    ideal_cap_min = (battery_max_cap - battery_max_cap*max_charge) * 3600 


    #Internal counters
    idx_counter = 0
    max_counter = 0    #Counts how often the EV is fully charged and discharged
    min_counter = 0
    
    first_idx = freq_in.first_valid_index()     
    last_idx = freq_in.last_valid_index()
    
       
    for i in range(first_idx,last_idx+1):

        if freq_in[i] <= 49.8:
            idx_counter += 1
            power_ratio.append(1)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0] + power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
            
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)

                
        elif freq_in[i] >= 49.99 and freq_in[i] <=50.01:
            idx_counter += 1
            power_ratio.append(0)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max * -1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
            
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)


        elif freq_in[i] >= 50.2:  
            idx_counter += 1
            power_ratio.append(-1)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)

        elif freq_in[i] > 49.8 and freq_in[i] < 49.99:
            idx_counter += 1
            x_coords, y_coords = zip(*POP0_droop_pos_coords)
            A = np.vstack([x_coords,np.ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]
            power_ratio.append(m*freq_in[i] + c)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
           
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
     
        else:  #freq_in[i] > 50.01 and freq_in[i] < 50.2:
             idx_counter += 1
             x_coords, y_coords = zip(*POP0_droop_neg_coords)
             A = np.vstack([x_coords,np.ones(len(x_coords))]).T
             m, c = lstsq(A, y_coords)[0]
             power_ratio.append(m*freq_in[i] + c)
             if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
             else:
                power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                                                                              
             if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
             elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
             SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
             
    return power_ratio , power_kWs_app, SOC , max_counter, min_counter



""" Droop Function DK """
def FCR_DK(freq_in,SOC_init,battery_max_cap,max_charge, charging_max):
    #POP specific linear functions
    POP0_droop_pos_coords = [(49.8,1),(49.98,0)]
    POP0_droop_neg_coords = [(50.02,0),(50.2,-1)]
    
  
    
    power_ratio = []
    #POP = 0
    
    """ SOC variables """
    SOC = []
    
    power_kWs = [SOC_init * battery_max_cap * 3600] #63000
    power_kWs_app = [] # emppty list needed for more accuracy
    SOC_max = battery_max_cap * 3600
    
    
    #Never allowing charge/discharge to reach beyond 100%
    ideal_cap_max = (battery_max_cap*max_charge) * 3600
    ideal_cap_min = (battery_max_cap - battery_max_cap*max_charge) * 3600 


    #Internal counters
    POP0_counter = 0
    idx_counter = 0
    max_counter = 0
    min_counter = 0
    
    
    first_idx = freq_in.first_valid_index()
    last_idx = freq_in.last_valid_index()
    


    for i in range(first_idx,last_idx+1):    
        if freq_in[i] <= 49.8:
            POP0_counter += 1
            idx_counter += 1
            power_ratio.append(1)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0] + power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
                
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
                    
        elif freq_in[i] >= 49.98 and freq_in[i] <=50.02:
            POP0_counter += 1
            idx_counter += 1
            power_ratio.append(0)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
            
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
        
        elif freq_in[i] >= 50.2:  
            POP0_counter += 1
            idx_counter += 1
            power_ratio.append(-1)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
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
           
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
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
                                                                              
             if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
             elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
             SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                
    return power_ratio , power_kWs_app, SOC, max_counter, min_counter 




""" Droop Function DE """
def FCR_D(freq_in,SOC_init,battery_max_cap,max_charge, charging_max):
    #POP specific linear functions
    POP0_droop_pos_coords = [(49.5,1),(49.9,0)]
    
   
    
    power_ratio = []
    #POP = 0
    
    """ SOC variables """
    SOC = []
 
    power_kWs = [SOC_init * battery_max_cap * 3600] #63000
    power_kWs_app = [] # emppty list needed for more accuracy
    SOC_max = battery_max_cap * 3600
    
    
    #Never allowing charge/discharge to reach 100%
    ideal_cap_max = (battery_max_cap*max_charge) * 3600
    ideal_cap_min = (battery_max_cap - battery_max_cap*max_charge) * 3600 


    #Internal counters
    POP0_counter = 0
    idx_counter = 0
    max_counter = 0
    min_counter = 0
    
    first_idx = freq_in.first_valid_index()
    last_idx = freq_in.last_valid_index()

#    first_idx = 0
#    for i in range(len(freq_in)):
    for i in range(first_idx,last_idx+1):        
        if freq_in[i] <= 49.5:
            POP0_counter += 1
            idx_counter += 1
            power_ratio.append(1)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0] + power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
                
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                    
        
        elif freq_in[i] >= 49.9:  
            POP0_counter += 1
            idx_counter += 1
            power_ratio.append(0)
            if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
            else:
                power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
            if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
            elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                power_kWs_app[idx_counter-1] = ideal_cap_min
                min_counter += 1
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
            SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
    
        
        else:  #freq_in[i] > 49.9 and freq_in[i] < 50.1:
             POP0_counter += 1
             idx_counter += 1
             x_coords, y_coords = zip(*POP0_droop_pos_coords)
             A = np.vstack([x_coords,np.ones(len(x_coords))]).T
             m, c = lstsq(A, y_coords)[0]
             power_ratio.append(m*freq_in[i] + c)
             if i == first_idx:
                power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
             else:
                power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                                                                              
             if power_kWs_app[idx_counter-1] > ideal_cap_max:
                max_counter += 1
                power_kWs_app[idx_counter-1] = ideal_cap_max
                power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
             elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                 power_kWs_app[idx_counter-1] = ideal_cap_min
                 min_counter += 1
                 power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
             SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
            
    return power_ratio , power_kWs_app, SOC , max_counter, min_counter



""" Droop Function DE """
def FCR_N(freq_in,SOC_init,battery_max_cap,max_charge, charging_max):
    #POP specific linear functions
    POP0_droop_pos_coords = [(49.9,1),(50.1,-1)]
    

    
    power_ratio = []
    #POP = 0
    
    """ SOC variables """
    SOC = []
    
    power_kWs = [SOC_init * battery_max_cap * 3600] #63000
    power_kWs_app = [] # emppty list needed for more accuracy
    SOC_max = battery_max_cap * 3600
    
    
    #Never allowing charge/discharge to reach 100%
    #max_charge = 0.95
    ideal_cap_max = (battery_max_cap*max_charge) * 3600
    ideal_cap_min = (battery_max_cap - battery_max_cap*max_charge) * 3600 
     
    #Internal counters
    POP0_counter = 0
    idx_counter = 0
    max_counter = 0
    min_counter = 0
    
    
    
    first_idx = freq_in.first_valid_index()
    last_idx = freq_in.last_valid_index()


    for i in range(first_idx,last_idx+1):    
            if freq_in[i] <= 49.9:
                POP0_counter += 1
                idx_counter += 1
                power_ratio.append(1)
                if i == first_idx:
                    power_kWs_app.append(power_kWs[0] + power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
                    
                if power_kWs_app[idx_counter-1] > ideal_cap_max:
                    max_counter += 1
                    power_kWs_app[idx_counter-1] = ideal_cap_max
                    power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                    
                elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                    power_kWs_app[idx_counter-1] = ideal_cap_min
                    min_counter += 1
                    power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
                        
            
            elif freq_in[i] >= 50.1:  
                POP0_counter += 1
                idx_counter += 1
                power_ratio.append(-1)
                if i == first_idx:
                    power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2]+power_ratio[idx_counter-1] * charging_max *-1)
                if power_kWs_app[idx_counter-1] > ideal_cap_max:
                    max_counter += 1
                    power_kWs_app[idx_counter-1] = ideal_cap_max
                    power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                
                elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                    power_kWs_app[idx_counter-1] = ideal_cap_min
                    min_counter += 1
                    power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                SOC.append(power_kWs_app[idx_counter-1]/SOC_max)
        
            
            else:  #freq_in[i] > 49.9 and freq_in[i] < 50.1:
                 POP0_counter += 1
                 idx_counter += 1
                 x_coords, y_coords = zip(*POP0_droop_pos_coords)
                 A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                 m, c = lstsq(A, y_coords)[0]
                 power_ratio.append(m*freq_in[i] + c)
                 if i == first_idx:
                    power_kWs_app.append(power_kWs[0]+power_ratio[idx_counter-1] * charging_max *-1)
                 else:
                    power_kWs_app.append(power_kWs_app[idx_counter-2] + power_ratio[idx_counter-1] * charging_max *-1)
                                                                                  
                 if power_kWs_app[idx_counter-1] > ideal_cap_max:
                     max_counter += 1
                     power_kWs_app[idx_counter-1] = ideal_cap_max
                     power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                         
                 elif power_kWs_app[idx_counter-1] < ideal_cap_min:
                    power_kWs_app[idx_counter-1] = ideal_cap_min
                    min_counter += 1
                    power_ratio[idx_counter-1] = ((power_kWs_app[idx_counter-1]-power_kWs_app[idx_counter-2])/(-1*charging_max))
                 SOC.append(power_kWs_app[idx_counter-1]/SOC_max)

    return power_ratio , power_kWs_app, SOC , max_counter, min_counter







