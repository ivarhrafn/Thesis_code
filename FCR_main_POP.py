
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})
import scipy as sc 
import numpy as np
from scipy.signal import argrelextrema
from matplotlib.font_manager import FontProperties
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from decimal import getcontext, Decimal
from sympy import pretty_print as pp, latex


from droop_functions import *
from FCR_DE_POP import FCR_DE_POP
from FCR_DK_POP import FCR_DK_POP
from FCR_N_POP import FCR_N_POP
from FCR_D_POP import FCR_D_POP
from FCR_N_POP_2step import FCR_N_POP_2step


""" 
Code Terminology:
    
    DOD = Depth of Discharge
    DOC = Depth of Charge (reverse DOD)
    DOB = Depth of Both (DOD + DOC)

"""




#Features
sns.set_style('darkgrid')
plt.style.use('seaborn-deep')




""" Read in Frequency Data """
#### DK2
df_hz_DK2 = pd.read_csv('DK2_Februar.csv' , delimiter = ',')
df_small_DK2 = df_hz_DK2[:]    #(3600*24*14):(3600*24*16)]

#### DK1
df_hz_DK1 = pd.read_csv('DK1_Februar.csv' , delimiter = ';')
df_small_DK1 = df_hz_DK1[:]




""" FCR Function Inputs """
time = df_small_DK1.iloc[:,0]

freq_DK2 = df_small_DK2.iloc[:,1]
freq_DK1 = df_small_DK1.iloc[:,2]

# If comparison between DK1 and DK2 is desired the length of freq_DK1 needs to be reduced
#freq_DK1 = df_small_DK1.iloc[:len(freq_DK2),2]


SOC_init = 0.50 #Initial SOC in p.u.
battery_max_cap = 24 #kWh ----- # Tests : 12, 24, 60 kWh
max_charge = 1 #Max and min capacity the battery
charging_max = 10 #Max charging in kW


""" FCR Functions and Extract Variables """
##### POP Functions #####
power_DE, kWs_DE, SOC_DE, POP0_count_DE, POP1_count_DE ,POP2_count_DE, max_count_DE , min_count_DE = np.array(FCR_DE_POP(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max ))
power_DK, kWs_DK, SOC_DK, POP0_count_DK, POP1_count_DK ,POP2_count_DK, max_count_DK , min_count_DK = np.array(FCR_DK_POP(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max ))
power_N, kWs_N, SOC_N, POP0_count_N, POP1_count_N ,POP2_count_N, max_count_N , min_count_N = np.array(FCR_N_POP(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max ))
power_D, kWs_D, SOC_D, POP0_count_D, POP1_count_D ,POP2_count_D, max_count_D , min_count_D = np.array(FCR_D_POP(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max ))


##### FCR Without POP #####
#power_DE, kWs_DE, SOC_DE, max_count_DE , min_count_DE = np.array(FCR_DE(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max))
#power_DK, kWs_DK, SOC_DK, max_count_DK , min_count_DK = np.array(FCR_DK(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max))
#power_N, kWs_N, SOC_N, max_count_N , min_count_N = np.array(FCR_N(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max))
#power_D, kWs_D, SOC_D, max_count_D , min_count_D = np.array(FCR_D(freq_DK1, SOC_init,battery_max_cap,max_charge, charging_max))


kWh_DE = np.array(kWs_DE)/3600; kWh_DK = np.array(kWs_DK)/3600; kWh_N = np.array(kWs_N)/3600; kWh_D = np.array(kWs_D)/3600
power_DE = np.array([power_DE])*10; power_DK = np.array([power_DK])*10; power_N = np.array([power_N])*10; power_D = np.array([power_D])*10;
power_DE = power_DE.reshape(len(SOC_DE),); power_DK = power_DK.reshape(len(SOC_DK),); power_N = power_N.reshape(len(SOC_N),); power_D = power_D.reshape(len(SOC_D),)



""" X axis for graphs """
x = np.array(range(len(power_DE)))


""" kWh/hour  """

def E_output(input_power):
    hour = 3600
   # minute = 60
    counter = 0
    if len(input_power) < hour:
        print('Len Power less than 3600')
    
    if len(input_power) % hour != 0: # If the lenght of the vector P divided by 3600 does not return an integer
        for n in range(1,len(input_power)):
            input_power[:-n]
            if len(input_power[:-n]) % hour == 0: # Remove the last values from the vector untill the divide returns an integer
                print('Its a Match!')#   break
                counter += 1
                input_P_h = input_power[:-n]
                tot_hour = int(len(input_power[:-n])/3600)
                break
    
    else:
        input_P_h = input_power
        tot_hour = int(len(input_power)/3600)
        
    
    
    print('Tot Hour:',tot_hour)
    
    power_hours_mat = np.reshape(input_P_h, (tot_hour,hour))
     
    
    # Positive sum:
    positives = np.where(power_hours_mat>0,power_hours_mat,0).sum(axis=1)        
    
    # Negative sum:
    negatives = np.where(power_hours_mat<0,power_hours_mat,0).sum(axis=1)
    
    return positives, negatives

         
        


""" Split time column for dates and time """
def Split_Time(time_column):
    a = []
    b = []
    first_idx = time_column.first_valid_index()
    last_idx = time_column.last_valid_index()

    for i in range(first_idx,last_idx):
        a.append(time_column[i].split(' ')[1])
        b.append(time_column[i].split(' ')[0])
        
    return a,b

x_tick = df_small_DK1.iloc[:,0]
time_x, date_x = Split_Time(x_tick)

""" Insert Vertical Line on Grapths every 24 h """
def vertical_line(when):
    coords_vertical = []
    h24 = 86400
    for i in range(len(when)):
        if (i+h24)%h24 == 0 and i != 0:
            coords_vertical.append(i)
           
    return coords_vertical


""" Algorithm 2 -- Monotonic Cycle Behavior -- Discharges (DOD) and Charges (DOC) """
### DOD based on energy -- outputs cycle in kWh
def DOD_calc_kWh(kWh, batt_cap):
    count1 = 0
    count2 = 0
    DOD = []
    DOD_thres = []
    calc = []
    triggers = []
    tirggers_thres = []
    for i in range(1,len(kWh)):
        if kWh[i] <= kWh[i-1]:              #Looks for downwards slope
            count1 +=1
            calc.append(kWh[i])
            if len(calc) == 1:
                set_pnt1 = i                #Starting point of slope saved
            if i < len(kWh) -1:
                if kWh[i+1] > kWh[i]:       #If upwards slope is detected, end cycle
                    count2 += 1
                    DOD.append(kWh[set_pnt1] - kWh[i])
                    triggers.append(i)
                    count1 = 0
                    calc = []
                    if (kWh[set_pnt1] - kWh[i]) > (0.1*batt_cap):    #Only register cycle period if its over 1% "Delta" SOC
                        DOD_thres.append(kWh[set_pnt1] - kWh[i])
                        tirggers_thres.append(i)
    
    return DOD, triggers, DOD_thres, tirggers_thres

### DOC based on energy -- outputs cycle in kWh
def DOC_calc_kWh(kWh, batt_cap):
    count1 = 0
    count2 = 0
    DOC = []
    DOC_thres = []
    calc = []
    triggers = []
    tirggers_thres = []
    for i in range(1,len(kWh)):
        if kWh[i] >= kWh[i-1]:
            count1 +=1
            calc.append(kWh[i])
            if len(calc) == 1:
                set_pnt1 = i
            if i < len(kWh) -1:
                if kWh[i+1] < kWh[i]:
                    count2 += 1
                    DOC.append(kWh[i]-kWh[set_pnt1])
                    triggers.append(i)
                    count1 = 0
                    calc = []
                    if (kWh[i]-kWh[set_pnt1]) > (0.1*batt_cap):
                        DOC_thres.append(kWh[i]-kWh[set_pnt1])
                        tirggers_thres.append(i)

    return DOC, triggers, DOC_thres, tirggers_thres

## DOD for output based on SOC
def DOD_calc(SOC):
    count1 = 0
    count2 = 0
    DOD = []
    DOD_thres = []
    calc = []
    triggers = []
    tirggers_thres = []
    for i in range(1,len(SOC)):
        if SOC[i] <= SOC[i-1]:
            count1 +=1
            calc.append(SOC[i])
            if len(calc) == 1:
                set_pnt1 = i
            if i < len(SOC) -1:
                if SOC[i+1] > SOC[i]:
                    count2 += 1
                    DOD.append(SOC[set_pnt1] - SOC[i])
                    triggers.append(i)
                    count1 = 0
                    calc = []
                    if (SOC[set_pnt1] - SOC[i]) > 0.01:
                        DOD_thres.append(SOC[set_pnt1] - SOC[i])
                        tirggers_thres.append(i)

    return DOD, triggers, DOD_thres, tirggers_thres



## DOD for output based on SOC
def DOC_calc(SOC):
    count1 = 0
    count2 = 0
    DOC = []
    DOC_thres = []
    calc = []
    triggers = []
    tirggers_thres = []
    for i in range(1,len(SOC)):
        if SOC[i] >= SOC[i-1]:
            count1 +=1
            calc.append(SOC[i])
            if len(calc) == 1:
                set_pnt1 = i
            if i < len(SOC) -1:
                if SOC[i+1] < SOC[i]:
                    count2 += 1
                    DOC.append(SOC[i]-SOC[set_pnt1])
                    triggers.append(i)
                    count1 = 0
                    calc = []
                    if (SOC[i]-SOC[set_pnt1]) > 0.01:
                        DOC_thres.append(SOC[i]-SOC[set_pnt1])
                        tirggers_thres.append(i)

    return DOC, triggers, DOC_thres, tirggers_thres




"""  Algorithm 3 -- Slope deviation under 1% SOC allowed -- Discharges (DOD) and Charges (DOC) -- """
## DOD based on SOC
def DOD_calc_no_micro(SOC):
      
    DOD = []

    
    calc = []
    calc_up = []
    set_pnt1 = 0
    set_pnt2 = 0
    
    for i in range(1,len(SOC)):
        if i < len(SOC)-1:
            if SOC[i] <= SOC[i-1]:                         # Looks for downwards of even slope
                calc.append(SOC[i])
                if len(calc) == 1:
                    set_pnt1 = i                           #Starting location of slope saved
                if SOC[i] < SOC[i+1]:                      #if SOC is higher next iteration enter new for loop
                    for j in range(i , len(SOC)):
                        if j < len(SOC)-1:
                            if SOC[j] <= SOC[j+1]:         #check for upward or even slope
                                calc_up.append(SOC[j])
                                if len(calc_up) == 1:
                                    set_pnt2 = i           #Save local minima point
                            else:
                                if SOC[j]-SOC[set_pnt2] > 0.01:                  #If upwards slope is greater than 1% "delta" SOC end cycle
                                    if SOC[set_pnt1] - SOC[set_pnt2] > 0.01:
                                        DOD.append(SOC[set_pnt1] - SOC[set_pnt2])   #Save DOD if its greater than 1% "delta" SOC
                                    
                                    calc = []       
                                    calc_up = []     #Reset cycle variables 
                                    set_pnt1 = 0
                                    set_pnt2 = 0
                                    break
                                else:
                                    calc_up = []    #Only reset local minima variables if cycle is not broken
                                    set_pnt2 = 0
                                    break
                                                        
    return DOD




##DOC based on SOC
def DOC_calc_no_micro(SOC):
      
    DOC = []    
    calc = []
    calc_up = []
    set_pnt1 = 0
    set_pnt2 = 0
    
    for i in range(1,len(SOC)):
        if i < len(SOC)-1:
            if SOC[i] >= SOC[i-1]:
                calc.append(SOC[i])
                if len(calc) == 1:
                    set_pnt1 = i   
                if SOC[i] > SOC[i+1]:
                    for j in range(i , len(SOC)):
                        if j < len(SOC)-1:
                            if SOC[j] >= SOC[j+1]:
                                calc_up.append(SOC[j])
                                if len(calc_up) == 1:
                                    set_pnt2 = i
                            else:
                                if SOC[set_pnt2]-SOC[j] > 0.01:
                                    if SOC[set_pnt2] - SOC[set_pnt1] > 0.01:
                                        DOC.append(SOC[set_pnt2] - SOC[set_pnt1])

                                    calc = []
                                    calc_up = []
                                    set_pnt1 = 0
                                    set_pnt2 = 0
                                    break
                                else:
                                    calc_up = []
                                    set_pnt2 = 0
                                    break
                                                        
    return DOC



### DOD based on energy -- outputs cycle in kWh
def DOD_calc_no_micro_kWh(SOC, batt_cap):
      
    DOD = []

    
    calc = []
    calc_up = []
    set_pnt1 = 0
    set_pnt2 = 0
    
    for i in range(1,len(SOC)):
        if i < len(SOC)-1:
            if SOC[i] <= SOC[i-1]:
                calc.append(SOC[i])
                if len(calc) == 1:
                    set_pnt1 = i   
                if SOC[i] < SOC[i+1]:
                    for j in range(i , len(SOC)):
                        if j < len(SOC)-1:
                            if SOC[j] <= SOC[j+1]:
                                calc_up.append(SOC[j])
                                if len(calc_up) == 1:
                                    set_pnt2 = i
                            else:
                                if SOC[j]-SOC[set_pnt2] > 0.01*batt_cap:
                                    if SOC[set_pnt1] - SOC[set_pnt2] > 0.01*batt_cap:
                                        DOD.append(SOC[set_pnt1] - SOC[set_pnt2])

                                    calc = []
                                    calc_up = []
                                    set_pnt1 = 0
                                    set_pnt2 = 0
                                    break
                                else:
                                    calc_up = []
                                    set_pnt2 = 0
                                    break
                                                        
    return DOD


### DOC based on energy -- outputs cycle in kWh
def DOC_calc_no_micro_kWh(SOC, batt_cap):
      
    DOC = []    
    calc = []
    calc_up = []
    set_pnt1 = 0
    set_pnt2 = 0
    
    for i in range(1,len(SOC)):
        if i < len(SOC)-1:
            if SOC[i] >= SOC[i-1]:
                calc.append(SOC[i])
                if len(calc) == 1:
                    set_pnt1 = i   
                if SOC[i] > SOC[i+1]:
                    for j in range(i , len(SOC)):
                        if j < len(SOC)-1:
                            if SOC[j] >= SOC[j+1]:
                                calc_up.append(SOC[j])
                                if len(calc_up) == 1:
                                    set_pnt2 = i
                            else:
                                if SOC[set_pnt2]-SOC[j] > 0.01*batt_cap:
                                    if SOC[set_pnt2] - SOC[set_pnt1] > 0.01*batt_cap:
                                        DOC.append(SOC[set_pnt2] - SOC[set_pnt1])

                                    calc = []
                                    calc_up = []
                                    set_pnt1 = 0
                                    set_pnt2 = 0
                                    break
                                else:
                                    calc_up = []
                                    set_pnt2 = 0
                                    break
                                                        
    return DOC






""" Count occurences of DOD measurments with a 2% interval   """
def count_DOD_size(DOD_vec):
    counter0 = 0; counter1 = 0; counter2 = 0; counter3 = 0; counter4 = 0; counter5 = 0; counter6 = 0;
    counter7 = 0; counter8 = 0; counter9 = 0; counter10 = 0; counter11 = 0; counter12 = 0; counter13 = 0;
    counter14 = 0; counter15 = 0; counter16 = 0; counter17 = 0; counter18 = 0; counter19 = 0; counter20 = 0;
    counter21 = 0; counter22 = 0; counter23 = 0; counter24 = 0; counter25 = 0; counter26 = 0; counter27 = 0;
    counter28 = 0; counter29 = 0; counter30 = 0; counter31 = 0; counter32 = 0; counter33 = 0; counter34 = 0;
    counter35 = 0; counter36 = 0; counter37 = 0; counter38 = 0; counter39 = 0; counter40 = 0; counter41 = 0; 
    counter42 = 0; counter43 = 0; counter44 = 0; counter45 = 0; counter46 = 0; counter47 = 0; counter48 = 0;
    counter49 = 0; 
   
    for i in range(len(DOD_vec)):
        if   DOD_vec[i] >= 1 and DOD_vec[i] < 2:   counter0 += 1
        elif DOD_vec[i] >= 2  and DOD_vec[i] < 4:   counter1 += 1
        elif DOD_vec[i] >= 4  and DOD_vec[i] < 6:   counter2 += 1
        elif DOD_vec[i] >= 6  and DOD_vec[i] < 8:   counter3 += 1
        elif DOD_vec[i] >= 8  and DOD_vec[i] < 10:  counter4 += 1
        elif DOD_vec[i] >= 10 and DOD_vec[i] < 12:  counter5 += 1
        elif DOD_vec[i] >= 12 and DOD_vec[i] < 14:  counter6 += 1
        elif DOD_vec[i] >= 14 and DOD_vec[i] < 16:  counter7 += 1
        elif DOD_vec[i] >= 16 and DOD_vec[i] < 18:  counter8 += 1
        elif DOD_vec[i] >= 18 and DOD_vec[i] < 20:  counter9 += 1
        elif DOD_vec[i] >= 20 and DOD_vec[i] < 22:  counter10 += 1
        elif DOD_vec[i] >= 22 and DOD_vec[i] < 24:  counter11 += 1
        elif DOD_vec[i] >= 24 and DOD_vec[i] < 26:  counter12 += 1
        elif DOD_vec[i] >= 26 and DOD_vec[i] < 28:  counter13 += 1
        elif DOD_vec[i] >= 28 and DOD_vec[i] < 30:  counter14 += 1
        elif DOD_vec[i] >= 30 and DOD_vec[i] < 32:  counter15 += 1
        elif DOD_vec[i] >= 32 and DOD_vec[i] < 34:  counter16 += 1
        elif DOD_vec[i] >= 34 and DOD_vec[i] < 36:  counter17 += 1
        elif DOD_vec[i] >= 36 and DOD_vec[i] < 38:  counter18 += 1
        elif DOD_vec[i] >= 38 and DOD_vec[i] < 40:  counter19+= 1
        elif DOD_vec[i] >= 40 and DOD_vec[i] < 42:  counter20 += 1
        elif DOD_vec[i] >= 42 and DOD_vec[i] < 44:  counter21+= 1
        elif DOD_vec[i] >= 44 and DOD_vec[i] < 46:  counter22 += 1
        elif DOD_vec[i] >= 46 and DOD_vec[i] < 48:  counter23 += 1
        elif DOD_vec[i] >= 48 and DOD_vec[i] < 50:  counter24 += 1
        elif DOD_vec[i] >= 50 and DOD_vec[i] < 52:  counter25 += 1
        elif DOD_vec[i] >= 52 and DOD_vec[i] < 54:  counter26 += 1
        elif DOD_vec[i] >= 54 and DOD_vec[i] < 56:  counter27 += 1
        elif DOD_vec[i] >= 56 and DOD_vec[i] < 58:  counter28 += 1
        elif DOD_vec[i] >= 58 and DOD_vec[i] < 60:  counter29 += 1
        elif DOD_vec[i] >= 60 and DOD_vec[i] < 62:  counter30 += 1
        elif DOD_vec[i] >= 62 and DOD_vec[i] < 64:  counter31 += 1
        elif DOD_vec[i] >= 64 and DOD_vec[i] < 66:  counter32 += 1
        elif DOD_vec[i] >= 66 and DOD_vec[i] < 68:  counter33 += 1
        elif DOD_vec[i] >= 68 and DOD_vec[i] < 70:  counter34 += 1
        elif DOD_vec[i] >= 70 and DOD_vec[i] < 72:  counter35 += 1
        elif DOD_vec[i] >= 72 and DOD_vec[i] < 74:  counter36 += 1
        elif DOD_vec[i] >= 74 and DOD_vec[i] < 76:  counter37 += 1
        elif DOD_vec[i] >= 76 and DOD_vec[i] < 78:  counter38 += 1
        elif DOD_vec[i] >= 78 and DOD_vec[i] < 80:  counter39 += 1
        elif DOD_vec[i] >= 80 and DOD_vec[i] < 82:  counter40 += 1
        elif DOD_vec[i] >= 82 and DOD_vec[i] < 84:  counter41 += 1
        elif DOD_vec[i] >= 84 and DOD_vec[i] < 86:  counter42 += 1
        elif DOD_vec[i] >= 86 and DOD_vec[i] < 88:  counter43 += 1
        elif DOD_vec[i] >= 88 and DOD_vec[i] < 90:  counter44 += 1
        elif DOD_vec[i] >= 90 and DOD_vec[i] < 92:  counter45 += 1
        elif DOD_vec[i] >= 92 and DOD_vec[i] < 92:  counter46 += 1
        elif DOD_vec[i] >= 94 and DOD_vec[i] < 96:  counter47 += 1
        elif DOD_vec[i] >= 96 and DOD_vec[i] < 98:  counter48 += 1
        elif DOD_vec[i] >= 98 and DOD_vec[i] < 100:  counter49 += 1
        
    count_vec = [counter0,  counter1,  counter2,  counter3,  counter4,  counter5,  counter6,
                 counter7,  counter8,  counter9,  counter10, counter11, counter12, counter13,
                 counter14, counter15, counter16, counter17, counter18, counter19, counter20,
                 counter21, counter22, counter23, counter24, counter25, counter26, counter27,
                 counter28, counter29, counter30, counter31, counter32, counter33, counter34,
                 counter35, counter36, counter37, counter38, counter39, counter40, counter40, 
                 counter41, counter42, counter43, counter44, counter45, counter46, counter47,
                 counter48, counter49]
                         
    return count_vec  
 

""" Generate X and Y variables for Cycle Plots """
def listing(y_list,DOD):
    
    no_0 = []
    no_0 = [g for g in y_list if g > 0] # 0s removed from 
    
    x =  np.linspace(1, 100  , len(y_list)) 
    
    # Find where there are no counts and remove that from the X axis  
    where0 = []
    for i in range(len(y_list)):          
        if y_list[i] == 0:                
            where0.append(i)    
    x = np.delete(x, where0, axis = 0)

    return x, no_0




""" Plot stacked bar plot """
def cumulated_power_plot(pos_vec, neg_vec):
    data = np.array([pos_vec, neg_vec])
    data_shape = np.shape(data)

    cum_neg = data.clip(max=0)
    cum_neg= np.cumsum(cum_neg, axis=0)
    d_neg = np.zeros(np.shape(data))
    d_neg[1:] = cum_neg[:-1]
    
    cumulated_data_neg_DE = d_neg
    
    cum_pos = data.clip(min=0)
    cum_pos= np.cumsum(cum_pos, axis=0)
    d_pos = np.zeros(np.shape(data))
    d_pos[1:] = cum_pos[:-1]
    
    cumulated_data_DE = d_pos
    
    row_mask = (data<0)
    cumulated_data_DE[row_mask] = cumulated_data_neg_DE[row_mask]
    data_stack = cumulated_data_DE
    
    return data, data_shape,data_stack



# biggest value
def E_biggest(pos, neg):
    g = pos.max()
    t = neg.min()
    if g > abs(t):
        return g
    else: return t

def E_biggest_pos(vec1, vec2):
    g = vec1.max()
    t = vec2.max()
    if g > (t):
        return g
    else: return t


"""" Power Sums """
# kWs
pos_E_DE, neg_E_DE = E_output(power_DE)
pos_E_DK, neg_E_DK = E_output(power_DK)
pos_E_N, neg_E_N = E_output(power_N)
pos_E_D, neg_E_D = E_output(power_D)


# convert to kWh
pos_E_DE_h = pos_E_DE/3600; neg_E_DE_h = neg_E_DE/3600
pos_E_DK_h = pos_E_DK/3600; neg_E_DK_h = neg_E_DK/3600
pos_E_N_h = pos_E_N/3600; neg_E_N_h = neg_E_N/3600
pos_E_D_h = pos_E_D/3600; neg_E_D_h = neg_E_D/3600

#calculate averages and standard deviation
pos_E_DE_ave = sc.average(pos_E_DE_h); neg_E_DE_ave = sc.average(neg_E_DE/3600)
pos_E_DK_ave = sc.average(pos_E_DK_h); neg_E_DK_ave = sc.average(neg_E_DK/3600)
pos_E_N_ave = sc.average(pos_E_N_h); neg_E_N_ave = sc.average(neg_E_N/3600)
pos_E_D_ave = sc.average(pos_E_D_h); neg_E_D_ave = sc.average(neg_E_D/3600)

pos_E_DE_std = sc.std(pos_E_DE_h); neg_E_DE_std = sc.std(neg_E_DE/3600)
pos_E_DK_std = sc.std(pos_E_DK_h); neg_E_DK_std = sc.std(neg_E_DK/3600)
pos_E_N_std = sc.std(pos_E_N_h); neg_E_N_std = sc.std(neg_E_N/3600)
pos_E_D_std = sc.std(pos_E_D_h); neg_E_D_std = sc.std(neg_E_D/3600)



abs_E_DE_neg = abs(neg_E_DE_h)
abs_E_DK_neg = abs(neg_E_DK_h)
abs_E_N_neg = abs(neg_E_N_h)
abs_E_D_neg = abs(neg_E_D_h)


big_DE = E_biggest_pos(pos_E_DE_h, abs_E_DE_neg)
big_DK = E_biggest_pos(pos_E_DK_h, abs_E_DK_neg)
big_N = E_biggest_pos(pos_E_N_h, abs_E_N_neg)


#Total Energy output
E_tot_DE_per_h = pos_E_DE_h + abs_E_DE_neg
E_tot_DK_per_h = pos_E_DK_h + abs_E_DK_neg
E_tot_N_per_h = pos_E_N_h + abs_E_N_neg

E_tot_DE = E_tot_DE_per_h.sum()
E_tot_DK = E_tot_DK_per_h.sum()
E_tot_N = E_tot_N_per_h.sum()
E_tot_D = pos_E_D_h.sum()

# Variables
E_tot = [E_tot_DK, E_tot_DE,E_tot_N,E_tot_D]
E_tot_pos = [pos_E_DK_h.sum(), pos_E_DE_h.sum(),pos_E_N_h.sum(),pos_E_D_h.sum()]
E_tot_neg = [neg_E_DK_h.sum(), neg_E_DE_h.sum(),neg_E_N_h.sum(),neg_E_D_h.sum()]

#Average

ave_E_vec = np.array([sc.average(E_tot_DK_per_h),sc.average(E_tot_DE_per_h),sc.average(E_tot_N_per_h),
    sc.average(pos_E_DK_h),sc.average(pos_E_DE_h), sc.average(pos_E_N_h), sc.average(pos_E_D_h),
    sc.average(neg_E_DK_h),sc.average(neg_E_DE_h),sc.average(neg_E_N_h)])

#Standard Dev
std_E_vec = np.array([sc.std(E_tot_DK_per_h),sc.std(E_tot_DE_per_h),sc.std(E_tot_N_per_h),
    sc.std(pos_E_DK_h),sc.std(pos_E_DE_h), sc.std(pos_E_N_h), sc.std(pos_E_D_h),
    sc.std(neg_E_DK_h),sc.std(neg_E_DE_h),sc.std(neg_E_N_h)])


"""           Energy Dataframes          """
# Pos and Neg
key_Energy_DK = {'PFR-DK Input': pos_E_DK_h, 'PFR-DK Output': neg_E_DK_h}
key_Energy_DE = {'PFR-DE Input': pos_E_DE_h, 'PFR-DE Output': neg_E_DE_h}
key_Energy_N = {'FCR-N Input': pos_E_N_h, 'FCR-N Output': neg_E_N_h}
key_Energy_D = {'PFR-D Input ': pos_E_D_h, 'PFR-D Output': neg_E_D_h}

#Abs
key_Energy_DK_abs = {'PFR-DK Input': pos_E_DK_h, 'PFR-DK Output': abs_E_DK_neg}
key_Energy_DE_abs = {'PFR-DE Input': pos_E_DE_h, 'PFR-DE Output': abs_E_DE_neg}
key_Energy_N_abs = {'FCR-N Input': pos_E_N_h, 'FCR-N Output': abs_E_N_neg}
key_Energy_D_abs = {'PFR-D Input ': pos_E_D_h, 'PFR-D Output': abs_E_D_neg}

# United DataFrames
key_Energy_allt_abs = {"PFR-DK Input": pos_E_DK_h, 'PFR-DK Output': abs_E_DK_neg,
'PFR-DE Input': pos_E_DE_h, 'PFR-DE Output': abs_E_DE_neg,
'FCR-N Input': pos_E_N_h, 'FCR-N Output': abs_E_N_neg, 'FCR-D Input': pos_E_D_h}


dada = [pos_E_DK_h,abs_E_DK_neg,pos_E_DE_h,abs_E_DE_neg,pos_E_N_h,abs_E_N_neg,pos_E_D_h]
Ene_vec = np.array(dada).flatten()


strat_vec = []
strat_vec.append(np.array(['PFR-DK']).repeat(len(pos_E_DK_h)*2))
strat_vec.append(np.array(['PFR-DE']).repeat(len(pos_E_DK_h)*2))
strat_vec.append(np.array(['FCR-N']).repeat(len(pos_E_DK_h)*2))
strat_vec.append(np.array(['FCR-D']).repeat(len(pos_E_DK_h)))
#Tuple to array
strat_vec = [t for xs in strat_vec for t in xs]  

flow_vec = []
flow_vec.append(np.array(['Input']).repeat(len(pos_E_DK_h)))
flow_vec.append(np.array(['Output']).repeat(len(pos_E_DK_h)))
flow_vec.append(np.array(['Input']).repeat(len(pos_E_DK_h)))
flow_vec.append(np.array(['Output']).repeat(len(pos_E_DK_h)))
flow_vec.append(np.array(['Input']).repeat(len(pos_E_DK_h)))
flow_vec.append(np.array(['Output']).repeat(len(pos_E_DK_h)))
flow_vec.append(np.array(['Input']).repeat(len(pos_E_DK_h)))
#Tuple to array
flow_vec = [t for xs in flow_vec for t in xs]


# Put Strat , flow and energy vectors
key_compare = {'Strategy': strat_vec, 'Energy [kWh/h]' : Ene_vec, 'Power Flow': flow_vec}

key_Energy_allt_pos = {'PFR-DK Input': pos_E_DK_h, 
'PFR-DE Input': pos_E_DE_h, 'FCR-N Input': pos_E_N_h, 'FCR-D Input': pos_E_D_h}

key_Energy_allt_neg = {'PFR-DK Output': neg_E_DK_h, 
'PFR-DE Output': neg_E_DE_h, 'FCR-N Output': neg_E_N_h}

key_Energy_allt_sum = {'PFR-DK': E_tot_DK_per_h, 
'PFR-DE': E_tot_DE_per_h, 'FCR-N': E_tot_N_per_h, 'FCR-D' : pos_E_D_h}


# Pos and Neg DataFrames
df_Energy_DK_h = pd.DataFrame(key_Energy_DK).T
df_Energy_DE_h = pd.DataFrame(key_Energy_DE).T
df_Energy_N_h = pd.DataFrame(key_Energy_N).T
df_Energy_D_h = pd.DataFrame(key_Energy_D).T

# Abs DataFrames
df_Energy_DK_h_abs = pd.DataFrame(key_Energy_DK_abs).T
df_Energy_DE_h_abs = pd.DataFrame(key_Energy_DE_abs).T
df_Energy_N_h_abs = pd.DataFrame(key_Energy_N_abs).T
df_Energy_D_h_abs = pd.DataFrame(key_Energy_D_abs).T

# United DataFrame 
df_E_allt_abs = pd.DataFrame(key_Energy_allt_abs)
df_E_allt_pos = pd.DataFrame(key_Energy_allt_pos)
df_E_allt_neg = pd.DataFrame(key_Energy_allt_neg)
df_E_allt_sum = pd.DataFrame(key_Energy_allt_sum)

df_E_compare = pd.DataFrame(key_compare)

""" Energy DataFrames END  """




""" Calling DOD Functons -- Algorithm 2 """


DOD_DE,triggered_DE, DOD_DE_thres,triggered_thres_DE = DOD_calc(SOC_DE)
DOD_DK,triggered_DK, DOD_DK_thres,triggered_thres_DK = DOD_calc(SOC_DK)
DOD_N,triggere_Nd, DOD_N_thres,triggered_thres_N = DOD_calc(SOC_N)
DOD_D,triggere_D, DOD_D_thres,triggered_thres_D = DOD_calc(SOC_D)




""" Calling DOC Functions  -- Algorithm 2  """
DOC_DE, DOC_triggered_DE, DOC_DE_thres, DOC_triggered_thres_DE = DOC_calc(SOC_DE)
DOC_DK, DOC_triggered_DK, DOC_DK_thres, DOC_triggered_thres_DK = DOC_calc(SOC_DK)
DOC_N, DOC_triggered_N, DOC_N_thres, DOC_triggered_thres_N = DOC_calc(SOC_N)


### kWh  
DOD_DE_kWh ,triggered_DE_kWh, DOD_DE_thres_kWh, triggered_thres_DE_kWh = DOD_calc_kWh(kWh_DE, battery_max_cap)
DOD_DK_kWh ,triggered_DK_kWh, DOD_DK_thres_kWh, triggered_thres_DK_kWh = DOD_calc_kWh(kWh_DK, battery_max_cap)
DOD_N_kWh ,triggered_N_kWh, DOD_N_thres_kWh, triggered_thres_N_kWh = DOD_calc_kWh(kWh_N, battery_max_cap)


DOC_DE_kWh , DOC_triggered_DE_kWh, DOC_DE_thres_kWh, DOC_triggered_thres_DE_kWh = DOC_calc_kWh(kWh_DE, battery_max_cap)
DOC_DK_kWh , DOC_triggered_DK_kWh, DOC_DK_thres_kWh, DOC_triggered_thres_DK_kWh = DOC_calc_kWh(kWh_DK, battery_max_cap)
DOC_N_kWh , DOC_triggered_N_kWh, DOC_N_thres_kWh, DOC_triggered_thres_N_kWh = DOC_calc_kWh(kWh_N, battery_max_cap)




""" Call Cycle function for kWh --- Algorithm 3  """
DOD_DE_kWh_m  = DOD_calc_no_micro_kWh(kWh_DE, battery_max_cap)
DOD_DK_kWh_m  = DOD_calc_no_micro_kWh(kWh_DK, battery_max_cap)
DOD_N_kWh_m   = DOD_calc_no_micro_kWh(kWh_N, battery_max_cap)

DOC_DE_kWh_m  = DOC_calc_no_micro_kWh(kWh_DE, battery_max_cap)
DOC_DK_kWh_m  = DOC_calc_no_micro_kWh(kWh_DK, battery_max_cap)
DOC_N_kWh_m   = DOC_calc_no_micro_kWh(kWh_N, battery_max_cap)





### No Thres No Nothin 
key_DOD_DE = {'PFR-DE': DOD_DE}
key_DOD_DK = {'PFR-DK': DOD_DK}
key_DOD_N = {'FCR-N': DOD_N}

key_DOC_DE = {'PFR-DE': DOC_DE}
key_DOC_DK = {'PFR-DK': DOC_DK}
key_DOC_N = {'FCR-N': DOC_N}



## 1% SOC Threshold
key_DOD_DE_thres = {'PFR-DE': DOD_DE_thres}
key_DOD_DK_thres = {'PFR-DK': DOD_DK_thres}
key_DOD_N_thres = {'FCR-N': DOD_N_thres}

key_DOC_DE_thres = {'PFR-DE': DOC_DE_thres}
key_DOC_DK_thres = {'PFR-DK': DOC_DK_thres}
key_DOC_N_thres = {'FCR-N': DOC_N_thres}

## 1% SOC Threshold -- Givemn in kWh
key_DOD_DE_kWh = {'PFR-DE': DOD_DE_thres_kWh}
key_DOD_DK_kWh = {'PFR-DK': DOD_DK_thres_kWh}
key_DOD_N_kWh = {'FCR-N': DOD_N_thres_kWh}

key_DOC_DE_kWh = {'PFR-DE': DOC_DE_thres_kWh}
key_DOC_DK_kWh = {'PFR-DK': DOC_DK_thres_kWh}
key_DOC_N_kWh = {'FCR-N': DOC_N_thres_kWh}


###  No micro Cycles and kWh ###
key_DOD_DE_kWh_m = {'PFR-DE': DOD_DE_kWh_m}
key_DOD_DK_kWh_m = {'PFR-DK': DOD_DK_kWh_m}
key_DOD_N_kWh_m = {'FCR-N': DOD_N_kWh_m}

key_DOC_DE_kWh_m = {'PFR-DE': DOC_DE_kWh_m}
key_DOC_DK_kWh_m = {'PFR-DK': DOC_DK_kWh_m}
key_DOC_N_kWh_m = {'FCR-N': DOC_N_kWh_m}


DOB_DE_m = DOD_DE_kWh_m + DOC_DE_kWh_m
DOB_DK_m = DOD_DK_kWh_m + DOC_DK_kWh_m
DOB_N_m = DOD_N_kWh_m + DOC_N_kWh_m

key_DOB_DE_kWh_m = {'PFR-DE': DOB_DE_m}
key_DOB_DK_kWh_m = {'PFR-DK': DOB_DK_m}
key_DOB_N_kWh_m = {'FCR-N': DOB_N_m}





df_DOD_DE = pd.DataFrame(key_DOD_DE)
df_DOD_DK = pd.DataFrame(key_DOD_DK)
df_DOD_N = pd.DataFrame(key_DOD_N)

df_DOC_DE_thres = pd.DataFrame(key_DOC_DE_thres)
df_DOC_DK_thres = pd.DataFrame(key_DOC_DK_thres)
df_DOC_N_thres = pd.DataFrame(key_DOC_N_thres)




# 1% SOC Threshold
df_DOD_DE_thres = pd.DataFrame(key_DOD_DE_thres)
df_DOD_DK_thres = pd.DataFrame(key_DOD_DK_thres)
df_DOD_N_thres = pd.DataFrame(key_DOD_N_thres)

df_DOC_DE_thres = pd.DataFrame(key_DOC_DE_thres)
df_DOC_DK_thres = pd.DataFrame(key_DOC_DK_thres)
df_DOC_N_thres = pd.DataFrame(key_DOC_N_thres)

## 1% SOC Threshold -- Given in kWh
df_DOD_DE_kWh = pd.DataFrame(key_DOD_DE_kWh)
df_DOD_DK_kWh = pd.DataFrame(key_DOD_DK_kWh)
df_DOD_N_kWh = pd.DataFrame(key_DOD_N_kWh)

df_DOC_DE_kWh = pd.DataFrame(key_DOC_DE_kWh)
df_DOC_DK_kWh = pd.DataFrame(key_DOC_DK_kWh)
df_DOC_N_kWh = pd.DataFrame(key_DOC_N_kWh)



### DOB -- No micro -- kWh
df_DOB_DE_kWh_m = pd.DataFrame(key_DOB_DE_kWh_m)
df_DOB_DK_kWh_m = pd.DataFrame(key_DOB_DK_kWh_m)
df_DOB_N_kWh_m = pd.DataFrame(key_DOB_N_kWh_m)








""""""" Data Visialization """""""

"""

Energy per hour


"""

### BarPlot for the 4 methods
objects = ('PFR-DK','PFR-DE','FCR-N', 'FCR-D')
y_pos = np.arange(len(objects))
fig = plt.figure() #dpi=200)
plt.bar(y_pos, E_tot, align = 'center', alpha = 0.5)
plt.yscale('log')
plt.ylabel('Total Energy Transfer [kWh]', fontsize = '13')
plt.xlabel('Strategy', fontsize = '13')
plt.xticks(y_pos, objects)
#plt.yscale('log', nonposy='clip')
plt.show()




"""               Grouped Histograms                """

fig = plt.figure() #dpi = 200)
ax = fig.add_axes([0,0,1,1])
bins = np.linspace(0, max(pos_E_N_h)+0.01 , 25)
plt.hist([pos_E_N_h, pos_E_DE_h,pos_E_DK_h], bins, label=['FCR-N', 'PFR-DE', 'PFR-DK'])
plt.legend(loc='upper right')
plt.yscale('log')
ax.set_xlabel('Energy Input to Grid [kWh/h]')
ax.set_ylabel('Occurrences')
plt.show()
#
##
fig = plt.figure() #dpi = 200)
ax = fig.add_axes([0,0,1,1])
bins = np.linspace(min(neg_E_N_h)+0.01, 0 , 25)
plt.hist([neg_E_N_h,neg_E_DE_h,neg_E_DK_h], bins, label=['FCR-N', 'PFR-DE', 'PFR-DK'])
plt.legend(loc='upper right')
fig.axes[0].invert_xaxis()
plt.yscale('log')
ax.set_xlabel('Energy Output from Grid [kWh/h]')
ax.set_ylabel('Occurrences')
plt.show()



"""" Joint Plot """

## DE
g = sns.jointplot(x=df_Energy_DE_h_abs.iloc[1], y=df_Energy_DE_h_abs.iloc[0] ,
                     kind='scatter',
                     ylim = (0, abs(big_DE)+0.05),
                     xlim = (0, 1*(abs(big_DE)+0.05)),
                     data=df_Energy_DE_h_abs)  
g.set_axis_labels('PFR-DE Output [kWh/h]', 'PFR-DE Input [kWh/h]', fontsize=13)
####g.ax_joint.set_xlabel('new x label', fontweight='bold')


#### DK
g = sns.jointplot(x=df_Energy_DK_h_abs.iloc[1], y=df_Energy_DK_h_abs.iloc[0] ,
                     kind='scatter',
                     ylim = (0, abs(big_DK)+0.05),
                     xlim = (0, 1*(abs(big_DK)+0.05)),
                     data=df_Energy_DK_h_abs)  
g.set_axis_labels('PFR-DK Output [kWh/h]', 'PFR-DK Input [kWh/h]', fontsize=13)

#### N
g = sns.jointplot(x=df_Energy_N_h_abs.iloc[1], y=df_Energy_N_h_abs.iloc[0] ,
                     kind='scatter',
                     ylim = (0, abs(big_N)+0.05),
                     xlim = (0, 1*(abs(big_N)+0.05)),
                     data=df_Energy_N_h_abs)  
g.set_axis_labels('FCR-N Output [kWh/h]', 'FCR-N Input [kWh/h]', fontsize=13)




""" Strategy Comparison - Box Plot  """
fig, ax = plt.subplots(dpi = 200)
sns.boxplot(x = 'Strategy', y = 'Energy [kWh/h]', data = df_E_compare, hue= 'Power Flow', palette='rainbow',orient='v')
ax.set_ylim(top = 8.3)
ax.set_ylabel('Energy [kWh/h]', fontsize = '13')
ax.set_xlabel('Strategy', fontsize = '13')




""" Energy Barplots - Matplotlib - """

### DE
fig = plt.figure() #dpi=200)
ax = plt.subplot(111)
data_DE,data_shape_DE,data_stack_DE = cumulated_power_plot(pos_E_DE_h, neg_E_DE_h)
for i in np.arange(0, data_shape_DE[0]):
    if i == 0:
        ax.bar(np.arange(data_shape_DE[1]), data_DE[i], bottom=data_stack_DE[i], label = 'PFR-DE - Input')
    elif i == 1:
         ax.bar(np.arange(data_shape_DE[1]), data_DE[i], bottom=data_stack_DE[i], label = 'PFR-DE - Output')
ax.set_ylabel("Energy [kWh/h]")
ax.set_xlabel("Time [Hours]")
ax.tick_params('y')
ax.set_ylim([-1*(abs(big_N)+0.05),abs(big_N)+0.05])
plt.legend()
plt.show()
#
fig = plt.figure() #dpi=200)
ax = plt.subplot(111)
data_DK,data_shape_DK,data_stack_DK = cumulated_power_plot(pos_E_DK_h, neg_E_DK_h)
for i in np.arange(0, data_shape_DE[0]):
    if i == 0:
        ax.bar(np.arange(data_shape_DK[1]), data_DK[i], bottom=data_stack_DK[i], label = 'PFR-DK - Input')
    elif i == 1:
         ax.bar(np.arange(data_shape_DK[1]), data_DK[i], bottom=data_stack_DK[i], label = 'PFR-DK - Output')
ax.set_ylabel("Energy [kWh/h]")
ax.set_xlabel("Time [Hours]")
ax.tick_params('y')
ax.set_ylim([-1*(abs(big_N)+0.05),abs(big_N)+0.05])
plt.legend()
plt.show()



fig = plt.figure() #dpi=200)
ax = plt.subplot(111)
data_N,data_shape_N,data_stack_N = cumulated_power_plot(pos_E_N_h, neg_E_N_h)
for i in np.arange(0, data_shape_N[0]):
    if i == 0:
        ax.bar(np.arange(data_shape_N[1]), data_N[i], bottom=data_stack_N[i], label = 'FCR-N - Input')
    elif i == 1:
         ax.bar(np.arange(data_shape_N[1]), data_N[i], bottom=data_stack_N[i], label = 'FCR-N - Output')
        
ax.set_ylabel("Energy [kWh/h]")
ax.set_xlabel("Time [Hours]")
ax.tick_params('y')
ax.set_ylim([-1*(abs(big_N)+0.05),abs(big_N)+0.05])
plt.legend()
plt.show()





"""

Data Visualization regarding Battery


"""


""" Plot Power Comparison """
fig = plt.figure() #dpi = 200)
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])
ax.plot(x ,  power_N, label = 'FCR-N') #, color = 'blue')
ax.plot(x ,  power_DE, label = 'PFR-DE') #, color = 'orange')
ax.plot(x ,  power_DK, label = 'PFR-DK' ) #, color = 'green')
ax.plot(x ,  power_D, label = 'FCR-D' )#, color = 'red')
ax.set_xlabel('Time', fontsize = 13)
ax.set_ylabel('Power [kW]', fontsize = 13)
#for ind in vertical_line(x):
#    plt.axvline(x=ind, color = 'grey')
plt.xticks(x[::(21600*4)], date_x[::(21600*4)], rotation = 'vertical')
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
#plt.legend(bbox_to_anchor=(1.05, 0.55), loc=2, borderaxespad=0.)
#ax.legend("title", prop=fontP)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=5)

""" Power 1 Day """

fig = plt.figure() #dpi = 200)
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])
#fontP = FontProperties()
#fontP.set_size('small')
ax.plot(x[(5*24*3600):(6*24*3600)] ,  power_N[(5*24*3600):(6*24*3600)], label = 'FCR-N') #, color = 'blue')
ax.plot(x[(5*24*3600):(6*24*3600)] ,  power_DE[(5*24*3600):(6*24*3600)], label = 'PFR-DE')
ax.plot(x[(5*24*3600):(6*24*3600)] ,  power_DK[(5*24*3600):(6*24*3600)], label = ' PFR-DK')
ax.plot(x[(5*24*3600):(6*24*3600)] ,  power_D[(5*24*3600):(6*24*3600)], label = 'FCR-D')
ax.set_xlabel('Time - 6. Feb 2017', fontsize = 13)
ax.set_ylabel('Power [kW]', fontsize = 13)
#for ind in vertical_line(x):
#    plt.axvline(x=ind, color = 'grey')
plt.xticks(x[(5*24*3600):(6*24*3600)][::(10800)], time_x[(5*24*3600):(6*24*3600)][::(10800)], rotation = 'vertical')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=5)


"""" Plot SOC Comparison """
fig = plt.figure() #dpi= 200) 
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])

ax.plot(x ,  SOC_N, label = 'FCR-N')
ax.plot(x ,  SOC_DE, label = 'PFR-DE')
ax.plot(x ,  SOC_DK, label = 'PFR-DK')
ax.plot(x ,  SOC_D, label = 'FCR-D')

ax.set_xlabel('Time', fontsize = 13)
ax.set_ylabel('SOC [p.u.]', fontsize = 13)

ax.set_ylim([0,1])
plt.xticks(x[::(21600*4)], date_x[::(21600*4)], rotation = 'vertical')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30),
          fancybox=True, shadow=True, ncol=5)


""" SOC - 1 Day """
fig = plt.figure() #dpi= 200) 
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])
ax.plot(x[(5*24*3600):(6*24*3600)] ,  SOC_N[(5*24*3600):(6*24*3600)], label = 'FCR-N')
ax.plot(x[(5*24*3600):(6*24*3600)] ,  SOC_DE[(5*24*3600):(6*24*3600)], label = 'PFR-DE')
ax.plot(x[(5*24*3600):(6*24*3600)] ,  SOC_DK[(5*24*3600):(6*24*3600)], label = 'PFR-DK')
ax.plot(x[(5*24*3600):(6*24*3600)] ,  SOC_D[(5*24*3600):(6*24*3600)], label = 'FCR-D')
ax.set_xlabel('Time - 6. Feb 2017', fontsize = 13)
ax.set_ylabel('SOC [p.u.]', fontsize = 13)

ax.set_ylim([0,1])
plt.xticks(x[(5*24*3600):(6*24*3600)][::(10800)], time_x[(5*24*3600):(6*24*3600)][::(10800)], rotation = 'vertical')
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32),
          fancybox=True, shadow=True, ncol=5)







""" Grouped Hist. DOD og DOC """
""" With SOC Threshold of 1 % """
### Set DOD to Percentage
DOD_DE_thres_p = np.array(DOD_DE_thres)*100; DOD_DK_thres_p = np.array(DOD_DK_thres)*100; DOD_N_thres_p = np.array(DOD_N_thres)*100;
DOD_DE_p = np.array(DOD_DE)*100; DOD_DK_p = np.array(DOD_DK)*100; DOD_N_p = np.array(DOD_N)*100; DOD_D_p = np.array(DOD_D)*100;



DOC_DE_p = np.array(DOC_DE)*100; DOC_DK_p = np.array(DOC_DK)*100; DOC_N_p = np.array(DOC_N)*100; 

key_DOD_DE_p = {'PFR-DE': DOD_DE_p}; df_DOD_DE_p = pd.DataFrame(key_DOD_DE_p)
key_DOD_DK_p = {'PFR-DK': DOD_DK_p}; df_DOD_DK_p = pd.DataFrame(key_DOD_DK_p)
key_DOD_N_p = {'FCR-N': DOD_N_p}; df_DOD_N_p = pd.DataFrame(key_DOD_N_p)

key_DOC_DE_p = {'PFR-DE': DOC_DE_p}; df_DOC_DE_p = pd.DataFrame(key_DOC_DE_p)
key_DOC_DK_p = {'PFR-DK': DOC_DK_p}; df_DOC_DK_p = pd.DataFrame(key_DOC_DK_p)
key_DOC_N_p = {'FCR-N': DOC_N_p}; df_DOC_N_p = pd.DataFrame(key_DOC_N_p)




    


""" No Micro Cycles Inside DOD  """

## 1 % Thres
DOD_DE_microless = DOD_calc_no_micro(SOC_DE)
DOD_DK_microless = DOD_calc_no_micro(SOC_DK)
DOD_N_microless = DOD_calc_no_micro(SOC_N)

DOC_N_microless = DOC_calc_no_micro(SOC_N)
DOC_DE_microless = DOC_calc_no_micro(SOC_DE)
DOC_DK_microless = DOC_calc_no_micro(SOC_DK)
   
DOD_DE_microless_p = np.array(DOD_DE_microless)*100; DOD_DK_microless_p = np.array(DOD_DK_microless)*100;  DOD_N_microless_p = np.array(DOD_N_microless)*100
DOC_DE_microless_p = np.array(DOC_DE_microless)*100; DOC_DK_microless_p = np.array(DOC_DK_microless)*100;  DOC_N_microless_p = np.array(DOC_N_microless)*100




"""    Monotonic Cylce Behavior  ----  Algorithm 2    """

## number of counts at each interval
dods_DE  = count_DOD_size(DOD_DE_p)
dods_DK  = count_DOD_size(DOD_DK_p)
dods_N  = count_DOD_size(DOD_N_p)


## X and Y variables for Graph
bins_N, ditch0_N = listing(dods_N, DOD_N_p)
bins_DE, ditch0_DE = listing(dods_DE, DOD_DE_p)
bins_DK, ditch0_DK = listing(dods_DK, DOD_DK_p)



## Present data on logarithmic scale
logy_N = np.log(ditch0_N)
logy_DE = np.log(ditch0_DE)
logy_DK = np.log(ditch0_DK)


## Polynomial fit for measurements --- needs to be scaled for logarithmic operation
coeffs_N = np.polyfit(bins_N,logy_N,deg=2)
coeffs_DE = np.polyfit(bins_DE,logy_DE,deg=2)
coeffs_DK = np.polyfit(bins_DK,logy_DK,deg=2)


poly_N = np.poly1d(coeffs_N)
poly_DE = np.poly1d(coeffs_DE)
poly_DK = np.poly1d(coeffs_DK)


yfit_N = lambda bins_N: np.exp(poly_N(bins_N))
yfit_DE = lambda bins_DE: np.exp(poly_DE(bins_DE))
yfit_DK = lambda bins_DK: np.exp(poly_DK(bins_DK))

## calculate R^2 to estimate goodness of fit 
r_squared_DK_micro = sk.metrics.r2_score(ditch0_DK,yfit_DK(bins_DK) )
print('DK score',r_squared_DK_micro)
r_squared_DE_micro = sk.metrics.r2_score(ditch0_DE,yfit_DE(bins_DE) )
print('DE score',r_squared_DE_micro)
r_squared_N_micro = sk.metrics.r2_score(ditch0_N, yfit_N(bins_N) )
print('N score',r_squared_N_micro)


## Plot poly. fit and scatter plot of measurements
fig = plt.figure(figsize=(8,4)) #dpi = 200)
ax = plt.gca()
plt.plot(bins_N,yfit_N(bins_N),  linewidth=2.0,
         label = 'FCR-N  \n3rd Order \nPoly. Fit  \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_N_micro,3)))
plt.scatter(bins_N, ditch0_N, color ='darkblue', label ='FCR-N \nMeasurement')

plt.plot(bins_DE,yfit_DE(bins_DE) , linewidth=2.0,
         label = 'PFR-DE \n3rd Order \nPoly. Fit \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_DE_micro,3)))
plt.scatter(bins_DE, ditch0_DE, color ='darkgreen' , marker='x' , label ='PFR-DE \nMeasurement')

plt.plot(bins_DK,yfit_DK(bins_DK), linewidth=2.0,
         label = 'PFR-DK \n3rd Order \nPoly. Fit \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_DK_micro,3)))
plt.scatter(bins_DK, ditch0_DK, color ='#800000'  ,   marker='^' , label ='PFR-DK \nMeasurement')
ax.set_ylim(0.5,1200)
ax.set_yscale('log')
ax.set_xlabel('Depth of Discharge [%]', fontsize=13)
ax.set_ylabel('Cycles', fontsize=13)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()







""" DOD ----------- Scatter and poly. fit when appying Algorithm 3  """

## number of counts at each interval
dods_DE_microless  = count_DOD_size(DOD_DE_microless_p)
dods_DK_microless  = count_DOD_size(DOD_DK_microless_p)
dods_N_microless  = count_DOD_size(DOD_N_microless_p)

## X and Y variables for Graph
bins_N_microless, ditch0_N_microless = listing(dods_N_microless, DOD_N_microless_p)
bins_DE_microless, ditch0_DE_microless = listing(dods_DE_microless, DOD_DE_microless_p)
bins_DK_microless, ditch0_DK_microless = listing(dods_DK_microless, DOD_DK_microless_p)

## Present data on logarithmic scale
logy_N_microless = np.log(ditch0_N_microless)
logy_DE_microless = np.log(ditch0_DE_microless)
logy_DK_microless = np.log(ditch0_DK_microless)


## Polynomial fit for measurements --- needs to be scaled for logarithmic operation
coeffs_N_microless = np.polyfit(bins_N_microless,logy_N_microless,deg=3)
coeffs_DE_microless = np.polyfit(bins_DE_microless,logy_DE_microless,deg=3)
coeffs_DK_microless = np.polyfit(bins_DK_microless,logy_DK_microless,deg=3)


poly_N_microless = np.poly1d(coeffs_N_microless)
poly_DE_microless = np.poly1d(coeffs_DE_microless)
poly_DK_microless = np.poly1d(coeffs_DK_microless)


yfit_N_microless = lambda bins_N_microless: np.exp(poly_N_microless(bins_N_microless))
yfit_DE_microless = lambda bins_DE_microless: np.exp(poly_DE_microless(bins_DE_microless))
yfit_DK_microless = lambda bins_DK_microless: np.exp(poly_DK_microless(bins_DK_microless))

## calculate R^2 to estimate goodness of fit
r_squared_DK_DOD = sk.metrics.r2_score(ditch0_DK_microless,yfit_DK_microless(bins_DK_microless) )
print('DK score',r_squared_DK_DOD)
r_squared_DE_DOD = sk.metrics.r2_score(ditch0_DE_microless,yfit_DE_microless(bins_DE_microless) )
print('DE score',r_squared_DE_DOD)
r_squared_N_DOD = sk.metrics.r2_score(ditch0_N_microless,yfit_N_microless(bins_N_microless) )
print('N score',r_squared_N_DOD)


## Plot poly. fit and scatter plot of measurements
fig = plt.figure(figsize=(8,4)) 
ax = plt.gca()
plt.plot(bins_N_microless,yfit_N_microless(bins_N_microless), linewidth=2.0,
           label = 'FCR-N  \n3rd Order \nPoly. Fit  \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_N_DOD,3)))
plt.scatter(bins_N_microless, ditch0_N_microless , color = 'darkblue', 
          label = 'FCR-N \nMeasurement')

plt.plot(bins_DE_microless,yfit_DE_microless(bins_DE_microless), linewidth=2.0,
        label = 'PFR-DE \n3rd Order \nPoly. Fit \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_DE_DOD,3)))
plt.scatter(bins_DE_microless, ditch0_DE_microless, color = 'darkgreen'  ,marker='x' , label = 'PFR-DE \nMeasurement')

plt.plot(bins_DK_microless,yfit_DK_microless(bins_DK_microless), linewidth=2.0,
        label = 'PFR-DK \n3rd Order \nPoly. Fit \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_DK_DOD,3)))
plt.scatter(bins_DK_microless, ditch0_DK_microless, color = '#800000',  marker='^' ,  label = 'PFR-DK \nMeasurement' ) #, label = 'PFR-DK')

ax.set_ylim(0.5,top = None)
ax.set_yscale('log')
ax.set_xlabel('Depth of Discharge [%]', fontsize=13)
ax.set_ylabel('Cycles', fontsize=13)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()








"""  DOC ----------- Scatter and poly. fit when appying Algorithm 3 """
 
## number of counts at each interval
dods_DOC_DE_microless  = count_DOD_size(DOC_DE_microless_p)
dods_DOC_DK_microless  = count_DOD_size(DOC_DK_microless_p)
dods_DOC_N_microless  = count_DOD_size(DOC_N_microless_p)

## X and Y variables for Graph
bins_DOC_N_microless, ditch0_DOC_N_microless = listing(dods_DOC_N_microless, DOC_N_microless_p)
bins_DOC_DE_microless, ditch0_DOC_DE_microless = listing(dods_DOC_DE_microless, DOC_DE_microless_p)
bins_DOC_DK_microless, ditch0_DOC_DK_microless = listing(dods_DOC_DK_microless, DOC_DK_microless_p)


## Present data on logarithmic scale
logy_N_microless_DOC = np.log(ditch0_DOC_N_microless)
logy_DE_microless_DOC = np.log(ditch0_DOC_DE_microless)
logy_DK_microless_DOC = np.log(ditch0_DOC_DK_microless)


## Polynomial fit for measurements --- needs to be scaled for logarithmic operation
coeffs_N_microless_DOC = np.polyfit(bins_DOC_N_microless,logy_N_microless_DOC,deg=2)
coeffs_DE_microless_DOC = np.polyfit(bins_DOC_DE_microless,logy_DE_microless_DOC,deg=2)
coeffs_DK_microless_DOC = np.polyfit(bins_DOC_DK_microless,logy_DK_microless_DOC,deg=2)


poly_N_microless_DOC = np.poly1d(coeffs_N_microless_DOC)
poly_DE_microless_DOC = np.poly1d(coeffs_DE_microless_DOC)
poly_DK_microless_DOC = np.poly1d(coeffs_DK_microless_DOC)


yfit_N_microless_DOC = lambda bins_DOC_N_microless: np.exp(poly_N_microless_DOC(bins_DOC_N_microless))
yfit_DE_microless_DOC = lambda bins_DOC_DE_microless: np.exp(poly_DE_microless_DOC(bins_DOC_DE_microless))
yfit_DK_microless_DOC = lambda bins_DOC_DK_microless: np.exp(poly_DK_microless_DOC(bins_DOC_DK_microless))

## calculate R^2 to estimate goodness of fit
r_squared_DK = sk.metrics.r2_score(ditch0_DOC_DK_microless,yfit_N_microless_DOC(bins_DOC_N_microless) )
print('DK score',r_squared_DK)
r_squared_DE = sk.metrics.r2_score(ditch0_DOC_DE_microless,yfit_DE_microless_DOC(bins_DOC_DE_microless) )
print('DE score',r_squared_DE)
r_squared_N = sk.metrics.r2_score(ditch0_DOC_N_microless,yfit_DK_microless_DOC(bins_DOC_DK_microless) )
print('N score',r_squared_N)

## Plot poly. fit and scatter plot of measurements
fig = plt.figure(figsize=(8,4)) 
ax = plt.gca()
plt.plot(bins_DOC_N_microless,yfit_N_microless_DOC(bins_DOC_N_microless), label = '3rd Order \n Poly. Fit \n FCR-N')
plt.scatter(bins_DOC_N_microless, ditch0_DOC_N_microless , label = 'FCR-N')

plt.plot(bins_DOC_DE_microless,yfit_DE_microless_DOC(bins_DOC_DE_microless), label = '3rd Order \n Poly. Fit \n PFR-DE')
plt.scatter(bins_DOC_DE_microless, ditch0_DOC_DE_microless, marker='x' , label = 'PFR-DE')

plt.plot(bins_DOC_DK_microless,yfit_DK_microless_DOC(bins_DOC_DK_microless), label = '3rd Order \n Poly. Fit \n PFR-DK')
plt.scatter(bins_DOC_DK_microless, ditch0_DOC_DK_microless, marker='^' , label = 'PFR-DK')

ax.set_yscale('log')
ax.set_xlabel('Charge Cycles [%]', fontsize=13)
ax.set_ylabel('Cycles', fontsize=13)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()




""" DOB ----------- Scatter and poly. fit when appying Algorithm 3  """

#Sum DOD and DOC
double_DOD_N = DOD_N_microless + DOC_N_microless
double_DOD_DE = DOD_DE_microless + DOC_DE_microless
double_DOD_DK = DOD_DK_microless + DOC_DK_microless


double_N_microless_p = np.array(double_DOD_N)*100
double_DE_microless_p = np.array(double_DOD_DE)*100
double_DK_microless_p = np.array(double_DOD_DK)*100



## number of counts at each interval
dods_double_N_microless = count_DOD_size(double_N_microless_p) 
dods_double_DE_microless = count_DOD_size(double_DE_microless_p) 
dods_double_DK_microless = count_DOD_size(double_DK_microless_p) 



## X and Y variables for Graph
bins_double_N_microless, ditch0_double_N_microless = listing(dods_double_N_microless, double_N_microless_p)
bins_double_DE_microless, ditch0_double_DE_microless = listing(dods_double_DE_microless, double_DE_microless_p)
bins_double_DK_microless, ditch0_double_DK_microless = listing(dods_double_DK_microless, double_DK_microless_p)



## Present data on logarithmic scale
logy_double_N_microless = np.log(ditch0_double_N_microless)
logy_double_DE_microless = np.log(ditch0_double_DE_microless)
logy_double_DK_microless = np.log(ditch0_double_DK_microless)

coeffs_double_N_microless = np.polyfit(bins_double_N_microless,logy_double_N_microless,deg=4)
coeffs_double_DE_microless = np.polyfit(bins_double_DE_microless,logy_double_DE_microless,deg=3)
coeffs_double_DK_microless = np.polyfit(bins_double_DK_microless,logy_double_DK_microless,deg=2)

poly_double_N_microless = np.poly1d(coeffs_double_N_microless)
poly_double_DE_microless = np.poly1d(coeffs_double_DE_microless)
poly_double_DK_microless = np.poly1d(coeffs_double_DK_microless)

yfit_double_N_microless = lambda bins_double_N_microless: np.exp(poly_double_N_microless(bins_double_N_microless))
yfit_double_DE_microless = lambda bins_double_DE_microless: np.exp(poly_double_DE_microless(bins_double_DE_microless))
yfit_double_DK_microless = lambda bins_double_DK_microless: np.exp(poly_double_DK_microless(bins_double_DK_microless))


## calculate R^2 to estimate goodness of fit
r_squared_DK = sk.metrics.r2_score(ditch0_double_DK_microless,yfit_double_DK_microless(bins_double_DK_microless) )
print('DK score',r_squared_DK)
r_squared_DE = sk.metrics.r2_score(ditch0_double_DE_microless,yfit_double_DE_microless(bins_double_DE_microless) )
print('DE score',r_squared_DE)
r_squared_N = sk.metrics.r2_score(ditch0_double_N_microless,yfit_double_N_microless(bins_double_N_microless) )
print('N score',r_squared_N)



## Plot poly. fit and scatter plot of measurements
fig = plt.figure(figsize=(8,4)) 
ax = plt.gca()
plt.plot(bins_double_N_microless,yfit_double_N_microless(bins_double_N_microless), linewidth=2.0,
                  label = 'FCR-N  \n3rd Order \nPoly. Fit  \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_N,3)))
plt.scatter(bins_double_N_microless, ditch0_double_N_microless, color ='darkblue', alpha= 1, label = 'FCR-N \nMeasurement')  
plt.plot(bins_double_DE_microless,yfit_double_DE_microless(bins_double_DE_microless), linewidth=2.0,
                  label = 'PFR-DE \n3rd Order \nPoly. Fit \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_DE,3))) 
plt.scatter(bins_double_DE_microless, ditch0_double_DE_microless, alpha= 1, color ='darkgreen', marker='x',label = 'PFR-DE \nMeasurement'  )

plt.plot(bins_double_DK_microless,yfit_double_DK_microless(bins_double_DK_microless), linewidth=2.0,
                  label = 'PFR-DK \n2nd Order \nPoly. Fit \nR\N{SUPERSCRIPT TWO} = {}'.format(round(r_squared_DK,3))) 
plt.scatter(bins_double_DK_microless, ditch0_double_DK_microless, alpha= 1, color ='#800000',marker='^', label = 'PFR-DK \nMeasurement'  )
ax.set_ylim(bottom= 0.5, top = None)
ax.set_yscale('log')
ax.set_xlabel('Depth of Discharge and Charging Periods [%]', fontsize=13)
ax.set_ylabel('Occurrences', fontsize=13)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()




""" Histogram presenting DOD with no threshold from Algorithm 2 -- Logarithmic """ 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
bins = np.linspace(0, max(DOD_N_p)+0.01 , 25)
plt.hist([DOD_N_p,DOD_DE_p,DOD_DK_p],bins, label=['FCR-N', 'PFR-DE', 'PFR-DK'])
plt.yscale('log', nonposy='clip')
plt.legend(loc='upper right')
ax.set_xlabel('Depth of Discharge [%]', fontsize=13)
ax.set_ylabel('Cycles', fontsize=13)
plt.show()





"""        Aglorithm 3 ---- Boxplot displaying cycle periods in kWH                 """

""" DOD and DOC BOXPLOT """
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plt.subplot(1,3,1)
sns.boxplot(data = df_DOB_N_kWh_m, palette='GnBu_d',orient='v')
sns.stripplot(data = df_DOB_N_kWh_m,jitter=True,color='skyblue', alpha=0.4,dodge=True)
ax1.set_ylabel('Charges and Discharges [kWh]', fontsize = '13')
ax1.set_ylim([0,max(DOD_N_kWh_m)+0.8])
plt.subplot(1,3,2)
sns.boxplot(data = df_DOB_DE_kWh_m, palette='BuGn_r' ,orient='v')
sns.stripplot(data = df_DOB_DE_kWh_m,jitter=True,color ='lightgreen' ,alpha=0.4 ,dodge=True)
ax2.set_ylim([0,max(DOD_N_kWh_m)+0.8])
plt.subplot(1,3,3)
sns.boxplot(data = df_DOB_DK_kWh_m, palette='hls',orient='v')
sns.stripplot(data = df_DOB_DK_kWh_m,jitter=True, color ='tomato',  alpha=0.4,  dodge=True)
ax3.set_ylim([0,max(DOD_N_kWh_m)+0.8])
fig.text(0.55, -0.02, 'Strategy', ha='center', fontsize = '13')
plt.tight_layout()










