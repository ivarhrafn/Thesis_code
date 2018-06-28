import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})
import scipy as sc 
import numpy as np
import math
from scipy.signal import argrelextrema
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from FCR_N_POP_2step import *
from scipy import stats
from scipy.stats import ks_2samp

#Features
sns.set_style('darkgrid')
plt.style.use('seaborn-deep')

#Load in V2G and frequency data
EV_data = pd.read_csv('combined.csv' , delimiter = ',')
df_hz_DK2 = pd.read_csv('DK2_Februar.csv' , delimiter = ',')


time_col0 = df_hz_DK2.iloc[:,0]



""" Split time column for dates and time """
def Split_Time(time_column):
    a = []
    b = []


    for i in time_column:
        a.append(i.split(' ')[1])
        b.append(i.split(' ')[0])
        
    return a,b



""" Rearrange Date order to compare time columns of frequency and V2G data """
def rearrange(time):
    a = []
    b = []
    c = []

    splice = []
    x_tick_new = []
    
    
    for i in time:
        a.append(i.split(' ')[1])
        b.append(i.split(' ')[0])


    for j in b:
        c.append(j.split('-'))
    
    for idx in range(len(c)):
        bb = c[idx][1], '-', c[idx][2], '-' , c[idx][0]
        splice.append(''.join(bb))
        reverse = splice[idx], ' ', a[idx]
        x_tick_new.append(''.join(reverse))
        
        
    return a, splice, x_tick_new


## Extract certain car from data
#'MCV2G03160002', 'MCV2G03160012', 'MCV2G04160002', 'MCV2G03160003',
#       'MCV2G03160004', 'MCV2G03160006', 'MCV2G03160007', 'MCV2G03160008',
#       'MCV2G04160001', 'MCV2G03160009'


""" Car 2 """
##  MCV2G03160012
EV_MCV2G03160012 = EV_data.loc[EV_data['CarName'] == 'MCV2G03160012']

#Extract data for when the car is providing service and remove error values
df_car_2 = pd.DataFrame(EV_MCV2G03160012)
df_car_2_avil = df_car_2.loc[df_car_2['Available'] == True]
df_car_2_avil = df_car_2_avil.loc[df_car_2_avil['ev_state'] == 'V2G']
df_car_2_avil = df_car_2_avil.loc[df_car_2_avil['EnergyStored'] != 21000]
df_car_2_avil = df_car_2_avil.loc[df_car_2_avil['EnergyStored'] != 0]


enery_stored = df_car_2_avil.iloc[:,7]
energy_cap = 24000

#Calculate SOC
SOC_car_2 = np.array(enery_stored) / energy_cap


#Time variables
x_tick_car2 = df_car_2_avil.iloc[:,0]
time_x_car2, date_x_car2 = Split_Time(x_tick_car2)
x = range(len(time_x_car2))








""" simulation Car 2 """

time_sim, date_sim, xticks_sim = rearrange(x_tick_car2)
df_xticks = pd.DataFrame(xticks_sim)


## Remove Dates from frequency data that are not in EV_data
df_freq_sim = df_hz_DK2[df_hz_DK2.iloc[:,0].isin(xticks_sim)]          
freq_DK2 = df_freq_sim.reset_index()
freq_DK2 = freq_DK2.iloc[:,2]
SOC_car_2 = SOC_car_2[:len(freq_DK2)]

SOC_car_2_ave = np.average(SOC_car_2.astype(float))
SOC_car_2_ave = np.full((len(SOC_car_2),) , SOC_car_2_ave)




# Run simulation
SOC_init = 0.7 #Initial SOC in p.u.
battery_max_cap = 24 #kWh ----- # Tests : 12, 24, 60 kWh
max_charge = 0.91 #Max and min capacity the battery
charging_max = 10 #Max charging in kW

power_N, kWs_N, SOC_N, POP0_count_N, POP1_count_N ,POP2_count_N, POP3_count_N ,POP4_count_N, max_count_N , min_count_N = np.array(FCR_N_POP_2step(freq_DK2, SOC_init,battery_max_cap,max_charge, charging_max))

kWh_N = np.array(kWs_N)/3600
power_N = np.array([power_N])*10
power_N = power_N.reshape(len(SOC_N),)





# Pearson correlation and regression errors
pearsonr = sc.stats.pearsonr(SOC_car_2, SOC_N)

y_true = SOC_car_2
y_pred = SOC_N
print(mean_squared_error(y_true, y_pred))
rms = math.sqrt(mean_squared_error(y_true, y_pred))
print(rms)
print(r2_score(y_true,y_pred))




#### SOC of car and simulated SOC

x_1 = range(len(freq_DK2))

fig = plt.figure() #dpi= 200) 
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])
ax.plot(x_1 ,  SOC_car_2, label = 'Car 2 - MCV2G03160012')
ax.plot(x_1, SOC_N, label = 'FCR - Simulated')
ax.set_xlabel('Time', fontsize = 12)
ax.set_ylabel('SOC [p.u.]', fontsize = 12)
#for ind in vertical_line(x):
#    plt.axvline(x=ind, color = 'grey')
ax.set_ylim([0,1])
plt.xticks(x[::(21600*4)], date_x_car2[::(21600*4)], rotation = 'vertical')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend()



## Extract certain periods
fig = plt.figure() #dpi= 200) 
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])
ax.plot(x_1[(16*24*3600):(21*24*3600)] ,  SOC_car_2[(16*24*3600):(21*24*3600)], label = 'Car 1 - MCV2G03160002', color =  'navy')
ax.plot(x_1[(16*24*3600):(21*24*3600)] ,  SOC_N[(16*24*3600):(21*24*3600)], label = 'FCR - Simulated', color = 'forestgreen')

ax.set_xlabel('Time - 16.-21. Feb 2017', fontsize = 13)
ax.set_ylabel('SOC [p.u.]', fontsize = 13)
#for ind in vertical_line(x):
#    plt.axvline(x=ind, color = 'grey')
ax.set_ylim([0,1])
plt.xticks(x_1[(16*24*3600):(21*24*3600)][::(10800)], time_x_car2[(16*24*3600):(21*24*3600)][::(10800)], rotation = 'vertical')
ax.legend(loc=3)


fig = plt.figure() #dpi= 200) 
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])
ax.plot(x_1[(7*24*3600 + 35000):(10*24*3600)] ,  SOC_car_2[(7*24*3600 + 35000):(10*24*3600)], label = 'Car 7 - MCV2G03160007', color =  'navy')
ax.plot(x_1[(7*24*3600 + 35000):(10*24*3600)] ,  SOC_N[(7*24*3600 + 35000):(10*24*3600)], label = 'FCR - Simulated', color = 'forestgreen')

ax.set_xlabel('Time - 16.-21. Feb 2017', fontsize = 13)
ax.set_ylabel('SOC [p.u.]', fontsize = 13)
#for ind in vertical_line(x):
#    plt.axvline(x=ind, color = 'grey')
ax.set_ylim([0,1])
#plt.xticks(x_1[(16*24*3600):(21*24*3600)][::(10800)], time_x_car2[(16*24*3600):(21*24*3600)][::(10800)], rotation = 'vertical')
ax.legend(loc=3)


#calulate pearson correlation and regression errors based on extraced periods
SOC_car_span = SOC_car_2[(16*24*3600):(21*24*3600)]
SOC_N_span = SOC_N[(16*24*3600):(21*24*3600)]


SOC_car_span1 = SOC_car_2[(7*24*3600 + 35000 + 10800 *3):(10*24*3600 + 10800)]
SOC_N_span1 = SOC_N[(7*24*3600 + 35000 + 10800 *3):(10*24*3600 + 10800)]
x_span1 = x_1[(7*24*3600 + 35000 + 10800 *3):(10*24*3600 + 10800)]




## Plot reduced period
fig = plt.figure() #dpi= 200) 
plt.cm.get_cmap(name=None, lut=None)
ax = fig.add_axes([0,0,1,1])
ax.plot(x_span1 ,  SOC_car_span1, label = 'Car 7 - MCV2G03160007', color =  'navy')
ax.plot(x_span1 ,  SOC_N_span1, label = 'FCR - Simulated', color = 'forestgreen')
ax.set_xlabel('Time - 17.-19. Feb 2017', fontsize = 13)
ax.set_ylabel('SOC [p.u.]', fontsize = 13)
#for ind in vertical_line(x):
#    plt.axvline(x=ind, color = 'grey')
ax.set_ylim([0,1])
plt.xticks(x_1[(7*24*3600 + 35000 + 10800 *3):(10*24*3600 + 10800)][::(10800)], 
               time_x_car2[(7*24*3600 + 35000 + 10800 *3):(10*24*3600 + 10800)][::(10800)], rotation = 'vertical')
ax.legend(loc=3)




## Vector of average SOC in order to compare RMSE
SOC_car_span_ave = np.average(SOC_car_span1.astype(float))
SOC_car_span_ave = np.full((len(SOC_car_span1),) , SOC_car_span_ave)


## Error measurements
pearsonr_span = sc.stats.pearsonr(SOC_car_span1, SOC_N_span1)
print('pearson', pearsonr_span)
y_true = SOC_car_span1
y_pred = SOC_N_span1
print(mean_squared_error(y_true, y_pred))
rms = math.sqrt(mean_squared_error(y_true, y_pred))
print(rms)
print(r2_score(y_true,y_pred))




