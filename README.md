# Thesis_code

This link provides the python code used to achive the results attained in my masters thesis.

The program FCR_main_POP.py is the main program. 
It imports the functions FCR_D_POP.py, FCR_N_POP.py, FCR_DE_POP.py, FCR_DK_POP.py,
and droop_functions.py.

droop_functions.py includes the four frequency regulation strategies under normal operation.

The FCR_xx_POP functions are the strategies with a 1-step POP implemented. 

FCR_main_POP.py can also import the function FCR_N_POP_2step.py. It is a 2-step POP algorithm for FCR-N.


The program V2G_data.py is tasked with comparing real V2G data to the simulation. It imports the funtion FCR_N_POP_2step.py to simulate the battery SOC under the same frequency signal as an individual car in V2G data.
