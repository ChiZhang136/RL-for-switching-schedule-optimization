# RL-for-switching-schedule-optimization
This script integrates Q-learning with neural network for locating optimal switching sequences of Lotka Volterra fishing problem.  
Results show that including subsystem index as part of 'state' outperform that without the index. To avoid sparse reward, we assign the terminal reward as its derivatives per time step. Otherwise, the system tends not to switch at all.  
Numerical example reference: https://mintoc.de/index.php/Lotka_Volterra_fishing_problem
 
