# Simulation Setup 1
This file describes the simulation setups and executions, required to generate all data from scratch, for a full evaluation 
of the relations between observation duration, attacker types and the estimated RoI. With the new generated data, a graph is 
created showing the relation between the observation time and the RoI sizes on different attacker types. The new created graph 
will be comparable to Figure 10 in the paper. Variations up to a factor of two are expected due to the published beam model.

This setup consists of four steps:
- Simulate the attacker types 1 & 2.
- Simulate the attacker type 3.
- Simulate the attacker type 4.
- Combine the generated data to a evaluation graph.

The total execution time of this simulation setup is ~580 hours. 

[Optional: The first three steps of running the simulation on different attacker types can be done in parallel to reduce the 
waiting time. To do so, it is recommended to go through the following instructions step by step. When a simulation is started, 
continue with the next steps and start the following simulation in parallel.]


## Simulate the attacker types 1 & 2

### Preparation:
Adapt the parameters in file `simulations_attackerTypes_fibo.py` in lines 301 - 307 in the following way:
- output_folder="my_simulations/duration_and_type"
- iterations_in = 20
- inter_obs_distance = [100]
- durations_in = [60, 180, 600, 1800, 3600, 7200, 14400]
- number_eves = 3
- noisy_prediction = True
- weak_events = True

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the simulation by executing the 
python command `python simulations_attackerTypes_fibo.py`.

### Expected Execution Time:
116 hours. It simulates 20 iterations * (60 + 180 + 600 + 1800 + 3600 + 7200 + 14400) seconds * 3 eavesdroppers = 1670400 seconds.
Using weak_events = True will require ~0.25 seconds execution time per second simulation. This results in ~417600 seconds 
of execution time (~116 hours).
       
### Results:
The simulation will write the resulting RoI sizes in the previously specified sub-folder "my_simulations/duration_and_type".
It creates seven files, each containing 20 RoI size pairs. (The first number is the RoI of a single observer, the second 
number is the RoI of all observers combined.)


## Simulate the attacker type 3

### Preparation:
Adapt the parameters in file `simulations_attackerTypes_fibo.py` in lines 301 - 307 in the following way:
- output_folder="my_simulations/duration_and_type"
- iterations_in = 20
- inter_obs_distance = [100]
- durations_in = [60, 180, 600, 1800, 3600, 7200, 14400]
- number_eves = 3
- noisy_prediction = False
- weak_events = True

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the simulation by executing the 
python command `python simulations_attackerTypes_fibo.py`.

### Expected Execution Time:
116 hours (same calculation as above).
       
### Results:
The simulation will write the resulting RoI sizes in 7 files in the sub-folder "my_simulations/duration_and_type".


## Simulate the attacker type 4

### Preparation:
Adapt the parameters in file `simulations_attackerTypes_fibo.py` in lines 301 - 307 in the following way:
- output_folder="my_simulations/duration_and_type"
- iterations_in = 20
- inter_obs_distance = [100]
- durations_in = [60, 180, 600, 1800, 3600, 7200, 14400]
- number_eves = 3
- noisy_prediction = False
- weak_events = False

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the simulation by executing the 
python command `python simulations_attackerTypes_fibo.py`.

### Expected Execution Time:
348 hours. It simulates 20 iterations * (60 + 180 + 600 + 1800 + 3600 + 7200 + 14400) seconds * 3 eavesdroppers = 1670400 seconds.
Using weak_events = False will require ~0,75 seconds execution time per second simulation. This results in ~1252800 seconds 
of execution time (~348 hours). 

[Optional: It is possible to parallelize this even further to reduce the waiting time: Reduce the number of iterations to 7 and
run three simulations with this parameters in parallel to meet the waiting time of the other simulations.]
       
### Results:
The simulation will write the resulting RoI sizes in 7 files in the sub-folder "my_simulations/duration_and_type".


## Combine the generated data to a evaluation graph

### Preparation:
Wait until all simulations are executed.
Adapt the parameters in file `graph_simulation_attackerTypes.py` in line 60 in the following way:
- folder_name = "my_simulations/duration_and_type"

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the graph creation by executing the 
python command `python graph_simulation_attackerTypes.py`.

### Expected Execution Time:
2 seconds. 
       
### Results:
The script will load the previously created files from the sub-folder "my_simulations/duration_and_type" and create a graph, that can be 
compared to Figure 10 in the paper.








