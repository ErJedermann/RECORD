# Simulation Setup 1
This file describes the simulation setups and executions, required to generate all data from scratch, for a full evaluation 
of the relations between inter observer distances, different observer amounts and the estimated RoI. With the new generated 
data, a graph is created, showing this relation. The new created graph will be comparable to Figure 13 in the paper. Variations 
up to a factor of two are expected due to the published beam model.

This setup consists of four steps:
- Simulate 3 observers at different inter observer distances.
- Simulate 6 observers at different inter observer distances.
- Simulate 12 observers at different inter observer distances.
- Combine the generated data to a evaluation graph.

The total execution time of this simulation setup is ~840 hours. 

[Optional: The first three steps of running the simulation on different observer numbers can be done in parallel to reduce the 
waiting time. To do so, it is recommended to go through the following instructions step by step. When a simulation is started, 
continue with the next steps and start the following simulation in parallel.]


## Simulate 3 observers at different inter observer distances.

### Preparation:
Adapt the parameters in file `simulations_attackerTypes_fibo.py` in lines 301 - 307 in the following way:
- output_folder="my_simulations/observer_distances_and_amount"
- iterations_in = 20
- inter_obs_distance = [100, 200, 300, 400, 500, 600, 700, 800]
- durations_in = [3600]
- number_eves = 3
- noisy_prediction = True
- weak_events = True

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the simulation by executing the 
python command `python simulations_attackerTypes_fibo.py`.

### Expected Execution Time:
120 hours. It simulates 20 iterations * 3600 seconds * 8 distances * 3 eavesdroppers = 1728000 seconds.
Using weak_events = True will require ~0.25 seconds execution time per second simulation. This results in ~432000 seconds 
of execution time (~120 hours).
       
### Results:
The simulation will write the resulting RoI sizes in the previously specified sub-folder "my_simulations/observer_distances_and_amount".
It creates eight files, each containing 20 RoI size pairs. (The first number is the RoI of a single observer, the second 
number is the RoI of all observers combined.)


## Simulate 6 observers at different inter observer distances.

### Preparation:
Adapt the parameters in file `simulations_attackerTypes_fibo.py` in lines 301 - 307 in the following way:
- output_folder="my_simulations/observer_distances_and_amount"
- iterations_in = 20
- inter_obs_distance = [100, 200, 300, 400, 500, 600, 700, 800]
- durations_in = [3600]
- number_eves = 6
- noisy_prediction = True
- weak_events = True

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the simulation by executing the 
python command `python simulations_attackerTypes_fibo.py`.

### Expected Execution Time:
240 hours. It simulates 20 iterations * 3600 seconds * 8 distances * 6 eavesdroppers = 3456000 seconds.
Using weak_events = True will require ~0.25 seconds execution time per second simulation. This results in ~864000 seconds 
of execution time (~240 hours).

[Optional: It is possible to parallelize this even further to reduce the waiting time: Reduce the number of iterations to 10 and
run two simulations with this parameters in parallel to meet the waiting time of the first simulation.]
       
### Results:
The simulation will write the resulting RoI sizes in 8 files in the sub-folder "my_simulations/observer_distances_and_amount".


## Simulate 12 observers at different inter observer distances.

### Preparation:
Adapt the parameters in file `simulations_attackerTypes_fibo.py` in lines 301 - 307 in the following way:
- output_folder="my_simulations/observer_distances_and_amount"
- iterations_in = 20
- inter_obs_distance = [100, 200, 300, 400, 500, 600, 700, 800]
- durations_in = [3600]
- number_eves = 12
- noisy_prediction = True
- weak_events = True

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the simulation by executing the 
python command `python simulations_attackerTypes_fibo.py`.

### Expected Execution Time:
480 hours. It simulates 20 iterations * 3600 seconds * 8 distances * 12 eavesdroppers = 6912000 seconds.
Using weak_events = True will require ~0.25 seconds execution time per second simulation. This results in ~1728000 seconds 
of execution time (~480 hours).

[Optional: It is possible to parallelize this even further to reduce the waiting time: Reduce the number of iterations to 5 and
run four simulations with this parameters in parallel to meet the waiting time of the first simulation.]
       
### Results:
The simulation will write the resulting RoI sizes in 8 files in the sub-folder "my_simulations/observer_distances_and_amount".


## Combine the generated data to a evaluation graph

### Preparation:
Wait until all simulations are executed.
Adapt the parameters in file `graph_simulation_receiverDistances.py` in line 60 in the following way:
- folder_name = "my_simulations/observer_distances_and_amount/"

### Execution Start:
Open the virtual environment via `source venv/bin/activate`, then start the graph creation by executing the 
python command `python graph_simulation_receiverDistances.py`.

### Expected Execution Time:
2 seconds. 
       
### Results:
The script will load the previously created files from the sub-folder "my_simulations/observer_distances_and_amount" and create 
a graph, that is comparable to Figure 13 in the paper.








