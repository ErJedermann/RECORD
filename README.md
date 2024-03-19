# RECORD: Simulation Code & Evaluation Graphs
This repository is associated with the paper *RECORD: A RECeption-Only Region Determination Attack on LEO Satellite Users* 
(https://www.usenix.org/conference/usenixsecurity24/presentation/jedermann) by Eric Jedermann, Martin Strohmeier, 
Vincent Lenders and Jens Schmitt. If you use this code, please cite our paper.

The repo holds the code that is used in the paper for the simulations (generate observation events, calculate the region of interest 
(RoI) based on the events and evaluate the RoI), the generated simulation data and the code for generating the 
evaluation graphs.

## Installation
For installing the dependencies run `pip install -r requirements.txt`. After this the python scripts `graph...py` and
`simulations...py` can be executed.

## Simulation scripts
The two scripts `simulations_attackerTypes_fibo.py` and `simulations_victimMovement_fibo.py` are responsible for
executing all the simulations, performed in the paper.
There are a number of parameters that are used to create the different scenarios covered by the paper:
   - `output_folder`: Path to store the result files. The simulation stores the results of each independent iteration.
                      With the `graph...py` files, the resulting data are loaded afterward to generate the graphs.
   - `iterations_in`: Number of iterations to repeat each scenario.
   - `inter_obs_distance`: The distances (in km) between neighbouring observers. In `simulations_attackerTypes_fibo.py`
                           it is possible to specify a list of distances, so a series of independent scenarios with
                           different inter-observer distances will be simulated sequentially.
   - `durations_in`: The duration (in seconds) of the simulated observation time. The simulation will calculate the
                     observation events in steps of 1 sec. In `simulations_attackerTypes_fibo.py` it is possible to
                     specify a list of durations, so a series of independent scenarios with different observation periods
                     will be simulated sequentially.
   - `number_eves`: How many observers (eavesdroppers) are simulated. The distance between the observers is given in the
                    parameter `inter_obs_distance`.
   - `noisy_prediction`: Specifies if a noisy (or noiseless) antenna model is used for RoI estimation (see section 5.1).
   - `point_estimator`: Specifies if the point estimator (as an additional step) is applied after the RoI estimation.

The `simulations_attackerTypes_fibo.py` script has some additional parameters:
   - `weak_events`: Specifies if weak or strong events are used for the RoI estimation (see section 5.1 in the paper).
   - `starlink_simulations`: If True, the Starlink beam model, Starlink satellites and Starlink specific simulation
                             parameters are loaded. (This option was introduced for convenience.)
   - `iter_border`: Specifies how many observers need to have some observation events (checks only if observation events
                    exist, not how many). This becomes important when the antenna beam gets small (Starlink) or the area
                    covered by the observers gets large (many observers or large inter-observer-distances).

The `simulations_victimMovement_fibo.py` script has some additional parameters:
   - `target_movement_radius`: The movement radius (in km), in which the target will wander around. In the paper, the
                               diameter is varied between 1 and 8 km (see section 5.5).
   - `target_movement_speed`: The maximal movement speed (in m/sec) of the target, while wandering around.

## Simulation procedure & involved scripts
For each individual simulation, the scripts `simulations_attackerTypes_fibo.py` and `simulations_victimMovement_fibo.py`
contain all important parameters (as described above). First they call the `user_position_setup.py` script (which uses the
`user_position_setup_sunflower.py` script) to create a setup of observer- and target-locations. With the specified
locations, a random time window (according to the date of the TLE data) is selected. The
`user_position_scenario_generator.py` script is used to generate the observation events for each previously specified
observation location. The observer locations and observation events are forwarded to the `user_position_estimator.py`
script, which does the main job of calculating the possible RoI. After calculating the RoI, the
`user_position_evaluator.py` script is used to evaluate the size of the resulting RoI.

- The `polygon_utilities.py` script provides useful polygon operations and is used in various other scripts. The RoI is 
stored as Polygon or MultiPolygon from the shapely-library. 
- The `TLE_calculator.py` script is used by the `user_position_scenario_generator.py` and `user_position_estimator.py`
scripts to calculate the positions of the satellites, based on the given TLEs. 
- The `generic_satellite.py` script is also used by the `user_Position_Scenario_Generator.py` and the 
`user_position_estimator.py` scripts. It provides some utility functions for calculating satellite positions. 
- The `plot_satellite_beams.py` script provides plotting utilities for visualizing satellite beams. This can be used in 
various scripts for debugging or illustration purposes. 
- The `plot_measurement_row.py` script generates the graphs in the paper. 

The scripts and files in the `beam_model` sub-folder are handling the beam model calculations:
- `beamModel_iridium.npy` is the beam model which is used to represent the ground truth beam patterns of the Iridium 
satellites (it is used during the creation of the observation events). The original beam model in the paper was built by
the procedures described in Section 3.1 and 4.2 of the paper. However, this is not the original beam model which was used in the 
paper and does not precisely represent the beam pattern of the real world Iridium satellites. Since the original beam 
model in the paper was representing the beam pattern of the real world Iridium satellites, it could be used to execute 
the RECORD attack in the real world. To give interested readers the possibility to perform their own simulations, but 
preventing an abuse of the beam model we published a beam model that was altered by adding some noise to the original 
model.
- `beamModel_iridium_noisy.npy` is a noisy version of the ground truth beam model above. This represents the imperfect 
beam model of the attacker. A normal distributed noise with a mean of 1 and a std of 0.1 was added. A noise of 1 is 
equivalent to 6.8 km error in the footprint, which is equivalent to one second of satellite movement.
- `Generic_rec_processed_beam_model.py` loads the beam models and calculates the beam footprints.
- `beamModel_coordinate_transformations.py` is a helping script for the `generic_rec_processed_beam_model.py` to handle 
the coordinate transformations between the model internal coordinates and the ITRS coordinate system.

## Data Collection
The results of a simulation are stored in the `simulation_data` folder. In this folder are several sub-folders, with 
simulation data ordered by different varying parameters. Inside the sub-folders the result files are stored. There are 
two types of result files:
   - The area of the RoI (in km²) is stored, which is the main evaluation criteria in the paper. In the file name, the
     parameters of the simulation are encoded (e.g. 100kmFibo_cont_60sec_3eves_weakEvents_noisyPrediction.csv):
      - `100kmFibo`: The distance (100 km) between the observers.
                     The observer-placement strategy is the Fibonacci grid method (see section 5.1)
      - `cont_60sec`: The simulation simulated a continuous observation time (60 seconds).
                      Alternative: `120x_30sec_rec_600sec_pause`: Fragmented observations with 120 fragments, each
                      fragment has 30 seconds observation time and 600 seconds pause between fragments (section 5.4).
      - `3eves`: The amount of observers used in the simulation (3).
      - `weakEvents`: This tag indicates that only weak observation events were used (see section 5.1)
      - `noisyPrediction`: This tag indicates that the noisy beam model was used for calculating the RoI (section 5.1).
     Inside the file each line is an independent iteration. The first number is the RoI area (in km²) of a single
     observer, while the second number is the combined RoI area (in km²) of all observers (of this iteration) together.
   - The point estimator: it gives the centroid of the RoI as an estimation for the targets location (section 4.6.2).
     Again, we encoded parameters of the simulation in the file name and used the same principle as described above
     (e.g. 100kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction_pointEstimator.csv):
     The `pointEstimator` tag indicates that the point estimator was applied and evaluated on the resulting RoIs. Inside
     the file each line is an independent iteration. The first number is the RoI area (in km²) of a single observer, the
     second number is the distance (in km) between the estimated point of this single observer RoI and the simulated
     target location. The third number is the combined RoI area (in km²) of all observers (of this iteration) together
     and the fourth number is the distance (in km) between the estimated point of this combined observers RoI and the
     simulated target location.

## Graphics
The graph...py files are used to evaluate the simulation results and to create the graphs in the paper. They load the
data from the simulation_data folder and print them using the `plot_measurement_row.py` script.

- `graph_simulation_attackerTypes.py`: Creates Figure 10 (chapter 5.2).
- `graph_simulation_long_attackerTypes.py`: Creates Figure 12 (chapter 5.2).
- `graph_simulation_paper_vs_published.py`: Compares the performance of the published beam model 
`beam_model/beamModel_iridium.npy` with the original beam model that was used in the paper. We did not publish the 
original model to avoid a simple tracking of Iridium devices in the real world, as mentioned in section 6.6 in the paper.
- `graph_simulation_receiverDistances.py`: Creates Figure 13 (chapter 5.3).
- `graph_simulation_receiverIntervals.py`: Creates the graph for the textual evaluation of fragmented observations 
(chapter 5.4).
- `graph_simulation_victimMovement.py`: Creates Figure 14 (chapter 5.5).
- `graph_simulation_vs_realWorld.py`: Creates Figure 11 (chapter 5.2).

## Published Beam Model
Comparison of the median RoI sizes [km²] of the paper vs the published beam model with attacker type 2
(`inter_obs_distance`=100 km, `number_eves`=3, `noisy_prediction`=true, `weak_events`=true):

| durations_in | paper  | published |
|--------------|--------|-----------|
| 1 min        | 307 k  | 460 k     |
| 3 min        | 122 k  | 161 k     |
| 10 min       | 25.3 k | 46.2 k    |
| 30 min       | 3.48 k | 9.89 k    |
| 1 h          | 700    | 1070      |
| 2 h          | 154    | 314       |
| 4 h          | 92     | 208       |
| 8 h          | 60.8   | 52.9      |
| 16 h         | 48.5   | 36        |

This shows that the resulting RoI produced by the published beam model can be compared to the original beam model in the
paper. They are not precisely matching the original model, which is expected since we added some noise to the data.
Still it is partially comparable as the results are in the same order of magnitude and the RoIs are decreasing with
increasing observation time.
