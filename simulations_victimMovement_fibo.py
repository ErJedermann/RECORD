import json
import logging
import random
import time
import numpy as np
import os
import csv
from shapely.geometry import Polygon as shPoly

import polygon_utilities as pu
import plot_satellite_beams
from user_position_estimator import UserPositionEstimator
from beam_model.Generic_rec_processed_beam_model import GenericRecordedProcessedModel, Polygon  # is required
from user_position_scenario_generator import UserPositionScenarioGenerator, RecordingEvents
import user_position_evaluator as eval
import user_position_setup
from datetime import datetime


def __call_estimator(estimator: UserPositionEstimator, tle_file: str, eve_lolah: (float, float, float),
                     eve_scen: dict, pre_poly: shPoly = None, victim_lolah: (float, float, float) = None,
                     debug_eval: bool = False) -> shPoly:
    est_poly = estimator.estimate_user_position(eve_pos_lolah=eve_lolah, TLE_file=tle_file, new_recordings=eve_scen,
                                                previous_polygon=pre_poly, dbg_plot=False, result_plot=False)
    if debug_eval:
        temp_valid, temp_area, temp_dist = eval.evaluate_polygon(est_poly, eve_lolah, victim_lolah)  # check for flaws
        if not temp_valid:
            print(f"WARNING: __call_estimator: found invalid estimation!")
            print(f"WARNING: __call_estimator: eve_loc: {eve_lolah}")
            print(f"WARNING: __call_estimator: victim_loc: {victim_lolah}")
            print(f"WARNING: __call_estimator: scenario: {eve_scen}")
            print(f"WARNING: __call_estimator: validity: {temp_valid} dist:{temp_dist}km, area{temp_area}km²")
            estimator.plot_beams(eve_lolah, est_poly, f"attacker with invalid victim-position ({temp_dist}km)",
                                 victim_lolah)
    return est_poly


def __execute_attacks_weak_events(estimator: UserPositionEstimator, tle_file: str,
                                  eve_locations: [(float, float, float)],
                                  eve_scenarios: [dict], victim_lolah: (float, float, float) = None,
                                  debug_eval: bool = False) -> [[float]]:
    # check that there are the same number of eavesdroppers as scenarios
    if len(eve_locations) != len(eve_scenarios):
        print(f"ERROR: execute_attacks: the number of eavesdroppers ({len(eve_locations)}) has to match the "
              f"number of eavesdropper-scenarios ({len(eve_scenarios)})!")
        exit(1)
    eve_all_polygons = []  # [eve1[atk1, atk4], eve2[atk1,..., atk4],...]
    for i in range(len(eve_locations)):
        if RecordingEvents.GENERAL_RECEIVING in list(eve_scenarios[i].keys()):
            temp_eve_lolah = eve_locations[i]
            # attacker1: single easy attacker (GENERAL_RECEIVING)
            atk1_scenario = {RecordingEvents.GENERAL_RECEIVING: eve_scenarios[i][RecordingEvents.GENERAL_RECEIVING]}
            atk1_polygon = __call_estimator(estimator, tle_file, temp_eve_lolah, atk1_scenario, None, victim_lolah,
                                            debug_eval=debug_eval)
        else:
            atk1_polygon = None
        eve_all_polygons.append([atk1_polygon])
    return eve_all_polygons


def __combine_scenarios(eve_scenarios_lst: [{}]) -> {}:
    # combines multiple eve-scenarios to one cumulative scenario
    combined_scenario = {}
    for i in range(len(eve_scenarios_lst)):
        temp_dict = eve_scenarios_lst[i]
        for key in list(temp_dict.keys()):
            if key in list(combined_scenario.keys()):
                comb_set = set(combined_scenario[key])
                comb_set.update(temp_dict[key])
                combined_scenario[key] = list(comb_set)
            else:
                combined_scenario[key] = temp_dict[key]
    return combined_scenario


def __generate_victim_movement(center: (float, float, float), circle_radius: float, duration: int, speed: float):
    # center [lon°, lat°, km], circle_radius [km], duration [sec], speed [km/sec]
    point_amount = 100
    points_lst = user_position_setup.get_many_victims_setup(amounts=point_amount, center_lolah=center, dist=None,
                                                            circle_radius=circle_radius)
    end_index = random.randint(0, len(points_lst) - 1)
    moved_locations = []
    while len(moved_locations) < duration:
        start_index = end_index
        while end_index == start_index:
            end_index = random.randint(0, len(points_lst) - 1)
        start_loc = points_lst[start_index]  # lolah
        end_loc = points_lst[end_index]
        move_distance = eval.great_circle_distance(start_loc, end_loc)
        move_duration = int(move_distance / speed)
        delta_lon = (end_loc[0] - start_loc[0]) / move_duration  # delta for the linear interpolation start -> end
        delta_lat = (end_loc[1] - start_loc[1]) / move_duration  # delta for the linear interpolation start -> end
        used_timesteps = min(duration - len(moved_locations), move_duration)
        for i in range(used_timesteps):
            temp_delta = [delta_lon * i, delta_lat * i, 0]
            temp_loc = [start_loc[0] + temp_delta[0], start_loc[1] + temp_delta[1], start_loc[2] + temp_delta[2]]
            moved_locations.append(temp_loc)
    return moved_locations


def __generate_random_waypoint_movement(center: (float, float, float), circle_radius: float, duration: int,
                                        min_speed: float = 1, max_speed: float = 13.889):
    # uses the random waypoint model from https://link.springer.com/chapter/10.1007/978-0-585-29603-6_5
    # center [lon°, lat°, km], circle_radius [km], duration [sec], speed [m/sec]
    # min_speed = average walking of human (3.6km/h), max_speed = speed of a car in a city (50 km/h)
    pause_time_max = duration * 0.05  # maximal wait 5% (3min in a 1h recording, average is 1.5 minutes)
    point_amount = 100
    points_lst = user_position_setup.get_many_victims_setup(amounts=point_amount, center_lolah=center, dist=None,
                                                            circle_radius=circle_radius)
    end_index = random.randint(0, len(points_lst) - 1)
    moved_locations = []
    while len(moved_locations) < duration:
        start_index = end_index
        while end_index == start_index:
            end_index = random.randint(0, len(points_lst) - 1)
        start_loc = points_lst[start_index]  # lolah
        end_loc = points_lst[end_index]
        move_distance = eval.great_circle_distance(start_loc, end_loc)
        move_speed = random.random() * abs(max_speed - min_speed) + min_speed
        move_speed_kms = move_speed / 1000  # m/s -> km/s
        move_duration = int(move_distance / move_speed_kms)
        delta_lon = (end_loc[0] - start_loc[0]) / move_duration  # delta for the linear interpolation start -> end
        delta_lat = (end_loc[1] - start_loc[1]) / move_duration  # delta for the linear interpolation start -> end
        used_timesteps = min(duration - len(moved_locations), move_duration)
        for i in range(used_timesteps):
            temp_delta = [delta_lon * i, delta_lat * i, 0]
            temp_loc = [start_loc[0] + temp_delta[0], start_loc[1] + temp_delta[1], start_loc[2] + temp_delta[2]]
            moved_locations.append(temp_loc)
        wait_timesteps = random.randrange(0, int(pause_time_max))
        wait_timesteps = min(duration - len(moved_locations), wait_timesteps)
        for i in range(wait_timesteps):
            moved_locations.append(end_loc)
    return moved_locations


def __evaluate_results(eves_locs: [(float, float, float)], all_results: [[shPoly]],
                       victim_all_locs: [(float, float, float)]):
    # combine the polygons
    eve1_lolah = eves_locs[0]
    eve1_itrs = pu.lonLatHeight_2_ITRS(eve1_lolah[0], eve1_lolah[1], eve1_lolah[2])
    single_poly = all_results[0][0]  # use the first eve atk1 for single-measurement
    combined_poly = all_results[0][-1]  # start with first eve, fully combined measurement
    for i_eve in range(len(all_results)):
        temp_result = all_results[i_eve]  # eve_result = (atk1_poly, atk4_poly)
        temp_eve = eves_locs[i_eve]
        eve_t_itrs = pu.lonLatHeight_2_ITRS(temp_eve[0], temp_eve[1], temp_eve[2])
        atk4_translated = pu.recenter_poly(eve_t_itrs, eve1_itrs, temp_result[-1])
        combined_poly = pu.polygons_intersection([combined_poly, atk4_translated])
    # strategy to evaluate the polygon with respect to the victim-movement:
    # 1. evaluate the final position
    # 2. evaluate the most east / west / north / south position
    # only when all positions are included, the estimation is valid
    valid_s, area_s, dist_s = eval.evaluate_polygon(single_poly, eve1_lolah, victim_all_locs[-1])  # use only last loc
    victim_all_locs = np.array(victim_all_locs)
    min_lon_index = np.argmin(victim_all_locs[:, 0])
    max_lon_index = np.argmax(victim_all_locs[:, 0])
    min_lat_index = np.argmin(victim_all_locs[:, 1])
    max_lat_index = np.argmax(victim_all_locs[:, 1])
    test_indices = [min_lon_index, max_lon_index, min_lat_index, max_lat_index]
    valid_c_corners = True
    area_c = 0
    dist_c = 0
    for temp_index in test_indices:
        valid_c_temp, area_c, dist_c_temp = eval.evaluate_polygon(combined_poly, eve1_lolah,
                                                                  victim_all_locs[temp_index])
        valid_c_corners = (valid_c_corners and valid_c_temp)
        if dist_c_temp > dist_c:
            dist_c = dist_c_temp
    valid_c_end, area_c, dist_c_end = eval.evaluate_polygon(combined_poly, eve1_lolah, victim_all_locs[-1])
    return valid_s, area_s, dist_s, valid_c_corners, area_c, dist_c, valid_c_end, dist_c_end, combined_poly


def write_scenario_json(filename: str, eve_locs: [(float, float, float)], victim_locs: [(float, float, float)],
                        duration: int, time_start: float, eve_amount: int, eve_distances: float, victim_type: int,
                        victim_r: float, victim_s: float,
                        area: float, distance_outside: float, polygon: shPoly):
    # convert the shPoly to a list of 2D-point-lists
    lst_2D_point_arrs = pu.get_shPoly_points_2D(polygon)
    lst_2D_point_lsts = []
    for temp_arr in lst_2D_point_arrs:
        temp_lst = []
        for temp_tuple in temp_arr:
            temp_lst.append(list(temp_tuple))
        lst_2D_point_lsts.append(temp_lst)
    victim_locs_2 = []
    for temp_vloc in victim_locs:
        victim_locs_2.append(list(temp_vloc))
    eve_locs_2 = []
    for temp_eloc in eve_locs:
        eve_locs_2.append(list(temp_eloc))
    setup_dict = {'duration': duration, 'start_time': time_start, 'eve_amount': eve_amount,
                  'inter_eve_distances': eve_distances, 'victim_type': victim_type, 'victim_movement_radius': victim_r,
                  'victim_speed': victim_s, 'RoI_area': area, 'distance_outside': distance_outside,
                  'eve_locations': eve_locs_2, 'victim_locations': victim_locs_2, 'polygon': lst_2D_point_lsts}
    setup_dict_str = json.dumps(setup_dict)  # , indent=4) + "\n"  # better readability
    with open(filename, 'w') as file_output:
        file_output.writelines(setup_dict_str)
    file_output.close()


def write_output_csv(filename: str, data_points: [float]):
    # Adds data to a csv-file. Expected data_points: [atk1area_new, atk2area_new, atk3area_new].
    filename = filename
    if os.path.exists(filename):
        # append to existing file
        with open(filename, 'a') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            # writer.writerows(data_points)
            writer.writerow(data_points)
        csvoutput.close()
    else:
        # create new file and create one line per data-point
        with open(filename, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            # writer.writerows(data_points)
            writer.writerow(data_points)
        csvoutput.close()


def __analyze_moving_victim(eve_distances: float, sniffing_duration: int, iterations: int, eve_amounts: int = 6,
                            victim_type: int = 1, area_output_folder: str = None,
                            noisy_prediction: bool = False, debug: bool = True, use_point_estimator: bool = False,
                            victim_movement_radius: float = 0.01,
                            victim_movement_speed: float = 1, save_result: bool = True):
    if area_output_folder is None:
        print(f"Warning: Ms_VM: an output-folder should be given to store the results of the computation.")
    else:
        if not os.path.exists(area_output_folder):
            os.makedirs(area_output_folder)
        if not os.path.isdir(area_output_folder):
            print(f"ERROR: Ms_VM: given output-folder ({area_output_folder}) is not a folder!")
            exit(1)
    # now start the computation
    grpm_file = "beam_model/beamModel_iridium.npy"
    noisy_grpm_file = "beam_model/beamModel_iridium_noisy.npy"
    tle_file = "beam_model/iridium_TLE_2022_02_14.tle"
    time_steps = 1  # sec
    time_min = 1644796800  # start of 14th Jan 2022
    time_max = 1644883200  # end of 14th Jan 2022
    all_events_list = [RecordingEvents.GENERAL_RECEIVING, RecordingEvents.SUDDEN_RECEIVING, RecordingEvents.SUDDEN_NOT,
                       RecordingEvents.NOT_AFTER_HAND, RecordingEvents.CONTINUOUS_HAND, RecordingEvents.NOT_DURING_COMM]
    scenarioGen = UserPositionScenarioGenerator(grpm_file, tle_file)
    if noisy_prediction:
        estimator = UserPositionEstimator(noisy_grpm_file)
    else:
        estimator = UserPositionEstimator(grpm_file)
    invalid_counter = 0
    print(
        f"Ms_VM: starting with {eve_amounts} eves @ {eve_distances}km, {sniffing_duration}sec, {victim_movement_radius}km victim movement .")
    it_counter = 0
    while it_counter < iterations:
        dateTimeObj = datetime.now()
        time_str = f"{dateTimeObj.hour}:{dateTimeObj.minute}:{dateTimeObj.second}"
        print(f"Ms_VM: {time_str} i:{it_counter + 1}/{iterations}")
        victim_loc, eves_locs = user_position_setup.get_large_scale_setup(eve_amounts=eve_amounts,
                                                                          inter_eve_dist=eve_distances,
                                                                          victim_loc=victim_type)
        # generate the movement of the victim around this initial victim_location
        victim_movement_locs = __generate_random_waypoint_movement(center=victim_loc, duration=sniffing_duration,
                                                                   circle_radius=victim_movement_radius,
                                                                   max_speed=victim_movement_speed)
        start_time = np.random.rand() * (time_max - time_min) + time_min
        scenarios = []
        for temp_eve in eves_locs:
            scen_lst = []
            for i in range(sniffing_duration):
                t_start = start_time + i
                t_end = t_start + 1
                scen_lst.append(scenarioGen.generate_scenario(temp_eve, victim_movement_locs[i], all_events_list,
                                                              t_start, t_end, 1))
            comb_scen = __combine_scenarios(scen_lst)
            scenarios.append(comb_scen)
        # ensure there is at least one successful recording event
        at_least_one_recorded = False
        for temp_scen in scenarios:
            if RecordingEvents.GENERAL_RECEIVING in list(temp_scen.keys()):
                at_least_one_recorded = True
                break
        if at_least_one_recorded:
            it_counter += 1
            victim_loc = victim_movement_locs[-1]  # use the last location of the victim for verification
            # execute the attackers
            all_results = __execute_attacks_weak_events(estimator=estimator, tle_file=tle_file,
                                                        eve_locations=eves_locs, eve_scenarios=scenarios,
                                                        victim_lolah=victim_loc, debug_eval=False)
            # check the results for validity
            foo = __evaluate_results(eves_locs, all_results, victim_movement_locs)
            valid_s, area_s, dist_s, valid_c_corners, area_c, dist_c, valid_c_end, dist_c_end, poly_c = foo
            if (not valid_c_corners) or (not valid_c_end):
                print(f"Ms_VM: invalid estimation! estimated area: {area_c}km², location outside: {dist_c}km")
                invalid_counter += 1
                figur_name = f"invalid combined estimation ({dist_c}km, {area_c}km²)"
            else:
                figur_name = f"valid combined estimation ({area_c}km²)"
            if debug:
                plot_satellite_beams.plot_polygon_manyEves_manyVic(eves_lolah=eves_locs, figure_name=figur_name,
                                                                   victims_lolah=victim_movement_locs, poly=poly_c)
            # write the output-files
            if area_output_folder is not None:
                if area_output_folder[-1] != os.sep:
                    area_output_folder = f"{area_output_folder}{os.sep}"
                filename = f"{victim_movement_radius}km_RWP_{eve_distances}km_cont_{sniffing_duration}sec_{eve_amounts}_sun_eves_{victim_type}VicType"
                if noisy_prediction:
                    filename = filename + f"_noisyPrediction"
                output_path = area_output_folder + filename + ".csv"
                write_output_csv(filename=output_path, data_points=[area_s, area_c, dist_c, dist_c_end])
            if use_point_estimator and area_output_folder is not None:
                output_path = area_output_folder + filename + "_pointEstimator" + ".csv"
                pE_dist_s = 0
                pE_dist_c = eval.evaluate_point_estimator(poly_c, eves_locs[0], victim_loc)
                write_output_csv(filename=output_path, data_points=[area_s, pE_dist_s, area_c, pE_dist_c])
            if area_output_folder is not None and save_result:
                timestamp = int(time.time())
                output_path = area_output_folder + "estimations/" + filename + str(timestamp) + ".json"
                write_scenario_json(filename=output_path, eve_locs=eves_locs, victim_locs=victim_movement_locs,
                                    duration=sniffing_duration, time_start=start_time, eve_amount=eve_amounts,
                                    eve_distances=eve_distances, victim_type=victim_type,
                                    victim_r=victim_movement_radius, victim_s=victim_movement_speed, area=area_c,
                                    distance_outside=max(dist_c, dist_c_end), polygon=poly_c)
    if debug:
        print(f"INFO: Ms_VM: invalid_counter={invalid_counter} ({invalid_counter / iterations * 100}%)")


if __name__ == '__main__':
    # set the logging level for shapely (critical > error >  > warning = warn > info > debug > not_set)
    logging.getLogger('shapely.geos').setLevel(logging.WARNING)
    t1 = time.time()
    output_folder = "simulation_data/victim_movement"
    iterations_in = 100
    inter_obs_distance = 400  # fix this to 400 to focus on the effect of target movement
    durations_in = 3600  # 1h sniffing
    number_eves = 6
    noisy_prediction = True
    target_movement_radius = 2  # km
    target_movement_speed = 50000 / 3600  # m/sec
    point_estimator = False

    __analyze_moving_victim(eve_distances=inter_obs_distance, sniffing_duration=durations_in, iterations=iterations_in,
                            eve_amounts=number_eves, area_output_folder=output_folder,
                            noisy_prediction=noisy_prediction, debug=False, use_point_estimator=point_estimator,
                            victim_movement_radius=target_movement_radius,
                            victim_movement_speed=target_movement_speed, save_result=True)

    t2 = time.time()
    print(f"started at {t1}, until {t2}. took {t2 - t1} sec")
