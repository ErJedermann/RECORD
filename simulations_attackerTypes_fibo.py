import logging
import time
import numpy as np
import os
import csv

from shapely.geometry import Polygon as shPoly

import polygon_utilities as pu
from user_position_estimator import UserPositionEstimator
from beam_model.Generic_rec_processed_beam_model import GenericRecordedProcessedModel, Polygon  # it is required
from user_position_scenario_generator import UserPositionScenarioGenerator, RecordingEvents
import user_position_evaluator as eval
import user_position_setup
from datetime import datetime


def __calc_not_during_comm_events(eve_scenarios: [dict]) -> [dict]:
    ndc_dicts_list = []
    for i in range(len(eve_scenarios)):
        ndc_events = []
        main_eve = eve_scenarios[i]
        if RecordingEvents.GENERAL_RECEIVING not in list(main_eve.keys()):
            continue
        major_gen_rec = main_eve[RecordingEvents.GENERAL_RECEIVING]
        for j in range(len(eve_scenarios)):
            if j != i:
                minor_eve = eve_scenarios[j]
                if RecordingEvents.GENERAL_RECEIVING not in list(minor_eve.keys()):
                    continue
                minor_gen_rec = minor_eve[RecordingEvents.GENERAL_RECEIVING]
                diff_list = list(set(minor_gen_rec) - set(major_gen_rec))
                if len(diff_list) > 0:
                    ndc_events.extend(diff_list)
        ndc_events = list(sorted(set(ndc_events)))
        ndc_dict = {RecordingEvents.NOT_DURING_COMM: ndc_events}
        ndc_dicts_list.append(ndc_dict)
    return ndc_dicts_list


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
            print(f"WARNING: __call_estimator: validity: {temp_valid} distance:{temp_dist}km, area{temp_area}km²")
            estimator.plot_beams(eve_lolah, est_poly, f"attacker with invalid victim-position ({temp_dist}km)",
                                 victim_lolah)
    return est_poly


# executing the single and full attackers
def __execute_attacks(estimator: UserPositionEstimator, tle_file: str, eve_locations: [(float, float, float)],
                      eve_scenarios: [dict], victim_lolah: (float, float, float) = None,
                      debug_eval: bool = False) -> [[float]]:
    # check that there are the same number of eavesdroppers as scenarios
    if len(eve_locations) != len(eve_scenarios):
        print(f"ERROR: execute_attacks: the number of eavesdroppers ({len(eve_locations)}) has to match the "
              f"number of eavesdropper-scenarios ({len(eve_scenarios)})!")
        exit(1)
    # calculate the NotDuringComm dictionaries for all
    ndc_dicts = __calc_not_during_comm_events(eve_scenarios)
    eve_all_polygons = []  # [eve1[atk1, atk2, atk3, atk4], eve2[atk1,..., atk4],...]
    for i in range(len(eve_locations)):
        temp_eve_lolah = eve_locations[i]
        # attacker1: single easy attacker (GENERAL_RECEIVING, SUDDEN_RECEIVING, SUDDEN_NOT)
        atk1_scenario = eve_scenarios[i].copy()
        eve1_nah_times = atk1_scenario.pop(RecordingEvents.NOT_AFTER_HAND, [])
        eve1_chand_times = atk1_scenario.pop(RecordingEvents.CONTINUOUS_HAND, [])
        atk1_polygon = __call_estimator(estimator, tle_file, temp_eve_lolah, atk1_scenario, None, victim_lolah,
                                        debug_eval=debug_eval)
        atk4_update_dict = {RecordingEvents.NOT_AFTER_HAND: eve1_nah_times,
                            RecordingEvents.CONTINUOUS_HAND: eve1_chand_times,
                            RecordingEvents.NOT_DURING_COMM: ndc_dicts[i][RecordingEvents.NOT_DURING_COMM]}
        atk4_polygon = __call_estimator(estimator, tle_file, temp_eve_lolah, atk4_update_dict, atk1_polygon,
                                        victim_lolah, debug_eval=debug_eval)
        eve_all_polygons.append([atk1_polygon, atk4_polygon])
    return eve_all_polygons


# more practical attackers to compare with real measurements:
# attacker 1 (simple, single): only general_receiving
def __execute_attacks_weak_events(estimator: UserPositionEstimator, tle_file: str,
                                  eve_locations: [(float, float, float)],
                                  eve_scenarios: [dict], victim_lolah: (float, float, float) = None,
                                  debug_eval: bool = False) -> [[float]]:
    # check that there are the same number of eavesdroppers as scenarios
    if len(eve_locations) != len(eve_scenarios):
        print(f"ERROR: execute_attacks: the number of eavesdroppers ({len(eve_locations)}) has to match the "
              f"number of eavesdropper-scenarios ({len(eve_scenarios)})!")
        exit(1)
    # calculate the NotDuringComm dictionaries for all
    eve_all_polygons = []  # [eve1[atk1, atk2, atk3, atk4], eve2[atk1,..., atk4],...]
    for i in range(len(eve_locations)):
        temp_eve_lolah = eve_locations[i]
        # attacker1: single easy attacker (GENERAL_RECEIVING)
        if RecordingEvents.GENERAL_RECEIVING in list(eve_scenarios[i].keys()):
            atk1_scenario = {RecordingEvents.GENERAL_RECEIVING: eve_scenarios[i][RecordingEvents.GENERAL_RECEIVING]}
            atk1_polygon = __call_estimator(estimator, tle_file, temp_eve_lolah, atk1_scenario, None, victim_lolah,
                                            debug_eval=debug_eval)
            eve_all_polygons.append([atk1_polygon])
        else:
            print(f"DEBUG: no GenRec-Events -> insert None-polygon")
            eve_all_polygons.append([None])
    return eve_all_polygons


def __write_output_csv(filename: str, data_points: [float]):
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


def __analyze_attackers_generic(distances: [int], sniffing_durations: [int], iterations: int, eve_amounts: int = 2,
                                do_snf_intervals: bool = False, inter_int_time: [int] = None,
                                int_rep: [int] = None, area_output_folder: str = None, weak_events: bool = False,
                                noisy_prediction: bool = False, use_point_estimator: bool = False,
                                iteration_acceptance_border: int = 1) -> dict:
    # a generic method that is able to execute all setups variations: varying distance, cont_time, interval_time
    # ensure that either the distances or the sniffing_durations is no vector
    if len(distances) > 1 and len(sniffing_durations) > 1:
        print(f"ERROR: generic: too many inputs: len(distances)={len(distances)} and len(sniffing_durations)="
              f"{len(sniffing_durations)}. One of both has to have one single entry (len = 1) to avoid too long "
              f"running simulations.")
        exit(1)
    if do_snf_intervals:
        if inter_int_time is None:
            print(f"ERROR: generic: when sniffing-intervals are activated, a inter-sniffing-interval-time "
                  f"(pause time) must be given!")
            exit(1)
        if int_rep is None:
            print(f"ERROR: generic: when sniffing-intervals are activated, a interval-repetition must be given!")
            exit(1)
        if len(sniffing_durations) != len(inter_int_time) or len(sniffing_durations) != len(int_rep):
            print(
                f"ERROR: generic: when sniffing-intervals are activated, the number of given sniffing-durations "
                f"({len(sniffing_durations)}), inter-interval-times ({len(inter_int_time)}) and interval-repetitions "
                f"({len(int_rep)}) must match!")
            exit(1)
    if area_output_folder is None:
        print(f"Warning: generic: an output-folder should be given to store the results of the computation.")
    else:
        if not os.path.exists(area_output_folder):
            os.makedirs(area_output_folder)
        if not os.path.isdir(area_output_folder):
            print(f"ERROR: generic: given output-folder ({area_output_folder}) is not a folder!")
            exit(1)
    # now start the computation
    bm_file = "beam_model/beamModel_iridium.npy"
    noisy_bm_file = "beam_model/beamModel_iridium_noisy.npy"
    tle_file = "beam_model/iridium_TLE_2022_02_14.tle"
    time_steps = 1  # sec
    time_min = 1644796800  # start of 14th Jan 2022
    time_max = 1644883200  # end of 14th Jan 2022
    all_events_list = [RecordingEvents.GENERAL_RECEIVING, RecordingEvents.SUDDEN_RECEIVING, RecordingEvents.SUDDEN_NOT,
                       RecordingEvents.NOT_AFTER_HAND, RecordingEvents.CONTINUOUS_HAND]
    scenarioGen = UserPositionScenarioGenerator(bm_file, tle_file)
    if noisy_prediction:
        estimator = UserPositionEstimator(noisy_bm_file)
    else:
        estimator = UserPositionEstimator(bm_file)
    print(f"starting with {distances}km, {sniffing_durations}sec and {eve_amounts}eves.")
    for it_dist in range(len(distances)):
        temp_dist = distances[it_dist]
        for it_time in range(len(sniffing_durations)):
            temp_duration = sniffing_durations[it_time]
            atk1s_results_list = []  # areas of attacker1 for single+simple performance measurement
            atk4c_results_list = []  # areas of attacker4 for multi+advanced performance measurement
            it_counter = 0
            while it_counter < iterations:
                # do a debug-output, so the user can see, if the program is running
                dateTimeObj = datetime.now()
                time_str = f"{dateTimeObj.hour}:{dateTimeObj.minute}:{dateTimeObj.second}"
                print(
                    f"generic: {time_str}: d:{it_dist + 1}/{len(distances)} t:{it_time + 1}/{len(sniffing_durations)} "
                    f"i:{it_counter + 1}/{iterations}")
                # get a new scenario, using the fibonacci (sunflower) method
                victim_loc, eves_locs = user_position_setup.get_large_scale_setup(eve_amounts=eve_amounts,
                                                                                  inter_eve_dist=temp_dist)
                if do_snf_intervals:
                    start_time = np.random.rand() * (time_max - time_min) + time_min
                    end_time = start_time + int_rep[it_time] * temp_duration + \
                               (int_rep[it_time] - 1) * inter_int_time[it_time]
                    scenarios = []
                    for temp_eve in eves_locs:
                        temp_scen = {}
                        for i in range(int_rep[it_time]):
                            t_start = start_time + i * (temp_duration + inter_int_time[it_time])
                            t_end = t_start + temp_duration
                            temp_scen.update(scenarioGen.generate_scenario(temp_eve, victim_loc, all_events_list,
                                                                           t_start, t_end, time_steps))
                        scenarios.append(temp_scen)
                else:
                    start_time = np.random.rand() * (time_max - time_min) + time_min
                    end_time = start_time + temp_duration
                    scenarios = []
                    for temp_eve in eves_locs:
                        temp_scen = scenarioGen.generate_scenario(temp_eve, victim_loc, all_events_list, start_time,
                                                                  end_time, time_steps, debug=False)
                        scenarios.append(temp_scen)
                # ensure that at enough observers have at least one successful recording
                successful_rec = 0
                for temp_scen in scenarios:
                    if RecordingEvents.GENERAL_RECEIVING in list(temp_scen.keys()):
                        successful_rec += 1
                if successful_rec >= iteration_acceptance_border:
                    it_counter += 1
                    # execute the attackers
                    if weak_events:
                        all_results = __execute_attacks_weak_events(estimator=estimator, tle_file=tle_file,
                                                                    eve_locations=eves_locs, eve_scenarios=scenarios,
                                                                    victim_lolah=victim_loc, debug_eval=False)
                    else:
                        all_results = __execute_attacks(estimator=estimator, tle_file=tle_file, eve_locations=eves_locs,
                                                        eve_scenarios=scenarios, victim_lolah=victim_loc,
                                                        debug_eval=False)
                    # check the results for completeness and select a proper eavesdropper to combine the results
                    at_least_one_poly = False
                    single_poly = None
                    eve_used_lolah = None
                    eve_used_itrs = None
                    combined_poly = None
                    for i_eve in range(len(all_results)):
                        eve_result = all_results[i_eve]  # eve_result = (atk1_poly, (atk2_poly, atk3_poly,) atk4_poly)
                        temp_eve = eves_locs[i_eve]
                        valid1, area1, dist1 = eval.evaluate_polygon(eve_result[0], temp_eve, victim_loc)
                        if valid1 and not at_least_one_poly:
                            at_least_one_poly = True
                            single_poly = eve_result[0]
                            combined_poly = eve_result[-1]
                            eve_used_lolah = temp_eve
                            eve_used_itrs = pu.lonLatHeight_2_ITRS(temp_eve[0], temp_eve[1], temp_eve[2])
                    if not at_least_one_poly:
                        print(f"WARNING: generic: invalid estimation!")
                        print(f"WARNING: generic: time: {start_time} - {end_time}")
                        print(f"WARNING: generic: victim_loc: {victim_loc}")
                        print(f"WARNING: generic: all_eves_loc: {eves_locs}")
                        print(f"WARNING: generic: all_scenarios: {scenarios}")
                    if at_least_one_poly:
                        # combine the results of multiple attackers
                        for i_eve in range(len(all_results)):
                            eve_result = all_results[i_eve]  # eve_result = (atk1_poly, atk4_poly)
                            temp_eve = eves_locs[i_eve]
                            eve_t_itrs = pu.lonLatHeight_2_ITRS(temp_eve[0], temp_eve[1], temp_eve[2])
                            atk4_translated = pu.recenter_poly(eve_t_itrs, eve_used_itrs, eve_result[-1])
                            combined_poly = pu.polygons_intersection([combined_poly, atk4_translated])
                    # evaluate the attack and write the result
                    valid_s, area_s, dist_s = eval.evaluate_polygon(single_poly, eve_used_lolah, victim_loc)
                    valid_c, area_c, dist_c = eval.evaluate_polygon(combined_poly, eve_used_lolah, victim_loc)
                    # write the output-files
                    if area_output_folder is not None:
                        if area_output_folder[-1] != os.sep:
                            area_output_folder = f"{area_output_folder}{os.sep}"
                        filename = f"{temp_dist}kmFibo_cont_{temp_duration}sec_{eve_amounts}eves"
                        if do_snf_intervals:
                            filename = f"{temp_dist}kmFibo_{int_rep[it_time]}x_{temp_duration}sec_rec_" \
                                       f"{inter_int_time[it_time]}sec_pause_{eve_amounts}eves"
                        if weak_events:
                            filename = filename + f"_weakEvents"
                        if noisy_prediction:
                            filename = filename + f"_noisyPrediction"
                        output_path = area_output_folder + filename + ".csv"
                        if (not at_least_one_poly) or (not valid_s) or (not valid_c):
                            output_path = output_path + "_invalid"
                        __write_output_csv(filename=output_path, data_points=[area_s, area_c])
                    if use_point_estimator and area_output_folder is not None:
                        output_path = area_output_folder + filename + "_pointEstimator" + ".csv"
                        if (not at_least_one_poly) or (not valid_s) or (not valid_c):
                            output_path = output_path + "_invalid"
                        pE_dist_s = eval.evaluate_point_estimator(single_poly, eve_used_lolah, victim_loc)
                        pE_dist_c = eval.evaluate_point_estimator(combined_poly, eve_used_lolah, victim_loc)
                        __write_output_csv(filename=output_path, data_points=[area_s, pE_dist_s, area_c, pE_dist_c])


if __name__ == '__main__':
    logging.getLogger('shapely.geos').setLevel(logging.WARNING)
    # Do simulations, where the number of receivers vary.
    # This is tailored towards long computations: starting multiple instances with slightly different parameters.
    # The output-files are later combined to one graph.
    t1 = time.time()
    output_folder = "simulation_data2/duration_and_type"
    iterations_in = 10
    inter_obs_distance = [100]  # distances between the observers
    durations_in = [60]  # seconds;  durations_in = [1*60, 3*60, 10*60, 30*60, 60*60, 4*60*60]
    number_eves = 3
    noisy_prediction = True
    weak_events = True
    point_estimator = False

    __analyze_attackers_generic(distances=inter_obs_distance, sniffing_durations=durations_in, iterations=iterations_in,
                                eve_amounts=number_eves, do_snf_intervals=False,
                                area_output_folder=output_folder, weak_events=weak_events,
                                noisy_prediction=noisy_prediction, use_point_estimator=point_estimator)
    t2 = time.time()
    print(f"started at {t1}, until {t2}. took {t2 - t1} sec")
