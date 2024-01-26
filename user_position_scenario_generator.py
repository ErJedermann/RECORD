# Idea for this Generator:
# input-parameters: eavesdropper-location, victim-location, TLE-file, starting-time, end-time, step-size
# output: a list of timestamps, where the eavesdropper can eavesdrop on the victims communication

from astropy import units as u
from astropy import coordinates as coord
from datetime import datetime
import numpy as np
from enum import Enum

from beam_model.Generic_rec_processed_beam_model import GenericRecordedProcessedModel
from TLE_calculator import TLE_calculator
from generic_satellite import Satellite


class RecordingEvents(Enum):
    GENERAL_RECEIVING = 1
    SUDDEN_RECEIVING = 2
    NOT_DURING_COMM = 3  # this shall be realized by the Attacker using multiple receivers
    NOT_AFTER_HAND = 4
    SUDDEN_NOT = 5
    CONTINUOUS_HAND = 6


class UserPositionScenarioGenerator:
    def __init__(self, genRecProcModel_file: str, tle_file: str):
        self.bm = GenericRecordedProcessedModel()
        self.bm.load_processed_model(genRecProcModel_file)
        self.tle_calculator = TLE_calculator(tle_file)

    def __find_visible_satellites(self, utc_time: int, tle_calc: TLE_calculator, rec_lolah: (float, float, float)):
        # go through all satellite in the TLE-file and find the visible ones
        temp_time = datetime.utcfromtimestamp(utc_time)
        jDay, jDayF = tle_calc.utc_time_to_jDay(temp_time.year, temp_time.month, temp_time.day,
                                                temp_time.hour, temp_time.minute, temp_time.second)
        raw_pos_TEME, raw_vel_TEME = tle_calc.calculate_one_position_all(jDay, jDayF)
        raw_pos_TEME = raw_pos_TEME[:, 0, :]
        raw_vel_TEME = raw_vel_TEME[:, 0, :]
        pos_ITRS, vel_ITRS = tle_calc.TEME_2_ITRS(jDay, jDayF, raw_pos_TEME, raw_vel_TEME)
        # receiver itrs_pos
        rec_earth = coord.EarthLocation(lon=rec_lolah[0] * u.deg, lat=rec_lolah[1] * u.deg, height=rec_lolah[2] * u.km)
        rec_ITRS = rec_earth.get_itrs()
        rec_ITRS = np.array([rec_ITRS.cartesian.x.value, rec_ITRS.cartesian.y.value, rec_ITRS.cartesian.z.value])
        # https://math.stackexchange.com/questions/2998875/how-to-determine-if-a-point-is-above-or-below-a-plane-defined-by-a-triangle
        # normalize the rec_ITRS to use it as the normal-vector of the tangential-plane
        normal = rec_ITRS / np.linalg.norm(rec_ITRS)
        signed_distances = pos_ITRS - rec_ITRS  # shape(x,3)
        signed_distances = np.dot(signed_distances, normal)
        positive_distances = signed_distances > 0
        index_positives = np.nonzero(positive_distances)
        index_positives = index_positives[0]
        visible_satellites = []
        for index in list(index_positives):
            sat_pos_TEME = raw_pos_TEME[index, :]
            sat_vel_TEME = raw_vel_TEME[index, :]
            sat_name = tle_calc.sat_names[index]
            sat = Satellite(sat_pos_TEME, sat_vel_TEME, sat_name, jDay, jDayF)
            sat_distance = signed_distances[index]
            visible_satellites.append((sat_distance, sat))
        visible_satellites = sorted(visible_satellites, reverse=True)  # close to the plane = close to the horizon
        visible_satellites = np.array(visible_satellites)
        visible_satellites = visible_satellites[:, 1]  # drop the distances
        return visible_satellites  # returns the satellites sorted, so that the first satellite is closest to azimuth

    def __add_solution(self, sol_dict: {}, time_value: float, event_type: Enum):
        if event_type in list(sol_dict.keys()):
            sol_dict[event_type].append(time_value)
        else:
            temp_lst = [time_value]
            sol_dict[event_type] = temp_lst

    def generate_scenario(self, eaves_lolah: (float, float, float), victim_lolah: (float, float, float),
                          event_types: [RecordingEvents], time_start: float, time_end: float = None,
                          time_step_size: float = 1, debug: bool = False) -> {}:
        if time_end is None:
            time_end = time_start + 3600
        solutions_dict = {}
        vic_old_beam_center = ""
        eve_was_receiving = None
        eve_is_receiving = False
        for curr_time in np.arange(time_start, time_end, time_step_size):
            # go through all victim-visible satellites and find all visible beams
            beams_victim = []
            visible_vic_sats = self.__find_visible_satellites(curr_time, self.tle_calculator, victim_lolah)
            for temp_sat in visible_vic_sats:
                beams_victim.extend(self.bm.get_beams(victim_lolah, temp_sat.pos_ITRS, temp_sat.vel_ITRS,
                                                      sat_name=f"{temp_sat.name}."))
            # go through all eve-visible satellites and find all visible beams
            beams_eve = []
            visible_eve_sats = self.__find_visible_satellites(curr_time, self.tle_calculator, eaves_lolah)
            for temp_sat in visible_eve_sats:
                beams_eve.extend(self.bm.get_beams(eaves_lolah, temp_sat.pos_ITRS, temp_sat.vel_ITRS,
                                                   sat_name=f"{temp_sat.name}."))
            beams_together = list(set(beams_victim).intersection(beams_eve))
            # find the beam, the victim is communicating with
            center_vic = "-no-beam-"
            for temp_sat in visible_vic_sats:
                center_vic = self.bm.get_beam(victim_lolah, temp_sat.pos_ITRS, temp_sat.vel_ITRS,
                                              sat_name=f"{temp_sat.name}.")
                if center_vic != "-no-beam-":
                    break
            if debug:
                print(f"DEBUG: UPSG: time={curr_time}: beams_victim:{beams_victim}, beams_eve:{beams_eve}")
                print(f"DEBUG: UPSG: time={curr_time}: beams_together:{beams_together}, center_vic:{center_vic}")
            if (len(beams_together) > 0) and (center_vic in beams_together):
                eve_is_receiving = True
            else:
                eve_is_receiving = False
            if RecordingEvents.GENERAL_RECEIVING in event_types:
                if eve_is_receiving:
                    self.__add_solution(solutions_dict, curr_time, RecordingEvents.GENERAL_RECEIVING)
            if RecordingEvents.SUDDEN_RECEIVING in event_types:
                if eve_is_receiving and (not eve_was_receiving) and (curr_time > time_start):
                    self.__add_solution(solutions_dict, curr_time, RecordingEvents.SUDDEN_RECEIVING)
            if RecordingEvents.NOT_AFTER_HAND in event_types:
                if (not eve_is_receiving) and eve_was_receiving and (vic_old_beam_center != center_vic) and (
                        center_vic != "-no-beam-") and (curr_time > time_start):
                    self.__add_solution(solutions_dict, curr_time, RecordingEvents.NOT_AFTER_HAND)
            if RecordingEvents.SUDDEN_NOT in event_types:
                if eve_was_receiving and (not eve_is_receiving) and (vic_old_beam_center == center_vic) and (
                        curr_time > time_start):
                    self.__add_solution(solutions_dict, curr_time, RecordingEvents.SUDDEN_NOT)
            if RecordingEvents.CONTINUOUS_HAND in event_types:
                if eve_was_receiving and eve_is_receiving and (vic_old_beam_center != center_vic) and (
                        curr_time > time_start):
                    self.__add_solution(solutions_dict, curr_time, RecordingEvents.CONTINUOUS_HAND)
            # update the past values
            eve_was_receiving = eve_is_receiving
            vic_old_beam_center = center_vic
        if debug:
            print(f"DEBUG: UPSG: scenario_dict={solutions_dict}")
        return solutions_dict  # {key=event_type: vale=[times_of_the_event]}

    def __generate_direct_events(self, eve_is_receiving: bool, eve_index: int, curr_time: float, time_start: float,
                                 center_vic: str, eve_was_receiving_lst: [bool], solutions_dict_lst: {},
                                 vic_old_beam_center_lst: [str], event_types: [RecordingEvents]):
        # !!! side effects !!! it uses 'eve_was_receiving_lst', 'solutions_dict_lst', 'vic_old_beam_center_lst' as
        # persistent storage over time (list-variables are passed by reference).
        # get the variables of the current eve
        solutions_dict = solutions_dict_lst[eve_index]
        eve_was_receiving = eve_was_receiving_lst[eve_index]
        vic_old_beam_center = vic_old_beam_center_lst[eve_index]
        # generate the events and insert them into the solutions-dict of this eve
        if RecordingEvents.GENERAL_RECEIVING in event_types:
            if eve_is_receiving:
                self.__add_solution(solutions_dict, curr_time, RecordingEvents.GENERAL_RECEIVING)
        if RecordingEvents.SUDDEN_RECEIVING in event_types:
            if eve_is_receiving and (not eve_was_receiving) and (curr_time > time_start):
                self.__add_solution(solutions_dict, curr_time, RecordingEvents.SUDDEN_RECEIVING)
        if RecordingEvents.NOT_AFTER_HAND in event_types:
            if (not eve_is_receiving) and eve_was_receiving and (vic_old_beam_center != center_vic) and (
                    center_vic != "-no-beam-") and (curr_time > time_start):
                self.__add_solution(solutions_dict, curr_time, RecordingEvents.NOT_AFTER_HAND)
        if RecordingEvents.SUDDEN_NOT in event_types:
            if eve_was_receiving and (not eve_is_receiving) and (vic_old_beam_center == center_vic) and (
                    curr_time > time_start):
                self.__add_solution(solutions_dict, curr_time, RecordingEvents.SUDDEN_NOT)
        if RecordingEvents.CONTINUOUS_HAND in event_types:
            if eve_was_receiving and eve_is_receiving and (vic_old_beam_center != center_vic) and (
                    curr_time > time_start):
                self.__add_solution(solutions_dict, curr_time, RecordingEvents.CONTINUOUS_HAND)
        # update the past: store the current variable-states in the list of old-variable-states
        solutions_dict_lst[eve_index] = solutions_dict
        eve_was_receiving_lst[eve_index] = eve_is_receiving
        vic_old_beam_center_lst[eve_index] = center_vic
        return

    def __generate_ndc_events(self, solutions_dict_lst: {}, event_types: [RecordingEvents]):
        if RecordingEvents.NOT_DURING_COMM not in event_types:
            return solutions_dict_lst
        gen_rec_lists = []
        for i in range(len(solutions_dict_lst)):
            temp_dict = solutions_dict_lst[i]
            if RecordingEvents.GENERAL_RECEIVING in list(temp_dict.keys()):
                temp_gen_rec_lst = temp_dict[RecordingEvents.GENERAL_RECEIVING]
                gen_rec_lists.append(temp_gen_rec_lst)
            else:
                gen_rec_lists.append([])
        # generate the ndc-events for every eve
        for i in range(len(solutions_dict_lst)):
            other_receive_times = set([])
            for j in range(len(solutions_dict_lst)):
                if j == i:
                    continue
                other_receive_times.update(set(gen_rec_lists[j]))  # combine the gen-rec-events of all other eves
            other_receive_times.discard(set(gen_rec_lists[i]))  # remove the gen-rec-events of this eve = ndc-events
            for curr_ndc in list(other_receive_times):
                self.__add_solution(solutions_dict_lst[i], curr_ndc, RecordingEvents.NOT_DURING_COMM)
        return solutions_dict_lst

    def generate_scenario_multi(self, eaves_lolah: [(float, float, float)], victim_lolah: (float, float, float),
                                event_types: [RecordingEvents], time_start: float, time_end: float = None,
                                time_step_size: float = 1, debug: bool = False) -> {}:
        # New version of the algorithm, handling multiple eves and generating NOT_DURING_COMM events.
        # Returns a list of dicts (one dict per eve).
        # Each dict contains the events of the according eve: [{Event: [timestamps]}]
        if time_end is None:
            time_end = time_start + 3600
        solutions_dict_lst = []
        vic_old_beam_center_lst = []
        eve_was_receiving_lst = []
        for i in range(len(eaves_lolah)):
            solutions_dict_lst.append({})
            vic_old_beam_center_lst.append("")
            eve_was_receiving_lst.append(None)
        for curr_time in np.arange(time_start, time_end, time_step_size):
            # go through all victim-visible satellites and find all visible beams
            beams_victim = []
            visible_vic_sats = self.__find_visible_satellites(curr_time, self.tle_calculator, victim_lolah)
            for temp_sat in visible_vic_sats:
                beams_victim.extend(self.bm.get_beams(victim_lolah, temp_sat.pos_ITRS, temp_sat.vel_ITRS,
                                                      sat_name=f"{temp_sat.name}."))
            # find the beam, the victim is communicating with
            center_vic = "-no-beam-"
            for temp_sat in visible_vic_sats:  # use the sorting of the satellites (first is closest to azimuth)
                center_vic = self.bm.get_beam(victim_lolah, temp_sat.pos_ITRS, temp_sat.vel_ITRS,
                                              sat_name=f"{temp_sat.name}.")
                if center_vic != "-no-beam-":
                    break
            # go through all eves
            for i in range(len(eaves_lolah)):
                eve_loc = eaves_lolah[i]
                # go through all eve-visible satellites and find all visible beams
                beams_eve = []
                visible_eve_sats = self.__find_visible_satellites(curr_time, self.tle_calculator, eve_loc)
                for temp_sat in visible_eve_sats:
                    beams_eve.extend(self.bm.get_beams(eve_loc, temp_sat.pos_ITRS, temp_sat.vel_ITRS,
                                                       sat_name=f"{temp_sat.name}."))
                beams_together = list(set(beams_victim).intersection(beams_eve))
                if debug:
                    print(f"DEBUG: UPSG: time={curr_time}: beams_victim:{beams_victim}, beams_eve:{beams_eve}")
                    print(f"DEBUG: UPSG: time={curr_time}: beams_together:{beams_together}, center_vic:{center_vic}")
                if (len(beams_together) > 0) and (center_vic in beams_together):
                    eve_is_receiving = True
                else:
                    eve_is_receiving = False
                self.__generate_direct_events(eve_is_receiving, i, curr_time, time_start, center_vic,
                                              eve_was_receiving_lst, solutions_dict_lst, vic_old_beam_center_lst,
                                              event_types)
        solutions_dict_lst = self.__generate_ndc_events(solutions_dict_lst, event_types)
        return solutions_dict_lst  # every eve got a dict: {key=event_type: vale=[times_of_the_event]}
