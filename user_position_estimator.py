# Input: Eavesdropper position, List of [UTC timestamps] when the victims downlink was eavesdropped
# Working steps:
#   for each timestamp
#       get the current satellites and beams the eavesdropper can receive
#       get the combined footprint of these beams for this timestamp
#   intersection of all combined footprints holds the possible location
# Output: The possible area, where the victim is located


from beam_model.Generic_rec_processed_beam_model import GenericRecordedProcessedModel, Polygon
from user_position_scenario_generator import RecordingEvents
from TLE_calculator import TLE_calculator
from generic_satellite import Satellite
import plot_satellite_beams
import polygon_utilities as pu

from astropy import units as u
from astropy import coordinates as coord
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon as shPoly
from shapely.geometry import MultiPolygon as shMPoly


class UserPositionEstimator:
    def __init__(self, beamModel_file: str):
        self.bm = GenericRecordedProcessedModel()
        self.bm.load_processed_model(beamModel_file)

    def __find_visible_satellites(self, utc_time: int, tle_calc: TLE_calculator, eve_lolah: (float, float, float)):
        # go through all satellite in the TLE-file and find the visible ones
        temp_time = datetime.utcfromtimestamp(utc_time)
        jDay, jDayF = tle_calc.utc_time_to_jDay(temp_time.year, temp_time.month, temp_time.day,
                                                temp_time.hour, temp_time.minute, temp_time.second)
        raw_pos_TEME, raw_vel_TEME = tle_calc.calculate_one_position_all(jDay, jDayF)
        raw_pos_TEME = raw_pos_TEME[:, 0, :]
        raw_vel_TEME = raw_vel_TEME[:, 0, :]
        pos_ITRS, vel_ITRS = tle_calc.TEME_2_ITRS(jDay, jDayF, raw_pos_TEME, raw_vel_TEME)
        # receiver itrs_pos
        rec_earth = coord.EarthLocation(lon=eve_lolah[0] * u.deg, lat=eve_lolah[1] * u.deg, height=eve_lolah[2] * u.km)
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
        return visible_satellites

    def __invert_beam_selection(self, beam_list: [str]) -> [str]:
        all_beams_list = list(self.bm.beam_polygon_dict.keys())
        all_beams_set = set(all_beams_list)
        input_set = set(beam_list)
        output_set = all_beams_set - input_set
        return list(output_set)

    def __footprint_dict_2_poly(self, eve_pos_lolah: (float, float, float), fp_poly_dict: dict) -> [shPoly]:
        # transforms the lists of itrs coordinates to shapely.polygon
        fp_poly_list = []
        eve_itrs = self.lonLatHeight_2_ITRS(eve_pos_lolah[0], eve_pos_lolah[1], eve_pos_lolah[2])
        for beam_key in list(fp_poly_dict.keys()):
            curr_beam_poly_list = fp_poly_dict[beam_key]
            for poly_index in range(len(curr_beam_poly_list)):
                curr_poly_itrs = curr_beam_poly_list[poly_index]
                curr_poly = pu.points_itrs_2_poly(eve_itrs, curr_poly_itrs)
                fp_poly_list.append(curr_poly)
        return fp_poly_list

    def __plot_beams(self, eve_lolah: (float, float, float), poly: shPoly, fig_name: str,
                     target: (float, float, float) = None):
        eve_itrs = self.lonLatHeight_2_ITRS(eve_lolah[0], eve_lolah[1], eve_lolah[2])
        map_points = pu.poly_get_points_itrs(eve_itrs, poly)
        print_dict = {"possible location": map_points}
        plot_satellite_beams.plot_polygon_footprints(eve_lolah, print_dict, figure_name=fig_name,
                                                         sat_name="eavesdropper", target_pos_lolah=target)

    def plot_beams(self, eve_lolah, poly: shPoly, fig_name: str, target_lolah = None):
        self.__plot_beams(eve_lolah, poly, fig_name, target_lolah)


    def __estimate_maximum_area(self, eve_pos_lolah: (float, float, float), tle_calc: TLE_calculator, rec_times: [float],
                                debug_plots: bool = False, result_plot: bool = False) -> shPoly:
        # Takes an eavesdropper position, a TLE file and some recording-times of the victim and returns a
        # polygon (shapely.polygon/shapely.multiPolygon) where the victim should be located in.
        possible_positions = []  # list of polygons with possible positions, one for each time-point
        for temp_time in rec_times:
            # find all visible satellites
            possible_positions_this_time = []
            visible_satellites = self.__find_visible_satellites(temp_time, tle_calc, eve_pos_lolah)
            # get from each satellite the beams
            for curr_sat in visible_satellites:
                eve_beams = self.bm.get_beams(eve_pos_lolah, curr_sat.pos_ITRS, curr_sat.vel_ITRS)
                if len(eve_beams) > 0:
                    # get the footprints of the beams
                    eve_fp_poly_dict = self.bm.calculate_footprint(curr_sat.pos_ITRS, curr_sat.vel_ITRS, eve_beams)
                    if debug_plots:
                        fig_name = f"max_area: Satellite footprints for sat:{curr_sat.name} at time {temp_time}"
                        plot_satellite_beams.plot_polygon_footprints(eve_pos_lolah, eve_fp_poly_dict, figure_name=fig_name, sat_name="eve")
                    # get the combination of all beams
                    eve_fp_polys = self.__footprint_dict_2_poly(eve_pos_lolah, eve_fp_poly_dict)
                    eve_combined_fp_poly = pu.polygons_union(eve_fp_polys)
                    if debug_plots:
                        plot_name = f"max_area: Union of the beams of sat:{curr_sat.name} at time {temp_time}"
                        self.__plot_beams(eve_pos_lolah, eve_combined_fp_poly, plot_name)
                    # store the combination of this sat at this time-point
                    possible_positions_this_time.append(eve_combined_fp_poly)
            # combine all possible positions of this time point (union, since you don't know which beam/sat it was)
            temp_positions = pu.polygons_union(possible_positions_this_time)
            possible_positions.append(temp_positions)
            if debug_plots:
                self.__plot_beams(eve_pos_lolah, temp_positions, f"max_area: Union of all satellites at {temp_time}")
        # get intersection of all (the victim has to be in all polygons)
        intersection = pu.polygons_intersection(possible_positions)
        if debug_plots or result_plot:
            self.__plot_beams(eve_pos_lolah, intersection, f"max_area: final possible positions")
        return intersection

    def __estimate_minimum_area(self, eve_pos_lolah: (float, float, float), tle_calc: TLE_calculator,
                                no_rec_times: [float], debug_plots: bool = False, result_plot: bool = False) -> shMPoly:
        # Calculate the minimum area, that is only covered by the recorded beams
        not_possible_positions = []
        for temp_time in no_rec_times:
            # get the maximal beam-polygon, where eve is in
            eve_poly = self.__estimate_maximum_area(eve_pos_lolah, tle_calc, [temp_time])
            if debug_plots:
                self.__plot_beams(eve_pos_lolah, eve_poly, f"min_area: Starting eve-pos at time {temp_time}")
            # then subtract all beams, where eve is not in, to get the minimum
            visible_satellites = self.__find_visible_satellites(temp_time, tle_calc, eve_pos_lolah)
            # get from each satellite the beams
            for curr_sat in visible_satellites:
                # get the list of all beams, where eve is not inside
                eve_beams = self.bm.get_beams(eve_pos_lolah, curr_sat.pos_ITRS, curr_sat.vel_ITRS)
                no_eve_beams = self.__invert_beam_selection(eve_beams)
                if len(no_eve_beams) > 0:
                    # get the footprints of the beams
                    no_eve_fp_poly_dict = self.bm.calculate_footprint(curr_sat.pos_ITRS, curr_sat.vel_ITRS,
                                                                        no_eve_beams)
                    if debug_plots:
                        fig_name = f"min_area: foreign footprints (sat:{curr_sat.name} at time {temp_time})"
                        plot_satellite_beams.plot_polygon_footprints(eve_pos_lolah, no_eve_fp_poly_dict, figure_name=fig_name)
                    # subtract the positions from eves polygon
                    no_eve_fp_poly_list = self.__footprint_dict_2_poly(eve_pos_lolah, no_eve_fp_poly_dict)
                    eve_poly = pu.polygons_subtract(eve_poly, no_eve_fp_poly_list)
                    if debug_plots:
                        fig_name = f"min_area: not possible = eve - foreign sat:{curr_sat.name} time:{temp_time}"
                        self.__plot_beams(eve_pos_lolah, eve_poly, fig_name)
            not_possible_positions.append(eve_poly)
            if debug_plots:
                self.__plot_beams(eve_pos_lolah, eve_poly, f"min_area: Impossible positions at time {temp_time}")
        # At the end a list of small polygons remain, with impossible positions.
        # The Union of them are all positions, where the victim can't be.
        not_possible_polygon = pu.polygons_union(not_possible_positions)
        if debug_plots or result_plot:
            self.__plot_beams(eve_pos_lolah, not_possible_polygon,
                              f"min_area: Remaining positions, that can't be the positions of the victim.")
        return not_possible_polygon

    def __estimate_sudden_rec(self, eve_pos_lolah: (float, float, float), tle_calc: TLE_calculator,
                              sudden_times: [float], time_step: int = 1, debug_plots: bool = False,
                              result_plot: bool = False) -> shMPoly:
        possible_positions = []
        for temp_time in sudden_times:
            entered_beams_polys = []
            # get two lists of satellites for rec-time and no-rec-time
            visible_rec_satellites = self.__find_visible_satellites(temp_time, tle_calc, eve_pos_lolah)
            rec_sat_names = []
            for i in range(len(visible_rec_satellites)):
                rec_sat_names.append(visible_rec_satellites[i].name)
            visible_not_rec_satellites = self.__find_visible_satellites(temp_time - time_step, tle_calc, eve_pos_lolah)
            not_rec_sat_names = []
            for i in range(len(visible_not_rec_satellites)):
                not_rec_sat_names.append(visible_not_rec_satellites[i].name)
            # go through all satellites and get from each satellite the beams
            for i_rec in range(len(visible_rec_satellites)):
                rec_sat = visible_rec_satellites[i_rec]
                rec_name = rec_sat_names[i_rec]
                if rec_name in not_rec_sat_names:
                    i_nrec = not_rec_sat_names.index(rec_name)
                    nrec_sat = visible_not_rec_satellites[i_nrec]
                    # compare the beams for a new entered beam
                    rec_beams = self.bm.get_beams(eve_pos_lolah, rec_sat.pos_ITRS, rec_sat.vel_ITRS)
                    not_rec_beams = self.bm.get_beams(eve_pos_lolah, nrec_sat.pos_ITRS, nrec_sat.vel_ITRS)
                    for temp_beam in rec_beams:
                        if temp_beam not in not_rec_beams:
                            # eve entered a new beam
                            fp_poly_dict = self.bm.calculate_footprint(rec_sat.pos_ITRS, rec_sat.vel_ITRS,
                                                                         [temp_beam])
                            if debug_plots:
                                fig_name = f"sudden_rec: eve entered new receiving beam {temp_beam} at sat {rec_name} in the second after {temp_time - time_step} time"
                                plot_satellite_beams.plot_polygon_footprints(eve_pos_lolah, fp_poly_dict,
                                                                                 figure_name=fig_name)
                            fp_polys = self.__footprint_dict_2_poly(eve_pos_lolah, fp_poly_dict)
                            rec_poly = pu.polygons_union(fp_polys)
                            entered_beams_polys.append(rec_poly)
                else:
                    rec_beams = self.bm.get_beams(eve_pos_lolah, rec_sat.pos_ITRS, rec_sat.vel_ITRS)
                    if len(rec_beams) > 0:
                        # get the footprints of the new satellites beams
                        fp_poly_dict = self.bm.calculate_footprint(rec_sat.pos_ITRS, rec_sat.vel_ITRS, rec_beams)
                        temp_new_polys = self.__footprint_dict_2_poly(eve_pos_lolah, fp_poly_dict)
                        if debug_plots:
                            fig_name = f"sudden_rec: footprint of apprearing sat:{rec_name} at time {temp_time - time_step})"
                            plot_satellite_beams.plot_polygon_footprints(eve_pos_lolah, fp_poly_dict,
                                                                             figure_name=fig_name)
                        temp_new_polys2 = pu.polygons_union(temp_new_polys)
                        entered_beams_polys.append(temp_new_polys2)
            # covered area of the first recording (time at sudden rec) minus the last not recording (time - time_step)
            first_rec_poly = self.__estimate_maximum_area(eve_pos_lolah, tle_calc, [temp_time], debug_plots=debug_plots)
            last_nrec_poly = self.__estimate_minimum_area(eve_pos_lolah, tle_calc, [temp_time - time_step], debug_plots=debug_plots)
            temp_result_poly = pu.polygons_subtract(first_rec_poly, [last_nrec_poly])
            if debug_plots:
                self.__plot_beams(eve_pos_lolah, temp_result_poly, f"sudden_rec: positons for case 1: victim entered eves beam (time {temp_time})")
            # if eve entered some new beams, their area is added to the possible positions
            if len(entered_beams_polys) > 0:
                input_list = [temp_result_poly]
                input_list.extend(entered_beams_polys)
                temp_result_poly = pu.polygons_union(input_list)
                if debug_plots:
                    self.__plot_beams(eve_pos_lolah, temp_result_poly, f"sudden_rec: positions for cas 2: eve also entered new beam (time {temp_time})")
            possible_positions.append(temp_result_poly)
        # At the end a list of possible positions remain. The intersection of them are all possible positions.
        result_polygon = pu.polygons_intersection(possible_positions)
        if debug_plots or result_plot:
            self.__plot_beams(eve_pos_lolah, result_polygon,
                              f"sudden_rec: Resulting positions from all SuddenReceiving-events.")
        return result_polygon

    def __estimate_not_after_handoff(self, eve_pos_lolah: (float, float, float), tle_calc: TLE_calculator,
                                     nah_times: [float], time_step: int = 1, debug_plots: bool = False,
                                     result_plot: bool = False) -> shMPoly:
        possible_positions = []
        for temp_time in nah_times:
            # covered area of the handoff recording (at time-1)
            handoff_poly = self.__estimate_maximum_area(eve_pos_lolah, tle_calc, [temp_time - time_step])
            nrec_poly = self.__estimate_minimum_area(eve_pos_lolah, tle_calc, [temp_time])
            temp_result_poly = pu.polygons_subtract(handoff_poly, [nrec_poly])
            if debug_plots:
                self.__plot_beams(eve_pos_lolah, temp_result_poly,
                                  f"not_hnd: Possible pos at time {temp_time}")
            possible_positions.append(temp_result_poly)
        # At the end a list of possible positions remain. The intersection of them are all possible positions.
        result_polygon = pu.polygons_intersection(possible_positions)
        if debug_plots or result_plot:
            self.__plot_beams(eve_pos_lolah, result_polygon,
                              f"not_hnd: Possible positions from all Not-After-Handoff-events.")
        return result_polygon

    def __estimate_sudden_not(self, eve_pos_lolah: (float, float, float), tle_calc: TLE_calculator,
                              sudden_times: [float], time_step: int = 1, debug_plots: bool = False,
                              result_plot: bool = False) -> shMPoly:
        possible_positions = []
        for temp_time in sudden_times:
            current_leaving_beams = []
            # get two lists of satellites for rec-time and no-rec-time
            visible_rec_satellites = self.__find_visible_satellites(temp_time - time_step, tle_calc, eve_pos_lolah)
            rec_sat_names = []
            for i in range(len(visible_rec_satellites)):
                rec_sat_names.append(visible_rec_satellites[i].name)
            visible_not_rec_satellites = self.__find_visible_satellites(temp_time, tle_calc, eve_pos_lolah)
            not_rec_sat_names = []
            for i in range(len(visible_not_rec_satellites)):
                not_rec_sat_names.append(visible_not_rec_satellites[i].name)
            # go through all satellites and get from each satellite the beams
            for i_rec in range(len(visible_rec_satellites)):
                rec_sat = visible_rec_satellites[i_rec]
                rec_name = rec_sat_names[i_rec]
                if rec_name in not_rec_sat_names:
                    i_nrec = not_rec_sat_names.index(rec_name)
                    nrec_sat = visible_not_rec_satellites[i_nrec]
                    # compare the beams for a left / abandoned beam
                    rec_beams = self.bm.get_beams(eve_pos_lolah, rec_sat.pos_ITRS, rec_sat.vel_ITRS)
                    not_rec_beams = self.bm.get_beams(eve_pos_lolah, nrec_sat.pos_ITRS, nrec_sat.vel_ITRS)
                    for temp_beam in rec_beams:
                        if temp_beam not in not_rec_beams:
                            fp_poly_dict = self.bm.calculate_footprint(rec_sat.pos_ITRS, rec_sat.vel_ITRS,
                                                                         [temp_beam])
                            if debug_plots:
                                fig_name = f"sudden_not: last sec of receiving beam {temp_beam} at sat:{rec_name} at time {temp_time - time_step})"
                                plot_satellite_beams.plot_polygon_footprints(eve_pos_lolah, fp_poly_dict, figure_name=fig_name)
                            fp_polys = self.__footprint_dict_2_poly(eve_pos_lolah, fp_poly_dict)
                            rec_poly = pu.polygons_union(fp_polys)
                            current_leaving_beams.append(rec_poly)
                else:
                    rec_beams = self.bm.get_beams(eve_pos_lolah, rec_sat.pos_ITRS, rec_sat.vel_ITRS)
                    if len(rec_beams) > 0:
                        # get the footprints of the beams
                        fp_poly_dict = self.bm.calculate_footprint(rec_sat.pos_ITRS, rec_sat.vel_ITRS, rec_beams)
                        temp_raw_polys = self.__footprint_dict_2_poly(eve_pos_lolah, fp_poly_dict)
                        if debug_plots:
                            fig_name = f"sudden_not: last-sec-footprint of disappearing sat:{rec_name} at time {temp_time - time_step})"
                            plot_satellite_beams.plot_polygon_footprints(eve_pos_lolah, fp_poly_dict, figure_name=fig_name)
                        temp_result_poly = pu.polygons_union(temp_raw_polys)
                        current_leaving_beams.append(temp_result_poly)
            current_combined_positions = pu.polygons_union(current_leaving_beams)
            possible_positions.append(current_combined_positions)
        # At the end a list of possible positions remain. The intersection of them are all possible positions.
        result_polygon = pu.polygons_intersection(possible_positions)
        if debug_plots or result_plot:
            self.__plot_beams(eve_pos_lolah, result_polygon,
                              f"sudden_not: Resulting positions from all Sudden-Not-events.")
        return result_polygon

    def __estimate_cont_handoff(self, eve_pos_lolah: (float, float, float), tle_calc: TLE_calculator,
                                sudden_times: [float], time_step: int = 1, debug_plots: bool = False,
                                result_plot: bool = False) -> shMPoly:
        possible_positions = []
        for temp_time in sudden_times:
            # get the list of all beams of all satellites at handoff-time
            handoff_satellites = self.__find_visible_satellites(temp_time - time_step, tle_calc, eve_pos_lolah)
            beam_poly_list = []
            for curr_sat in handoff_satellites:
                eve_beams = self.bm.get_beams(eve_pos_lolah, curr_sat.pos_ITRS, curr_sat.vel_ITRS)
                for temp_beam in eve_beams:
                    fp_poly_dict = self.bm.calculate_footprint(curr_sat.pos_ITRS, curr_sat.vel_ITRS, [temp_beam])
                    fp_polys = self.__footprint_dict_2_poly(eve_pos_lolah, fp_poly_dict)
                    fp_union = pu.polygons_union(fp_polys)
                    beam_poly_list.append(fp_union)
            # calculate the pairwise intersections
            intersection_list = []
            for i in range(len(beam_poly_list)):
                for j in range(i, len(beam_poly_list)):
                    temp_intersection = pu.polygons_intersection([beam_poly_list[i], beam_poly_list[j]])
                    intersection_list.append(temp_intersection)
            # the union of all sections is the solution for this time
            temp_result_poly = pu.polygons_union(intersection_list)
            possible_positions.append(temp_result_poly)
        # At the end a list of possible positions remain. The intersection of them are all possible positions.
        result_polygon = pu.polygons_intersection(possible_positions)
        if debug_plots or result_plot:
            self.__plot_beams(eve_pos_lolah, result_polygon,
                              f"cont_hnd: Resulting positions from all Continuous-Handoff-events.")
        return result_polygon

    # You can give pre-computed polygons to speed up the computation (avoid re-computing everything)
    def estimate_user_position(self, eve_pos_lolah: (float, float, float), TLE_file: str,
                               new_recordings: dict, previous_polygon: shPoly = None, result_plot: bool = False,
                               dbg_plot: bool = False) \
            -> shPoly:
        tle_calc = TLE_calculator(tleFile=TLE_file)
        recording_event_list = list(new_recordings.keys())
        event_amounts = []
        if dbg_plot:
            print(f"DEBUG: UPE.estimate_user_position(): scenario={new_recordings}")
        if RecordingEvents.GENERAL_RECEIVING in recording_event_list:
            pos_times = new_recordings[RecordingEvents.GENERAL_RECEIVING]
            # print(f"DEBUG: GENERAL_RECEIVINGs: {len(pos_times)}")
            event_amounts.append(len(pos_times))
            if len(pos_times) > 0:
                result_polygon = self.__estimate_maximum_area(eve_pos_lolah, tle_calc, pos_times, result_plot=dbg_plot, debug_plots=dbg_plot)
                if previous_polygon is None:
                    previous_polygon = result_polygon
                else:
                    previous_polygon = previous_polygon.intersection(result_polygon)
        if type(previous_polygon) is not shPoly and type(previous_polygon) is not shMPoly:
            print(
                f"ERROR: User_Position_Estimator: No positions after GENERAL_RECEIVING! But a first guess is required!")
            return None
        if RecordingEvents.SUDDEN_RECEIVING in recording_event_list:
            su_rec_times = new_recordings[RecordingEvents.SUDDEN_RECEIVING]
            # print(f"DEBUG: SUDDEN_RECEIVING: {len(su_rec_times)}")
            event_amounts.append(len(su_rec_times))
            if len(su_rec_times) > 0:
                su_rec_polygon = self.__estimate_sudden_rec(eve_pos_lolah, tle_calc, su_rec_times, debug_plots=dbg_plot, result_plot=dbg_plot)
                previous_polygon = pu.polygons_intersection([previous_polygon, su_rec_polygon])
        if RecordingEvents.NOT_DURING_COMM in recording_event_list:
            not_rec_times = new_recordings[RecordingEvents.NOT_DURING_COMM]
            # print(f"DEBUG: NOT_DURING_COMM: {len(not_rec_times)}")
            event_amounts.append(len(not_rec_times))
            if len(not_rec_times) > 0:
                imp_polygon = self.__estimate_minimum_area(eve_pos_lolah, tle_calc, not_rec_times, debug_plots=dbg_plot, result_plot=dbg_plot)
                previous_polygon = pu.polygons_subtract(previous_polygon, [imp_polygon])
        if RecordingEvents.NOT_AFTER_HAND in recording_event_list:
            nah_times = new_recordings[RecordingEvents.NOT_AFTER_HAND]
            # print(f"DEBUG: NOT_AFTER_HAND: {len(nah_times)}")
            event_amounts.append(len(nah_times))
            if len(nah_times) > 0:
                nah_polygon = self.__estimate_not_after_handoff(eve_pos_lolah, tle_calc, nah_times, debug_plots=dbg_plot, result_plot=dbg_plot)
                previous_polygon = pu.polygons_intersection([previous_polygon, nah_polygon])
        if RecordingEvents.SUDDEN_NOT in recording_event_list:
            su_not_times = new_recordings[RecordingEvents.SUDDEN_NOT]
            # print(f"DEBUG: SUDDEN_NOT: {len(su_not_times)}")
            event_amounts.append(len(su_not_times))
            if len(su_not_times) > 0:
                su_not_polygon = self.__estimate_sudden_not(eve_pos_lolah, tle_calc, su_not_times, debug_plots=dbg_plot, result_plot=dbg_plot)
                previous_polygon = pu.polygons_intersection([previous_polygon, su_not_polygon])
        if RecordingEvents.CONTINUOUS_HAND in recording_event_list:
            chnd_times = new_recordings[RecordingEvents.CONTINUOUS_HAND]
            # print(f"DEBUG: CONTINUOUS_HAND: {len(chnd_times)}")
            event_amounts.append(len(chnd_times))
            if len(chnd_times) > 0:
                con_hnd_poly = self.__estimate_cont_handoff(eve_pos_lolah, tle_calc, chnd_times, debug_plots=dbg_plot, result_plot=dbg_plot)
                previous_polygon = pu.polygons_intersection([previous_polygon, con_hnd_poly])
        if result_plot:
            figure_name = f"update locations, used events [{recording_event_list}] with quantities [{event_amounts}]"
            self.__plot_beams(eve_pos_lolah, previous_polygon, figure_name)
        return previous_polygon




    def lonLatHeight_2_ITRS(self, lon: float, lat: float, height: float) -> (float, float, float):
        # lon is +east-west, lat is +north-south
        # lon in deg, lat in deg, height in km
        location = coord.EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=height * u.km)
        return location.itrs.x.value,location.itrs.y.value, location.itrs.z.value
