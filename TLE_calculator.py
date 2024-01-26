import sgp4.api
from sgp4.api import Satrec, SatrecArray, SGP4_ERRORS
from sgp4.api import jday
import skyfield.sgp4lib as sgp4lib

import numpy as np

# The SGP4 propagator returns raw x,y,z Cartesian coordinates in a “True Equator Mean Equinox” (TEME)
# reference frame that’s centered on the Earth but does not rotate with it — an “Earth centered inertial” (ECI)
# reference frame.

# The purpose of this class is to load a set of satellites from TLEs and calculate their position at a given point in
# time. The returned positions are in TEME frame.
class TLE_calculator:
    def __init__(self, tleFile: str, warnings: bool=True, verbose: bool=True):
        self.tleFile = tleFile
        self.warnings = warnings
        self.verbose = verbose
        self.valid_days = 7
        self.__parseFile()
        self.one_sec_jDay = 1 / 86400  # one second in julian day

    def __parseFile(self):
        file = open(self.tleFile, "r")
        lines = file.readlines()
        elements = int (len(lines) / 3)
        sat_list = []
        sat_names = []
        for i in range(elements):
            line_name = lines[i*3]
            line_one = lines[i*3 + 1]
            line_two = lines[i*3 + 2]
            if line_name[-1] is "\n":
                line_name = line_name[:-1]
                line_name = line_name.lstrip()
                line_name = line_name.rstrip()
            if line_one[-1] is "\n":
                line_one = line_one[:-1]
            if line_two[-1] is "\n":
                line_two = line_two[:-1]
            tempSat = Satrec.twoline2rv(line_one, line_two)
            sat_list.append(tempSat)
            sat_names.append(line_name)
        self.sat_names = sat_names
        self.satList = sat_list
        self.satrec = SatrecArray(sat_list)
        if self.verbose:
            print(f"INFO:TLE_calculator: {len(self.satList)} sats parsed")

    def get_min_max_epoch(self) -> ((int, float), (int, float)):
        minYr = 999
        minDay = 999
        maxYr = 000
        maxDay = 0
        for sat in self.satList:
            tempYr = sat.epochyr
            tempDay = sat.epochdays
            if tempYr < minYr:
                minYr = tempYr
                minDay = tempDay
            elif tempYr == minYr and tempDay < minDay:
                minDay = tempDay
            if tempYr > maxYr:
                maxYr = tempYr
                maxDay = tempDay
            elif tempYr == maxYr and tempDay > maxDay:
                maxDay = tempDay
        return (minYr, minDay), (maxYr, maxDay)

    def calculate_one_position_single(self, satellite: Satrec, jDay: float, jDayF: float):
        # return pos, vel in km/s in TEME
        if self.warnings and abs(jDay+jDayF - self.satList[0].jdsatepoch) > self.valid_days:
            print(f"WARNING: TLEcalculator.calculate_position: Large difference between TLE-epoch and given time: given={jDay+jDayF}, epoch={self.satList[0].jdsatepoch}.")
        err, pos, vel = satellite.sgp4(jDay, jDayF)
        return err, pos, vel

    def calculate_one_position_all(self, jTimeDay: float, jTimeFr: float=0.0):
        # return pos, vel in km/s in TEME
        if self.warnings and abs(jTimeDay+jTimeFr - self.satList[0].jdsatepoch) > self.valid_days:
            print(f"WARNING: TLEcalculator.calculate_position: Large difference between TLE-epoch and given time: given={jTimeDay+jTimeFr}, epoch={self.satList[0].jdsatepoch}.")
        jd = np.array([jTimeDay])
        fr = np.array([jTimeFr])
        err, pos, vel = self.satrec.sgp4(jd, fr)
        return pos, vel

    def calculate_multi_positions_all(self, jTimeDay: np.ndarray, jTimeFr: np.ndarray):
        # return pos, vel in km/s in TEME
        err, pos, vel = self.satrec.sgp4(np.array(jTimeDay), np.array(jTimeFr))
        return pos, vel

    def TEME_2_ITRS(self, jDay: float, jDayF: float, pos_TEME: [float], vel_TEME: [float]):
        pos_TEME = np.array(pos_TEME)
        vel_TEME = np.array(vel_TEME)
        pos_ITRS = []
        vel_ITRS = []
        for sat_index in range(len(pos_TEME)):
            pos_temp, vel_temp = sgp4lib.TEME_to_ITRF(jDay + jDayF, np.asarray(pos_TEME[sat_index,:]), np.asarray(vel_TEME[sat_index,:]) * 86400)
            vel_temp = vel_temp / 86400
            pos_ITRS.append(pos_temp)
            vel_ITRS.append(vel_temp)
        return np.array(pos_ITRS), np.array(vel_ITRS)

    def epoch_to_utc(self, epochYr: int, epochDay: float)-> (int, int, int, int, int):
        # returns (month, day, hour, minute, second)
        return sgp4.api.days2mdhms(epochYr, epochDay)

    def utc_time_to_jDay(self, year: int, month: int, day: int, hour:int, minute: int, second: float) -> (float, float):
        return jday(year, month, day, hour, minute, second)


    






