
from astropy.time import Time
from astropy import units as u
from astropy import coordinates as coord
import numpy as np
import skyfield.sgp4lib as sgp4lib

class Satellite:
    def __init__(self, pos_TEME: (float, float, float), vel_TEME: (float, float, float),
                 name: str, jDay: float, jDayF: float):
        self.pos_TEME = np.array(pos_TEME)
        self.vel_TEME = np.array(vel_TEME)
        self.name = name
        self.jDay = jDay
        self.jDayF = jDayF
        pos_ITRS, vel_ITRS = sgp4lib.TEME_to_ITRF(self.jDay + self.jDayF, np.asarray(self.pos_TEME),
                                                          np.asarray(self.vel_TEME) * 86400)
        self.pos_ITRS = pos_ITRS
        self.vel_ITRS = vel_ITRS / 86400
        self.pos_lonLatHeight = None

    def get_position_lonLatHeight(self) -> (float, float, float):
        if self.pos_lonLatHeight is None:
            time1 = Time(self.jDay + self.jDayF, format='jd')
            itrs = coord.ITRS(self.pos_ITRS[0] * u.km, self.pos_ITRS[1] * u.km, self.pos_ITRS[2] * u.km,
                                  self.vel_ITRS[0] * u.km / u.s, self.vel_ITRS[1] * u.km / u.s, self.vel_ITRS[2] * u.km / u.s,
                                  obstime=time1)
            self.pos_lonLatHeight = (itrs.earth_location.geodetic.lon.value, itrs.earth_location.geodetic.lat.value,
                                     itrs.earth_location.geodetic.height.value)
        return self.pos_lonLatHeight

    def get_psudo_velocity_lonLatHeight(self, time_delta: float=1) -> (float, float, float):
        # This is the position (in Lon,Lat,Hei) in one second, linearly interpolated with the current velocity.
        time1 = Time(self.jDay + self.jDayF, format='jd')
        delta_pos = self.pos_ITRS + self.vel_ITRS*time_delta
        delta_itrs = coord.ITRS(delta_pos[0] * u.km, delta_pos[1] * u.km, delta_pos[2] * u.km,
                                0 * u.km / u.s, 0 * u.km / u.s, 0 * u.km / u.s, obstime=time1)
        delta_lonLatHeight = (delta_itrs.earth_location.geodetic.lon.value,
                                 delta_itrs.earth_location.geodetic.lat.value,
                                 delta_itrs.earth_location.geodetic.height.value)
        return delta_lonLatHeight

    def calculate_distance_lonLatHeight(self, other_lon: float, other_lat: float, other_height: float) -> float:
        other_earth = coord.EarthLocation(lon=other_lon * u.deg, lat=other_lat * u.deg, height=other_height * u.km)
        other_ITRS = other_earth.get_itrs(obstime=Time(self.jDay + self.jDayF, format='jd'))
        other_ITRS = other_ITRS.cartesian
        distance = np.sqrt((self.pos_ITRS[0] - other_ITRS.x.value)**2 +
                           (self.pos_ITRS[1] - other_ITRS.y.value)**2 +
                           (self.pos_ITRS[2] - other_ITRS.z.value)**2)
        return distance


