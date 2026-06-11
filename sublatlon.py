# two different methods to compute the viewing and lighting directions for an asteroid, given its rotational state
# one with astroquery, one with kete

import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.io import ascii
from astropy.time import Time
from scipy.spatial.transform import Rotation as R
from astropy.table import Table
import kete
import pandas as pd


def sublatlon_Horizons(aster, observer, epoch, rot, vector_mode = False):
    # aster and observer are JPL horizons codes for the asteroid and observer respectively
    # epoch is epoch in JD (also for Horizons) OR horizons style dictionary

    # rot is a tuple containing the spin state parameters of the asteroid in the order
    # (lambda, beta, bigT, t0, phi0), both t0 and bigT should be in JD/days
    lamb = rot[0]
    beta = rot[1]
    bigT = rot[2]
    t0 = rot[3]
    phi0 = rot[4]
    
    horiz_obs_aster = Horizons(id=aster, location=observer, epochs=epoch)
    obs_aster = horiz_obs_aster.vectors()

    horiz_sun_aster = Horizons(id=aster, location='500@10', epochs=epoch)
    sun_aster = horiz_sun_aster.vectors()


    jepoch = obs_aster['datetime_jd'].data.data
    # now it points from the asteroid to the observer
    vect_aster_obs = np.array([-obs_aster['x'].data.data + obs_aster['lighttime'].data.data * obs_aster['vx'].data.data,\
                            -obs_aster['y'].data.data + obs_aster['lighttime'].data.data * obs_aster['vy'].data.data,\
                            -obs_aster['z'].data.data + obs_aster['lighttime'].data.data * obs_aster['vz'].data.data])
    vect_aster_obs = vect_aster_obs / np.sqrt(np.sum(vect_aster_obs**2, axis=0))

    vect_aster_sun = np.array([-sun_aster['x'].data.data + sun_aster['lighttime'].data.data * sun_aster['vx'].data.data,\
                            -sun_aster['y'].data.data + sun_aster['lighttime'].data.data * sun_aster['vy'].data.data,\
                            -sun_aster['z'].data.data + sun_aster['lighttime'].data.data * sun_aster['vz'].data.data])
    vect_aster_sun = vect_aster_sun / np.sqrt(np.sum(vect_aster_sun**2, axis=0))

    r1 = R.from_euler('z', 0-lamb, degrees=True)
    r2 = R.from_euler('y', 0-(90 - beta), degrees=True)
    r3 = R.from_euler('z', -2*np.pi * (jepoch - obs_aster['lighttime'].data.data - t0)/bigT)
    r4 = R.from_euler('z', -phi0, degrees = True)
    
    new_vect_aster_obs = np.transpose(r4.apply(r3.apply(r2.apply(r1.apply(np.transpose(vect_aster_obs))))))
    new_vect_aster_sun = np.transpose(r4.apply(r3.apply(r2.apply(r1.apply(np.transpose(vect_aster_sun))))))
    if vector_mode:
        t = Table([jepoch, new_vect_aster_obs[0], new_vect_aster_obs[1], new_vect_aster_obs[2], \
                   new_vect_aster_sun[0], new_vect_aster_sun[1], new_vect_aster_sun[2]],\
                   names = ['jd', 'Asteroid-to-observer x', 'Asteroid-to-observer y', 'Asteroid-to-observer z',\
                  'Asteroid-to-Sun x', 'Asteroid-to-Sun y', 'Asteroid-to-Sun z'])
        return(t)
    obs_longitude = (np.arctan2(new_vect_aster_obs[1], new_vect_aster_obs[0]) * 360 / (2 * np.pi)) % 360
    obs_latitude = 90 - np.arccos(new_vect_aster_obs[2]) * 360 / (2*np.pi)
    sun_longitude = (np.arctan2(new_vect_aster_sun[1], new_vect_aster_sun[0]) * 360 / (2 * np.pi)) % 360
    sun_latitude = 90 - np.arccos(new_vect_aster_sun[2]) * 360 / (2*np.pi)
    t = Table([jepoch, obs_latitude, obs_longitude, sun_latitude, sun_longitude], \
              names = ['jd', 'Sub-observer latitude', 'Sub-observer longitude', 'Sub-solar latitude', 'Sub-solar longitude'])
    return(t)

def sublatlon_kete(aster, observer, epoch, rot, vector_mode = False):    

    if type(epoch) == float or type(epoch) == int:
        epoch = [epoch]

    obj = kete.HorizonsProperties.fetch(aster).elements


    eph = np.zeros((len(epoch), 13))
    for i in range(len(epoch)):
        t = epoch[i]
        state = kete.propagate_n_body(obj.state, t)
        sun2obs = kete.spice.mpc_code_to_ecliptic(observer, t).pos
        sun2obj = state.pos
        obs2obj = -sun2obs + sun2obj
        # LT correction because I can. velocity is in AU/day, c = 173.14463 AU/day
        sun2obj_lt_sun = kete.propagate_two_body(state, t - sun2obj.r / (173.14463)).pos
        sun2obj_lt_obs = kete.propagate_two_body(state, t - obs2obj.r / (173.14463)).pos
        sun2obs_lt = kete.spice.mpc_code_to_ecliptic(observer, t - obs2obj.r/ (173.14463)).pos
        obs2obj_lt = -sun2obs_lt + sun2obj_lt_obs
        eph[i][0] = t
        eph[i][1] = (-sun2obj_lt_sun).angle_between(-obs2obj_lt)
        eph[i][2] = sun2obj_lt_sun.r
        eph[i][3] = obs2obj_lt.r
        eph[i][4], eph[i][5], eph[i][6] = sun2obj_lt_sun.raw
        eph[i][7], eph[i][8], eph[i][9] = obs2obj_lt.raw
        # everything above is ecliptic, only ra and dec and equatorial. 
        obs2obj_eq = obs2obj_lt.as_equatorial 
        eph[i][10] = obs2obj_eq.ra % 360 
        eph[i][11] = obs2obj_eq.dec
        eph[i][12] = (obs2obj_lt - sun2obj_lt_sun).angle_between(obs2obj_lt)
    ephemeris = pd.DataFrame(eph, columns=['jd', 'phase', 'r', 'delta', 'sun2obj_x', 'sun2obj_y', 'sun2obj_z',\
                                  'obs2obj_x', 'obs2obj_y', 'obs2obj_z', 'RA', 'Dec', 'elongation'])
    lamb = rot[0]
    beta = rot[1]
    bigT = rot[2]
    t0 = rot[3]
    phi0 = rot[4]
    # now it is a unit vector (ecliptic), points from the asteroid to the observer/sun
    sun_vect = np.array([-ephemeris.sun2obj_x.values, -ephemeris.sun2obj_y.values, -ephemeris.sun2obj_z.values])
    sun_vect = sun_vect / np.sqrt(np.sum(sun_vect**2, axis = 0))
    obs_vect = np.array([-ephemeris.obs2obj_x.values, -ephemeris.obs2obj_y.values, -ephemeris.obs2obj_z.values])
    obs_vect = obs_vect / np.sqrt(np.sum(obs_vect**2, axis = 0))
    # speed of light again
    dt_sun = ephemeris['r']/173.14463
    dt_obs = ephemeris['delta']/173.14463
    r1 = R.from_euler('z', 0-lamb, degrees=True)
    r2 = R.from_euler('y', 0-(90 - beta), degrees=True)
    r3_obs = R.from_euler('z', -2*np.pi * (ephemeris['jd'] - dt_obs - t0)/bigT)
    r4 = R.from_euler('z', -phi0, degrees = True)

    new_sun = np.transpose(r4.apply(r3_obs.apply(r2.apply(r1.apply(np.transpose(sun_vect))))))
    new_obs = np.transpose(r4.apply(r3_obs.apply(r2.apply(r1.apply(np.transpose(obs_vect))))))

    if vector_mode:
        t = Table([epoch, new_obs[0], new_obs[1], new_obs[2], \
                   new_sun[0], new_sun[1], new_sun[2]],\
                   names = ['jd', 'Asteroid-to-observer x', 'Asteroid-to-observer y', 'Asteroid-to-observer z',\
                  'Asteroid-to-Sun x', 'Asteroid-to-Sun y', 'Asteroid-to-Sun z'])
        return(t)
    obs_longitude = (np.arctan2(new_obs[1], new_obs[0]) * 360 / (2 * np.pi)) % 360
    obs_latitude = 90 - np.arccos(new_obs[2]) * 360 / (2*np.pi)
    sun_longitude = (np.arctan2(new_sun[1], new_sun[0]) * 360 / (2 * np.pi)) % 360
    sun_latitude = 90 - np.arccos(new_sun[2]) * 360 / (2*np.pi)
    t = Table([epoch, obs_latitude, obs_longitude, sun_latitude, sun_longitude], \
              names = ['jd', 'Sub-observer latitude', 'Sub-observer longitude', 'Sub-solar latitude', 'Sub-solar longitude'])
    return(t)
    return(new_sun, new_obs)
