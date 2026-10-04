import os

os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeohash_fast


def process_gps(gps, cand_edge):
    """
    Process GPS data by encoding geohashes, mapping edges, and creating a GeoDataFrame.

    Parameters
    ----------
    gps : pandas.DataFrame
        DataFrame containing GPS data with columns 'lon', 'lat'.

    cand_edge : dict
        Dictionary mapping geohashes to candidate edges.

    Returns
    -------
    gps : gpd.GeoDataFrame
        GeoDataFrame with additional columns 'id', 'geohash_int', and 'geometry'.
    
    gps_mm : pandas.DataFrame
        DataFrame containing aggregated GPS data per trip, with columns 'traj', 'first_emission', 'last_emission', 'sub_edges'.
    
    dic_geohash : dict
        Dictionary mapping geohashes to integer indices.
    
    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.
    """
    gps.reset_index(drop=True, inplace=True)
    gps['id'] = range(len(gps))
    gps["id"] = gps["id"].astype(np.uint32)
    gps['geohash']   = pygeohash_fast.encode_many(gps['lon'].values, gps['lat'].values, 8)
    all_geohashes = set(gps['geohash'] )
    dic_geohash = dict(zip(all_geohashes, range(len(all_geohashes))))
    dic_candidates = {k: cand_edge[k] for k in cand_edge.keys() & all_geohashes}

    gps['geohash_int'] = gps['geohash'].map(dic_geohash)
    gps.set_index('geohash_int', inplace=True)
    gps = gpd.GeoDataFrame(gps, geometry=gpd.points_from_xy(gps['lon'], gps['lat']), crs=4326)
    gps.to_crs(3035, inplace=True)
    gps_mm = trips(gps, dic_candidates)
    gps.drop(columns=['geohash'], inplace=True)
    return(gps, gps_mm, dic_geohash, dic_candidates)

def trips(gps, dic_candidates):
    """
    Per trip (ID_trip, sorted), its points, first and last point ids and candidate edges, as version 0.1.1 built
    them with a groupby: trips without any candidate edge are left out.

    Parameters
    ----------
    gps : pandas.DataFrame
        GPS data with columns 'lon', 'lat', 'ID_trip', 'id' and 'geohash'.

    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.

    Returns
    -------
    gps_mm : pandas.DataFrame
        Columns 'traj' (array of lon, lat), 'first_emission', 'last_emission' and 'sub_edges'.
    """
    codes, keys = pd.factorize(gps['ID_trip'], sort=True)
    kept = codes >= 0                                   # groupby leaves out missing ID_trip
    codes = codes[kept]
    order = np.argsort(codes, kind='stable')            # the points of each trip, in their order
    bounds = np.r_[0, np.cumsum(np.bincount(codes, minlength=len(keys)))]
    points = np.column_stack((gps['lon'].to_numpy(), gps['lat'].to_numpy())).astype(float)[kept][order]
    ids = gps['id'].to_numpy()[kept][order]
    geohashes = gps['geohash'].to_numpy()[kept][order]

    traj, first, last, sub_edges, with_candidates = [], [], [], [], []
    for k in range(len(keys)):
        lo, hi = bounds[k], bounds[k + 1]
        # 0.1.1 summed the candidate lists of the points, then took list(set(...)): adding each geohash's list once,
        # in the order of the points, inserts the same new elements in the same order, so the set is the same
        edges, seen = set(), set()
        for geohash in geohashes[lo:hi]:
            if geohash not in seen:
                seen.add(geohash)
                candidates = dic_candidates.get(geohash)
                if candidates is not None:
                    edges.update(candidates)
        traj.append(points[lo:hi])
        first.append(ids[lo])
        last.append(ids[hi - 1])
        sub_edges.append(list(edges))
        with_candidates.append(bool(edges))
    gps_mm = pd.DataFrame({'traj': pd.Series(traj, dtype=object), 'first_emission': np.array(first, dtype=ids.dtype),
                           'last_emission': np.array(last, dtype=ids.dtype),
                           'sub_edges': pd.Series(sub_edges, dtype=object)})
    gps_mm.index = pd.Index(keys, name='ID_trip')
    return gps_mm[np.array(with_candidates, dtype=bool)]
