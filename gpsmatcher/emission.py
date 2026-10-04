import os

from scipy.sparse import csr_matrix

os.environ["USE_PYGEOS"] = "1"
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import pickle

from gpsmatcher.utils import load_param, print_step, save_param


def chunks_dist_computation(chunk_gps, cand_edges, radius=150, alpha=0.1):
    """
    Compute distances between GPS points in a chunk and candidate edges.

    Parameters
    ----------
    chunk_gps : geopandas.GeoDataFrame
        GPS points in the current chunk.

    cand_edges : pandas.DataFrame
        Candidate edges with associated geometries.

    radius : float, optional
        Maximum distance for a GPS point to be matched to an edge. Default is 150 meters.

    alpha : float, optional
        Tuning parameter for the distance calculation. Default is 0.1.

    Returns
    -------
    dist_df : pandas.DataFrame
        DataFrame with columns 'id', 'edge', and 'dist' representing GPS points and distance to the candidate edges.
    """
    
    dist_df = chunk_gps[["id", "geometry"]].join(cand_edges, how='inner')
    dist_df.reset_index(drop=True, inplace=True)
    dist_df['dist'] = dist_df['geometry'].distance(dist_df['geometry_edge'])
    dist_df.drop(columns=['geometry', 'geometry_edge'], inplace=True)
    dist_df = dist_df[dist_df['dist'] < radius]
    dist_df['dist'] =  ((np.exp(-0.5* ((dist_df['dist'].values / 1000)/alpha)**2)/(np.sqrt(2*np.pi) * alpha))*100).astype(np.int16)
    return(dist_df[['id','edge','dist']])


def edge_geometry(G):
    """
    Geometry of the edges of a graph in EPSG:3035, as emission_matrix needs it: computed once per graph and passed to
    emission_matrix (and mm_gps), it saves rebuilding it at every call.

    Parameters
    ----------
    G : networkx.Graph
        Road network represented as a graph.

    Returns
    -------
    geom_df : geopandas.GeoDataFrame
        Columns 'edge' (edge_id) and 'geometry', one row per edge of G.
    """
    geom_df = (nx.to_pandas_edgelist(G)).rename(columns={'edge_id': 'edge'})
    geom_df = gpd.GeoDataFrame(geom_df[['edge', 'geometry']], geometry=geom_df['geometry'], crs=4326)
    geom_df.to_crs(3035, inplace=True)
    return geom_df

_edge_geometry = edge_geometry


def _chunk_emissions(chunk_gps, cand_pairs, lines, bounds, radius, alpha):
    """
    chunks_dist_computation on integer columns: the same pairs (point, candidate edge) in the same order, the same
    distances (shapely), the same values. The distance is not computed for a pair whose bounding box distance is
    already beyond the radius (plus 1 micrometre against rounding): such a pair is left out either way.
    """
    pairs = pd.DataFrame({'id': chunk_gps['id'].to_numpy(), 'point': np.arange(len(chunk_gps))},
                         index=chunk_gps.index).join(cand_pairs, how='inner')
    point, row = pairs['point'].to_numpy(), pairs['row'].to_numpy()
    x, y = shapely.get_x(np.asarray(chunk_gps.geometry.values)), shapely.get_y(np.asarray(chunk_gps.geometry.values))
    x, y, box = x[point], y[point], bounds[row]
    dx = np.maximum(np.maximum(box[:, 0] - x, x - box[:, 2]), 0)
    dy = np.maximum(np.maximum(box[:, 1] - y, y - box[:, 3]), 0)
    near = np.sqrt(dx ** 2 + dy ** 2) < radius + 1e-6
    dist = np.full(len(pairs), np.inf)
    dist[near] = shapely.distance(np.asarray(chunk_gps.geometry.values)[point[near]], lines[row[near]])
    kept = dist < radius
    dist = dist[kept]
    dist = ((np.exp(-0.5* ((dist / 1000)/alpha)**2)/(np.sqrt(2*np.pi) * alpha))*100).astype(np.int16)
    return pd.DataFrame({'id': pairs['id'].to_numpy()[kept], 'edge': pairs['edge'].to_numpy()[kept], 'dist': dist})


def emission_matrix(gps, G, dic_geohash, dic_candidates, alpha = 0.1, radius = 150, chunk=True, nb_chunks = 20, show_print=True, edge_geometry=None):  
    """
    Generate an emission matrix for map-matching based on GPS data and candidate edges.

    Parameters
    ----------
    gps : geopandas.GeoDataFrame
        GPS points with 'id' and 'geometry' columns.

    G : networkx.Graph
        Road network represented as a graph.

    dic_geohash : dict
        Dictionary mapping geohashes to integers.

    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.

    alpha : float, optional
        Tuning parameter for the distance calculation. Default is 0.1.

    radius : float, optional
        Maximum distance for a GPS point to be matched to an edge. Default is 150 meters.

    chunk : bool, optional
        Whether to process GPS data in chunks. Default is True.

    nb_chunks : int, optional
        Number of chunks to split the GPS data into if 'chunk' is True. Default is 20.

    edge_geometry : geopandas.GeoDataFrame, optional
        Output of edge_geometry(G), computed once per graph; computed here if not given.

    Returns
    -------
    emission_matrix : scipy.sparse.csr_matrix
        Emission matrix representing the likelihood of GPS points emitting from edges.
    """

    #with open('debug_emmision.pickle', 'wb') as handle:
    #    pickle.dump((gps, G, dic_geohash, dic_candidates), handle, protocol=pickle.HIGHEST_PROTOCOL)
    if show_print:
        print_step("Start process emission")
    geom_df = _edge_geometry(G) if edge_geometry is None else edge_geometry
    cand_edges = pd.DataFrame.from_dict(dic_candidates.items())
    cand_edges.columns=['geohash', 'edge']
    cand_edges['geohash_int'] = cand_edges['geohash'].map(dic_geohash)
    cand_edges.drop(columns=["geohash"], inplace=True)
    cand_edges = cand_edges.explode("edge")
    cand_edges.reset_index(drop=True, inplace=True)
    cand_edges = cand_edges.astype(np.uint32)
    if chunk:
        # as below, with the geometries taken by position after the joins instead of carried through them
        lines = np.asarray(geom_df.geometry.values)
        cand_pairs = cand_edges.merge(pd.DataFrame({'edge': geom_df['edge'].to_numpy(), 'row': np.arange(len(geom_df))}),
                                      on='edge')
        cand_pairs.set_index('geohash_int', inplace=True)
        bounds = shapely.bounds(lines)
        gps_splited = np.array_split(gps, nb_chunks)
        dist_df = pd.concat([_chunk_emissions(chunk_gps, cand_pairs, lines, bounds, radius, alpha)
                             for chunk_gps in gps_splited])
    else:
        cand_edges = cand_edges.merge(geom_df, on='edge')
        cand_edges.set_index('geohash_int', inplace=True)
        cand_edges.rename(columns={'geometry': 'geometry_edge'}, inplace=True)
        dist_df = gps[["id", "geometry"]].join(cand_edges, how='inner')
        dist_df.reset_index(drop=True, inplace=True)
        dist_df['dist'] = dist_df['geometry'].distance(dist_df['geometry_edge'])
        dist_df.drop(columns=['geometry', 'geometry_edge'], inplace=True)
        dist_df = dist_df[dist_df['dist'] < radius]
        dist_df['dist'] =  ((np.exp(-0.5* ((dist_df['dist'].values / 1000)/alpha)**2)/(np.sqrt(2*np.pi) * alpha))*100).astype(np.int16)
    nb_rows, nb_cols = int(dist_df['id'].max() + 1), len(G.edges())
    emission_matrix = csr_matrix((dist_df['dist'].values, ( dist_df['id'].values, dist_df['edge'].values)), shape=(nb_rows,nb_cols))
    return(emission_matrix)