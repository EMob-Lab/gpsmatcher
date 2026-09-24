import os

from scipy.sparse import csr_matrix

os.environ["USE_PYGEOS"] = "1"
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
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


def emission_matrix(gps, G, dic_geohash, dic_candidates, alpha = 0.1, radius = 150, chunk=True, nb_chunks = 20, show_print=True):  
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

    Returns
    -------
    emission_matrix : scipy.sparse.csr_matrix
        Emission matrix representing the likelihood of GPS points emitting from edges.
    """

    #with open('debug_emmision.pickle', 'wb') as handle:
    #    pickle.dump((gps, G, dic_geohash, dic_candidates), handle, protocol=pickle.HIGHEST_PROTOCOL)
    if show_print:
        print_step("Start process emission")
    geom_df = (nx.to_pandas_edgelist(G)).rename(columns={'edge_id': 'edge'})
    geom_df = gpd.GeoDataFrame(geom_df[['edge', 'geometry']], geometry=geom_df['geometry'], crs=4326)
    geom_df.to_crs(3035, inplace=True)
    cand_edges = pd.DataFrame.from_dict(dic_candidates.items())
    cand_edges.columns=['geohash', 'edge']
    cand_edges['geohash_int'] = cand_edges['geohash'].map(dic_geohash)
    cand_edges.drop(columns=["geohash"], inplace=True)
    cand_edges = cand_edges.explode("edge")
    cand_edges.reset_index(drop=True, inplace=True)
    cand_edges = cand_edges.astype(np.uint32)
    cand_edges = cand_edges.merge(geom_df, on='edge')
    cand_edges.set_index('geohash_int', inplace=True)
    cand_edges.rename(columns={'geometry': 'geometry_edge'}, inplace=True)
    if chunk:
        gps_splited = np.array_split(gps, nb_chunks)
        dist_df = []
        for idx, chunk_gps in enumerate(gps_splited):
            dist_df.append(chunks_dist_computation(chunk_gps, cand_edges, radius=radius, alpha=alpha))
        dist_df = pd.concat(dist_df)

    else:
        dist_df = gps[["id", "geometry"]].join(cand_edges, how='inner')
        dist_df.reset_index(drop=True, inplace=True)
        dist_df['dist'] = dist_df['geometry'].distance(dist_df['geometry_edge'])
        dist_df.drop(columns=['geometry', 'geometry_edge'], inplace=True)
        dist_df = dist_df[dist_df['dist'] < radius]
        dist_df['dist'] =  ((np.exp(-0.5* ((dist_df['dist'].values / 1000)/alpha)**2)/(np.sqrt(2*np.pi) * alpha))*100).astype(np.int16)
    nb_rows, nb_cols = int(dist_df['id'].max() + 1), len(G.edges())
    emission_matrix = csr_matrix((dist_df['dist'].values, ( dist_df['id'].values, dist_df['edge'].values)), shape=(nb_rows,nb_cols))
    return(emission_matrix)