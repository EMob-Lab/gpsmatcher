"""
Reference copies of gpsmatcher 0.1.1 (tag v0.1.1): map_matching.py, gps.py and emission.py, function by function, so
that the identity tests compare the current code with the released one. Never edit them: they are the reference.
They call each other, not the package's versions of these functions; print_step is the package's (unchanged).
"""
import itertools

import geopandas as gpd
import networkx as nx
import numba
import numpy as np
import pandas as pd
import pygeohash_fast
from scipy.sparse import csr_matrix

from gpsmatcher.utils import print_step


# --- gpsmatcher/map_matching.py, get_predecessor ---
def get_predecessor(G, edge, edge_pre):
    if edge[0] == edge_pre[1]:
        return(edge_pre)
    else:
        path = nx.shortest_path(G, edge_pre[1], edge[0])
        return((path[-2], path[-1]))


# --- gpsmatcher/map_matching.py, get_sucessor ---
def get_sucessor(G, edge, edge_suc):
    if edge[1] == edge_suc[0]:
        return(edge_suc)
    else:
        path = nx.shortest_path(G, edge[1], edge_suc[0])
        return((path[0], path[1]))


# --- gpsmatcher/map_matching.py, get_new_edge_id ---
def get_new_edge_id(cand_edgeid, emit_p, edges):
    nb_rows, nb_cols = np.shape(cand_edgeid)
    new_edgeid = np.zeros((nb_rows,))
    for i in range(nb_rows):
        val_max = -1
        for j in range(nb_cols):
            edgeid = cand_edgeid[i, j]
            if edgeid != -1:
                val = emit_p[i, edgeid]
                if val > val_max:
                    val_max = val
                    edgeid_min = edgeid
        if val_max != -1:
            new_edgeid[i] = edgeid_min
        else:
            new_edgeid[i] = edges[i]
    return(new_edgeid)


# --- gpsmatcher/map_matching.py, correct_edge ---
def correct_edge(graph, states, state, sub_edges, id2edges, edges2id, emit_p):
    edge2state = dict(zip(sub_edges, states))
    edges = [id2edges[sub_edges[int(i)]] for i in state]
    edges_no_redundancy = [i[0] for i in itertools.groupby(edges)]
    edges_no_redundancy_count = [len(list(i[1])) for i in itertools.groupby(edges)]

    pre_edges = [-1]  + edges_no_redundancy[0:-1]
    next_edges = edges_no_redundancy[1:] + [-1] 
    sucessors = [get_sucessor(graph, edges_no_redundancy[i], next_edges[i]) for i in range(len(edges_no_redundancy) - 1)] + [-1]
    predecessors = [-1] + [get_predecessor(graph, edges_no_redundancy[i], pre_edges[i]) for i in range(1, len(edges_no_redundancy))]  
    cand_edges = [[sucessors[i], edges_no_redundancy[i], predecessors[i]] for i in range(len(sucessors) )] 
    # a predecessor or successor that is not a candidate edge of the trace is ignored (-1), as at the path ends
    cand_edgeid = [[edge2state.get(edges2id[i], -1) if i!=-1 else -1 for i in sub_list ] for sub_list in cand_edges]
    cand_edgeid = np.array(cand_edgeid)
    cand_edgeid = np.repeat(cand_edgeid, edges_no_redundancy_count, axis=0)
    new_edgeid = get_new_edge_id(cand_edgeid, emit_p, edges)
    return(new_edgeid)


# --- gpsmatcher/map_matching.py, format_result ---
def format_result(gps, gps_mm):
    """
    Format map-matching results.

    Parameters
    ----------
    gps : pandas.DataFrame
        GPS data.

    gps_mm : pandas.DataFrame
        Map-matched GPS data.

    Returns
    -------
    gps : pandas.DataFrame
        Updated GPS data with edge information.

    gps_mm : pandas.DataFrame
        Formatted map-matching results.
    """
    gps.reset_index(drop=True, inplace=True)
    gps_mm['shortest_path_nodes'] = gps_mm.apply(lambda row: row['map_match'][1], axis=1)
    gps_mm['edges'] = gps_mm.apply(lambda row: row['map_match'][0], axis=1)
    gps = gps[['lon', 'lat', 'ID_trip']]
    gps['edge'] = gps_mm['edges'].explode().reset_index(drop=True)
    #gps_mm = gps_mm["shortest_path_nodes"].reset_index()
    return(gps, gps_mm)


# --- gpsmatcher/map_matching.py, mm_gps ---
def mm_gps(gps, G_mm, trans, dic_candidates, id2edges, edges2id, alpha=0.1, radius=150):
    """
    Given the precomputation, map-match gps data (without multiprocessing)

    Parameters
    ----------
    gps : pandas.DataFrame
        GPS data.

    G_mm : networkx.DiGraph
        Processed road network graph for map-matching.

    trans : scipy.sparse.csr_matrix
        Transition matrix for map-matching.

    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.

    id2edges : dict
        Dictionary mapping edge_id to edge.

    alpha : float, optional
        Parameter for emission matrix computation.

    radius : int, optional
        Radius for candidate edges computation.

    Returns
    -------
    gps : pandas.DataFrame
        Updated GPS data with edge information.

    gps_mm : pandas.DataFrame
        Map-matched GPS data. Each ID_trip with most likely path in the graph.
    """
    gps, gps_mm, dic_geohash, dic_candidates = process_gps(gps, dic_candidates)
    emit = emission_matrix(gps, G_mm, dic_geohash, dic_candidates, alpha = alpha, radius = radius)
    gps_mm['map_match'] = gps_mm.apply(lambda row: one_traj_mm(row['traj'], G_mm, trans, emit, id2edges, edges2id, row['sub_edges'], row['first_emission'], row['last_emission']), axis=1)
    gps, gps_mm = format_result(gps, gps_mm)
    return(G_mm, gps, gps_mm)


# --- gpsmatcher/map_matching.py, one_traj_mm ---
def one_traj_mm(GPS_traj, graph, transition_matrix, emission_matrix, id2edges, edges2id,sub_edges, start, end):
    """
    Perform map-matching on a single GPS trajectory.

    Parameters
    ----------
    GPS_traj : numpy.ndarray
        Array containing GPS observations.

    graph : networkx.DiGraph
        The road network graph.

    transition_matrix : scipy.sparse.csr_matrix
        Transition matrix for the graph.

    emission_matrix : scipy.sparse.csr_matrix
        Emission matrix for the graph.

    id2edges : dict
        Dictionary mapping edge_id to edge of the graph.

    sub_edges : list
        List of edge IDs candidates considered for map-matching.

    start : int
        Start index in emission_matrix.

    end : int
        End index in emission_matrix).

    Returns
    -------
    edges : list
        List of edges matched to the GPS trajectory.

    path : list
        List of nodes representing the matched path on the road network graph.
    """
    try:
        emit_p = (emission_matrix[start:end+1, : ][:, sub_edges].toarray())/100
        trans_p = (transition_matrix[sub_edges, :][:, sub_edges].toarray())/100
        start_p = np.ones(len(sub_edges))/len(sub_edges)
        obs = np.array([i for i in range(len(GPS_traj))])
        states = np.array([i for i in range(len(sub_edges))])
        (V,prob, state,path) = fast_viterbi(obs, states, start_p, trans_p, emit_p)
        if prob != 0:
            edges = [id2edges[sub_edges[int(i)]] for i in state]
            new_edges = correct_edge(graph, states, state, sub_edges,id2edges, edges2id,emit_p)
            new_edges = [id2edges[sub_edges[int(i)]] for i in new_edges]
            edges_without_redundancy = [i[0] for i in itertools.groupby(new_edges)]
            #edges_without_redundancy = [i[0] for i in itertools.groupby(edges)] #be careful
            path = [nx.shortest_path(graph, edges_without_redundancy[i][1], edges_without_redundancy[i+1][0]) for i in range(len(edges_without_redundancy) - 1)]
            path = [edges_without_redundancy[0][0]] + [item for sublist in path for item in sublist] + [edges_without_redundancy[-1][1]]
            return(new_edges, path)
        else:
            return([np.nan]*len(GPS_traj),[])
    except:
        return(["Problem"]*len(GPS_traj), ["Problem"])


# --- gpsmatcher/map_matching.py, fast_viterbi ---
@numba.jit(nopython=True)
def fast_viterbi(obs, states, start_p, trans_p, emit_p):
    """
    Implement the Viterbi algorithm for map-matching.

    Parameters
    ----------
    obs : numpy.ndarray
        Array containing observation indices.

    states : numpy.ndarray
        Array containing state indices.

    start_p : numpy.ndarray
        Initial state probabilities.

    trans_p : numpy.ndarray
        Transition probabilities.

    emit_p : numpy.ndarray
        Emission probabilities.

    Returns
    -------
    V : numpy.ndarray
        Viterbi matrix.

    prob : float
        Probability of the most likely path.

    seq_nodes : numpy.ndarray
        Sequence of node indices representing the most likely path.

    path : numpy.ndarray
        most likely path matrix.
    """
    nb_obs = obs.shape[0]
    nb_states = states.shape[0]

    V = np.zeros((nb_obs, nb_states))
    path = np.zeros((1,nb_states))

    # Initialize base cases (t == 0)
    for idx, y in enumerate(states):
        V[0, y] = start_p[y] * emit_p[ obs[0],y]
        path[0, idx] = y

    for itime in range(1, nb_obs):   
        states_cand = np.where(emit_p[ obs[itime],:] != 0)[0]
        newpath = np.zeros((itime + 1 ,states_cand.shape[0]))
        states_bis =np.where(V[itime-1, :] != 0)[0]
        for idx, y in enumerate(states_cand):
            prob = 0
            for y0 in states_bis:
                if V[itime-1, y0] * trans_p[y0,y] * emit_p[ obs[itime],y] > prob:
                    prob = V[itime-1, y0] * trans_p[y0,y] * emit_p[ obs[itime],y]
                    state = y0
            V[itime, y] = prob
            if prob>0: #check this if problem
                row = np.where(path[-1,:] == state)[0][0]
                newpath[:itime, idx] = path[:, row]
                newpath[-1, idx] = y
        # Don't need to remember the old paths
        path = newpath

    idx_max = np.argmax(V[itime,:])
    (prob, state) = V[itime, idx_max], idx_max
    if prob !=0:
        row = np.where(path[-1,:] == state)[0][0]
        seq_nodes = path[:, row]
    else:
        path = np.zeros((1,nb_states))
        seq_nodes = np.zeros((1,nb_states))[:,0]
    return (V,prob, seq_nodes,path)

# --- gpsmatcher/gps.py, process_gps ---
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

    gps['edge'] = gps['geohash'].map(dic_candidates)
    gps['geohash_int'] = gps['geohash'].map(dic_geohash)
    gps.set_index('geohash_int', inplace=True)
    gps = gpd.GeoDataFrame(gps, geometry=gpd.points_from_xy(gps['lon'], gps['lat']), crs=4326)
    gps.to_crs(3035, inplace=True)
    gps['points'] = list(zip(gps['lon'], gps['lat']))
    
    gps_mm = gps.groupby('ID_trip').agg({'points' : lambda row: np.array(list(row), dtype=float),  
                                            "id":['first', 'last'], 
                                            "edge":  sum})
    gps_mm.columns = ['traj', 'first_emission', 'last_emission', 'sub_edges']
    gps_mm = gps_mm[gps_mm['sub_edges']!=0]
    gps_mm['sub_edges'] = gps_mm['sub_edges'].apply(lambda row: list(set(row)))
    gps.drop(columns=['geohash', 'points', "edge"], inplace=True)
    return(gps, gps_mm, dic_geohash, dic_candidates)

# --- gpsmatcher/emission.py, chunks_dist_computation ---
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


# --- gpsmatcher/emission.py, emission_matrix ---
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
