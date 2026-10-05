# gpsmatcher

gpsmatcher is a Python library for map-matching GPS traces to a transportation network. 
The library boasts high-speed processing, achieving approximately 5000 to 50000 points per second, depending on the specified parameters. 
Users have the flexibility to employ their custom graph or utilize OpenStreetMap (OSM) data via the osmnx library.



## Installation

You can easily install gpsmatcher using the Python package manager pip:

```bash
pip install gpsmatcher
```

gpsmatcher also depends on recent functionalities of pygeohash, not yet available on PyPi. Please install it directly from GitHub:

```bash
pip install 'git+https://github.com/wdm0006/pygeohash.git@a89e6e794f0dde4e066048ec16c867e945a680c1'
```

## Usage

Here is a minimal example code for matching GPS points to a transportation network from osmnx:
```python
import osmnx as ox
import pandas as pd
import networkx as nx
from gpsmatcher.map_matching import gps_file_mm

# It is easy to work from osmnx graph. Edges are road sections and nodes intersections. Weight have to be travel time.
# You can also work with your own grah, be sure it is a networkx graph with travel time as weight. Also, the coordinates of the nodes have
# to be set with attributes 'x' and 'y' in the graph
G = ox.graph_from_place('Lyon', network_type = 'drive', simplify=False)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
G = ox.simplify_graph(G)
G = nx.DiGraph(G)
edges2tt =nx.get_edge_attributes(G, 'travel_time')
nx.set_edge_attributes(G, edges2tt, 'weight')

# Exemple of gps data
gps = pd.DataFrame([(1, 4.861566712580813, 45.74692597206301),
                    (1, 4.8625001213195755, 45.74584036222316),
                    (1, 4.864549329010423, 45.74592271957176),
                    (1, 4.866566350193037 , 45.745293805830876),
                    (1, 4.8672851822053556, 45.744327960213006),]
                   , columns = ['ID_trip', 'lon', 'lat'])

# Perform map matching
G, gps_mm, gps_mm_sp = gps_file_mm(G, gps)
```
When employing your custom graph, it is imperative to confirm that the graph includes travel time specified as edge weights with networkx. 
Moreover, ensure that the coordinates of the nodes are set with attributes 'x' and 'y' within the nodes of the graph. 
Optionally, you can add your own geometry using the "geometry" attribute for the edges. 
If not provided, the geometry of the edges will be simplified based on the nodes' coordinates.

Here is the miniaml example code of matching gps points to a transportation network using networkx.

```python
import pandas as pd
import networkx as nx
from gpsmatcher.map_matching import gps_file_mm

edges_weight = [(3991897779, 11520092777, 11.33),
                 (11520092763, 11520092764, 13.64),
                 (11520092764, 11520092765, 14.04),
                 (11520092765, 11520092766, 14.69),
                 (11520092766, 11520092767, 15.26),
                 (11520092767, 11520092768, 15.57),
                 (11520092768, 11520092769, 15.77),
                 (11520092769, 11520092770, 15.59),
                 (11520092770, 3991897832, 15.29),
                 (3991897832, 11520092781, 11.9),
                 (11520092777, 3991897855, 11.27),
                 (11520092781, 3991897779, 11.6),
                 (11520092782, 11520092783, 14.4),
                 (11520092783, 11520092784, 14.75),
                 (11520092784, 11520092785, 15.41),
                 (11520092785, 11520092786, 15.81),
                 (11520092786, 11520092787, 15.61),
                 (3991897855, 11520092782, 14.19)] 

lon_lat_coord = {3991897779: (4.825, 45.64),
                 11520092777: (4.828, 45.642),
                 11520092763: (4.8, 45.605),
                 11520092764: (4.803, 45.608),
                 11520092765: (4.807, 45.611),
                 11520092766: (4.81, 45.614),
                 11520092767: (4.812, 45.618),
                 11520092768: (4.814, 45.622),
                 11520092769: (4.816, 45.626),
                 11520092770: (4.818, 45.63),
                 3991897832: (4.821, 45.634),
                 11520092781: (4.823, 45.637),
                 3991897855: (4.83, 45.645),
                 11520092782: (4.833, 45.648),
                 11520092783: (4.836, 45.651),
                 11520092784: (4.839, 45.655),
                 11520092785: (4.841, 45.659),
                 11520092786: (4.843, 45.663),
                 11520092787: (4.845, 45.667)}

G = nx.DiGraph()
G.add_weighted_edges_from(edges_weight)
lon_coord = {key: value[0] for key, value in lon_lat_coord.items()}
lat_coord = {key: value[1] for key, value in lon_lat_coord.items()}
nx.set_node_attributes(G, lon_coord, 'x')
nx.set_node_attributes(G, lat_coord, 'y')

# Exemple of gps data
gps = pd.DataFrame([(1, 4.799618, 45.605151),
                    (1, 4.814109, 45.622385),
                    (1, 4.823339, 45.637911),
                    (1, 4.83678 , 45.65168),
                    (1, 4.844435, 45.666307),]
                   , columns = ['ID_trip', 'lon', 'lat'])

# Perform map matching
G, gps_mm, gps_mm_sp = gps_file_mm(G, gps, folder_name="mini_graph", save =False, show_print=False)
```

## Matching many batches on the same graph

`gps_file_mm` prepares the graph and matches in one call. To match several batches of traces on the same graph,
prepare it once with `mm_precomputation` and call `mm_gps` for each batch. The geometry of the edges can also be
computed once and passed to each call:

```python
from gpsmatcher.emission import edge_geometry
from gpsmatcher.map_matching import mm_gps, mm_precomputation

G_mm, trans, dic_candidates, id2edges, edges2id = mm_precomputation(G, save=False, show_print=False)
geometry = edge_geometry(G_mm)
for gps in batches:
    _, gps_matched, gps_mm = mm_gps(gps, G_mm, trans, dic_candidates, id2edges, edges2id, edge_geometry=geometry)
```

## Tests

```bash
pip install -e . pytest
python -m pytest tests
```

`tests/test_identity.py` compares the current code with the functions of version 0.1.1 (`tests/reference_0_1_1.py`)
on generated graphs and traces (`tests/generate.py`): any change must keep the results identical, or say why not.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License

[GNU LGPLv3](https://choosealicense.com/licenses/lgpl-3.0/)