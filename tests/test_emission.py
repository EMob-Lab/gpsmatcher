import geopandas as gpd
import networkx as nx

from generate import make_case
from gpsmatcher.emission import edge_geometry


def test_edge_geometry_is_the_frame_emission_matrix_builds():
    case = make_case(4)
    frame = edge_geometry(case.G_mm)
    assert list(frame.columns) == ['edge', 'geometry']
    assert frame.crs.to_epsg() == 3035
    edges = nx.to_pandas_edgelist(case.G_mm)
    assert list(frame['edge']) == list(edges['edge_id'])
    assert frame.geometry.geom_equals_exact(gpd.GeoSeries(edges['geometry'], crs=4326).to_crs(3035),
                                            tolerance=0).all()
