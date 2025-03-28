import copy
from typing import Dict, Tuple, Text, List, Optional
import random
import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Dict
import math
import geopandas as gpd
import libpysal
import math
import momepy
import networkx as nx
import pandas as pd
from geopandas import GeoSeries, GeoDataFrame
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint, LineString, MultiLineString
from shapely.ops import snap, polygonize

from urban_planning.envs import city_config
from ReinforcementLearning.utils import load_yaml, load_pickle, simplify_by_angle, simplify_by_distance, get_boundary_edges, \
    slice_polygon_from_edge, slice_polygon_from_corner, get_intersection_polygon_with_maximum_area

def set_land_use_array_from_dict(land_use_array: np.ndarray, land_use_dict: Dict, land_use_id_map: Dict) -> None:
    for land_use, value in land_use_dict.items():
        land_use_array[land_use_id_map[land_use]] = value

class PlanClient(object):
    """Defines the PlanClient class."""
    PLAN_ORDER = np.array([
        city_config.HOSPITAL_L,
        city_config.SCHOOL,
        city_config.HOSPITAL_S,
        city_config.RECREATION,
        city_config.RESIDENTIAL,
        city_config.GREEN_L,
        city_config.WASTEMGMT,
        city_config.BUSINESS,
        city_config.GREEN_S,
        city_config.OFFICE], dtype=np.int32)
    EPSILON = 1E-4
    DEG_TOL = 1
    SNAP_EPSILON = 1

    def __init__(self, objectives_plan_file: Text, init_plan_file: Text) -> None:
        """Creates a PlanClient client object.

        Args:
            objectives_plan_file: Path to the file of community objectives.
            init_plan_file: Path to the file of initial plan.
        """
        file_path = 'urban_planning/cfg/**/{}.yaml'.format(objectives_plan_file)
        self.objectives = load_yaml(file_path)
        file_path = 'urban_planning/cfg/**/{}.pickle'.format(init_plan_file)
        self.init_plan = load_pickle(file_path)
        self.init_objectives()
        self.init_constraints()
        self.restore_plan()

    def init_objectives(self) -> None:
        """Initializes objectives of different land uses."""
        objectives = self.objectives
        self._grid_cols = objectives['community']['grid_cols']
        self._grid_rows = objectives['community']['grid_rows']
        self._cell_edge_length = objectives['community']['cell_edge_length']
        self._cell_area = self._cell_edge_length ** 2

        land_use_types_to_plan = objectives['objectives']['land_use']
        land_use_to_plan = np.array(
            [city_config.LAND_USE_ID_MAP[land_use] for land_use in land_use_types_to_plan],
            dtype=np.int32)
        custom_planning_order = objectives['objectives'].get('custom_planning_order', False)
        if custom_planning_order:
            self._plan_order = land_use_to_plan
        else:
            self._plan_order = self.PLAN_ORDER[np.isin(self.PLAN_ORDER, land_use_to_plan)]

        self._required_plan_ratio = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_plan_ratio = objectives['objectives']['ratio']
        set_land_use_array_from_dict(self._required_plan_ratio, required_plan_ratio, city_config.LAND_USE_ID_MAP)

        self._required_plan_count = np.zeros(city_config.NUM_TYPES, dtype=np.int32)
        required_plan_count = objectives['objectives']['count']
        set_land_use_array_from_dict(
            self._required_plan_count, required_plan_count, city_config.LAND_USE_ID_MAP)

    def init_constraints(self) -> None:
        """Initializes constraints of different land uses."""
        objectives = self.objectives
        self.init_specific_constraints(objectives['constraints'])
        self.init_common_constraints()

    def init_specific_constraints(self, constraints: Dict) -> None:
        """Initializes constraints of specific land uses.

        Args:
            constraints: Constraints of specific land uses.
        """
        self._required_max_area = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_max_area = constraints['max_area']
        set_land_use_array_from_dict(self._required_max_area, required_max_area, city_config.LAND_USE_ID_MAP)

        self._required_min_area = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_min_area = constraints['min_area']
        set_land_use_array_from_dict(self._required_min_area, required_min_area, city_config.LAND_USE_ID_MAP)

        self._required_max_edge_length = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_max_edge_length = constraints['max_edge_length']
        set_land_use_array_from_dict(
            self._required_max_edge_length, required_max_edge_length, city_config.LAND_USE_ID_MAP)

        self._required_min_edge_length = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_min_edge_length = constraints['min_edge_length']
        set_land_use_array_from_dict(
            self._required_min_edge_length, required_min_edge_length, city_config.LAND_USE_ID_MAP)

    def init_common_constraints(self) -> None:
        """Initializes common constraints of difference land uses."""
        self._common_max_area = self._required_max_area[self._plan_order].max()
        self._common_min_area = self._required_min_area[self._plan_order].min()
        self._common_max_edge_length = self._required_max_edge_length[self._plan_order].max()
        self._common_min_edge_length = self._required_min_edge_length[self._plan_order].min()
        self._min_edge_grid = round(self._common_min_edge_length / self._cell_edge_length)
        self._max_edge_grid = round(self._common_max_edge_length / self._cell_edge_length)

        self.max_roads = 15
        self.current_roads = 0
        self.ids = []

    def get_common_max_area(self) -> float:
        """Returns the required maximum area of all land uses."""
        return self._common_max_area

    def get_common_max_edge_length(self) -> float:
        """Returns the required maximum edge length of all land uses."""
        return self._common_max_edge_length

    def calculate_road_volume_for_polygon(self,polygon,polygon_type):
        """Calculate road volume for a specified polygon index."""
        # Extract the polygon geometry using the index

        if polygon.geom_type != "Polygon":
            print(f"Does not correspond to a Polygon.")
            return
         # Assuming 'type' contains the polygon type

        # Get the exterior boundary of the polygon as LineString (broken into segments)
        # boundary_coords = list(polygon.exterior.coords)
        # polygon_sides = [LineString([boundary_coords[i], boundary_coords[i + 1]]) for i in range(len(boundary_coords) - 1)]

        # Filter for LineString geometries in the GeoDataFrame
        # linestrings = gdf[gdf['geometry'].geom_type == "LineString"]

        # Find matching LineString geometries, including swapped coordinates
        # matching_lines = linestrings[linestrings.geometry.apply(
        #     lambda line: any(
        #         line.equals_exact(side, tolerance=2) or
        #         line.overlaps(side) or
        #         line.equals_exact(LineString(side.coords[::-1]), tolerance=2) for side in polygon_sides
        #     )
        # )]

        # Calculate population volume
        density = city_config.DENSITY.get(polygon_type, 0)  # Get density based on polygon type
        area_in_hundred_square_meters = polygon.area / 10000  # Convert area from m² to ha
        population_volume = area_in_hundred_square_meters * density  # Volume based on density

        return population_volume




        # Calculate the total length of matching lines
        # total_length = sum(line.length for line in matching_lines.geometry)


        # if total_length > 0:
        #     # Divide the population volume by the total length of matching lines
        #     volume_per_length_unit = population_volume / total_length

        #     # Update road volume for each matching LineString based on its length
        #     for line in matching_lines.geometry:
        #         line_index = gdf[gdf.geometry == line].index[0]  # Get the index of the LineString
        #         gdf.at[line_index, 'volume'] += volume_per_length_unit * line.length  # Update the road volume

    def calculate_volume(self):
        gdf = self._gdf.copy()
        # Calculate road volume for every polygon in the GeoDataFrame using the index
        polygon_indices = gdf[gdf['geometry'].geom_type == "Polygon"].index
        for polygon_index in polygon_indices:
            polygon_row = gdf.loc[polygon_index]
            polygon = polygon_row['geometry']
            polygon_type = polygon_row['type']
            gdf.at[polygon_index,'population'] = self.calculate_road_volume_for_polygon(polygon,polygon_type)
        return gdf['population']

    def _add_domain_features(self) -> None:
        """Adds domain features to the gdf."""
        self._gdf['rect'] = momepy.Rectangularity(self._gdf[self._gdf.geom_type == 'Polygon']).series
        self._gdf['eqi'] = momepy.EquivalentRectangularIndex(self._gdf[self._gdf.geom_type == 'Polygon']).series
        self._gdf['sc'] = momepy.SquareCompactness(self._gdf[self._gdf.geom_type == 'Polygon']).series
        self._gdf['population'] = self.calculate_volume()
        # self._gdf.loc[self._gdf['geometry'].geom_type == 'Point', 'population'] = np.random.randint(200, 1001, size=len(self._gdf[self._gdf['geometry'].geom_type == 'Point']))

    
    def get_init_plan(self) -> Dict:
        """Returns the initial plan."""
        return self.init_plan

    def restore_plan(self) -> None:
        """Restore the initial plan."""
        self._initial_gdf = self.init_plan['gdf']
        self._gdf = copy.deepcopy(self._initial_gdf)
        self._add_domain_features()
        self._load_concept(self.init_plan.get('concept', list()))
        self._rule_constraints = self.init_plan.get('rule_constraints', False)
        self._init_stats()
        self._init_counter()

    def load_plan(self, gdf: GeoDataFrame) -> None:
        """Loads the given plan.

        Args:
            gdf: The plan to load.
        """
        self._gdf = copy.deepcopy(gdf)

    def _load_concept(self, concept: List) -> None:
        """Initializes the planning concept of the plan.

        Args:
            concept: The planning concept.
        """
        self._concept = concept

    def _init_stats(self) -> None:
        """Initialize statistics of the plan."""
        gdf = self._gdf[self._gdf['existence'] == True]
        total_area = gdf.area.sum()*self._cell_area
        outside_area = gdf[gdf['type'] == city_config.OUTSIDE].area.sum()*self._cell_area
        self._community_area = total_area - outside_area

        self._required_plan_area = self._community_area * self._required_plan_ratio
        self._plan_area = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        self._plan_ratio = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        self._plan_count = np.zeros(city_config.NUM_TYPES, dtype=np.int32)
        self._compute_stats()

    def _compute_stats(self) -> None:
        """Update statistics of the plan."""
        gdf = self._gdf[self._gdf['existence'] == True]
        for land_use in city_config.LAND_USE_ID:
            area = gdf[gdf['type'] == land_use].area.sum() * self._cell_area
            self._plan_area[land_use] = area
            self._plan_ratio[land_use] = area / self._community_area
            self._plan_count[land_use] = len(gdf[gdf['type'] == land_use])

    def _update_stats(self, land_use_type: int, land_use_area: float) -> None:
        """Update statistics of the plan given new land_use.

        Args:
            land_use_type: land use type of the new land use.
            land_use_area: area of the new land use.
        """
        self._plan_count[land_use_type] += 1

        self._plan_area[land_use_type] += land_use_area
        self._plan_ratio[land_use_type] = self._plan_area[land_use_type]/self._community_area

        self._plan_area[city_config.FEASIBLE] -= land_use_area
        self._plan_ratio[city_config.FEASIBLE] = self._plan_area[city_config.FEASIBLE]/self._community_area

    def _init_counter(self):
        """Initialize action ID counter."""
        self._action_id = self._gdf.index.max()

    def _counter(self):
        """Return counter and add one."""
        self._action_id += 1
        return self._action_id

    def unplan_all_land_use(self) -> None:
        """Unplan all land use"""
        self._gdf = copy.deepcopy(self._initial_gdf)
        self._add_domain_features()
        self._compute_stats()
        self._init_counter()

    def freeze_land_use(self, land_use_gdf: GeoDataFrame) -> None:
        """Freeze the given land use.

        Args:
            land_use_gdf: The land use to freeze.
        """
        self._initial_gdf = copy.deepcopy(land_use_gdf)

    def fill_leftover(self) -> None:
        """Fill leftover space."""
        self._gdf.loc[(self._gdf['type'] == city_config.FEASIBLE) & (self._gdf['existence'] == True),
                      'type'] = city_config.GREEN_S

    def snapshot(self):
        """Snapshot the gdf."""
        snapshot = copy.deepcopy(self._gdf)
        return snapshot

    def build_all_road(self):
        """Build all road"""
        self._gdf.loc[(self._gdf['type'] == city_config.BOUNDARY) & (self._gdf['existence'] == True),
                      'type'] = city_config.ROAD

    def is_land_use_done(self) -> bool:
        """Check if the land_use planning is done."""
        ratio_satisfication = (self._plan_ratio - self._required_plan_ratio >= -self.EPSILON)[self._plan_order].all()
        count_satisfication = (self._plan_count >= self._required_plan_count)[self._plan_order].all()
        done = ratio_satisfication and count_satisfication
        return done

    def get_gdf(self) -> GeoDataFrame:
        """Return the current GDF."""
        return self._gdf

    def _get_current_gdf_and_graph(self) -> Tuple[GeoDataFrame, nx.Graph]:
        """Return the current GDF and graph.

        Returns:
            gdf: current GDF.
            graph: current graph.
                   Nodes are land_use, road intersections and road segments. Edges are spatial contiguity.
        """
        gdf = copy.deepcopy(self._gdf[self._gdf['existence'] == True])
        w = libpysal.weights.fuzzy_contiguity(gdf)
        graph = w.to_networkx()
        self._current_gdf = gdf
        self._current_graph = graph
        return gdf, graph

    def _filter_block_by_rule(self,
                              gdf: GeoDataFrame, feasible_blocks_id: np.ndarray, land_use_type: int) -> np.ndarray:
        """Filter feasible blocks by rule.

        Args:
            gdf: current GDF.
            feasible_blocks_id: feasible blocks ID.
            land_use_type: land use type.

        Returns:
            filtered_blocks_id: filtered blocks.
        """
        if land_use_type == city_config.SCHOOL:
            hospital_l = gdf[gdf['type'] == city_config.HOSPITAL_L].unary_union
            near_hospital_l = gdf[(gdf.geom_type == 'Polygon') & (gdf.intersects(hospital_l))].index.to_numpy()
            filtered_blocks_id = np.setdiff1d(feasible_blocks_id, near_hospital_l)
        elif land_use_type == city_config.HOSPITAL_S:
            school = gdf[(gdf['type'] == city_config.SCHOOL) | (gdf['type'] == city_config.HOSPITAL_L) | (gdf['type'] == city_config.HOSPITAL_S)].unary_union
            near_school = gdf[(gdf.geom_type == 'Polygon') & (gdf.intersects(school))].index.to_numpy()
            filtered_blocks_id = np.setdiff1d(feasible_blocks_id, near_school)
        else:
            filtered_blocks_id = feasible_blocks_id
        return filtered_blocks_id

    def _get_graph_edge_mask(self, land_use_type: int) -> np.ndarray:
        """Return the edge mask of the graph.

        Args:
            land_use_type: land use type of the new land use.

        Returns:
            edge_mask: edge mask of the graph.
        """
        gdf, graph = self._get_current_gdf_and_graph()
        current_graph_edges = np.array(graph.edges)
        current_graph_nodes_id = gdf.index.to_numpy()
        self._current_graph_edges_with_id = current_graph_nodes_id[current_graph_edges]

        feasible_blocks_id = gdf[
            (gdf['type'] == city_config.FEASIBLE) &
            (gdf.area * self._cell_area >= self._required_min_area[land_use_type])].index.to_numpy()
        intersections_id = gdf[gdf.geom_type == 'Point'].index.to_numpy()

        if self._rule_constraints:
            feasible_blocks_id = self._filter_block_by_rule(gdf, feasible_blocks_id, land_use_type)

        edge_mask = np.logical_or(
            np.logical_and(
                np.isin(self._current_graph_edges_with_id[:, 0], feasible_blocks_id),
                np.isin(self._current_graph_edges_with_id[:, 1], intersections_id)
            ),
            np.logical_and(
                np.isin(self._current_graph_edges_with_id[:, 1], feasible_blocks_id),
                np.isin(self._current_graph_edges_with_id[:, 0], intersections_id)
            )
        )

        return edge_mask

    def get_current_land_use_and_mask(self) -> Tuple[Dict, np.ndarray]:
        """Return the current land use and mask.

        Returns:
            land_use: current land use.
            mask: current mask.
        """
        land_use = dict()
        remaining_plan_area = (self._required_plan_area - self._plan_area)[self._plan_order]
        remaining_plan_count = (self._required_plan_count - self._plan_count)[self._plan_order]
        land_use_type = self._plan_order[np.logical_or(remaining_plan_area > self.EPSILON, remaining_plan_count > 0)][0]
        land_use['type'] = land_use_type
        mask = self._get_graph_edge_mask(land_use_type)
        land_use['x'] = 0.5
        land_use['y'] = 0.5
        land_use['area'] = self._required_max_area[land_use_type]
        land_use['length'] = 4*self._required_max_edge_length[land_use_type]
        land_use['width'] = self._required_max_edge_length[land_use_type]
        land_use['height'] = self._required_max_edge_length[land_use_type]
        land_use['rect'] = 1.0
        land_use['eqi'] = 1.0
        land_use['sc'] = 1.0
        return land_use, mask

    def get_current_road_mask(self) -> np.ndarray:
        """Return the current road mask.

        Returns:
            mask: current road mask.
        """
        gdf, graph = self._get_current_gdf_and_graph()
        self._current_graph_nodes_id = current_graph_nodes_id = gdf.index.to_numpy()
        boundary_id = gdf[gdf['type'] == city_config.BOUNDARY].index.to_numpy()
        mask = np.isin(current_graph_nodes_id, boundary_id)

        return mask

    def _simplify_polygon(self,
                          polygon: Polygon,
                          intersection: Point) -> Tuple[Polygon, GeoSeries, Text, List, float]:
        """Simplify polygon.

        Args:
            polygon: polygon to simplify.
            intersection: intersection point.

        Returns:
            polygon: simplified polygon.
            polygon_boundary: GeoSeries of boundary edges of the simplified polygon.
            relation: relation between the simplified polygon and the intersection point.
            edges: list of boundary edges of the simplified polygon that intersects with the intersection point.
        """
        cached_polygon = polygon
        polygon = simplify_by_angle(polygon.normalize(), deg_tol=self.DEG_TOL)
        simple_coords = MultiPoint(list(polygon.exterior.coords))
        polygon_boundary = get_boundary_edges(polygon, 'GeoSeries')

        error_msg = 'Original polygon: {}'.format(cached_polygon)
        error_msg += '\nSimplified polygon: {}'.format(polygon)
        error_msg += '\nIntersection: {}'.format(intersection)

        if simple_coords.distance(intersection) > self.EPSILON:
            boundary_distance = polygon_boundary.distance(intersection)
            distance = boundary_distance.min()
            if (boundary_distance < distance + self.EPSILON).sum() > 1:
                raise ValueError(error_msg + '\nIntersection within edge is near two edges.')
            relation = 'edge'
            edges = polygon_boundary[boundary_distance < distance + self.EPSILON].to_list()
        elif simple_coords.contains(intersection):
            polygon_intersection_relation = polygon_boundary.intersects(intersection)
            if polygon_intersection_relation.sum() != 2:
                raise ValueError(error_msg + '\nThe corner intersection must intersects with two edges.')
            relation = 'corner'
            edges = polygon_boundary[polygon_intersection_relation].to_list()
            distance = 0.0
        else:
            raise ValueError(error_msg + '\nIntersection is not corner or within edge.')

        return polygon, polygon_boundary, relation, edges, distance

    def _slice_polygon(self, polygon: Polygon, intersection: Point, land_use_type: int) -> Polygon:
        """Slice the polygon from the given intersection.

        Args:
            polygon: polygon to be sliced.
            intersection: intersection point.
            land_use_type: land use type of the new land use.

        Returns:
            sliced_polygon: sliced polygon.
        """
        search_max_length = self._required_max_edge_length[land_use_type] + self._common_min_edge_length
        search_max_area = self._required_max_area[land_use_type]
        search_min_area = self._required_min_area[land_use_type]
        polygon, polygon_boundary, relation, edges, distance = self._simplify_polygon(polygon, intersection)
        gdf = self._current_gdf
        all_intersections = gdf[gdf.geom_type == 'Point']
        min_edge_length = self._required_min_edge_length[land_use_type]
        max_edge_length = self._required_max_edge_length[land_use_type]
        if relation == 'edge':
            edge = edges[0]
            land_use_polygon = slice_polygon_from_edge(
                polygon, polygon_boundary, edge, intersection, all_intersections, distance, self.EPSILON,
                self._cell_edge_length, min_edge_length, max_edge_length, search_max_length,
                search_max_area, search_min_area)
        elif relation == 'corner':
            edge_1_intersection = MultiPoint(edges[0].coords).difference(intersection)
            edge_1 = LineString([intersection, edge_1_intersection])
            edge_2_intersection = MultiPoint(edges[1].coords).difference(intersection)
            edge_2 = LineString([intersection, edge_2_intersection])
            land_use_polygon = slice_polygon_from_corner(
                polygon, polygon_boundary, intersection, edge_1, edge_1_intersection, edge_2, edge_2_intersection,
                all_intersections, self.EPSILON, self._cell_edge_length,
                min_edge_length, max_edge_length, search_max_length,
                search_max_area, search_min_area)
        else:
            raise ValueError('Relation must be edge or corner.')

        land_use_polygon = get_intersection_polygon_with_maximum_area(land_use_polygon, polygon)
        return land_use_polygon

    def _add_remaining_feasible_blocks(self, feasible_polygon: Polygon, land_use_polygon: Polygon) -> None:
        """Add remaining feasible blocks back to gdf.

        Args:
            feasible_polygon: feasible polygon.
            land_use_polygon: land use polygon.
        """
        intersections = self._gdf[(self._gdf['existence'] == True) & (self._gdf.geom_type == 'Point')].unary_union
        feasible_polygon = snap(feasible_polygon, intersections, self.SNAP_EPSILON/self._cell_edge_length)
        remaining_feasibles = feasible_polygon.difference(land_use_polygon)

        error_msg = 'feasible region: {}'.format(feasible_polygon)
        error_msg += '\nland_use region: {}'.format(land_use_polygon)
        error_msg += '\nremaining feasible region: {}'.format(remaining_feasibles)

        if remaining_feasibles.area > 0:
            if remaining_feasibles.geom_type in ['Polygon', 'MultiPolygon']:
                if remaining_feasibles.geom_type == 'Polygon':
                    remaining_feasibles = MultiPolygon([remaining_feasibles])
                for remaining_feasible in list(remaining_feasibles.geoms):
                    self._update_gdf(
                        remaining_feasible, city_config.FEASIBLE, build_boundary=False, error_msg='Remaining feasible.')
            else:
                raise ValueError(error_msg + '\nRemaining feasible region is neither Polygon nor MultiPolygon.')
        elif not land_use_polygon.equals(feasible_polygon):
            raise ValueError(
                error_msg + '\nThe area of remaining feasible region is 0, but land_use does not equals to feasible.')

    def _simplify_snap_polygon(self, polygon: Polygon) -> Tuple[Polygon, MultiPoint, List]:
        """Simplify the polygon and snap it to existing intersections.

        Args:
            polygon: polygon to be simplified and snapped.

        Returns:
            polygon: the simplified polygon.
            intersections: existing intersections.
            new_intersections: new intersections.
        """
        cached_polygon = polygon
        polygon = polygon.normalize().simplify(self.SNAP_EPSILON/self._cell_edge_length, preserve_topology=True)
        cached_polygon_simplify = polygon
        polygon = simplify_by_distance(polygon, self.EPSILON)
        cached_polygon_simplify_distance = polygon
        existing_intersections = self._gdf[
            (self._gdf.geom_type == 'Point') & (self._gdf['existence'] == True)].unary_union
        polygon = snap(polygon, existing_intersections, self.SNAP_EPSILON/self._cell_edge_length)
        if polygon.is_empty:
            return None, None, None
        if polygon.geom_type != 'Polygon':
            error_msg = 'Original land_use polygon: {}'.format(cached_polygon)
            error_msg += '\nLand_use polygon after simplify: {}'.format(cached_polygon_simplify)
            error_msg += '\nLand_use polygon after simplify by distance: {}'.format(cached_polygon_simplify_distance)
            error_msg += '\nLand_use polygon after snap: {}'.format(polygon)
            raise ValueError(error_msg + '\nLand_use polygon is not a polygon after simplify and snap.')
        intersections = MultiPoint(polygon.exterior.coords[:-1])
        new_intersections = intersections.difference(existing_intersections)
        if new_intersections.is_empty:
            new_intersections = []
        elif new_intersections.geom_type == 'MultiPoint':
            new_intersections = list(new_intersections.geoms)
        elif new_intersections.geom_type == 'Point':
            new_intersections = [new_intersections]
        else:
            error_msg = 'New intersections: {}'.format(new_intersections)
            error_msg += '\nType of new intersections: {}'.format(new_intersections.geom_type)
            raise ValueError(error_msg + '\nThe type of new intersections is not point or multipoint or empty.')
        return polygon, intersections, new_intersections
    
    def _update_population_point(self,point:Point):
        points = self._gdf[self._gdf['geometry'].geom_type == 'Point']
        matching_point = points[points['geometry'] == point]
        
        if not matching_point.empty:
            idx = matching_point.index[0]
            existing_population = self._gdf.loc[idx, 'population']
            
            connected_roads = self._gdf[
                (self._gdf['type'] == 2) &  # Select roads
                (self._gdf['geometry'].apply(lambda geom: geom.geom_type == 'LineString' and point in geom.coords))  # Check connection
            ]

            road_count = len(connected_roads) + 1  # Include the new road
            
            # Recalculate average population
            new_population = (existing_population / road_count)*0.4
            self._gdf.loc[idx, 'population'] = new_population

        else:
            print("Point not found, adding new point with population 100.")
            # new_point_row = {
            #     'geometry': point,
            #     'type': 15,  # Define a default type if needed, or leave as None
            #     'population': 100,
            #     'existence': True  # Assuming default existence value
            # }

    def _add_new_intersections(self,
                               land_use_polygon: Polygon,
                               intersections: MultiPoint,
                               new_intersections: List) -> None:
        """Add new intersections to gdf.

        Args:
            land_use_polygon: polygon of land use to be updated.
            intersections: existing intersections.
            new_intersections: new intersections.
        """
        if len(new_intersections) == len(intersections.geoms):
            error_msg = 'New intersections:'
            for new_intersection in new_intersections:
                error_msg += '\n{}'.format(new_intersection)
            raise ValueError(error_msg + '\nAll new intersections without any old intersections!')
        for new_intersection in new_intersections:
            intersection_gdf = GeoDataFrame(
                [[self._counter(), city_config.INTERSECTION, True, new_intersection]],
                columns=['id', 'type', 'existence', 'geometry']).set_index('id')
            self._gdf = pd.concat([self._gdf, intersection_gdf])
            roads_or_boundaries = self._gdf[(self._gdf.geom_type == 'LineString') & (self._gdf['existence'] == True)]
            within_existing_roads_or_boundaries = roads_or_boundaries.distance(new_intersection) < self.EPSILON
            if within_existing_roads_or_boundaries.any():
                road_or_boundary_to_split = roads_or_boundaries[within_existing_roads_or_boundaries]
                if len(road_or_boundary_to_split) > 1:
                    error_msg = 'polygon: {}'.format(land_use_polygon)
                    error_msg += '\nnew intersection: {}'.format(new_intersection)
                    error_msg += '\nroad or boundary to split:'
                    for var in range(len(road_or_boundary_to_split)):
                        error_msg += '\n{}'.format(road_or_boundary_to_split['geometry'].iloc[var])
                    error_msg += '\nNew intersection is located at more than 1 existing roads or boundaries.'
                    raise ValueError(error_msg)
                road_or_boundary_to_split_linestring = road_or_boundary_to_split['geometry'].iloc[0]
                road_or_boundary_to_split_type = road_or_boundary_to_split['type'].iloc[0]
                road_or_boundary_1 = LineString([road_or_boundary_to_split_linestring.coords[0], new_intersection])
                road_or_boundary_2 = LineString([road_or_boundary_to_split_linestring.coords[1], new_intersection])
                
                road_or_boundary_gdf = GeoDataFrame(
                    [[self._counter(), city_config.ROAD, True, road_or_boundary_1,100],
                     [self._counter(), city_config.ROAD, True, road_or_boundary_2,100]],
                    columns=['id', 'type', 'existence', 'geometry','population']).set_index('id')

                self._gdf = pd.concat([self._gdf, road_or_boundary_gdf])
                road_or_boundary_to_split_id = road_or_boundary_to_split.index[0]
                self._gdf.at[road_or_boundary_to_split_id, 'existence'] = False
            self._gdf['geometry'] = self._gdf['geometry'].apply(lambda x: snap(x, new_intersection, self.EPSILON))
    
    def _add_new_intersections_road(self,
                               land_use_polygon: Polygon,
                               intersections: MultiPoint,
                               new_intersections: List) -> None:
        """Add new intersections to gdf.

        Args:
            land_use_polygon: polygon of land use to be updated.
            intersections: existing intersections.
            new_intersections: new intersections.
        """
        if len(new_intersections) == len(intersections.geoms):
            error_msg = 'New intersections:'
            for new_intersection in new_intersections:
                error_msg += '\n{}'.format(new_intersection)
            raise ValueError(error_msg + '\nAll new intersections without any old intersections!')
        for new_intersection in new_intersections:
            intersection_gdf = GeoDataFrame(
                [[self._counter(), city_config.INTERSECTION, True, new_intersection,100]],
                columns=['id', 'type', 'existence', 'geometry','population']).set_index('id')
            self._gdf = pd.concat([self._gdf, intersection_gdf])
            roads_or_boundaries = self._gdf[(self._gdf.geom_type == 'LineString') & (self._gdf['existence'] == True)]
            within_existing_roads_or_boundaries = roads_or_boundaries.distance(new_intersection) < self.EPSILON
            if within_existing_roads_or_boundaries.any():
                road_or_boundary_to_split = roads_or_boundaries[within_existing_roads_or_boundaries]
                if len(road_or_boundary_to_split) > 1:
                    error_msg = 'polygon: {}'.format(land_use_polygon)
                    error_msg += '\nnew intersection: {}'.format(new_intersection)
                    error_msg += '\nroad or boundary to split:'
                    for var in range(len(road_or_boundary_to_split)):
                        error_msg += '\n{}'.format(road_or_boundary_to_split['geometry'].iloc[var])
                    error_msg += '\nNew intersection is located at more than 1 existing roads or boundaries.'
                    raise ValueError(error_msg)
                road_or_boundary_to_split_linestring = road_or_boundary_to_split['geometry'].iloc[0]
                new_population = road_or_boundary_to_split['population']/2
                road_or_boundary_to_split_type = road_or_boundary_to_split['type'].iloc[0]
                road_or_boundary_1 = LineString([road_or_boundary_to_split_linestring.coords[0], new_intersection])
                road_or_boundary_2 = LineString([road_or_boundary_to_split_linestring.coords[1], new_intersection])
                
                road_or_boundary_gdf = GeoDataFrame(
                    [[self._counter(), city_config.ROAD, True, road_or_boundary_1,new_population],
                     [self._counter(), city_config.ROAD, True, road_or_boundary_2,new_population]],
                    columns=['id', 'type', 'existence', 'geometry','population']).set_index('id')
                
                self._gdf = pd.concat([self._gdf, road_or_boundary_gdf])
                road_or_boundary_to_split_id = road_or_boundary_to_split.index[0]
                self._gdf.at[road_or_boundary_to_split_id, 'existence'] = False
            self._gdf['geometry'] = self._gdf['geometry'].apply(lambda x: snap(x, new_intersection, self.EPSILON))

    def _add_new_boundaries(self, land_use_polygon: Polygon) -> None:
        """Add new boundaries to gdf.

        Args:
            land_use_polygon: polygon of land use to be updated.
        """
        new_boundaries = get_boundary_edges(land_use_polygon, 'MultiLineString')
        roads_or_boundaries = self._gdf[(self._gdf.geom_type == 'LineString')
                                        & (self._gdf['existence'] == True)].unary_union
        # new_population = roads_or_boundaries['population'].sum()/len(roads_or_boundaries)
        new_boundaries = new_boundaries.difference(roads_or_boundaries)
        if new_boundaries.is_empty:
            new_boundaries = []
        elif new_boundaries.geom_type == 'MultiLineString':
            new_boundaries = list(new_boundaries.geoms)
        elif new_boundaries.geom_type == 'LineString':
            new_boundaries = [new_boundaries]
        else:
            error_msg = 'New boundaries: {}'.format(new_boundaries)
            error_msg += '\nType of new boundaries: {}'.format(new_boundaries.geom_type)
            raise ValueError(error_msg + '\nNew boundaries is not linestring or multilinestring or empty.')

        for new_boundary in new_boundaries:
            roads_or_boundaries = self._gdf[(self._gdf.geom_type == 'Point') & (self._gdf['existence'] == True)]
            within_existing_roads_or_boundaries = roads_or_boundaries.distance(new_boundary) < self.EPSILON
            boundary = roads_or_boundaries[within_existing_roads_or_boundaries]
            new_population = boundary['population'].mean()
            if len(new_boundary.coords) > 2:
                error_msg = 'New boundary: {}'.format(new_boundary)
                raise ValueError(error_msg + '\nNumber of coords of new boundary is greater than 2.')
            boundary_gdf = GeoDataFrame(
                [[self._counter(), city_config.ROAD, True, new_boundary,new_population]],
                columns=['id', 'type', 'existence', 'geometry','population']).set_index('id')
            
            self._gdf = pd.concat([self._gdf, boundary_gdf])

    def _add_land_use_polygon(self, land_use_polygon: Polygon, land_use_type: int) -> None:
        """Add land use polygon to gdf.

        Args:
            land_use_polygon: polygon of land use to be updated.
            land_use_type: land use type of the new land use.
        """
        land_use_gdf = GeoDataFrame(
            [[self._counter(), land_use_type, True, land_use_polygon]],
            columns=['id', 'type', 'existence', 'geometry']).set_index('id')
        land_use_gdf['rect'] = momepy.Rectangularity(land_use_gdf).series
        land_use_gdf['eqi'] = momepy.EquivalentRectangularIndex(land_use_gdf).series
        land_use_gdf['sc'] = momepy.SquareCompactness(land_use_gdf).series
        land_use_gdf['population'] = self.calculate_road_volume_for_polygon(land_use_polygon,land_use_type)
        self._gdf = pd.concat([self._gdf, land_use_gdf])

    def _update_gdf_without_building_boundaries(self,
                                          land_use_polygon: Polygon,
                                          land_use_type: int,
                                          new_intersections: List,
                                          error_msg: Text = '') -> None:
        """Update the gdf without building boundaries.

        Args:
            land_use_polygon: polygon of land use to be updated.
            land_use_type: land use type of the new land use.
            new_intersections: new intersections.
            error_msg: error message.
        """
        if len(new_intersections) > 0:
            error_msg += '\nUpdate polygon: {}'.format(land_use_polygon)
            raise ValueError(error_msg + '\nUpdate polygon without building boundaries creates new points.')
        self._add_land_use_polygon(land_use_polygon, land_use_type)

    def _update_gdf(self,
                    land_use_polygon: Polygon,
                    land_use_type: int,
                    build_boundary: bool = True,
                    error_msg: Text = '') -> Polygon:
        """Update the GDF.

        Args:
            land_use_polygon: polygon of the new land use.
            land_use_type: land use type of the new land use.
            build_boundary: whether to build boundary.
            error_msg: error message.

        Returns:
            land_use_polygon: polygon of the new land use, might be different from the original one due to snapping.
        """
        land_use_polygon, intersections, new_intersections = self._simplify_snap_polygon(land_use_polygon)
        if land_use_polygon is None:
            error_msg = f'Type {land_use_type}\n'
            raise ValueError(error_msg + 'Empty after simplify and snap.')

        if not build_boundary:
            self._update_gdf_without_building_boundaries(land_use_polygon, land_use_type, new_intersections, error_msg)
            return land_use_polygon

        self._add_new_intersections_road(land_use_polygon, intersections, new_intersections)
        self._add_new_boundaries(land_use_polygon)
        self._add_land_use_polygon(land_use_polygon, land_use_type)

        return land_use_polygon
    
    def _add_new_roads(self,land_use_polygon:Polygon):
        new_boundaries = get_boundary_edges(land_use_polygon, 'MultiLineString')
        roads_or_boundaries = self._gdf[(self._gdf.geom_type == 'LineString')
                                        & (self._gdf['existence'] == True)].unary_union
        new_boundaries = new_boundaries.difference(roads_or_boundaries)
        if new_boundaries.is_empty:
            new_boundaries = []
        elif new_boundaries.geom_type == 'MultiLineString':
            new_boundaries = list(new_boundaries.geoms)
        elif new_boundaries.geom_type == 'LineString':
            new_boundaries = [new_boundaries]
        else:
            error_msg = 'New boundaries: {}'.format(new_boundaries)
            error_msg += '\nType of new boundaries: {}'.format(new_boundaries.geom_type)
            raise ValueError(error_msg + '\nNew boundaries is not linestring or multilinestring or empty.')

        for new_boundary in new_boundaries:
            if len(new_boundary.coords) > 2:
                error_msg = 'New boundary: {}'.format(new_boundary)
                raise ValueError(error_msg + '\nNumber of coords of new boundary is greater than 2.')
            id = self._counter()
            boundary_gdf = GeoDataFrame(
                [[id, city_config.ROAD, False, new_boundary,5]],
                columns=['id', 'type', 'existence', 'geometry','population']).set_index('id')
           
            self._gdf = pd.concat([self._gdf, boundary_gdf])
            self.ids.append(id)

    def _update_gdf_road(self,
                    land_use_polygon: Polygon,
                    land_use_type: int,
                    build_road: bool = True,
                    error_msg: Text = ''):
        land_use_polygon, intersections, new_intersections = self._simplify_snap_polygon(land_use_polygon)
        if land_use_polygon is None:
            error_msg = f'Type {land_use_type}\n'
            raise ValueError(error_msg + 'Empty after simplify and snap.')

        if not build_road:
            self._update_gdf_without_building_boundaries(land_use_polygon, land_use_type, new_intersections, error_msg)
            return land_use_polygon
        
        for point in intersections.geoms:
            self._update_population_point(point)

        self._add_new_intersections_road(land_use_polygon, intersections, new_intersections)
        self._add_new_roads(land_use_polygon)
        self._add_land_use_polygon(land_use_polygon, land_use_type)

        return land_use_polygon

    def _get_chosen_feasible_block_and_intersection(self, action: int) -> Tuple[int, int]:
        """Get the chosen feasible block and intersection

        Args:
            action: the chosen graph edge.

        Returns:
            Tuple of the chosen (feasible_block_id, intersection_id).
        """
        chosen_pair = self._current_graph_edges_with_id[action]
        if self._gdf.loc[chosen_pair[0]]['type'] == city_config.FEASIBLE:
            return chosen_pair[0], chosen_pair[1]
        else:
            return chosen_pair[1], chosen_pair[0]

    def _use_whole_feasible(self, feasible_polygon: Polygon, land_use_type: int) -> Polygon:
        """Use the whole feasible block.

        Args:
            feasible_polygon: polygon of the feasible block.
            land_use_type: land use type of the new land use.
        """
        land_use_polygon = feasible_polygon
        land_use_polygon = self._update_gdf(
            land_use_polygon, land_use_type, build_boundary=False, error_msg='Whole feasible.')
        return land_use_polygon

    def _place_land_use(self, land_use_type: int, feasible_id: int, intersection_id: int) -> Tuple[float, int]:
        """Place the land use at the given action position.

        Args:
          land_use_type: The type of the land use to be placed.
          feasible_id: The id of the feasible block.
          intersection_id: The id of the intersection.

        Returns:
            The area of the land use.
            The actual land use type.
        """
        actual_land_use_type = land_use_type
        feasible_polygon = self._gdf.loc[feasible_id, 'geometry']
        if feasible_polygon.area*self._cell_area <= self._required_max_area[land_use_type]:
            land_use_polygon = self._use_whole_feasible(feasible_polygon, land_use_type)
        else:
            intersection = self._gdf.loc[intersection_id, 'geometry']
            land_use_polygon = self._slice_polygon(feasible_polygon, intersection, land_use_type)
            if land_use_polygon.area < self.EPSILON:
                error_msg = 'feasible polygon: {}'.format(feasible_polygon)
                error_msg += '\nintersection: {}'.format(intersection)
                error_msg += '\nland_use polygon: {}'.format(land_use_polygon)
                raise ValueError(error_msg + '\nThe area of sliced land_use_polygon is near 0.')
            if (feasible_polygon.area - land_use_polygon.area)*self._cell_area <= self._common_min_area:
                land_use_polygon = self._use_whole_feasible(feasible_polygon, land_use_type)
            else:
                if land_use_polygon.area*self._cell_area < self._required_min_area[land_use_type]:
                    land_use_polygon = self._update_gdf(land_use_polygon, city_config.GREEN_S)
                    actual_land_use_type = city_config.GREEN_S
                else:
                    land_use_polygon = self._update_gdf(land_use_polygon, land_use_type)

                self._add_remaining_feasible_blocks(feasible_polygon, land_use_polygon)

        self._gdf.at[feasible_id, 'existence'] = False

        land_use_area = land_use_polygon.area*self._cell_area
        return land_use_area, actual_land_use_type

    def place_land_use(self, land_use: Dict, action: int) -> None:
        """Place the land use at the given action position.

        Args:
          land_use: A dict containing the type, x, y, area, width and height of the current land use.
          action: The action to take (an integer indicating the chosen graph edge).

        Returns:
            True if the land_use is successfully placed, False otherwise.
        """
        feasible_id, intersection_id = self._get_chosen_feasible_block_and_intersection(action)
        land_use_area, actual_land_use_type = self._place_land_use(land_use['type'], feasible_id, intersection_id)
        self._update_stats(actual_land_use_type, land_use_area)


        min_area_threshold = 200.0  # Example value, in the same units as your geometries

        # Filter the GeoDataFrame for existing geometries
        gdf = self._gdf[self._gdf['existence'] == True]

        # Filter for polygons and exclude those with an area smaller than the threshold
        polygons = gdf[(gdf['geometry'].geom_type == 'Polygon') & (gdf['geometry'].area > 750)]

        # Count new roads with specific criteria
        # new_road_count = len(gdf[(gdf['type'] == 2) & (gdf['population'] == 5)])
        print("number of roads built are ", self.current_roads)
        if len(polygons) >= 1 and self.current_roads < self.max_roads:
            random_polygon = polygons.sample(1, random_state=random.randint(0, 10000)).iloc[0]
            polygon_coords = list(random_polygon['geometry'].exterior.coords)
            
            intersection_point = random.sample(polygon_coords, 1)
            try:
                land_use_polygon = self._slice_polygon(random_polygon['geometry'], Point(intersection_point[0][0],intersection_point[0][1]), random_polygon['type'])

                if land_use_polygon.area < self.EPSILON:
                    error_msg = 'feasible polygon: {}'.format(random_polygon['geometry'])
                    error_msg += '\nintersection: {}'.format(intersection_point)
                    error_msg += '\nland_use polygon: {}'.format(land_use_polygon)
                    raise ValueError(error_msg + '\nThe area of sliced land_use_polygon is near 0.')
                else:
                    self._update_gdf_road(land_use_polygon,random_polygon['type'])
                    self.current_roads = len(self.ids)
                    self._update_population_point(intersection_point)

            except Exception:
                print("Could Not Build Road")



        
        # previous_lines = gdf[(gdf['type'] == 16) & (gdf['geometry'].geom_type == 'LineString')]
        # drainage_line_count = len(previous_lines)

        # if drainage_line_count >= 30:
        #     return 

        # if not previous_lines.empty:
        #     start_point = previous_lines.iloc[-1]['geometry'].coords[-1]
        # else:
        #     random_polygon = gdf[gdf['geometry'].geom_type == 'Polygon'].sample(1).iloc[0]
        #     start_point = random.choice(list(random_polygon['geometry'].exterior.coords))


        # candidate_lines = gdf[(gdf['geometry'].geom_type == 'LineString') &
        #                (gdf['geometry'].apply(lambda line: isinstance(line, LineString) and start_point in line.coords))]


        # if not previous_lines.empty:
        #     candidate_lines = candidate_lines[candidate_lines['geometry'] != previous_lines.iloc[-1]['geometry']]

        # if not candidate_lines.empty:
        #     random_line = candidate_lines.sample(1).iloc[0]
        #     end_point = random_line['geometry'].coords[-1]

        #     if end_point == start_point:
        #         random_line = candidate_lines.sample(1).iloc[0]
        #         end_point = random_line['geometry'].coords[-1]

        #     print("start point and end point are : ",start_point,end_point)
        
        #     new_drainage_line = LineString([start_point, end_point])
                
            # new_row = {
            #         'type': 16,
            #         'existence': True,
            #         'geometry': new_drainage_line,
            #         'population': 0
            #     }
                

            # new_row_gdf = gpd.GeoDataFrame([new_row], geometry='geometry')
            # self._gdf = pd.concat([gdf, new_row_gdf], ignore_index=True)
            # print("New drainage line added to gdf.")
            # else:
            #     print("Invalid points for LineString:", start_point, end_point)
        
        # else:
        #     random_line = gdf[gdf['geometry'].geom_type == 'LineString'].sample(1).iloc[0]
        #     start_point = random_line['geometry'].coords[0]
        #     end_point = random_line['geometry'].coords[-1]
            

        #     new_drainage_line = LineString([start_point, end_point])
        #     new_row = {
        #         'type': 16,
        #         'existence': True,
        #         'geometry': new_drainage_line,
        #         'population': 0
        #     }
            

        #     new_row_gdf = gpd.GeoDataFrame([new_row], geometry='geometry')
        #     self._gdf = pd.concat([gdf, new_row_gdf], ignore_index=True)
        #     print("Fallback drainage line added to gdf.")


    def _get_chosen_boundary(self, action: int) -> int:
        """Get the chosen boundary.

        Args:
            action: the chosen graph node.

        Returns:
            The chosen boundary.
        """
        chosen_boundary = self._current_graph_nodes_id[action]
        if self._gdf.loc[chosen_boundary, 'type'] != city_config.BOUNDARY:
            raise ValueError('The build road action is not boundary node.')
        return chosen_boundary

    def build_road(self, action: int) -> None:
        """Build the road at the given action position.

        Args:
          action: The action to take (the chosen node to build road).

        Returns:
            True if the road is successfully built, False otherwise.
        """
        chosen_boundary = self._get_chosen_boundary(action)
        self._gdf.loc[chosen_boundary, 'type'] = city_config.ROAD

    def get_requirements(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the planning requirements.

        Returns:
            A tuple of the requirements of land_use ratio and land_use count.
        """
        return self._required_plan_ratio, self._required_plan_count

    def get_plan_ratio_and_count(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the planning ratio and count.

        Returns:
            A tuple of the ratio and count of land_use.
        """
        return self._plan_ratio, self._plan_count

    @staticmethod
    def _get_road_boundary_graph(gdf) -> nx.MultiGraph:
        """Return the road and boundary graph."""
        road_boundary_gdf = gdf[(gdf['type'] == city_config.ROAD) | (gdf['type'] == city_config.BOUNDARY)]
        road_boundary_graph = momepy.gdf_to_nx(road_boundary_gdf.reset_index(), approach='primal', length='length')
        return road_boundary_graph

    @staticmethod
    def _get_domain_features(gdf: GeoDataFrame) -> np.ndarray:
        """Get the domain knowledge features.

        Args:
            gdf: the GeoDataFrame.

        Returns:
            The domain knowledge features.
        """
        domain_gdf = gdf[['rect', 'eqi', 'sc']].fillna(0.5)
        domain_features = domain_gdf.to_numpy()
        return domain_features

    def get_graph_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the graph features.

        Returns:
            A tuple of the graph features which contains the followings.
            1. node type: the type of the nodes.
            2. node coordinates: the x-y coordinate of the nodes.
            3. node area: the area of the nodes.
            4. node length: the length of the nodes.
            5. node width: the width of the nodes.
            6. node height: the height of the nodes.
            7. edges: the adjacency list.
        """
        gdf = self._current_gdf
        graph = self._current_graph
        node_type = gdf['type'].to_numpy(dtype=np.int32)
        node_coordinates = np.column_stack((gdf.centroid.x/self._grid_cols, gdf.centroid.y/self._grid_rows))
        node_area = gdf.area.to_numpy(dtype=np.float32)*self._cell_area
        node_length = gdf.length.to_numpy(dtype=np.float32)*self._cell_edge_length
        bounds = gdf.bounds
        node_width = (bounds['maxx'] - bounds['minx']).to_numpy(dtype=np.float32)*self._cell_edge_length
        node_height = (bounds['maxy'] - bounds['miny']).to_numpy(dtype=np.float32)*self._cell_edge_length
        node_domain = self._get_domain_features(gdf)

        edges = np.array(graph.edges)

        return node_type, node_coordinates, node_area, node_length, node_width, node_height, node_domain, edges

    def _get_road_graph(self) -> nx.MultiGraph:
        """Return the road graph."""

        #adding my own code to add roads to reduce traffic

        max_roads_per_polygon=5
        gdf = self._gdf[self._gdf['existence'] == True]
        polygons_with_population = gdf[gdf['population'].notna()]
        
        if polygons_with_population.empty:
            print("No populated polygons found, no roads to add.")
            return
        

        existing_roads = gdf[gdf['type'] == 2]['geometry'].tolist()  # All existing roads
        new_roads = []
        for _, polygon in polygons_with_population.iterrows():
            population = polygon['population']
            
            # Find candidate target polygons (lower population)
            less_populated_polygons = gdf[(gdf['population'].notna()) & 
                                        (gdf['population'] < population)]
            
            if less_populated_polygons.empty:
                continue  # No lower population areas to connect to

            # Calculate distances to potential target polygons
            polygon_centroid = polygon['geometry'].centroid
            less_populated_polygons['distance'] = less_populated_polygons['geometry'].centroid.distance(polygon_centroid)

            # Sort by proximity (closer polygons first)
            less_populated_polygons = less_populated_polygons.sort_values(by='distance')

            roads_created = 0
            for _, target_polygon in less_populated_polygons.iterrows():
                if roads_created >= max_roads_per_polygon:
                    break  # Limit the number of roads per polygon
                
                # Extract the vertices or midpoints of polygon edges (randomly)
                poly_coords = list(polygon['geometry'].exterior.coords)
                target_coords = list(target_polygon['geometry'].exterior.coords)
                
                start_point = random.choice(poly_coords)  # Choose random vertex from the polygon
                end_point = random.choice(target_coords)  # Choose random vertex from the target polygon
                
                # Create a new LineString representing the road
                new_road_geometry = LineString([start_point, end_point])
                
                # Check if road already exists (either in the same or reverse direction)
                if any(new_road_geometry.equals(road) or new_road_geometry.equals(LineString([end_point, start_point]))
                    for road in existing_roads):
                    continue  # Skip if this road already exists
                
                # Add the new road to the gdf with NaN population
                new_roads.append({
                    'type': 2,  # Assuming 2 is the type for roads
                    'existence': True,
                    'geometry': new_road_geometry,
                    'population': float('nan')  # Set population for roads to NaN
                })
                
                existing_roads.append(new_road_geometry)  # Add the new road to the existing roads list
                roads_created += 1
        
        # Append new roads to the gdf
        if new_roads:
            new_roads_gdf = gpd.GeoDataFrame(new_roads, geometry='geometry')
            self._gdf = pd.concat([self._gdf, new_roads_gdf], ignore_index=True)



        road_gdf = self._gdf[(self._gdf['type'] == city_config.ROAD) & (self._gdf['existence'] == True)]
        road_graph = momepy.gdf_to_nx(road_gdf, approach='primal', length='length', multigraph=False)
        return road_graph

    def get_road_network_reward(self) -> Tuple[float, Dict]:
        """Get the road network reward.

        Returns:
            The road network reward.
        """
        gdf = self._gdf[self._gdf['existence'] == True]
        road_graph = self._get_road_graph()

        # connectivity of road network 
        connectivity_reward = 1.0/nx.number_connected_components(road_graph)

        # density of road network
        road_length = gdf[gdf['type'] == city_config.ROAD].length
        road_total_length_km = road_length.sum()*self._cell_edge_length/1000
        community_area_km = self._community_area/1000/1000
        road_network_density = road_total_length_km/community_area_km
        density_reward = road_network_density/10.0

        # dead end penalty
        degree_sequence = np.array([d for n, d in road_graph.degree()], dtype=np.int32)
        num_dead_end = np.count_nonzero(degree_sequence == 1)
        dead_end_penalty = 1.0/(num_dead_end + 1)

        # penalty for short/long road
        road_gdf = gdf[gdf['type'] == city_config.ROAD]
        road_gdf = momepy.remove_false_nodes(road_gdf)
        road_length = road_gdf.length
        num_short_roads = len(road_length[road_length*self._cell_edge_length < 100])
        short_road_penalty = 1.0/(num_short_roads + 1)
        num_long_roads = len(road_length[road_length*self._cell_edge_length > 600])
        long_road_penalty = 1.0/(num_long_roads + 1)

        # penalty for road distance
        road_gdf = gdf[gdf['type'] == city_config.ROAD]
        blocks = polygonize(road_gdf['geometry'])
        block_bounds = [block.bounds for block in blocks]
        block_width_height = np.array([(b[2] - b[0], b[3] - b[1]) for b in block_bounds], dtype=np.float32)
        num_large_blocks = np.count_nonzero(
            np.logical_or(
                block_width_height[:, 0]*self._cell_edge_length > 800,
                block_width_height[:, 1]*self._cell_edge_length > 800))
        road_distance_penalty = 1.0/(num_large_blocks + 1)

        road_network_reward = 1.0 * connectivity_reward + 1.0 * density_reward + 1.0 * dead_end_penalty + \
            1.0 * short_road_penalty + 1.0 * long_road_penalty + 1.0 * road_distance_penalty
        road_network_reward = road_network_reward/6.0
        info = {'connectivity_reward': connectivity_reward,
                'density_reward': density_reward,
                'dead_end_penalty': dead_end_penalty,
                'short_road_penalty': short_road_penalty,
                'long_road_penalty': long_road_penalty,
                'road_distance_penalty': road_distance_penalty}

        return road_network_reward, info
    


    


    def get_life_circle_reward(self, weight_by_area: bool = False) -> Tuple[float, Dict]:
        """Get the reward of the life circle.
        Returns:
            The reward of the life circle.
        """
        print("i am in reward funtion")
        gdf = self._gdf[self._gdf['existence'] == True].copy(deep=True)
        gdf['population'] = gdf['population'].astype(float)


        new_roads = gdf[ (gdf['type'] == 2)]
        high_traffic_points = gdf[(gdf['geometry'].geom_type == 'Point') & (gdf['population'].astype(float) >= 600)]
        low_traffic_points = gdf[(gdf['geometry'].geom_type == 'Point') & (gdf['population'].astype(float) < 400)]
        
        # print("hight_traffic points are : ",high_traffic_points)

        total_proximity_score = 0
        total_balance_score = 0


        for _, road in new_roads.iterrows():
            road_geometry = road['geometry']

            # Proximity to High-Traffic Points: Sum of proximity scores based on inverse distances
            proximity_scores = []
            for _, point in high_traffic_points.iterrows():
                distance = road_geometry.distance(point['geometry'])
                # print("distance btw ",road_geometry," and ",point,"is : ",distance)
                if distance > 0:
                    proximity_scores.append(100 / distance)  # Inverse distance reward
                
            road_proximity_score = np.mean(proximity_scores) if proximity_scores else 0
            total_proximity_score += road_proximity_score

            # Load Balancing Efficiency: Check distances to low-traffic points to ensure efficient traffic diversion
            load_balance_scores = []
            for _, low_traffic_point in low_traffic_points.iterrows():
                low_traffic_distance = road_geometry.distance(low_traffic_point['geometry'])
                if low_traffic_distance > 0:
                    load_balance_scores.append(100 / low_traffic_distance) 

            road_balance_score = np.mean(load_balance_scores) if load_balance_scores else 0
            total_balance_score += road_balance_score


        avg_proximity_reward = total_proximity_score / len(new_roads) if len(new_roads) > 0 else 0
        avg_balance_reward = total_balance_score / len(new_roads) if len(new_roads) > 0 else 0


        print(f"Proximity Reward for new roads set: {avg_proximity_reward}")
        print(f"Load Balancing Reward for new roads set: {avg_balance_reward}")


        final_reward = avg_proximity_reward + avg_balance_reward




        # population_density_mapping = {
        # city_config.RESIDENTIAL: 100,  # Example: 100 people per unit area
        # city_config.BUSINESS: 50,
        # city_config.WASTEMGMT: 10,
        # city_config.GREEN_L: 5,
        # city_config.GREEN_S: 5,
        # city_config.SCHOOL: 200,
        # city_config.HOSPITAL_L: 300,
        # city_config.HOSPITAL_S: 150,
        # city_config.RECREATION: 20,
        # city_config.OFFICE: 75,
        # city_config.FEASIBLE: 50 }


        # def calculate_population(row):
        #     if row['type'] in population_density_mapping and row['geometry'].geom_type == 'Polygon':
        #         area = row['geometry'].area  # Calculate the area on the fly
        #         return area * population_density_mapping[row['type']]  # Return population based on area and type
        #     return np.nan 
        
        # gdf['population'] = gdf.apply(calculate_population, axis=1)


        residential_centroid = gdf[gdf['type'] == city_config.RESIDENTIAL].centroid
        residential_area = gdf[gdf['type'] == city_config.RESIDENTIAL].area.to_numpy()
        num_public_service = 0
        minimum_public_service_distances = []
        public_service_pairwise_distances = []
        public_service_area = 0.0


        def gini_coefficient(population):
            sorted_pop = np.sort(population)
            n = len(sorted_pop)
            cumulative_pop = np.cumsum(sorted_pop) / np.sum(sorted_pop)
            relative_ranks = np.arange(1, n + 1) / n
            gini = 1 - 2 * np.sum((relative_ranks - cumulative_pop) / n)
            return gini



        polygon_gdf = gdf[gdf['population'].notna()]  # Only consider rows with population data
        if not polygon_gdf.empty:
            gini_value = gini_coefficient(polygon_gdf['population'].values)
            # Convert Gini into an even distribution reward: 1 - Gini (lower Gini is better)
            even_distribution_reward = 1 - gini_value
        else:
            even_distribution_reward = 0.3



        for public_service in city_config.PUBLIC_SERVICES_ID:
            if not isinstance(public_service, tuple):
                public_service_gdf = gdf[gdf['type'] == public_service]
            else:
                public_service_gdf = gdf[gdf['type'].isin(public_service)]
            public_service_centroid = public_service_gdf.centroid.unary_union

            num_same_public_service = len(public_service_gdf)
            if num_same_public_service > 0:
                distance = residential_centroid.distance(public_service_centroid).to_numpy()
                minimum_public_service_distances.append(distance)
                num_public_service += 1
                public_service_area += public_service_gdf.area.sum()*self._cell_area

                if num_same_public_service > 1:
                    public_service_x = public_service_gdf.centroid.x.to_numpy()
                    public_service_y = public_service_gdf.centroid.y.to_numpy()
                    public_service_xy = np.stack([public_service_x, public_service_y], axis=1)
                    pair_distance = cdist(public_service_xy, public_service_xy)
                    average_pair_distance = np.mean(pair_distance[pair_distance > 0])
                    public_service_pairwise_distances.append(average_pair_distance)

        if num_public_service > 0:
            public_service_distance = np.column_stack(minimum_public_service_distances)
            life_circle_15min = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 1000, axis=1)/num_public_service
            life_circle_10min = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 500, axis=1)/num_public_service
            life_circle_5min = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 300, axis=1)/num_public_service
            if not weight_by_area:
                efficiency_reward = life_circle_10min.mean()
            else:
                efficiency_reward = np.average(life_circle_10min, weights=residential_area)
            reference_distance = math.sqrt(self._grid_cols**2 + self._grid_rows**2)
            decentralization_reward = np.array(public_service_pairwise_distances).mean()/reference_distance
            utility_reward = public_service_area/self._community_area
            reward = efficiency_reward + 0.05 * decentralization_reward + 0.75 * even_distribution_reward + final_reward
            info = {'life_circle_15min': life_circle_15min.mean(),
                    'life_circle_10min': life_circle_10min.mean(),
                    'life_circle_5min': life_circle_5min.mean(),
                    'life_circle_10min_area': np.average(life_circle_10min, weights=residential_area),
                    'decentralization_reward': decentralization_reward,
                    'utility': utility_reward,
                    'even_population_distribution_reward': even_distribution_reward ,
                    'traffic_reward': final_reward # Add to info dictionary
                    }
            
            life_circle_10min_all = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 500, axis=0)/public_service_distance.shape[0]
            for index, service_name in enumerate(city_config.PUBLIC_SERVICES):
                info[service_name] = life_circle_10min_all[index]
            return reward, info
        else:
            return 0.0, dict()
        





    def get_greenness_reward(self) -> float:
        """Get the reward of the greenness.

        Returns:
            The reward of the greenness.
        """
        gdf = self._gdf[self._gdf['existence'] == True]
        green_id = city_config.GREEN_ID
        green_gdf = gdf[(gdf['type'].isin(green_id)) & (gdf.area*self._cell_area >= city_config.GREEN_AREA_THRESHOLD)]
        green_cover = green_gdf.buffer(300/self._cell_edge_length).unary_union
        residential = gdf[gdf['type'] == city_config.RESIDENTIAL].unary_union
        green_covered_residential = green_cover.intersection(residential)
        reward = green_covered_residential.area / residential.area
        return reward
    
    def get_wastemgmt_reward(self) -> float:
        """Get the reward for waste management placement based on proximity to residential areas and green spaces.

        Returns:
            The reward for the waste management service.
        """

        gdf = self._gdf[self._gdf['existence'] == True]
        

        wastemgmt_gdf = gdf[gdf['type'] == city_config.WASTEMGMT]
        residential_gdf = gdf[gdf['type'] == city_config.RESIDENTIAL]
        green_gdf = gdf[gdf['type'].isin(city_config.GREEN_ID)]

        if wastemgmt_gdf.empty or residential_gdf.empty or green_gdf.empty:
            return 0.0


        wastemgmt_centroid = wastemgmt_gdf.centroid.unary_union
        residential_centroid = residential_gdf.centroid.unary_union
        green_centroid = green_gdf.centroid.unary_union

        distance_to_residential = wastemgmt_centroid.distance(residential_centroid)
   
        distance_to_green = wastemgmt_centroid.distance(green_centroid)


        min_residential_distance_threshold = 500 / self._cell_edge_length  

        min_green_distance_threshold = 300 / self._cell_edge_length  

        residential_penalty = 0.0
        green_space_reward = 0.0

        if distance_to_residential < min_residential_distance_threshold:
            residential_penalty = -0.25 * (min_residential_distance_threshold - distance_to_residential) / min_residential_distance_threshold

        if distance_to_green < min_green_distance_threshold:
            green_space_reward = 1.0 * (min_green_distance_threshold - distance_to_green) / min_green_distance_threshold

        final_reward = green_space_reward * 0.7 + residential_penalty * 0.3

        # final_reward = max(0.0, min(1.0, final_reward))

        return final_reward
    



    def get_drainage_reward(self) -> float:
        gdf = self._gdf[self._gdf['existence'] == True]
        drainage_lines = gdf[gdf['type'] == 16]  # Drainage lines
        # wastemgmt_areas = gdf[gdf['type'] == 6]  # WASTEMGMT areas
        waterbodies = gdf[gdf['type'] == 14]     # WATERBODY areas
        road_network = gdf[gdf['type'] == 2]     # ROAD lines
        intersection_points = gdf[gdf['type'] == 15]

        # 1. Coverage of WASTEMGMT areas
        # intersections = drainage_lines.geometry.intersection(wastemgmt_areas.unary_union)
        # intersected_length = intersections.length.sum()
        # print("Total intersected length with wastemgmt areas:", intersected_length)

        # Normalize intersected length
        max_intersected_length = 1000  # Define a reasonable maximum based on your dataset
        # wastemgmt_coverage = intersected_length / max_intersected_length

        # 2. Proximity to Water Bodies
        distances_to_waterbody = [
            drainage_line.geometry.distance(waterbodies.unary_union)
            for _, drainage_line in drainage_lines.iterrows()
        ]
        average_distance_to_waterbody = (
            sum(distances_to_waterbody) / len(distances_to_waterbody)
            if distances_to_waterbody else max_intersected_length
        )
        

        # Normalize proximity to water bodies
        max_proximity_distance = 1000 # Define a reasonable maximum
        proximity_to_waterbody = average_distance_to_waterbody / max_proximity_distance
        print("proximity to waterbody", proximity_to_waterbody)

        # 3. Connectivity and Integration with Roads and Intersections
        connections_to_road = sum(
            drainage_line.geometry.intersects(road_network.unary_union) for _, drainage_line in drainage_lines.iterrows()
        )
        connections_to_intersections = sum(
            drainage_line.geometry.intersects(intersection_points.unary_union) for _, drainage_line in drainage_lines.iterrows()
        )
        max_connections = len(drainage_lines)  # Maximum possible connections
        zone_balance = (connections_to_road + connections_to_intersections) / (2 * max_connections) if max_connections > 0 else 0

        # 4. Penalty for overlap with sensitive areas (example sensitive types: GREEN_L, HOSPITAL_L, SCHOOL)
        sensitive_zones = gdf[gdf['type'].isin([7, 10, 9])]
        sensitive_intersections = drainage_lines.geometry.intersection(sensitive_zones.unary_union)
        sensitive_length_overlap = sensitive_intersections.length.sum()
        max_sensitive_length = 1000  # Define a reasonable maximum length for normalization
        penalty = sensitive_length_overlap / max_sensitive_length
        print("Penalty for sensitive areas based on length:", penalty)

        # Weighted combination (normalized to range [0, 1])
        final_reward = (
            # 0.3 * wastemgmt_coverage +
            0.3 * (1 - proximity_to_waterbody) +  # Inverse of proximity for reward
            0.3 * zone_balance -
            0.1 * penalty
        )

        return max(0, min(final_reward, 1))  #





    def get_concept_reward(self) -> Tuple[float, Dict]:
        """Get the reward of the planning concept.

        Returns:
            The reward of the concept.
            The information of the concept reward.
        """
        if len(self._concept) == 0:
            raise ValueError('The concept list is empty.')
        gdf = self._gdf[(self._gdf['existence'] == True) & (self._gdf.geom_type == 'Polygon')]
        reward = 0.0
        info = dict()
        for i, concept in enumerate(self._concept):
            if concept['type'] == 'center':
                center_reward, center_info = self._get_center_concept_reward_info(gdf, concept)
                reward += center_reward
                info['{}_center'.format(i)] = center_info
            elif concept['type'] == 'axis':
                axis_reward, axis_info = self._get_axis_concept_reward_info(gdf, concept)
                reward += axis_reward
                info['{}_axis'.format(i)] = axis_info
            else:
                raise ValueError(f'The concept type {concept["type"]} is not supported.')
        reward /= len(self._concept)
        return reward, info

    def _get_center_concept_reward_info(self, gdf: GeoDataFrame, concept: Dict) -> Tuple[float, Dict]:
        """Get the reward of the center concept.

        Args:
            gdf: The GeoDataFrame of the city.
            concept: The concept.

        Returns:
            The reward of the center concept.
            The information of the center concept.
        """
        center = concept['geometry']
        distance_threshold = concept['distance']
        center_circle = center.buffer(distance_threshold/self._cell_edge_length)
        center_gdf = gdf[gdf.intersects(center_circle)]
        related_land_use = concept['land_use']
        center_related_gdf = center_gdf[center_gdf['type'].isin(related_land_use)]
        center_related_land_use_ratio = len(center_related_gdf)/len(center_gdf)
        reward = center_related_land_use_ratio

        info = dict()
        info['center'] = (center.x, center.y)
        info['distance_threshold'] = distance_threshold
        info['related_land_use'] = related_land_use
        info['related_land_use_ratio'] = center_related_land_use_ratio
        return reward, info

    def _get_axis_concept_reward_info(self, gdf: GeoDataFrame, concept: Dict) -> Tuple[float, Dict]:
        """Get the reward of the axis concept.

        Args:
            gdf: The GeoDataFrame of the city.
            concept: The concept.

        Returns:
            The reward of the axis concept.
            The information of the axis concept.
        """
        axis = concept['geometry']
        distance_threshold = concept['distance']
        axis_band = axis.buffer(distance_threshold/self._cell_edge_length, cap_style=2, join_style=2)
        axis_gdf = gdf[gdf.intersects(axis_band)]
        related_land_use = concept['land_use']
        axis_related_gdf = axis_gdf[axis_gdf['type'].isin(related_land_use)]
        if len(axis_related_gdf) > 0:
            axis_related_land_use_ratio = len(axis_related_gdf)/len(axis_gdf)
            axis_related_land_use_type = axis_related_gdf['type'].nunique()/len(related_land_use)
            axis_related_land_use_project = axis_related_gdf.centroid.apply(lambda x: axis.project(x, normalized=True))
            axis_related_land_use_expand = axis_related_land_use_project.max() - axis_related_land_use_project.min()
            reward = (axis_related_land_use_ratio + axis_related_land_use_type + axis_related_land_use_expand)/3
            info = dict()
            info['axis'] = axis.coords[:]
            info['distance_threshold'] = distance_threshold
            info['related_land_use'] = related_land_use
            info['related_land_use_ratio'] = axis_related_land_use_ratio
            info['related_land_use_type'] = axis_related_land_use_type
            info['related_land_use_expand'] = axis_related_land_use_expand
        else:
            reward = 0.0
            info = dict()
            info['axis'] = axis.coords[:]
            info['distance_threshold'] = distance_threshold
            info['related_land_use'] = related_land_use
            info['related_land_use_ratio'] = 0.0
            info['related_land_use_type'] = 0.0
            info['related_land_use_expand'] = 0.0

        return reward, info
