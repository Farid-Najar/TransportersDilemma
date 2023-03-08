from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from sortedcontainers import SortedList
import numpy as np

from typing import List


class Transporter:
    def __init__(self, 
                 distance_matrix,
                 delegation_cost = 100, 
                 transporter_hub : int = 85, 
                 max_capacity : int|List[int] = 50,
                 num_vehicles :int = 1,
                 ):
        
        self.data = dict()
        self.orders = [] # List of nodes to visit
        
        if type(max_capacity) == List:
            self.data['vehicle_capacities'] = max_capacity
        elif type(max_capacity) == int:
            self.data['vehicle_capacities'] = [max_capacity for _ in range(num_vehicles)]
        else:
            raise("max_capacity must be an integer or a list of integers !")
        
        self.max_capacity = 100
        self.capacity = 0
        self.delegation_cost = delegation_cost
        self.distance_matrix = distance_matrix
        
        self.nodes = SortedList([transporter_hub])
        
        self.last_cost = 0
        self.cost_history = [0]
        self.data['demands'] = []
        self.data['num_vehicles'] = num_vehicles
        self.data['depot'] = 0
        
    def reset(self):
        self.capacity = 0
        self.orders.clear()
        
    def new_order(self, node, quantity):
        self.nodes.add(node)
        idx = self.nodes.index(node)
        self.orders.insert(idx, quantity)
        self.cost_history.append(self.last_cost)
        
    
    def compute_cost(self, node : int, quantity : int):
        """Solve the CVRP problem and computes the additional cost"""
        # Instantiate the data problem.
        data = self.data
        
        new_nodes = self.nodes.copy()
        new_nodes.add(node)
        idx = new_nodes.index(node)
        
        data['demands'] = self.orders.copy()
        data['demands'].insert(idx, quantity)
        
        data['distance_matrix'] = self.distance_matrix[np.ix_(new_nodes, new_nodes)]
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])
    
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
    
    
        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
    
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    
        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
    
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')
    
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(1)
    
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
    
        # Print solution on console.
        self.last_cost = self.delegation_cost + self.cost_history[-1]
        
        if not solution:
            return self.delegation_cost
        
        self.last_cost = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += data['distance_matrix'][previous_index-1][index-1]
                # print(route_distance)
            self.last_cost += route_distance
            
        return self.last_cost - self.cost_history[-1]
        

if __name__ == '__main__':
    import networkx as nx
    size = 5
    G = nx.grid_2d_graph(size, size)
    distance_matrix = nx.floyd_warshall_numpy(G)
    T1 = Transporter(distance_matrix, transporter_hub=0)
    c = T1.compute_cost(1, 5)
    print(c)
    T1.new_order(1, 5)
    c = T1.compute_cost(3, 5)
    print(c)
    print('total cost : ', T1.last_cost)
