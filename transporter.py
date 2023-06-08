from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from sortedcontainers import SortedList
import numpy as np

from typing import List

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))
    
def solve(data):
    """Returns the CVRP solution and routing"""
    
    # # Sets a time limit of 10 seconds.
    # search_parameters.time_limit.seconds = 10
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    # print('n vehicles : ', data['num_vehicles'])
    # print('n vehicles : ', data['distance_matrix'])
    
    
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    
    # Create and register a transit callback.
    def distance_callback(from_index, to_index, cost_unit = 1):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]*cost_unit
    
    # transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # print('test', distance_callback(5,7))
    
    # Define cost of each arc.
    # routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    for m in range(data['num_vehicles']):
        transit_callback_index = routing.RegisterTransitCallback(
            lambda from_idx, to_idx : distance_callback(
                from_idx, to_idx, 
                data['cost_per_unit'][m])
        )
        routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, m)
    
    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )
    
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(2)
    
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    # Print solution on console.
    #if solution:
    #    print_solution(data, manager, routing, solution)
    
    return solution, routing, manager

class Transporter:
    def __init__(self, 
                 distance_matrix,
                 cost_per_unit = None,
                #  emission_per_unit = None,
                 time_matrix = None,
                 omission_cost = 100, 
                 transporter_hub : int = 85, 
                 max_capacity : int|List[int] = 15,
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
        self.omission_cost = omission_cost
        self.distance_matrix = distance_matrix
        self.transporter_hub = transporter_hub
        self.nodes = SortedList()
        
        if cost_per_unit is None:
            self.data['cost_per_unit'] = np.ones(num_vehicles, dtype=int)
        else:
            self.data['cost_per_unit'] = cost_per_unit.copy()
            
        # if emission_per_unit is None:
        #     self.data['emission_per_unit'] = np.ones(num_vehicles, dtype=int)
        # else:
        #     self.data['emission_per_unit'] = cost_per_unit.copy()
        
        if time_matrix is None:
            self.time_matrix = distance_matrix/40 #In cities, the average speed is 40 km/h
        else:
            self.time_matrix = time_matrix
        
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
        
    def compute_marginal_cost(self, node : int, quantity : int):
        """Solve the CVRP problem and computes the additional cost"""
        # Instantiate the data problem.
        data = self.data
        
        new_nodes = self.nodes.copy()
        new_nodes.add(node)
        idx = new_nodes.index(node)
        
        data['demands'] = self.orders.copy()
        data['demands'].insert(idx, quantity)
        
        l = [self.transporter_hub] + list(new_nodes)
        data['distance_matrix'] = self.distance_matrix[np.ix_(l, l)]
        
        solution, routing, manager = solve(data)
        
        # Print solution on console.
        self.last_cost = self.omission_cost + self.cost_history[-1]
        
        if not solution:
            return self.omission_cost*len(new_nodes)
        
        #TODO
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
    

    def compute_cost(self, nodes, quantities):
        """Solve the CVRP problem and computes the total cost and time per vehicle"""
        # Instantiate the data problem.
        data = self.data
        
        data['demands'] = [0] + list(quantities)
        
        l = [self.transporter_hub] + list(nodes)
        data['distance_matrix'] = self.distance_matrix[np.ix_(l, l)]
        
        # print(data['distance_matrix'])
        
        solution, routing, manager = solve(data)
    
        
        # If no solution; we penalize all packages (it's a rare event)
        if not solution:
            return self.omission_cost*len(nodes), 0
        
        route_distance = np.zeros(data['num_vehicles'])
        route_time = np.zeros(data['num_vehicles'])
        text = 'Routes :'
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            text += '\nvehicle ' + str(vehicle_id) + '\n'
            text += '0'
                
            while not routing.IsEnd(index):
                previous_index = index
                # print(index)
                index = solution.Value(routing.NextVar(index))
                from_node = manager.IndexToNode(previous_index)
                to_node = manager.IndexToNode(index)
                
                text += ' -> ' + str(to_node)
                
                route_distance[vehicle_id] += data['distance_matrix'][from_node][to_node]
                # routing.GetArcCostForVehicle(
                #     previous_index, index, vehicle_id
                # )
                route_time[vehicle_id] += self.time_matrix[from_node][to_node]
                # print(route_distance)
                
        return route_distance, route_time, text
         

if __name__ == '__main__':
    import networkx as nx
    size = 15
    G = nx.grid_2d_graph(size, size)
    distance_matrix = nx.floyd_warshall_numpy(G)
    T1 = Transporter(distance_matrix, num_vehicles=3)
    # c = T1.compute_marginal_cost(1, 5)
    # print(c)
    # T1.new_order(1, 5)
    # c = T1.compute_marginal_cost(3, 5)
    # print(c)
    
    nodes = np.random.choice(len(G.nodes), size=10, replace=False)
    quantities = np.ones(10, dtype=int)
    c = T1.compute_cost(nodes, quantities, print_res=True)
    print(c)
    print('total cost : ', np.sum(c[0]))
