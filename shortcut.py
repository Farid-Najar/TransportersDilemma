from dataclasses import dataclass
import sys
import os
from copy import deepcopy
from SA_baseline import recuit
# direc = os.path.dirname(__file__)
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, '/Users/faridounet/PhD/TransportersDilemma')

import numpy as np
from numba import njit
from assignment import RemoveActionEnv
from numba.typed import List

@dataclass
class Table:
    values : np.ndarray
    sol : np.ndarray
    max_omitted : np.int64
# @njit
def compute_delta(cost_matrix, route, k):
    #compute the difference in cost when removing elements in the tour
    delta = np.zeros((len(route), k+1))#malloc(t.length*sizeof(float*));
    
    # observation[int(self.initial_routes[m, j])-1] = self.initial_routes[m, j-1] + self.initial_routes[m, j+1] -(
    #                             costs_matrix[m, int(self.initial_routes[m, j-2]), int(self.initial_routes[m, j+2])]
    #                     )
    #no need for delta[0]
    for i in range(1, len(delta)):
        current_vertex = route[i]
        sum = cost_matrix[current_vertex, route[i-1]]
        # //printf("\n Removing before vertex %d :",i);
        for j in range(1, len(delta[i])):
            if(route[i-j]):#//the vertex is not 0, hence can be removed
                sum += cost_matrix[route[i-j-1], route[i-j]]
                delta[i, j] = sum - cost_matrix[route[i-j-1], current_vertex]
                # //printf("(%d,%f)  ", j ,delta[i][j]);
            else:#//the vertex is 0, we cannot remove
                delta[i][j] = -1 #//special value to escape from the computation
                break
    return delta


# @njit
def compute_smallest_cost(cost_matrix, route, excess): 
    K = len(route) - 2
    delta = compute_delta(cost_matrix, route, K+1)
    # print(delta)
    # values, sol = init_table(len(route),K+1)
    sol = np.zeros((K+1, len(route)), dtype=np.int64)
    values = np.zeros((K+1, len(route)))
    max_omitted = 0
    for k in range(1, K+1):
        if values[k-1, len(route)-1] >= excess:
            break
        # while the pollution constraint is violated, try to remove one additional package
        max_omitted = k
        for i in range(k+1, len(route)):
            #we may begin from k+1, because we need to remove k elements before it and vertex 0 cannot be removed
            # loop to determine the optimal number of elements to remove before i
            for j in range(k+1):
                #break when an element cannot be removed
                if delta[i][j] == -1:
                    break
                
                val = delta[i, j] + values[k-j, i-j-1]
                if(val > values[k, i]):
                    values[k, i] = val
                    sol[k, i] = j
        
    # // //print for debugging
    # // for(int i = 0; i <= tab.max_omitted; i++){
    # //     printf("\n Number of ommited packages %d, gain %f",i,tab.values[i][t.length -1]);
    # // }
    # // int *s = get_solution(tab,t.length);
    # // printf("Position of packages omitted in the solution: ");
    # // for(int i = 0; i < tab.max_omitted; i++){
    # //     printf("%d ",s[i]);
    # // }
    return sol, values, max_omitted

# @njit
def value(value_tables, coeff, types, sol, excess):
    #evaluate the value and pollution of a solution
    gain = 0
    # pol = 0
    for i in range(types):
        idx = min(sol[i], len(value_tables[i])-1)
        gain += value_tables[i][idx]
        # pol += value_tables[i][sol[i]]#coeff[i]*
    return gain if gain > excess else 0

# @njit
def best_combination(k, types, current_type, value_tables, coeff, excess, sol, max_val, max_sol):
    #extremly simplistic enumeration of the way to generate k
    total = 0
    for i in range(current_type):
        total += sol[i]
    #printf("Total %d current type %d\n",total,current_type);
    
    if(current_type == types -1):    
        sol[current_type] = k - total
        val = value(value_tables,coeff,types,sol,excess)
        #print("value : %f \n",val)
        if (val > max_val):
            max_val = val
            max_sol = sol.copy()
            
        return sol, max_val, max_sol#a solution is found, stop
    
    for i in range(k - total + 1):
        sol[current_type]=i
        return best_combination(k, types, current_type+1, value_tables, coeff, excess, sol, max_val, max_sol)

# @njit
def get_solution_single_type(sol, k, tour_length):
    #get an optimal solution with k omitted packages from a full dynamic table
    solution = np.zeros(k, dtype=np.int64)
    position = tour_length-1
    #printf("\n K max considéré: %d Position max : %d\n",k,position);
    while(k):
        #continue until it has found all packets to remove
        to_remove = sol[k][position]
        #printf("%d %d %d\n",position,k,to_remove);
        for i in range(1, to_remove+1):
            k -= 1
            solution[k] = position -i
        
        position -= to_remove+1

    return solution


# @njit
def get_solution_multiple_types(sol, max_sol, routes, types):
    #get a solution from a full dynamic table
    solution = []#np.zeros((types, len(routes[0])), dtype=np.int64)
    for i in range(types):
        solution.append(get_solution_single_type(sol[i],max_sol[i],len(routes[i])))
    return solution
    
# @njit
def multi_types(cost_matrix, routes, coeff, excess):
    types = len(cost_matrix)
    K = len(routes[0])-2
    #types is the number of types (and thus of tour and coeff)

    sol = np.zeros((types, K+1, len(routes[0])), dtype=np.int64)
    values = np.zeros((types, K+1, len(routes[0])), dtype=float)
    max_omitted = np.zeros(types, dtype=np.int64)
    
    weight = coeff/np.sum(coeff)
    
    value_tables = []#List()
    
    for i in range(types):
        sol[i], values[i], max_omitted[i] = compute_smallest_cost(cost_matrix[i], routes[i], excess)#*weight[i])
        #extract the best combination of omission between the different types
        value_tables.append([values[i, j, -1] for j in range(max_omitted[i]+1)])
        
        # print(values[i, 1])
        # print(values[i, 1, -1])
    
    solution = np.zeros(types, dtype=np.int64)
    max_sol = np.zeros(types, dtype=np.int64)
    print('tot omitted : ', np.sum(max_omitted))
    max_val = 0
    for k in range(1, K):
        #we could begin with a larger k, compute by how much
        #printf("k: %d \n",k);
        solution, max_val, max_sol = best_combination(k, types, 0, value_tables, coeff, excess, solution, max_val, max_sol)
        if(max_val != 0):
            break
        
    final_sol = get_solution_multiple_types(sol, max_sol, routes, types)
    # printf("best solution of value: %f\n",*max_val);
    for i in range(types):
        print(max_sol[i], "packages omitted in tour ", i, " : ", end='')
        for j in range(max_sol[i]):
            print(final_sol[i][j], end='')
        print()
        
    a = np.array([
        routes[i, final_sol[i][j]]-1
        for i in range(len(final_sol))
        for j in range(len(final_sol[i]))
    ], dtype=np.int64)#np.zeros(len(cost_matrix)-1)
    
    # for i in range(types):
    #     for j in range(max_sol[i]):
    #         a[solution[i][j]] = 1.
            
    return a#, max_val, max_sol


if __name__ == '__main__':
    env = RemoveActionEnv()
    _, info = env.reset()
    print(info['excess_emission'])
    routes = np.array([
        [
            env._env.initial_routes[m, i] 
            for i in range(0, len(env._env.initial_routes[m]), 2)
        ]
        for m in range(len(env._env.initial_routes))
    ], dtype=int)
    
    action_SA, *_ = recuit(deepcopy(env._env), 5000, 1,0.9999, H=100_000)
    print('sa : ', len(action_SA) - np.sum(action_SA))
    
    print(info['excess_emission'])
    coeff = env._env._game.emissions_KM
    CM = np.array([
        env._env.distance_matrix*coeff[i]
        for i in range(len(coeff))
    ])
    a = multi_types(CM, routes, coeff, info['excess_emission'])
    print(env.step(a))