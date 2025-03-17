from uxsim import *
import numpy as np
from random import choice

from area_1 import *

FIGSIZE = (9,12)
tmax=1000


# Calculate Distance
def make_distance_matrix(netlinks, trans_node_coordinates):
    distmat = {}

    for node1, node2 in netlinks:
        n1 = trans_node_coordinates[node1]
        n2 = trans_node_coordinates[node2]

        dist = pow(pow(n1[0] - n2[0] , 2) + pow(n1[1] - n2[1] , 2) , 0.5)
        dist = round(dist, 2)

        distmat[(node1, node2)] = dist
        distmat[(node2, node1)] = dist

    return distmat

world_link_dists = make_distance_matrix(all_links, translated_node_coord_dic)


def add_world_nodes(w: World, nodes, trans_node_coord_dic):
    warnings.warn("Signal nodes have not yet been delt with yet")
    for node in nodes:
        coor = trans_node_coord_dic[node]
        w.addNode(node, coor[0], coor[1])
    return w

def add_world_nodes_V2(w: World, nodes, trans_node_coord_dic,
                       signal_node_group_dic):
    for node in nodes:
        coor = trans_node_coord_dic[node]

        if node in signal_node_group_dic:
            w.addNode(node, coor[0], coor[1], 
                      signal=signal_node_group_dic[node])
        else:
            w.addNode(node, coor[0], coor[1])

    return w

def link_to_name(link: tuple, direction: int):
    if direction == 1:
        return f"{link[0]}_{link[1]}_UP"
    elif direction == -1:
        return f"{link[0]}_{link[1]}_DOWN"

def add_world_links(w: World, links, freeflows, distdic, signal_links, siggroups):
    links_added_up = {}
    links_added_down = {}

    for link in links:
        name1 = link_to_name(link, 1) # f"{link[0]}_{link[1]}_UP"
        name2 = link_to_name(link, -1) # f"{link[1]}_{link[0]}_DOWN"
        links_added_up[name1] = link
        links_added_down[name2] = link

        ff = freeflows[link]
        if link in signal_links:
            group = siggroups[link]
        else:
            group = 0

        w.addLink(name1, link[0], link[1], distdic[link], 
                  free_flow_speed=ff, signal_group=group)
        w.addLink(name2, link[1], link[0], distdic[link],
                  free_flow_speed=ff, signal_group=group)

    return w, links_added_up, links_added_down

def custom_distribution(x):
        if   0 <= x <= 4:
            return 50
        elif 4 <= x <= 6:
            return 110
        elif 6 <= x <= 9:
            return 170
        elif 9 <= x <= 16:
            return 110
        elif 16 <= x <= 19:
            return 170
        elif 19 <= x <= 21:
            return 100
        elif x > 21:
            return 50
        else:
            return 50

def get_traffic_distribution_2():
    values = np.array(list(range(0,24)))
    weights = []

    for val in values:
        weights.append( custom_distribution(val) )

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()  # Normalize to sum to 1

    return values, weights

import random
random.seed(42)

# randomized_start_times = random.choices(start_basket, start_basket_p, k=500)
# randomized_stop_times =  random.choices(stop_basket, stop_basket_p, k=500)

def add_world_demand(w: World, gen_srcs, gen_sinks, hour_res=100, k=10):
    ''' k denotes how many src-sink pairs to take. Hour resolution
        denotes how many seconds of simulation are given to each hour.
        Therefore, if the simulation is set to run for 24*100 hours, 
        this res shall be 100.'''
    warnings.warn("This function contains itertools.product. In development pls use another function")
    hours , probs = get_traffic_distribution_2()
    distributed_time = random.choices(hours, probs, k=2000)

    from itertools import product
    
    all_travel_combinations = list(product(gen_srcs, gen_sinks))
    some_travel_combinations = random.choices(all_travel_combinations,
                                                k=k)

    for i,time in enumerate(distributed_time):
        start_time = time * hour_res
        stop_time = (time + 1) * hour_res
        flow_rate = 0.05

        chosen_combination = some_travel_combinations[i % k]
        src, dest = chosen_combination

        w.adddemand(src, dest, start_time, stop_time, flow_rate)

    return w

# Testing 

def pair_time_to_name(src, dest, time):
    return f"{src}_{dest}_{time}"

def add_personal_tests(w: World, personal_pairs, depart_times):
    warnings.warn("The attribute preferred_links has not been set!")
    warnings.warn("This will send cars from src to dest but without a deterministic route!")
    warnings.warn("Depart times is a list containing one time depart times, can be changed!")

    vehicles = []
    for pair, time in zip(personal_pairs , depart_times):
        src, dest = pair
        name = pair_time_to_name(src, dest, time)
        # names.append(name)
        veh = w.addVehicle(src, dest, time, name=name)
        vehicles.append(veh)
        if veh is None:
            print("Something Went Wrong")
    
    return w , vehicles















# SX = 0.45
# SY = 0.6285

# world_nodes = ['1', '2', '3', '4', '5', '6','6.5', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
#                '17', '18', '19', '20', '21', '22', '23', '24', 'start_4', 'end_4']

# world_node_coords = [(7000, 4243),(5681, 4109),(5684, 4459),(5683,4988),(6212,4015),
#                      (5721,1418),(6020,1343),(5820, 1063),(5412,569),(5056,436),
#                      (4836,1063),(5371,994),(3774,1153),(2669,1297),
#                      (2105,837),(2061,363),(1753,801),(2037,1411),
#                      (2469,2745),(2405,3455),(3474,3275),(3565,3777),
#                      (3470,5787),(5673,5751),(6997,5759),(3275,243),(3425, 6899),
#                      (3842,100),(3434,6994)]

# translated_world_node_coords = [(SX * x, SY*(9923 - y)) 
#                 for x,y in world_node_coords]


# world_links = [('1', '5'), ('5', '2'), ('2', '3'), ('3', '4'), ('6', '6.5'), ('6', '11'), 
#                ('6.5', '7'),('7', '8'), ('8', '9'), ('8', '11'), ('9', '10'), ('11', '10'),
#                 ('13', '14'),('13', '17'), ('14', '15'), ('15', '16'), ('16', '17'), 
#                 ('17', '18'), ('18', '20'), ('18', '19')]

# signal_links = [('start_4','12'),('12','21'),('10','12'),('12','13'),('21','22'),
#                 ('19','21'),('22','23'),('22','end_4'),('23','24'),('4','23')]
# signal_groups = [0,0,1,1,0,1,1,0,1,1]

# all_links = world_links + signal_links

# # Attributes
# world_link_freeflow = [50.0*0.277] * len(world_links)
# signal_link_freeflow = [0.277*i for i in [90.0,90,55,50,90,30,60,90.0,60,50]] # km/h to m/s

# all_links_freeflow = world_link_freeflow + signal_link_freeflow

# # Make attributes into dict
# all_links_ff_dic =  dict(zip(all_links, all_links_freeflow))
# signal_link_group_dic = dict(zip(signal_links, signal_groups))
# node_coord_dic = dict(zip(world_nodes, world_node_coords)) # Not affected by signals
# translated_node_coord_dic = dict(zip(world_nodes, translated_world_node_coords))



# world_links = [(1,5),(5,2),(2,3),(3,4),(6,12),(6,11),(12,7),(7,8),(8,9),(8,11),(9,10),
#               (11,10),(13,14),(13,17),(14,15),(15,16),(16,17),(17,18),(18,20),(18,19)]
# world_nodes = [str(num) for num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
# world_nodes += ["start_4", "end_4"]
# # print(world_nodes)

# world_link_caps = [2.0] * len(world_links)
# signal_link_caps = [10,10,4,2,10,1,4,10,4]

# signal_link_caps_dic = dict(zip(signal_links, signal_link_caps))
# world_link_caps_dic = dict(zip(world_links, world_link_caps))


# world_link_ff_dic = dict(zip(world_links, world_link_freeflow))
# signal_link_ff_dic = dict(zip(signal_links, signal_link_freeflow))




# for node1, node2 in world_links:
#     n1 = node_coord_dic[node1]
#     n2 = node_coord_dic[node2]

#     dist = pow(pow(n1[0] - n2[0] , 2) + pow(n1[1] - n2[1] , 2) , 0.5)
#     try:
#         distmat[node1][node2] = dist
#     except Exception:
#         distmat[node1] = {}
#         distmat[node1][node2] = dist

#     try:
#         distmat[node2][node1] = dist
#     except Exception:
#         distmat[node2] = {}
#         distmat[node2][node1] = dist

# print(distmat)

# start_basket   = [0,1,2,3,4,5,6,7,8,9,10,11,12]
# start_basket_p = [0.03225806, 0.03225806, 0.03225806, 0.03225806, 
#                   0.06451613, 0.06451613, 0.12903226, 0.12903226, 
#                   0.12903226, 0.09677419, 0.09677419, 0.09677419, 
#                   0.06451613]

# stop_basket    =      [23.59,23,22,21,20,19,18,17,16,15,14,13]
# # stop_basket_weights = [1,     1, 1, 2, 2, 3, 4, 4, 4, 2, 3, 3]
# stop_basket_p = [0.03333333, 0.03333333, 0.03333333, 0.06666667, 
#                  0.06666667, 0.1       , 0.13333333, 0.13333333, 
#                  0.13333333, 0.06666667, 0.1       , 0.1       ]


# def add_world_demand(w: World, gen_srcs, gen_sinks, hour_res=100, k=10):
#     ''' k denotes how many src-sink pairs to take. Hour resolution
#         denotes how many seconds of simulation are given to each hour.
#         Therefore, if the simulation is set to run for 24*100 hours, 
#         this res shall be 100.'''
#     # global randomized_start_times, randomized_stop_times
#     global r_start , r_stop
#     from itertools import product
    
#     all_travel_combinations = list(product(gen_srcs, gen_sinks))
#     some_travel_combinations = random.choices(all_travel_combinations,
#                                                 k=k)

#     for i in range(len(randomized_start_times)):
#         start_time = randomized_start_times[i] * hour_res
#         stop_time = randomized_stop_times[i] * hour_res
#         flow_rate = 0.008

#         chosen_combination = some_travel_combinations[i % k]
#         src, dest = chosen_combination

#         w.adddemand(src, dest, start_time, stop_time, flow_rate)

#     return w
