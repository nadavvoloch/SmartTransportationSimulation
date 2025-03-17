# What must area.py contain?
# All documentation regarding the world needed to be simulated.
# Nodes, edges, traffic, signals and more

SX = 0.45
SY = 0.6285

world_nodes = ['1', '2', '3', '4', '5', '6','6.5', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
               '17', '18', '19', '20', '21', '22', '23', '24', 'start_4', 'end_4']

world_node_coords = [(7000, 4243),(5681, 4109),(5684, 4459),(5683,4988),(6212,4015),
                     (5721,1418),(6020,1343),(5820, 1063),(5412,569),(5056,436),
                     (4836,1063),(5371,994),(3774,1153),(2669,1297),
                     (2105,837),(2061,363),(1753,801),(2037,1411),
                     (2469,2745),(2405,3455),(3474,3275),(3565,3777),
                     (3470,5787),(5673,5751),(6997,5759),(3275,243),(3425, 6899),
                     (3842,100),(3434,6994)]

translated_world_node_coords = [(SX * x, SY*(9923 - y)) 
                for x,y in world_node_coords]


world_links = [('1', '5'), ('5', '2'), ('2', '3'), ('3', '4'), ('6', '6.5'), ('6', '11'), 
               ('6.5', '7'),('7', '8'), ('8', '9'), ('8', '11'), ('9', '10'), ('11', '10'),
                ('13', '14'),('13', '17'), ('14', '15'), ('15', '16'), ('16', '17'), 
                ('17', '18'), ('18', '20'), ('18', '19')]

signal_links = [('start_4','12'),('12','21'),('10','12'),('12','13'),('21','22'),
                ('19','21'),('22','23'),('22','end_4'),('23','24'),('4','23')]
signal_nodes = ['12','21','22','23']
signal_nodes_groups = [[30,15],[30,20],[30,30],[30,20]]
signal_groups = [0,0,1,1,0,1,1,0,1,1]

# Attributes
world_link_freeflow = [50.0*0.277] * len(world_links)
signal_link_freeflow = [0.277*i for i in [90.0,90,55,50,90,30,60,90.0,60,50]] # km/h to m/s


# Necessary Unions
all_links = world_links + signal_links
all_links_freeflow = world_link_freeflow + signal_link_freeflow


# Aggregate all data into dictionaries
# Make attributes into dict
all_links_ff_dic =  dict(zip(all_links, all_links_freeflow))
signal_link_group_dic = dict(zip(signal_links, signal_groups))
# node_coord_dic = dict(zip(world_nodes, world_node_coords)) # Not affected by signals
translated_node_coord_dic = dict(zip(world_nodes, translated_world_node_coords))
signal_node_group_dic = dict(zip(signal_nodes, signal_nodes_groups))



#### Flow / Demand Related Information
general_sinks   = ['1','24','start_4','end_4','14','20','8']
general_sources = ['24','1','6','7','16','18','20']

# specific_routes = [('',''),('',''),('',''),('',''),('',''),('',''),]


## Testing Related
testing_sources = general_sources[:3]
testing_sinks = general_sinks[:3]

testing_personal_pairs = [('1','12'),('1','17'),('24','6')]
testing_personal_srcs  = [pair[0] for pair in testing_personal_pairs]
testing_personal_dests = [pair[1] for pair in testing_personal_pairs]
personal_pair_times = [800,300,100]

