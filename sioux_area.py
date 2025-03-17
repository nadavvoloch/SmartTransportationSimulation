
world_nodes = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                '21', '22', '23', '24']

world_node_coords = [(90,62),(473,62),(89,146),(203,146),
               (333,147),(471,148),(608,218),(473,221),
               (334,223),(332,297),(201,294),(90,296),
               (90,635),(201,460),(333,459),(471,296),(471,371),
               (607,294),(471,459),(471,634),(334,635),
               (333,537),(201,535),(201,635)]


SX, SY = 1, 1
translated_world_node_coords = [(SX * x, SY *(707 - y)) for x,y in world_node_coords]

world_links = [('1','2'),('1','3'),('2','6'),('3','4'),('3','12'),('4','5'),
               ('4','11'),('5','9'),('5','6'),('6','8'),('7','8'),('7','18'),
               ('8','9'),('8','16'),('9','10'),('10','11'),('10','16'),
               ('10','15'),('11','12'),('11','14'),('12','13'),('13','24'),
               ('14','15'),('14','23'),('15','22'),('15','19'),('16','17'),
               ('16','18'),('17','19'),('18','20'),('19','20'),('20','21'),
               ('20','22'),('21','24'),('21','22'),('22','23'),('23','24')]

signal_links = []

world_link_freeflow = [50.0*0.277] * len(world_links)

# Necessary Unions
all_links = world_links + signal_links
all_links_freeflow = world_link_freeflow


## Aggregate all data into dictionaries
## Make attributes into dict

all_links_ff_dic =  dict(zip(all_links, all_links_freeflow))
signal_link_group_dic = {}
translated_node_coord_dic = dict(zip(world_nodes, translated_world_node_coords))
signal_node_group_dic = {}



#### Flow / Demand Related Information
general_sinks   = ['1','2','7','18','20','13']
general_sources = ['24','3','12','21','10','9']

## Testing Related
testing_sources = general_sources
testing_sinks = general_sinks

testing_personal_pairs = [('1','20'),('13','2'),('1','10')]
testing_personal_srcs  = [pair[0] for pair in testing_personal_pairs]
testing_personal_dests = [pair[1] for pair in testing_personal_pairs]

