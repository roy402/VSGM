import argparse
import urllib.request
import re
import fastText_embedding
import json
from bs4 import BeautifulSoup
from alfred_dataset_vocab_analysis import load_object
from count_subgoal_action_distribution import get_subgoal_lowaction_define
# save path
save_path = "./data_dgl/"
# object relation
objects_txt = "./data/objects.txt"
url_object_object = "https://ai2thor.allenai.org/ithor/documentation/objects/actionable-properties/"
url_object_room = "https://ai2thor.allenai.org/ithor/documentation/objects/object-locations/"
# verb relation
SUB_GOAL_DISTRIBUTION = "data_dgl/subgoal_distribution.json"
SUB_GOAL, LOW_ACT_NAMES = get_subgoal_lowaction_define()
NODE_PATH_SUBGOAL = save_path + "subgoal.csv"
NODE_PATH_LOWACTION = save_path + "lowaction.csv"
EDGE_PATH_LOWACTION_BEHAVE_SUBGOAL = save_path + "lowaction-interact-subgoal.csv"
EDGE_PATH_SUBGOAL_BEHAVE_LOWACTION = save_path + "subgoal-interact-lowaction.csv"


def create_node_object(rgb_feature=False):
    # load
    ft_model = fastText_embedding.load_model()
    objects = load_object(objects_txt)
    # process
    string_csv = "Id,name,feature" + ",feature" * 299 + ",\n"
    for i, name in enumerate(objects):
        string_csv += ','.join([str(i), name]) + ","
        vectors = fastText_embedding.get_faxtText_embedding(ft_model, name)
        string_csv += ','.join(str(vector) for vector in vectors)
        string_csv += "\n"
    # csv
    nodes = open(save_path + "object.csv", "w")
    nodes.write(string_csv)


# Table of Object Actionable Properties
def create_node_attribute():
    # load url dataframe
    source = urllib.request.urlopen(url_object_object).read()
    soup = BeautifulSoup(source, 'lxml')
    tablelist = soup.find_all(class_='tableWrapper')
    # {'UsedUp', 'Breakable', 'Openable', 'Fillable', 'Pickupable', 'Sliceable', 'Moveable', 'Cookable', 'Toggleable', 'Receptacle', 'Dirty'}
    attribute_name = set()
    for i, td in enumerate(tablelist[0].find_all('td')):
        if (i+1) % 3 == 0:
            td = re.findall('[^-,\ *]*', td.text)
            attribute_name.update(td)
    attribute_name.remove("")
    print(attribute_name)
    # load model
    ft_model = fastText_embedding.load_model()
    # process
    string_csv = "Id,name,feature" + ",feature" * 299 + ",\n"
    for i, name in enumerate(attribute_name):
        string_csv += ','.join([str(i), name]) + ","
        vectors = fastText_embedding.get_faxtText_embedding(ft_model, name)
        string_csv += ','.join(str(vector) for vector in vectors)
        string_csv += "\n"
    # csv
    attribute = open(save_path + "attribute.csv", "w")
    attribute.write(string_csv)


def create_node_room():
    # load
    ft_model = fastText_embedding.load_model()
    objects = ["Bedroom", "Kitchen", "LivingRoom", "Bathroom"]
    # process
    string_csv = "Id,name,feature" + ",feature" * 299 + ",\n"
    for i, name in enumerate(objects):
        string_csv += ','.join([str(i), name]) + ","
        vectors = fastText_embedding.get_faxtText_embedding(ft_model, name)
        string_csv += ','.join(str(vector) for vector in vectors)
        string_csv += "\n"
    # csv
    nodes = open(save_path + "room.csv", "w")
    nodes.write(string_csv)


def create_node_subgoal():
    # load
    ft_model = fastText_embedding.load_model()
    # process
    string_csv = "Id,name,feature" + ",feature" * 299 + ",\n"
    for i, sg_name in enumerate(SUB_GOAL):
        string_csv += ','.join([str(i), sg_name]) + ","
        vectors = fastText_embedding.get_faxtText_embedding(ft_model, sg_name)
        string_csv += ','.join(str(vector) for vector in vectors)
        string_csv += "\n"
    # csv
    csv_nodes = open(NODE_PATH_SUBGOAL, "w")
    csv_nodes.write(string_csv)


def create_node_low_action():
    # load
    ft_model = fastText_embedding.load_model()
    # process
    string_csv = "Id,name,feature" + ",feature" * 299 + ",\n"
    for i, act_name in enumerate(LOW_ACT_NAMES):
        string_csv += ','.join([str(i), act_name]) + ","
        vectors = fastText_embedding.get_faxtText_embedding(ft_model, act_name)
        string_csv += ','.join(str(vector) for vector in vectors)
        string_csv += "\n"
    # csv
    csv_nodes = open(NODE_PATH_LOWACTION, "w")
    csv_nodes.write(string_csv)


# relation from visual_genome
def create_relation_object_object_by_visual_genome():
    """
    Src = [1, 2, 3], Dst = [4, 5, 6]
    """
    import numpy
    Src = []
    Dst = []
    relationship_matrics = numpy.genfromtxt(
        './visual_genome/relationship_matrics.csv', delimiter=',')
    relationship_matrics = numpy.asarray(relationship_matrics, dtype=int)
    for i in range(relationship_matrics.shape[0]):
        for j in range(relationship_matrics.shape[1]):
            if relationship_matrics[i, j] > 0:
                Src.append(i)
                Dst.append(j)
    # process
    string_csv = "Src,Dst" + ",\n"
    for src, dst in zip(Src, Dst):
        string_csv += ','.join([str(src), str(dst)]) + ","
        string_csv += "\n"
    # csv
    object_relation = open(save_path + "object-interact-object.csv", "w")
    object_relation.write(string_csv)


# Default Receptacle Restrictions
# url_object_object
def create_relation_object_object_by_AI2_THOR():
    """
    Src = [1, 2, 3], Dst = [4, 5, 6]
    """
    pass


def create_relation_attribute_object():
    """
    Src: attribute
    Dst: object
    Src = [1, 2, 3], Dst = [4, 5, 6]
    """
    row_objects = open(save_path + "object.csv", "r").readlines()
    row_attribute = open(save_path + "attribute.csv", "r").readlines()
    dict_objects = {}
    dict_attribute = {}
    for rows in row_objects:
        row_list = [row.strip().lower() for row in rows.split(",")]
        dict_objects[row_list[1]] = row_list[0]
    for rows in row_attribute:
        row_list = [row.strip().lower() for row in rows.split(",")]
        dict_attribute[row_list[1]] = row_list[0]
    # {'name': 'id', 'alarmclock': '0', 'apple': '1', 'armchair': '2', 'baseballbat': '3', 'basketball': '4', 'bathtub': '5', 'bathtubbasin': '6', 'bed': '7', 'blinds': '8', 'book': '9', 'boots': '10', 'bowl': '11', 'box': '12', 'bread': '13', 'butterknife': '14', 'cabinet': '15', 'candle': '16', 'cart': '17', 'cd': '18', 'cellphone': '19', 'chair': '20', 'cloth': '21', 'coffeemachine': '22', 'countertop': '23', 'creditcard': '24', 'cup': '25', 'curtains': '26', 'desk': '27', 'desklamp': '28', 'dishsponge': '29', 'drawer': '30', 'dresser': '31', 'egg': '32', 'floorlamp': '33', 'footstool': '34', 'fork': '35', 'fridge': '36', 'garbagecan': '37', 'glassbottle': '38', 'handtowel': '39', 'towelholder': '101', 'houseplant': '41', 'kettle': '42', 'keychain': '43', 'knife': '44', 'ladle': '45', 'laptop': '46', 'laundryhamper': '47', 'hamperlid': '48', 'lettuce': '49', 'lightswitch': '50', 'microwave': '51', 'mirror': '52', 'mug': '53', 'newspaper': '54', 'ottoman': '55', 'painting': '56', 'pan': '57', 'papertowel': '58', 'papertowelroll': '59', 'pen': '60', 'pencil': '61', 'peppershaker': '62', 'pillow': '63', 'plate': '64', 'plunger': '65', 'poster': '66', 'pot': '67', 'potato': '68', 'remotecontrol': '69', 'safe': '70', 'saltshaker': '71', 'scrubbrush': '72', 'shelf': '73', 'showerdoor': '74', 'showerglass': '75', 'sink': '76', 'sinkbasin': '77', 'soapbar': '78', 'soapbottle': '79', 'sofa': '80', 'spatula': '81', 'spoon': '82', 'spraybottle': '83', 'statue': '84', 'stoveburner': '85', 'stoveknob': '86', 'diningtable': '87', 'coffeetable': '88', 'sidetable': '89', 'teddybear': '90', 'television': '91', 'tennisracket': '92', 'tissuebox': '93', 'toaster': '94', 'toilet': '95', 'toiletpaper': '96', 'toilethanger': '97', 'toiletpaperroll': '98', 'tomato': '99', 'towel': '100', 'tvstand': '102', 'vase': '103', 'watch': '104', 'wateringcan': '105', 'window': '106', 'winebottle': '107'}
    # {'name': 'id', 'usedup': '0', 'breakable': '1', 'openable': '2', 'fillable': '3', 'pickupable': '4', 'sliceable': '5', 'moveable': '6', 'cookable': '7', 'toggleable': '8', 'receptacle': '9', 'dirty': '10'}
    print(dict_objects)
    print(dict_attribute)
    # load url dataframe
    Src = []
    Dst = []
    source = urllib.request.urlopen(url_object_object).read()
    soup = BeautifulSoup(source, 'lxml')
    tablelist = soup.find_all(class_='tableWrapper')
    table = tablelist[0].find('tbody').find_all('tr')
    # dict_objects & dict_attribute interact
    for i, tr in enumerate(table):
        tr = tr.find_all('td')
        objects = tr[1].text.lower()
        attributes = re.findall('[^-,\ *]*', tr[2].text.lower())
        print(objects)
        for attribute in attributes:
            if attribute in dict_attribute and objects in dict_objects:
                Src.append(dict_attribute[attribute])
                Dst.append(dict_objects[objects])
                print("object: {}, Id: {}, attribute: {}, Id: {}".format(
                    objects, dict_objects[objects], attribute, dict_attribute[attribute]))
    # process
    string_csv = "Src,Dst" + ",\n"
    for src, dst in zip(Src, Dst):
        string_csv += ','.join([str(src), str(dst)]) + ","
        string_csv += "\n"
    # csv
    attribute_object_relation = open(save_path + "attribute-behave-object.csv", "w")
    attribute_object_relation.write(string_csv)


def create_relation_object_room():
    row_objects = open(save_path + "object.csv", "r").readlines()
    row_room = open(save_path + "room.csv", "r").readlines()
    dict_objects = {}
    dict_room = {}
    for rows in row_objects[1:]:
        # ['0', 'alarmclock', '0.039934035', '0.009737755', '-0.005974793'...]
        row_list = [row.strip() for row in rows.split(",")]
        dict_objects[row_list[1].lower()] = [row_list[0], 0]
        # 'AlarmClock' -> alarm
        # dangerous: ori dict would be replace
        # subword_object = re.findall('[A-Z][^A-Z]*', row_list[1])[0]
        # dict_objects[subword_object.lower()] = [row_list[0], 0]
    for rows in row_room:
        row_list = [row.strip().lower() for row in rows.split(",")]
        dict_room[row_list[1]] = row_list[0]
    dict_room["livin"] = dict_room["livingroom"]
    dict_room["living"] = dict_room["livingroom"]
    print(dict_objects)
    print(dict_room)
    # load url dataframe
    Src = []
    Dst = []
    source = urllib.request.urlopen(url_object_room).read()
    soup = BeautifulSoup(source, 'lxml')
    tablelist = soup.find_all(class_='tableWrapper')
    table = tablelist[0].find('tbody').find_all('tr')
    # dict_objects & dict_attribute interact
    for i, tr in enumerate(table):
        tr = tr.find_all('td')
        objects = tr[1].text.lower()
        rooms = re.findall('[^(-,\ *)]*', tr[2].text.lower())
        # print(rooms)
        for room in rooms:
            if room == "" or room == "room" or room == "some":
                continue
            if room in dict_room and objects in dict_objects:
                Src.append(dict_room[room])
                Dst.append(dict_objects[objects][0])
                dict_objects[objects][1] += 1
                print("object: {}, Id: {}, room: {}, Id: {}".format(
                    objects, dict_objects[objects], room, dict_room[room]))
    for k, i in dict_objects.items():
        if i[1] < 1:
            print("=== Can't find === object: {}, Id: {}, count: {}".format(k, i[0], i[1]))
    # process
    string_csv = "Src,Dst" + ",\n"
    for src, dst in zip(Src, Dst):
        string_csv += ','.join([str(src), str(dst)]) + ","
        string_csv += "\n"
    # csv
    object_room_relation = open(save_path + "room-interact-object.csv", "w")
    object_room_relation.write(string_csv)


def create_relation_lowaction_subgoal():
    # load json data
    with open(SUB_GOAL_DISTRIBUTION) as f:
        json_subgoal = json.load(f)
    row_lowaction = open(NODE_PATH_LOWACTION, "r").readlines()
    row_subgoal = open(NODE_PATH_SUBGOAL, "r").readlines()
    dict_lowaction = {}
    dict_subgoal = {}
    for rows in row_lowaction:
        # row_list = [0, LookDown, ...]
        row_list = [row.strip() for row in rows.split(",")]
        dict_lowaction[row_list[1]] = row_list[0]
    for rows in row_subgoal:
        row_list = [row.strip() for row in rows.split(",")]
        dict_subgoal[row_list[1]] = row_list[0]
    # {'name': 'Id', 'LookDown': '0', 'MoveAhead': '1', 'RotateLeft': '2', 'LookUp': '3', 'RotateRight': '4', 'PickupObject': '5', 'SliceObject': '6', 'OpenObject': '7', 'PutObject': '8', 'CloseObject': '9', 'ToggleObjectOn': '10', 'ToggleObjectOff': '11'}
    # {'name': 'Id', 'GotoLocation': '0', 'PickupObject': '1', 'PutObject': '2', 'CoolObject': '3', 'HeatObject': '4', 'CleanObject': '5', 'SliceObject': '6', 'ToggleObject': '7'}
    print(dict_lowaction)
    print(dict_subgoal)
    # load url dataframe
    Src = []
    Dst = []
    t_Src = []
    t_Dst = []
    for sg_name, json_dict_lowact in json_subgoal.items():
        # import pdb; pdb.set_trace()
        # json_dict_lowact = {'total_num': 1001, 'LookDown': 0, 'MoveAhead': 0, 'RotateLeft': 0, 'LookUp': 0, 'RotateRight': 0, 'PickupObject': 0, 'SliceObject': 1001, 'OpenObject': 117, 'PutObject': 0, 'CloseObject': 106, 'ToggleObjectOn': 0, 'ToggleObjectOff': 0}
        print(json_dict_lowact)
        for str_lowaction, value in json_dict_lowact.items():
            if str_lowaction in dict_lowaction and value > 0:
                Src.append(dict_lowaction[str_lowaction])
                Dst.append(dict_subgoal[sg_name])
                t_Src.append(dict_subgoal[sg_name])
                t_Dst.append(dict_lowaction[str_lowaction])
                print("lowaction: {}, Id: {}, subgoal: {}, Id: {}"
                      .format(str_lowaction, dict_lowaction[str_lowaction], sg_name, dict_subgoal[sg_name]))
    # process
    string_csv = "Src,Dst" + ",\n"
    string_csv_r = "Src,Dst" + ",\n"
    for src, dst, t_src, t_dst in zip(Src, Dst, t_Src, t_Dst):
        string_csv += ','.join([str(src), str(dst)]) + ","
        string_csv += "\n"
        string_csv_r += ','.join([str(t_src), str(t_dst)]) + ","
        string_csv_r += "\n"
    # csv
    csv_edges = open(EDGE_PATH_LOWACTION_BEHAVE_SUBGOAL, "w")
    csv_edges.write(string_csv)
    # csv
    csv_edges = open(EDGE_PATH_SUBGOAL_BEHAVE_LOWACTION, "w")
    csv_edges.write(string_csv_r)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', action='store_true')
    parser.add_argument('--verb', action='store_true')
    args = parser.parse_args()

    '''
    object relation
    '''
    if args.object:
        # create node
        create_node_object()
        create_node_attribute()
        create_node_room()

        # # create relation
        create_relation_object_object_by_visual_genome()
        create_relation_attribute_object()
        create_relation_object_room()
    '''
    verb relation: alfred sub-goal & low action relation
    '''
    if args.verb:
        create_node_subgoal()
        create_node_low_action()
        create_relation_lowaction_subgoal()
