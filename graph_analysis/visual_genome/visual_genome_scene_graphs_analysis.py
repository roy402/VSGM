import time
import numpy
from tqdm import trange
SceneGraphsJsonFile = "./scene_graphs.json"
AlfredObjectsFile = "../data/objects.txt"


def test_VG_api():
    # https://github.com/ranjaykrishna/visual_genome_python_driver
    from visual_genome import api
    import pdb
    pdb.set_trace()
    ids = api.get_all_image_ids()
    # 108077
    print(ids)
    # 'visual_genome.models.Image'
    # id: 61512, coco_id: 248774, flickr_id: 6273011878, width: 1024, url: https://cs.stanford.edu/people/rak248/VG_100K/61512.jpg
    image = api.get_image_data(id=61512)
    print(image)
    # 'list'
    # [id: 1, x: 511, y: 241, width: 206,height: 320, phrase: A brown, sleek horse with a bridle, image: 61512, id: 6, x: 291, y: 214, width: 431,height: 345, phrase: two horses stand on the grass, image: 61512, id: 11, x: 297, y: 235, width: 424,height: 321, phrase: Two horses in a pasture, image: 61512,
    regions = api.get_region_descriptions_of_image(id=61512)
    print(regions[0])
    # 'visual_genome.models.Graph'
    # Region Graph
    graph = api.get_scene_graph_of_image(id=61512)
    # [horse, grass, horse, bridle, truck, sign, gate, truck, tire, trough, window, door, building, halter, mane, mane, leaves, fence]
    print(graph.objects)
    # [3015675: horse is brown, 3015676: horse is spotted, 3015677: horse is red, 3015678: horse is dark brown, 3015679: truck is red, 3015680: horse is brown, 3015681: truck is red, 3015682: sign is blue, 3015683: gate is red, 3015684: truck is white, 3015685: tire is blue, 3015686: gate is wooden, 3015687: horse is standing, 3015688: truck is red, 3015689: horse is brown and white, 3015690: building is tan, 3015691: halter is red, 3015692: horse is brown, 3015693: gate is wooden, 3015694: grass is grassy, 3015695: truck is red, 3015696: gate is orange, 3015697: halter is red, 3015698: tire is blue, 3015699: truck is white, 3015700: trough is white, 3015701: horse is brown and cream, 3015702: leaves is green, 3015703: grass is lush, 3015704: horse is enclosed, 3015705: horse is brown and white, 3015706: horse is chestnut, 3015707: gate is red, 3015708: leaves is green, 3015709: building is brick, 3015710: truck is large, 3015711: gate is red, 3015712: horse is chestnut colored, 3015713: fence is wooden]
    print(graph.attributes)
    # [3199950: horse stands on top of grass, 3199951: horse is in grass, 3199952: horse is wearing bridle, 3199953: trough is for horse, 3199954: window is next to door, 3199955: building has door, 3199956: horse is nudging horse, 3199957: horse has mane, 3199958: horse has mane, 3199959: trough is for horse]
    print(graph.relationships)


class GenomeProcess(object):
    """docstring for GenomeProcess"""

    def __init__(self):
        """
        relationship_matrics : For GCN Adjacency matrix
        """
        super(GenomeProcess, self).__init__()
        self.alfred_objects = load_alfred_object()
        self.alfred_objects_index = {name: index for index, name in enumerate(self.alfred_objects)}
        self.relationship_matrics = numpy.zeros(
            (len(self.alfred_objects), len(self.alfred_objects)), dtype=int)

    # For test to watch scene graph objects
    def get_scene_graph_existance_objects(self):
        def _get_all_objects(self, data):
            objects = set()
            for i in trange(0, len(data), desc='i'):
                graph_objects = data[i]["objects"]
                graph_relationships = data[i]["relationships"]
                for graph_object in graph_objects:
                    if "names" in graph_object:
                        objects.update(graph_object["names"])
            with open('all_objects.txt', 'w') as f:
                for item in objects:
                    try:
                        f.write("%s\n" % item)
                    except Exception as e:
                        print(item)
        # relationships, imgids, objects
        data = read_json(SceneGraphsJsonFile)
        _get_all_objects(data)

    def count_alfred_object_relationships_by_VG_scene_graphs(self):
        def count_objects(relationship_matrics, alfred_objects, alfred_objects_index, scene_graphs_data):
            """
            intersection_objects : Objects appear at the same image
            """
            for i in trange(0, len(scene_graphs_data), desc='i'):
                graph_objects = scene_graphs_data[i]["objects"]
                intersection_objects = find_graph_objects_in_alfred_objects(
                    graph_objects, alfred_objects)
                # if scene_graphs_data
                for i_object in intersection_objects:
                    for j_object in intersection_objects:
                        relationship_matrics[alfred_objects_index[i_object],
                                             alfred_objects_index[j_object]] += 1
            return relationship_matrics

        def find_graph_objects_in_alfred_objects(graph_objects, alfred_objects):
            objects = set()
            for graph_object in graph_objects:
                if "names" in graph_object:
                    objects.update(graph_object["names"])
            intersection = list(objects.intersection(alfred_objects))
            return intersection
        self.scene_graphs_data = read_json(SceneGraphsJsonFile)
        self.relationship_matrics = count_objects(
            self.relationship_matrics, self.alfred_objects, self.alfred_objects_index, self.scene_graphs_data)
        self.relationship_matrics = numpy.asarray(self.relationship_matrics, dtype=int)
        print("\n=== The word doesn't exist at Visual Genome=== ")
        for i in range(self.relationship_matrics.shape[0]):
            if self.relationship_matrics[i, i] == 0:
                print(self.alfred_objects[i])
        with open("relationship_matrics.csv", "w") as csv:
            numpy.savetxt(csv, self.relationship_matrics, fmt='%i', delimiter=",")
        with open("relationship_matrics_with_word.csv", "w") as csv:
            csv.write("," + ",".join(self.alfred_objects) + "\n")
            for ind, row in enumerate(self.relationship_matrics):
                csv.write(self.alfred_objects[ind] + ", ")
                csv.write(','.join(str(n) for n in row) + "\n")

    def get_adjacency_matrix(self):
        relationship_matrics = numpy.genfromtxt('relationship_matrics.csv', delimiter=',')
        relationship_matrics = numpy.asarray(relationship_matrics, dtype=int)
        for i in range(relationship_matrics.shape[0]):
            for j in range(relationship_matrics.shape[1]):
                relationship_matrics[i, j] = 1 if relationship_matrics[i, j] > 0 else 0
        with open("A.csv", "w") as csv:
            numpy.savetxt(csv, relationship_matrics, fmt='%i', delimiter=",")


def read_json(path):
    import json
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data

# for match visual genome format
# otherwise it would be "The word doesn't exist at Visual Genome"
def load_alfred_object(file=AlfredObjectsFile):
    import re
    objects = open(file).readlines()
    objects = [o.strip() for o in objects]
    for i in range(len(objects)):
        o = re.findall('[A-Z][^A-Z]*', objects[i])
        o = " ".join(o).lower()
        objects[i] = o
    return objects


if __name__ == '__main__':
    # count time
    start = time.time()
    gp = GenomeProcess()
    gp.count_alfred_object_relationships_by_VG_scene_graphs()
    gp.get_adjacency_matrix()

    end = time.time()
    print(end - start)
