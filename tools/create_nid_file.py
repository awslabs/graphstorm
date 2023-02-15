import json
import os
import argparse


argparser = argparse.ArgumentParser("create nid file")

argparser.add_argument("--node-file", type=str, required=True,
                           help="path to nodes-xx")
argparser.add_argument("--id-field", type=str, default="id", help="the id field")
args = argparser.parse_args()


file_list = os.listdir(args.node_file)
file_list.sort()

ent_name = args.node_file.split('-')[1]

data = []
for i, name in enumerate(file_list):
    if i % 10 == 0:
        print (i)
    with open(os.path.join(args.node_file, name),'r') as infile:
        for line in infile:
            data.append(json.loads(line)[args.id_field])

with open("nid-%s.txt" % ent_name, "w") as f:
    for line in data:
        assert line is not None
        f.write(json.dumps(line) + "\n")
