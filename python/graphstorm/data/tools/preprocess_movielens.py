"""tools to preprocess the movielen100k data for later use
"""
import json
import os
import time
import argparse
import math


def process_file_user(tfile):
    '''Process a file of users.

    Each row in the file is a user record.
    '''
    user_edges = []
    users = []
    with open(tfile, 'r', encoding="ISO-8859-1") as f:
        start = time.time()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            uid, age, gender, occupation, _ = line.split('|')
            user_edges.append({'src_id': uid, 'dst_id': occupation})
            users.append({'id': uid, 'age': age, 'gender': gender})
        print('processing {} lines takes {:.3f} seconds'.format(len(lines), time.time() - start))

    out_nodes = {}
    out_edges = {}
    out_nodes['user'] = users
    out_edges[('user', 'has-occupation', 'occupation')] = user_edges
    return out_nodes, out_edges

def parse_record_movie(record):
    '''Parse the record of a user.
    '''
    fields = record.split('|')
    mid = fields[0]
    title = fields[1]
    date = fields[2]
    # TODO(zhengda) just take one of the genres as the label.
    genre = None
    for i, v in enumerate(fields[5:]):
        if int(v) == 1:
            genre = i
            break
    assert genre is not None

    return {'id': mid, 'title': title, 'date': date, 'genre': genre}

def process_file_movie(tfile):
    '''Process a file of movies.

    Each row in the file is a movie record.
    '''
    movies = []
    with open(tfile, 'r', encoding="ISO-8859-1") as f:
        start = time.time()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            movie_record = parse_record_movie(line)
            movies.append(movie_record)
        print('processing {} lines takes {:.3f} seconds'.format(len(lines), time.time() - start))

    out_nodes = {}
    out_edges = {}
    out_nodes['movie'] = movies
    return out_nodes, out_edges

def parse_record_occupation(record):
    '''Parse the record of a occupation.
    '''
    return {'id': record}

def process_file_occupation(tfile):
    '''Process a file of occupations.

    Each row in the file is an occupation.
    '''
    occupations = []
    with open(tfile, 'r', encoding="ISO-8859-1") as f:
        start = time.time()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            occupation_record  = parse_record_occupation(line)
            occupations.append(occupation_record)
        print('processing {} lines takes {:.3f} seconds'.format(len(lines), time.time() - start))

    out_nodes = {}
    out_edges = {}
    out_nodes['occupation'] = occupations
    return out_nodes, out_edges

def parse_record_rating(record):
    '''Parse the record of a user.
    '''
    user, movie, rating, timestamp = record.split('\t')

    return {'src_id': user, 'dst_id': movie, 'rate': rating, 'time': timestamp}

def process_file_ratings(tfile):
    '''Process a file of ratings.

    Each row in the file is a rating.
    '''
    rating_edges = []
    with open(tfile, 'r', encoding="ISO-8859-1") as f:
        start = time.time()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            rating_record = parse_record_rating(line)
            rating_edges.append(rating_record)
        print('processing {} lines takes {:.3f} seconds'.format(len(lines), time.time() - start))

    out_nodes = {}
    out_edges = {}
    out_edges['user', 'rating', 'movie'] = rating_edges
    return out_nodes, out_edges

def write_list(data, output_file):
    """save data to the local file
    """
    with open(output_file, 'w', encoding="utf-8") as f:
        for d in data:
            line = json.dumps(d)
            f.write(line + '\n')

def split_list(data, size):
    """split data into different partitions
    """
    res = []
    for x in range(0, len(data), size):
        if x + size <= len(data):
            res.append(data[x:(x+size)])
        else:
            res.append(data[x:])
    return res

def write_list_split(data, output_dir, size_per_file):
    """save split data to local files
    """
    if size_per_file > 0:
        data_list = split_list(data, size_per_file)
    else:
        data_list = [data]
    os.makedirs(output_dir, exist_ok = True)
    for i, data_piece in enumerate(data_list):
        with open(os.path.join(output_dir, str(i) + '.json'), 'w', encoding="utf-8") as f:
            for d in data_piece:
                line = json.dumps(d)
                f.write(line + '\n')

def write_nodes(out_nodes, output_dir, num_splits):
    """save nodes into local files
    """
    for ntype in out_nodes:
        data = out_nodes[ntype]
        if len(data) == 0:
            continue
        node_dir = os.path.join(output_dir, 'nodes-' + ntype)
        os.makedirs(node_dir, exist_ok = True)
        print('node dir:', node_dir)
        write_list_split(data, node_dir,
            int(math.ceil(len(data) / num_splits)))
        write_list([node['id'] for node in data],
                os.path.join(output_dir, 'nid-{}.txt'.format(ntype)))

def write_edges(out_edges, output_dir, num_splits):
    """save edges into local files
    """
    for etype in out_edges:
        data = out_edges[etype]
        if len(data) == 0:
            continue
        etype_dir = os.path.join(output_dir, 'edges-' + '_'.join(etype))
        os.makedirs(etype_dir, exist_ok = True)
        print('etype dir:', etype_dir)
        write_list_split(data, etype_dir,
            int(math.ceil(len(data) / num_splits)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BertModel')
    parser.add_argument('--input_path', type=str,
        help='The path of the input directory that contains all files.')
    parser.add_argument('--output_path', type=str,
        help='The path of the output directory')
    parser.add_argument('--num_split_files', type=int, default=1,
        help="Generate multiple files for distributed data loading test")
    args = parser.parse_args()

    nodes, edges = process_file_user(os.path.join(args.input_path, 'u.user'))
    nodes1, edges1 = process_file_movie(os.path.join(args.input_path, 'u.item'))
    nodes2, edges2 = process_file_occupation(os.path.join(args.input_path, 'u.occupation'))
    nodes3, edges3 = process_file_ratings(os.path.join(args.input_path, 'u.data'))
    nodes.update(nodes1)
    nodes.update(nodes2)
    nodes.update(nodes3)
    edges.update(edges1)
    edges.update(edges2)
    edges.update(edges3)

    write_nodes(nodes, args.output_path, args.num_split_files)
    write_edges(edges, args.output_path, args.num_split_files)
