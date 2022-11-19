"""tools to preprocess the yelp data for later use
"""
import json
import os
import time
import argparse

def parse_record_user(record, out_nodes, out_edges):
    '''Parse the record of a user.
    '''
    friends = record['friends']
    del record['friends']
    user = record['user_id']
    del record['user_id']
    record['id'] = user
    out_nodes.append(record)
    for friend in friends.split(','):
        friend = friend.strip()
        out_edges.append({'src_id': user, 'dst_id': friend})

def process_file_user(tfile):
    '''Process a file of users.

    Each row in the file is a user record.
    '''
    file_edges = []
    users = []
    with open(tfile, 'r', encoding="utf-8") as f:
        start = time.time()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            record = json.loads(line)
            parse_record_user(record, users, file_edges)
        print('processing {} lines takes {:.3f} seconds'.format(len(lines), time.time() - start))

    out_nodes = {}
    out_edges = {}
    out_nodes['user'] = users
    # use {}, instead of set(), to build a set object as the pylint standard suggested
    user_set = {user['id'] for user in users}
    tmp_edges = []
    # Some of the friend nodes do not exist. We need to filter those friendship edges.
    for edge in file_edges:
        if edge['src_id'] in user_set and edge['dst_id'] in user_set:
            tmp_edges.append(edge)
    out_edges[('user', 'friendship', 'user')] = tmp_edges
    print('There are {} users and {} valid friendships'.format(len(users), len(tmp_edges)))
    return out_nodes, out_edges

def get_binary_attr_values(data, attr_names):
    '''Get the values of binary attributes from the input dict.

    The value of a binary attribute is either 1 or 0.

    It's very likely that an attribute does not exist in the dict. If it doesn't exist,
    its value is 0. If the attribute exists, its value is stored as a string. We consider
    the string of 'True' or 'true' to indicate 1 and all other strings to indicate 0.
    '''
    vals = []
    for attr in attr_names:
        if attr not in data:
            val = 0
        elif isinstance(data[attr], str):
            val = int(data[attr] in ['True', 'true'])
        else:
            isinstance(data[attr], bool)
            val = int(data[attr])
        vals.append(val)
    return vals

def get_category_values(data, attr_names, cat_maps):
    '''Get the values of categorical attributes.

    In the input data, the categorical attribute contains a string. We need to map the string
    to an integer. As such, we need to maintain a map for each categoriacl attribute, whose key
    is the string and the value is an integer that is unique for the string. We build this
    categorical map on the fly.
    The string in a categorical attribute may be either u'xxx' or 'xxx'. This function treats
    them as the same categorical value. Each categorical attribute has the default value of 0,
    which indicates a missing value for the attribute.
    '''
    vals = []
    for attr in attr_names:
        # cat_maps contains the mappings from strings to integers
        # for every categorical attributes.
        if attr in cat_maps:
            cat_map = cat_maps[attr]
        else:
            cat_map = {'None': 0}
            cat_maps[attr] = cat_map

        # map a string to an integer. If the categorial attribute doesn't exist
        # in the record, we set its value to 0.
        if attr not in data:
            vals.append(0)
        else:
            val = data[attr]
            if val[:2] == "u'":
                val = val[2:-1]
            elif val[:1] == "'":
                val = val[1:-1]
            # 0 is used as the default value if a record doesn't contain the categorial attribute.
            if val not in cat_map:
                cat_map[val] = len(cat_map)
            vals.append(cat_map[val])
    return vals

def contain_all_attrs(data, attr_names):
    '''Verify if the input data has expected attributes.

    This function is a sanity check to make sure we have collected all attributes
    from the input data.
    '''
    return all(attr in attr_names for attr in data)

def load_non_json_data(data):
    '''Extract values from a format incompatible to JSON.

    The JSON format requires the keys and values to have double quotes.
    Some values of the input data are not stored in a format incompatible to JSON.
    This function is to extract keys and values from this special format.
    '''
    data = data.strip()[1:-1]
    ret_vals = {}
    for entry in data.split(','):
        parts = entry.split(':')
        if len(parts) < 2:
            continue
        key, val = parts
        key = key.strip()[1:-1]
        val = val.strip()
        ret_vals[key] = val in ('True', 'true')
    return ret_vals

def get_additional_binary_attrs(record, attr_name_map):
    '''Get additional binary attributes from the JSON-incompatible format.
    '''
    vals = []
    for attr_type in attr_name_map:
        attr_names = attr_name_map[attr_type]
        if 'attributes' in record and record['attributes'] is not None \
                and attr_type in record['attributes'] \
                and record['attributes'][attr_type] != "None":
            attrs = load_non_json_data(record['attributes'][attr_type])
            assert contain_all_attrs(attrs, attr_names)
            vals.extend(get_binary_attr_values(attrs, attr_names))
        else:
            vals.extend([0 for attr in attr_names])
    return vals

def parse_record_business(record, cat_map, out_nodes, out_edges):
    '''Parse the record of business.
    '''
    bin_attr_names = ['RestaurantsTableService', 'BikeParking', 'BusinessAcceptsCreditCards',
                  'RestaurantsReservations', 'WheelchairAccessible', 'Caters',
                  'OutdoorSeating', 'RestaurantsGoodForGroups', 'HappyHour',
                  'BusinessAcceptsBitcoin', 'HasTV', 'DogsAllowed', 'RestaurantsTakeOut',
                  'GoodForKids', 'ByAppointmentOnly', 'AcceptsInsurance',
                  'GoodForDancing', 'BYOB', 'CoatCheck', 'DriveThru', 'Corkage',
                  'RestaurantsCounterService', 'Open24Hours']
    cat_attr_names = ['WiFi', 'Alcohol', 'NoiseLevel', 'RestaurantsAttire', 'RestaurantsDelivery',
            'Smoking', 'BYOBCorkage', 'AgesAllowed']
    num_attr_names = ['RestaurantsPriceRange2']
    attr_name_map = {
            'BusinessParking': ['garage', 'street', 'validated', 'lot', 'valet'],
            'Ambience': ['touristy', 'hipster', 'romantic', 'divey', 'intimate',
                         'trendy', 'upscale', 'classy', 'casual'],
            'GoodForMeal': ['dessert', 'latenight', 'lunch', 'dinner', 'brunch', 'breakfast'],
            'HairSpecializesIn': ['straightperms', 'coloring', 'extensions', 'africanamerican',
                                  'curly', 'kids', 'perms', 'asian'],
            'BestNights': ['monday', 'tuesday', 'friday', 'wednesday', 'thursday', 'sunday',
                           'saturday'],
            'Music': ['dj', 'background_music', 'no_music', 'jukebox', 'live', 'video', 'karaoke'],
            "DietaryRestrictions": ['dairy-free', 'gluten-free', 'vegan', 'kosher', 'halal',
                                    'soy-free', 'vegetarian'],
    }

    vals = []
    if 'attributes' in record and record['attributes'] is not None:
        vals.extend(get_binary_attr_values(record['attributes'], bin_attr_names))
        cat_vals = get_category_values(record['attributes'], cat_attr_names, cat_map)
        assert contain_all_attrs(record['attributes'],
                bin_attr_names + cat_attr_names + num_attr_names + list(attr_name_map.keys()))
    else:
        vals.extend([0 for _ in bin_attr_names])
        cat_vals = [0 for _ in cat_attr_names]

    vals.extend(get_additional_binary_attrs(record, attr_name_map))
    del record['attributes']

    # process the categories
    if 'categories' in record and record['categories'] is not None:
        for category in record['categories'].split(','):
            category = category.strip()
            out_edge = {'src_id': record['business_id'], 'dst_id': category}
            out_edges[('business', 'incategory', 'category')].append(out_edge)
        del record['categories']

    # process the location.
    if 'city' in record and record['city'] is not None:
        out_edges[('business', 'in', 'city')].append({'src_id': record['business_id'],
                                                      'dst_id': record['city']})
        del record['city']
        del record['state']
        del record['postal_code']
        del record['latitude']
        del record['longitude']
        del record['address']

    business_data = {}
    business_data['numeric_vals'] = ','.join([str(val) for val in vals])
    business_data['category_vals'] = ','.join([str(val) for val in cat_vals])
    for name in record:
        if name != 'business_id':
            business_data[name] = record[name]
    business_data['id'] = record['business_id']
    out_nodes['business'].append(business_data)

def process_file_business(tfile):
    '''Process the file that contains business

    Each row in the file contains a record of business.
    '''
    out_nodes = {'business': [], 'category': [], 'city': []}
    out_edges = {('business', 'incategory', 'category'): [],
                 ('business', 'in', 'city'): []}
    cat_map = {}
    with open(tfile, 'r', encoding="utf-8") as f:
        start = time.time()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            record = json.loads(line)
            parse_record_business(record, cat_map, out_nodes, out_edges)
        print('processing {} lines takes {:.3f} seconds'.format(len(lines), time.time() - start))
        print('There are {} business'.format(len(out_nodes['business'])))

    uniq_cate = {edge['dst_id'] for edge in out_edges[('business', 'incategory', 'category')]}
    categories = list(uniq_cate)
    uniq_cities = {edge['dst_id'] for edge in out_edges[('business', 'in', 'city')]}
    cities = list(uniq_cities)
    print('There are {} categories and {} cities'.format(len(categories), len(cities)))
    # Save category data, which only contains the name of the category.
    # We need the category name to generate word embeddings.
    out_nodes['category'] = [{'id': category} for category in categories]
    out_nodes['city'] = [{'id': city} for city in cities]

    return out_nodes, out_edges

def parse_record_review(record, in_node_sets, out_nodes, out_edges):
    '''Parse the record of a review.
    '''
    user_id = record['user_id']
    business_id = record['business_id']
    review_id = record['review_id']
    assert user_id in in_node_sets['user']
    assert business_id in in_node_sets['business']
    del record['user_id']
    del record['business_id']
    del record['review_id']
    record['id'] = review_id
    out_nodes['review'].append(record)
    out_edges[('user', 'write', 'review')].append({'src_id': user_id, 'dst_id': review_id})
    out_edges[('review', 'on', 'business')].append({'src_id': review_id, 'dst_id': business_id})

def process_file_review(tfile, in_node_sets):
    '''Process the file of reviews.
    '''
    out_nodes = {'review': []}
    out_edges = {('user', 'write', 'review'): [],
                 ('review', 'on', 'business'): []}
    with open(tfile, 'r', encoding="utf-8") as f:
        start = time.time()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            record = json.loads(line)
            parse_record_review(record, in_node_sets, out_nodes, out_edges)
        print('processing {} lines takes {:.3f} seconds'.format(len(lines), time.time() - start))
        print('There are {} reviews'.format(len(out_nodes['review'])))

    return out_nodes, out_edges

def write_list(data, output_file):
    """save data to local files
    """
    with open(output_file, 'w', encoding="utf-8") as f:
        for d in data:
            line = json.dumps(d)
            f.write(line + '\n')

def split_list(data, size):
    """split data into partitions
    """
    res = []
    for x in range(0, len(data), size):
        if x + size <= len(data):
            res.append(data[x:(x+size)])
        else:
            res.append(data[x:])
    return res

def write_list_split(data, output_dir, size_per_file):
    """save data partitions into local files
    """
    if size_per_file > 0:
        data_list = split_list(data, size_per_file)
    else:
        data_list = [data]
    os.makedirs(output_dir, exist_ok = True)
    for i, data_partition in enumerate(data_list):
        with open(os.path.join(output_dir, str(i) + '.json'), 'w', encoding="utf-8") as f:
            for d in data_partition:
                line = json.dumps(d)
                f.write(line + '\n')

def write_nodes(out_nodes, output_dir, file_size):
    """save nodes to local files
    """
    for n_type in out_nodes:
        if len(out_nodes[n_type]) == 0:
            continue
        node_dir = os.path.join(output_dir, 'nodes-' + n_type)
        os.makedirs(node_dir, exist_ok = True)
        print('node dir:', node_dir)
        write_list_split(out_nodes[n_type], node_dir, file_size)
        write_list([node['id'] for node in out_nodes[n_type]],
                os.path.join(output_dir, 'nid-{}.txt'.format(n_type)))

def write_edges(out_edges, output_dir):
    """save edges to local files
    """
    for etype in out_edges:
        if len(out_edges[etype]) == 0:
            continue
        etype_dir = os.path.join(output_dir, 'edges-' + '_'.join(etype))
        os.makedirs(etype_dir, exist_ok = True)
        print('etype dir:', etype_dir)
        write_list(out_edges[etype], os.path.join(etype_dir, '0.json'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BertModel')
    parser.add_argument('--input_path', type=str,
                         help='The path of the input directory that contains all files.')
    parser.add_argument('--output_path', type=str,
                         help='The path of the output directory')
    parser.add_argument('--node_file_size', type=int,
                         help='The number of records in each node file.')
    args = parser.parse_args()

    business_file_path = os.path.join(args.input_path, 'yelp_academic_dataset_business.json')
    nodes, edges = process_file_business(business_file_path)
    user_file_path = os.path.join(args.input_path, 'yelp_academic_dataset_user.json')
    nodes1, edges1 = process_file_user(user_file_path)
    nodes.update(nodes1)
    edges.update(edges1)
    node_sets = {}
    for ntype in nodes:
        # use {}, instead of set(), to build a set object as the pylint standard suggested
        node_sets[ntype] = {node['id'] for node in nodes[ntype]}
    review_file_path = os.path.join(args.input_path, 'yelp_academic_dataset_review.json')
    nodes1, edges1 = process_file_review(review_file_path, node_sets)
    nodes.update(nodes1)
    edges.update(edges1)

    write_nodes(nodes, args.output_path, int(args.node_file_size))
    write_edges(edges, args.output_path)
