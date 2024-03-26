import random
import json
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# text processing function
def text_process(text):
    p_text = ' '.join(text.split('\r\n'))
    p_text = ' '.join(p_text.split('\n\r'))
    p_text = ' '.join(p_text.split('\n'))
    p_text = ' '.join(p_text.split('\t'))
    p_text = ' '.join(p_text.split('\rm'))
    p_text = ' '.join(p_text.split('\r'))
    p_text = ''.join(p_text.split('$'))
    p_text = ''.join(p_text.split('*'))

    return p_text

def construct_one_domain(full_file_path, asin_dict, asin_data, edge_dict, dup_asin, brand_dict, category_dict_list, item_brand, item_category):
    """ 
    Process review-related edges and node attributes and save them into parquet files.
    The results are saved into parquet files.
    Input example: {"reviewTime": "12 11, 2015", "reviewerID": "A27BTSGLXK2C5K", "asin": "B017O9P72A", 
                    "reviewerName": "Jacob M. Wessler", "reviewText": "Alexa is not able to control my
                    lights. If I ask her to tell me what LIFX can do, she will give me an example with
                    one of my group names. If I use that exact same group name in a new request, she'll
                    await that she doesn't recognize the name. This skill is VERY buggy and has not yet
                    worked for me. I even rest Alexa, uninstalled LIFX, and set everything up again.", 
                    "summary": "VERY Buggy, doesn't work.", "unixReviewTime": 1449792000}
    Output example:
        review.parquet
            format (review_id is consisited by reviewerID and asin): 
                review_id review_text review_summary
        write_review.parquet
            format:
                reviewerID review_id
        receive_review.parquet
            format:
                asin review_id
    """
    # read raw data
    with open(full_file_path) as f:
        raw_data = {}
        for line in f:
            tmp = json.loads(line)
            raw_data[f"{tmp['reviewerID']}_{tmp['asin']}"] = tmp

    review_id, review_text, summary = [], [], []
    write_review, receive_review = ([], []), ([], [])
    for idd, data in raw_data.items():
        if 'reviewText' in data:
            review_id.append(idd)
            review_text.append(data['reviewText'])
            if 'summary' in data:
                summary.append(data['summary'])
            else:
                summary.append('')
            write_review[0].append(data['reviewerID'])
            write_review[1].append(idd)
            receive_review[0].append(data['asin'])
            receive_review[1].append(idd)
    df = pd.DataFrame({'customer_id': list(set(write_review[0]))})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'processed_data/amazon_review/customer.parquet')

    df = pd.DataFrame({'review_id': review_id, 'text': review_text, 'summary': summary})
    table = pa.Table.from_pandas(df)
    rows_per_file = 500000

    # Calculate the total number of files needed
    num_files = len(df) // rows_per_file + (len(df) % rows_per_file > 0)

    # Iterate through chunks and write to separate Parquet files
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = (i + 1) * rows_per_file if (i + 1) * rows_per_file < len(df) else len(df)
        
        # Extract a chunk of the DataFrame
        chunk_df = df.iloc[start_idx:end_idx]

        # Convert the chunk to a pyarrow.Table
        chunk_table = pa.Table.from_pandas(chunk_df)
        
        # Write the table to a Parquet file
        file_path = f'processed_data/amazon_review/review/review_{i + 1}.parquet'
        pq.write_table(chunk_table, file_path)

        print(f"Parquet file {file_path} created for rows {start_idx + 1} to {end_idx}")
    
    df = pd.DataFrame({'customer': write_review[0], 'review': write_review[1]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'processed_data/amazon_review/write_review.parquet')
    df = pd.DataFrame({'item': receive_review[0], 'review': receive_review[1]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'processed_data/amazon_review/receive_review.parquet')

def construct_graph(full_file_path):
    random.seed(2023)
    
    asin_dict = dict()
    dup_asin = set()
    asin_data = dict()
    brand_dict = defaultdict(int)
    category_dict = [defaultdict(int),defaultdict(int), defaultdict(int)]
    item_brand = dict()
    item_category = [dict(), dict(), dict()]
    edge_dict = {
        ('item', 'also_buy', 'item'): [],
        ('item', 'also_view', 'item'): [],
    }
    cnt = 0

    construct_one_domain(full_file_path, asin_dict, asin_data, edge_dict, dup_asin, brand_dict, category_dict, item_brand, item_category)
    print(f"number of asins:{len(asin_dict)}, number of edges:{[(k,len(v)) for k,v in edge_dict.items()]}")

if __name__ == '__main__':
    file_path = 'raw_data/All_Amazon_Review.json'
    construct_graph(file_path)