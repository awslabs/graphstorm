"""tools to test the preprocess of movielens100k data
"""
import argparse
from graphstorm.data import MovieLens100kNCDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='query-asin')
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument('--bert_model_name',type=str,default="bert-base-uncased")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--retain_original_features",
                        type=lambda x: (str(x).lower() in ['true', '1']), default=False)
    parser.add_argument("--user_age_as_label",
                        type=lambda x: (str(x).lower() in ['true', '1']), default=False)
    parser.add_argument("--user_text",
                        type=lambda x: (str(x).lower() in ['true', '1']), default=False,
                        help="use occupation field as text field for user nodes")
    args = parser.parse_args()
    # only for test

    dataset = MovieLens100kNCDataset(raw_dir=args.filepath,
                                    bert_model_name=args.bert_model_name,
                                    max_sequence_length=args.max_sequence_length,
                                    retain_original_features=args.retain_original_features,
                                    user_text=args.user_text,
                                    user_age_as_label=args.user_age_as_label)

    dataset.save_graph(args.savepath)
