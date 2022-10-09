import argparse

from graphstorm.data.ogbn_datasets import OGBTextFeatDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='query-asin')
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--edge_pct", type=float, default=1)
    parser.add_argument("--dataset",type=str,default="ogbn-arxiv")
    parser.add_argument('--bert_model_name',type=str,default="bert-base-uncased")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--retain_original_features", type=lambda x: (str(x).lower() in ['true', '1']), default=False)
    args = parser.parse_args()
    # only for test
    dataset = OGBTextFeatDataset(args.filepath,
                                 edge_pct=args.edge_pct,
                                 dataset=args.dataset,
                                 bert_model_name=args.bert_model_name,
                                 max_sequence_length=args.max_sequence_length,
                                 retain_original_features=args.retain_original_features)
    dataset.save_graph(args.savepath)
