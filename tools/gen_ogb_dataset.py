"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Tool to create the ogb example datasets.
"""
import argparse

from graphstorm.data.ogbn_datasets import OGBTextFeatDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ogb')
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--edge_pct", type=float, default=1)
    parser.add_argument("--dataset",type=str,default="ogbn-arxiv")
    parser.add_argument('--bert_model_name',type=str,default="bert-base-uncased")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--retain_original_features", type=lambda x: (str(x).lower() in ['true', '1']), default=True)
    args = parser.parse_args()
    # only for test
    dataset = OGBTextFeatDataset(args.filepath,
                                 args.dataset,
                                 edge_pct=args.edge_pct,
                                 bert_model_name=args.bert_model_name,
                                 max_sequence_length=args.max_sequence_length,
                                 retain_original_features=args.retain_original_features)
    dataset.save_graph(args.savepath)
