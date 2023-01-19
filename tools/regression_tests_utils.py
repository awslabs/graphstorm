import json
import boto3
import datetime
import argparse

def save_model_results_in_s3(path, graph_name, task_type):
    """
    This function writes the model configuration and the test metric results to the S3 location
    Args:
        path: the path to the regression results
        graph_name: the name of the graph
        task_type: the type of the task

    Returns:

    """
    s3_rec = boto3.resource('s3')
    file_prefix = 'performance-history/{}/{}/{}/'.\
        format(graph_name,task_type,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    f = open(path)
    reported_results = json.load(f)
    # save test_metric
    s3_rec.Object('gsf-regression', file_prefix+"regression_results.json").put(
        Body=(bytes(json.dumps(reported_results).encode('UTF-8')))
    )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Regression tests checker")
    argparser.add_argument("--graph_name", type=str, required=True, help="The name of the graph dataset")
    argparser.add_argument("--filepath", type=str, default=None, help='The path of the regression_results json file.')
    argparser.add_argument('--task_type', type=str, required=True,
                           help='The type of the task.')
    args = argparser.parse_args()
    save_model_results_in_s3(args.filepath, args.graph_name, args.task_type)
    print("Results saved in S3.")
