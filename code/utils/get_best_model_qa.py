import os
import json
from shutil import copy

def main():
    metric_file = "all_results.json"
    experiments_dir = "/data/jcanete/all_results/sqac/distillbeto"
    copy_best_model_to = "/data/jcanete/best_models/sqac/distillbert-base-spanish-uncased-finetuned-qa-sqac"
    
    best_metric = -1e5
    best_model_path = None

    for model_dir in os.listdir(experiments_dir):
        model_dir_path = os.path.join(experiments_dir, model_dir)
        if not os.path.isdir(model_dir_path):
            continue
        metric_file_path = os.path.join(model_dir_path, metric_file)
        with open(metric_file_path) as f:
            data = json.load(f)
            metric = data["eval_f1"]
            if metric > best_metric:
                best_metric = metric
                best_model_path = model_dir_path

    print("Best model: {}".format(best_model_path))
    print("Best metric: {}".format(best_metric))
    print("Copying to: {}".format(copy_best_model_to))

    for _file in os.listdir(best_model_path):
        file_path = os.path.join(best_model_path, _file)
        if os.path.isdir(file_path):
            continue
        copy(file_path, copy_best_model_to)

if __name__ == "__main__":
    main()