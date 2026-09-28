import os
import argparse
import yaml
import numpy as np
import pandas as pd
import h5py
import mlflow
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluation.metrics import get_core_metrics, evaluate_mass_sculpting

def evaluate_model(config):
    # This is a placeholder for the evaluation script
    # It will load the trained model, run on test set, and compute metrics
    print(f"Evaluating {config['model']['name']}...")
    
    # Example metrics computation
    # metrics = get_core_metrics(y_test, y_score)
    # df = pd.DataFrame([metrics])
    # df.to_csv('reports/tables/metrics.csv', index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    evaluate_model(config)
