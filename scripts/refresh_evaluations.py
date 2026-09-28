"""Re-score completed checkpoints after evaluation-code/reporting changes."""
import copy
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate import evaluate_model

CONFIGS={
    'M1_Autoencoder':'m1_autoencoder','M2_IsolationForest':'m2_isolation_forest',
    'M3_DeepSVDD':'m3_deep_svdd','M4_DeepSAD':'m4_deep_sad',
    'M5_Supervised':'m5_supervised','M6_MassAware':'m6_mass_aware'}


def main():
    path='reports/tables/results.csv'
    rows=pd.read_csv(path).drop_duplicates(['model','f_train','k_labels','seed'])
    for _,row in rows.iterrows():
        key=row['model']; config_path=f"configs/{CONFIGS[key]}.yaml"
        with open(config_path,encoding='utf-8') as f: cfg=copy.deepcopy(yaml.safe_load(f))
        cfg['training']['seed']=int(row['seed'])
        cfg['data']['f_train']=float(row['f_train'])
        cfg['data']['k_labels']=int(row['k_labels'])
        variant=f"f{cfg['data']['f_train']:g}_k{cfg['data']['k_labels']}"
        checkpoint=os.path.join('models',f"{key}_{variant}_{cfg['training']['seed']}.pt")
        if not os.path.exists(checkpoint):
            print(f'Skip missing checkpoint: {checkpoint}',flush=True); continue
        print(f"Refreshing evaluation: {key} {variant} seed={cfg['training']['seed']}",flush=True)
        evaluate_model(cfg)


if __name__=='__main__': main()
