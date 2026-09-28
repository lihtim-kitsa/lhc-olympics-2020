import os
import subprocess
import glob

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("Starting full reproduction pipeline...")
    
    # 1. Download data
    run_command("python scripts/download_data.py")
    
    # 2. Build features (this might take a while on full data)
    run_command("python scripts/build_features.py")
    
    # 3. Create splits
    run_command("python src/data/make_dataset.py")
    
    # 4. Train and evaluate models
    configs = glob.glob("configs/m*.yaml")
    
    for config_file in sorted(configs):
        print(f"\n--- Processing {config_file} ---")
        run_command(f"python scripts/train.py --config {config_file}")
        run_command(f"python scripts/evaluate.py --config {config_file}")
        
    print("\nReproduction complete. Results saved in reports/tables/results.csv and MLflow.")

if __name__ == '__main__':
    main()
