import os
import h5py
import numpy as np

def create_dummy_data():
    os.makedirs('data/raw', exist_ok=True)
    
    # 2-prong dataset
    filename = 'data/raw/events_anomalydetection_v2.h5'
    n_events = 10000
    n_features = 2101
    
    if not os.path.exists(filename):
        print(f"Creating dummy data at {filename}...")
        with h5py.File(filename, 'w') as f:
            # Generate random kinematics
            # For 700 particles: pt, eta, phi
            data = np.zeros((n_events, n_features), dtype=np.float32)
            
            # Simple simulation: just 2 jets worth of particles
            for i in range(n_events):
                # jet 1
                n1 = np.random.randint(10, 30)
                pt1 = np.random.uniform(50, 500, n1)
                eta1 = np.random.normal(0, 0.1, n1)
                phi1 = np.random.normal(0, 0.1, n1)
                
                # jet 2
                n2 = np.random.randint(10, 30)
                pt2 = np.random.uniform(50, 500, n2)
                eta2 = np.random.normal(0, 0.1, n2)
                phi2 = np.random.normal(np.pi, 0.1, n2)
                
                parts = []
                for pt, eta, phi in zip(np.concatenate([pt1, pt2]), 
                                        np.concatenate([eta1, eta2]), 
                                        np.concatenate([phi1, phi2])):
                    parts.extend([pt, eta, phi])
                
                data[i, :len(parts)] = parts
                
                # Assign label: 5% signal
                if np.random.rand() < 0.05:
                    data[i, -1] = 1.0 # signal
                    
            f.create_dataset('table', data=data)
            
    # 3-prong dataset
    filename3 = 'data/raw/events_anomalydetection_Z_XY_qqq.h5'
    if not os.path.exists(filename3):
        print(f"Creating dummy data at {filename3}...")
        with h5py.File(filename3, 'w') as f:
            data = np.zeros((2000, n_features), dtype=np.float32)
            for i in range(2000):
                data[i, 0] = np.random.uniform(50, 500) # pt
                data[i, 1] = 0.0 # eta
                data[i, 2] = 0.0 # phi
                
                data[i, 3] = np.random.uniform(50, 500)
                data[i, 4] = 0.1
                data[i, 5] = np.pi
                data[i, -1] = 1.0 # all signal
                
            f.create_dataset('table', data=data)

if __name__ == '__main__':
    create_dummy_data()
