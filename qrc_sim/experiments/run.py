
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from qrc_sim.encoders import AngleEncoder, ReuploadingEncoder
from qrc_sim.reservoirs import RandomReservoir, RandomCRotReservoir
from qrc_sim.readout import ReadoutModel
from qrc_sim.simulator import QRCSimulator
from qrc_sim.tasks.memory import MemoryTask
from qrc_sim.tasks.parity import ParityTask
from qrc_sim.tasks.narma import NARMADataset

def main():
    parser = argparse.ArgumentParser(description="QRC-Lab Experiment Runner")
    parser.add_argument("--task", type=str, required=True, choices=['memory', 'parity', 'narma10'], help="Task to run")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=3, help="Reservoir depth")
    parser.add_argument("--encoder", type=str, default="angle", choices=['angle', 'reuploading'], help="Encoder type")
    parser.add_argument("--scale", type=float, default=1.0, help="Encoder scale factor")
    parser.add_argument("--res_type", type=str, default="simple", choices=['simple', 'crot'], help="Reservoir type")
    parser.add_argument("--enc_layers", type=int, default=1, help="Encoder layers (if reuploading)")
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots (if backend=shots)")
    parser.add_argument("--backend", type=str, default="ideal", choices=['ideal', 'shots'], help="Simulation backend")
    parser.add_argument("--mode", type=str, default="carry", choices=['carry', 'reset_each_step', 'reupload_k'], help="State update mode")
    parser.add_argument("--reupload_k", type=int, default=3, help="K for reupload mode")
    parser.add_argument("--noise_prob", type=float, default=0.0, help="Depolarizing noise probability")
    parser.add_argument("--bases", type=str, default="Z", help="Measurement bases (comma-separated, e.g. 'Z,X,Y')")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Output directory")
    
    args = parser.parse_args()
    
    # 1. Prepare Task
    print(f"Loading task: {args.task}")
    if args.task == 'memory':
        task = MemoryTask(length=1000, delay=1)
    elif args.task == 'parity':
        task = ParityTask(length=1000, delay=1)
    elif args.task == 'narma10':
        task = NARMADataset(length=2000, order=10)
        
    (X_train, y_train), (X_test, y_test) = task.generate()
    print(f"Data shapes: Train X{X_train.shape} y{y_train.shape}, Test X{X_test.shape} y{y_test.shape}")
    
    # 2. Setup QRC Components
    if args.encoder == 'angle':
        encoder = AngleEncoder(args.n_qubits, scale=args.scale)
    else:
        encoder = ReuploadingEncoder(args.n_qubits, layers=args.enc_layers, scale=args.scale)
        
    if args.res_type == 'simple':
        reservoir = RandomReservoir(args.n_qubits, args.depth, entanglement='ring')
    else:
        reservoir = RandomCRotReservoir(args.n_qubits, args.depth)
    
    # Observables: Multi-basis support
    base_list = [b.strip().upper() for b in args.bases.split(',')]
    obs_list = []
    for b in base_list:
        obs_list += [(b, i) for i in range(args.n_qubits)]
    
    # Add some ZZ neighbors in Z basis by default
    obs_list += [('ZZ', i, (i+1)%args.n_qubits) for i in range(args.n_qubits)]
    
    # Noise Model
    noise_model = None
    if args.noise_prob > 0:
        noise_model = QRCSimulator.create_depolarizing_noise(args.noise_prob)
    
    # Simulator
    sim = QRCSimulator(encoder, reservoir, obs_list, 
                       backend_config=args.backend,
                       shots=args.shots,
                       state_update_mode=args.mode,
                       reupload_k=args.reupload_k,
                       noise_model=noise_model) 

    
    print("Running Reservoir on Train Set...")
    train_features = sim.run_sequence(X_train)
    
    print("Running Reservoir on Test Set...")
    test_features = sim.run_sequence(X_test)
    
    # 3. Readout Training
    print("Training Readout...")
    readout = ReadoutModel(model_type='ridge', alpha=1e-6)
    readout.fit(train_features, y_train)
    
    train_score = readout.score(train_features, y_train)
    test_score = readout.score(test_features, y_test)
    
    print(f"Train R2/Acc: {train_score:.4f}")
    print(f"Test R2/Acc: {test_score:.4f}")
    
    if args.task == 'parity':
        # Accuracy check roughly
        preds = readout.predict(test_features)
        preds_bin = (preds > 0.5).astype(int)
        acc = np.mean(preds_bin == y_test)
        print(f"Binary Accuracy: {acc:.4f}")
    
    # 4. Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], label='Target', alpha=0.7)
    plt.plot(readout.predict(test_features)[:100], label='Prediction', alpha=0.7)
    plt.title(f"Task: {args.task} | Backend: {args.backend} | Test Score: {test_score:.3f}")
    plt.legend()
    plt.savefig(f"{args.output_dir}/results_{args.task}_{timestamp}.png")
    
    # Save Config + Metrics
    results = {
        "config": vars(args),
        "metrics": {
            "train_score": train_score,
            "test_score": test_score
        }
    }
    
    with open(f"{args.output_dir}/results_{args.task}_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
