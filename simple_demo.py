#!/usr/bin/env python3
"""
Simplified Federated Learning Demo for Presentation
Shows the core concepts without complex client-server timing issues
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from models import SimpleCNN
from dataset_loader import load_imagefolder_dataloaders, load_global_test_loader, train_one_epoch, evaluate
from utils import get_device, get_parameters_from_model, set_parameters_to_model
from datetime import datetime


def simulate_federated_round(client_models, client_loaders, global_model, device, criterion, round_num):
    """Simulate one round of federated learning"""
    print(f"\n🔄 FEDERATED ROUND {round_num}")
    print("=" * 50)
    
    # Step 1: Send global model to all clients
    global_params = get_parameters_from_model(global_model)
    client_updates = []
    client_sizes = []
    
    # Step 2: Each client trains locally
    for cid, (model, (train_loader, test_loader)) in enumerate(zip(client_models, client_loaders), 1):
        print(f"\n📱 CLIENT {cid} LOCAL TRAINING:")
        
        # Load global parameters
        set_parameters_to_model(model, global_params)
        
        # Local training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Local evaluation
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"   Local train loss: {train_loss:.4f}")
        print(f"   Local test loss: {test_loss:.4f}, acc: {test_acc:.4f}")
        
        # Collect updates
        client_updates.append(get_parameters_from_model(model))
        client_sizes.append(len(train_loader.dataset))
    
    # Step 3: Server aggregates (FedAvg)
    print(f"\n🌐 SERVER AGGREGATION:")
    total_samples = sum(client_sizes)
    aggregated_params = []
    
    for i in range(len(client_updates[0])):
        # Weighted average of parameters
        weighted_sum = np.zeros_like(client_updates[0][i])
        for client_update, client_size in zip(client_updates, client_sizes):
            weight = client_size / total_samples
            weighted_sum += weight * client_update[i]
        aggregated_params.append(weighted_sum)
    
    # Step 4: Update global model
    set_parameters_to_model(global_model, aggregated_params)
    
    return aggregated_params


def main():
    print("🌊 FEDERATED LEARNING FOR FLOOD DAMAGE DETECTION")
    print("=" * 60)
    print("Demonstrating privacy-preserving collaborative AI")
    print("✅ Each client keeps data private")
    print("✅ Only model weights are shared") 
    print("✅ Collective intelligence without data sharing")
    print()
    
    device = get_device()
    criterion = nn.CrossEntropyLoss()
    
    # Load data for each client
    print("📁 LOADING CLIENT DATA:")
    client_loaders = []
    for cid in [1, 2, 3]:
        try:
            train_loader, test_loader, num_classes = load_imagefolder_dataloaders(
                f"data/client_{cid}/train", f"data/client_{cid}/test", batch_size=16
            )
            client_loaders.append((train_loader, test_loader))
            print(f"   Client {cid}: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test images")
        except Exception as e:
            print(f"   ❌ Client {cid}: Failed to load data - {e}")
    
    if not client_loaders:
        print("❌ No client data available. Please run dataset preparation first.")
        return
    
    # Load global test data
    try:
        global_test_loader, num_classes = load_global_test_loader("data/global_test", batch_size=16)
        print(f"   Global test: {len(global_test_loader.dataset)} images")
    except Exception as e:
        print(f"   ❌ Global test data not available - {e}")
        return
    
    # Initialize models
    print(f"\n🤖 INITIALIZING MODELS:")
    global_model = SimpleCNN(num_classes=num_classes).to(device)
    client_models = [SimpleCNN(num_classes=num_classes).to(device) for _ in client_loaders]
    
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"   SimpleCNN architecture: {total_params:,} parameters")
    print(f"   {len(client_models)} client models initialized")
    
    # Initial global evaluation
    print(f"\n🎯 INITIAL GLOBAL EVALUATION:")
    initial_loss, initial_acc = evaluate(global_model, global_test_loader, device, criterion)
    print(f"   Initial global accuracy: {initial_acc:.4f}")
    
    accuracies = [initial_acc]
    
    # Update Streamlit metrics file with timestamp and training_complete flag
    def update_streamlit_metrics(accuracies, training_complete=False):
        """Update metrics.json for Streamlit dashboard with timestamp and completion flag"""
        metrics = {
            "accuracies": accuracies,
            "training_complete": training_complete,
            "last_updated": datetime.now().isoformat(),
            "rounds_expected": 3  # match number of federated rounds
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"   📊 Streamlit metrics updated: {len(accuracies)} points, training_complete={training_complete}")
    
    # Save initial accuracy
    update_streamlit_metrics(accuracies, training_complete=False)
    
    # Run federated learning rounds
    num_rounds = 3
    for round_num in range(1, num_rounds + 1):
        simulate_federated_round(client_models, client_loaders, global_model, device, criterion, round_num)
        
        # Global evaluation after round
        global_loss, global_acc = evaluate(global_model, global_test_loader, device, criterion)
        accuracies.append(global_acc)
        
        print(f"🎯 GLOBAL TEST ACCURACY AFTER ROUND {round_num}: {global_acc:.4f}")
        
        if round_num > 1:
            improvement = global_acc - accuracies[-2]
            print(f"   Improvement: {improvement:+.4f}")
        
        # Update Streamlit metrics after each round
        is_last_round = round_num == num_rounds
        update_streamlit_metrics(accuracies, training_complete=is_last_round)
        
        # Pause for live dashboard demonstration
        print(f"   ⏸️  Pause for live dashboard update (5 seconds)...")
        for i in range(5, 0, -1):
            print(f"   ⏱️  Continuing in {i}s...", end="\r")
            time.sleep(1)
        print("   ✅ Continuing to next round...")
    
    # Final results
    print(f"\n🏆 FINAL RESULTS:")
    print("=" * 40)
    for i, acc in enumerate(accuracies):
        if i == 0:
            print(f"Initial:   {acc:.4f}")
        else:
            improvement = acc - accuracies[0]
            print(f"Round {i}:   {acc:.4f} ({improvement:+.4f})")
    
    final_improvement = accuracies[-1] - accuracies[0]
    print(f"\nTotal improvement: {final_improvement:.4f}")
    print(f"Privacy preserved: ✅ No raw images shared")
    print(f"Collaboration achieved: ✅ Better accuracy through federation")
    
    # Privacy statistics
    print(f"\n🔒 PRIVACY STATISTICS:")
    model_size_kb = total_params * 4 / 1024  # 4 bytes per float32
    avg_images_per_client = sum(len(loader[0].dataset) for loader in client_loaders) / len(client_loaders)
    avg_image_size_kb = 25  # Approximate
    total_image_data_kb = avg_images_per_client * avg_image_size_kb
    
    print(f"   Model weights shared: ~{model_size_kb:.1f} KB per round")
    print(f"   Average client data: ~{total_image_data_kb:.1f} KB")
    print(f"   Privacy factor: {total_image_data_kb/model_size_kb:.1f}x smaller transmission")
    
    print(f"\n✨ FEDERATED LEARNING DEMONSTRATION COMPLETE! ✨")
    print(f"\n🌐 STREAMLIT DASHBOARD: Run `streamlit run streamlit_app.py`")
    print(f"📊 Live accuracy chart available in browser")
    print(f"📈 {len(accuracies)} data points ready for visualization")


if __name__ == "__main__":
    main()
