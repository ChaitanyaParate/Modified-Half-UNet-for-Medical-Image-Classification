import torch
from dataset import BrainTumorDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from dataset import BrainTumorDataset

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None):
    print("=> Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])

    print("Checkpoint loaded successfully!")

def get_loaders(data_dir, transform, batch_size, num_workers, pin_memory):
    full_dataset = BrainTumorDataset(data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    thresholds = [ 0.4, 0.5, 0.6]
    best_acc = 0
    best_t = 0.5

    with torch.no_grad():
        for t in thresholds:
            num_correct = 0
            num_samples = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                
                preds = (torch.sigmoid(model(x)) > t).float()
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            
            acc = float(num_correct) / num_samples * 100
            print(f"Threshold {t:.1f} → Accuracy: {acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_t = t

    
    model.train()

    return best_acc, best_t
