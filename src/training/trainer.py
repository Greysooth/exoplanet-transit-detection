import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def get_class_weights(y, device):
    counts = np.bincount(y)
    inv_weights = 1.0 / counts
    norm_weights = inv_weights / inv_weights.sum()
    return torch.tensor(norm_weights, dtype=torch.float32).to(device)


def train_single_stage(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=1e-4,
    patience=7,
    is_multimodal=False,
    save_path=None
):
    """
    Standard single-stage end-to-end training loop with class weighting,
    gradient clipping, and learning rate scheduling on plateau.
    """
    model.to(device)
    y_train_all = []
    for batch in train_loader:
        y_train_all.extend(batch[-1].numpy())
    weights = get_class_weights(np.array(y_train_all), device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if is_multimodal:
                bx, bmeta, by = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                logits = model(bx, bmeta)
            else:
                bx, by = batch[0].to(device), batch[1].to(device)
                logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                if is_multimodal:
                    bx, bmeta, by = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    logits = model(bx, bmeta)
                else:
                    bx, by = batch[0].to(device), batch[1].to(device)
                    logits = model(bx)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        current_val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        scheduler.step(current_val_f1)

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_weights, save_path)

    return model, best_val_f1


def train_two_stage(
    model,
    train_loader,
    val_loader,
    device,
    epochs_s1=30,
    epochs_s2a=10,
    epochs_s2b=30,
    lr_s1=1e-4,
    lr_s2a=5e-4,
    lr_s2b=5e-5,
    save_path=None
):
    """
    Two-Stage Disentangled Representation Training:
    - Stage 1: Pre-train CNN on light curves only (mode="stage1")
    - Stage 2A: Freeze CNN, train metadata projection and fusion head
    - Stage 2B: End-to-end joint fine-tuning with conservative learning rate
    """
    model.to(device)
    y_train_all = []
    for batch in train_loader:
        y_train_all.extend(batch[2].numpy())
    weights = get_class_weights(np.array(y_train_all), device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- STAGE 1: Spatial representation pre-training ---
    opt_s1 = torch.optim.Adam(model.parameters(), lr=lr_s1, weight_decay=1e-4)
    for epoch in range(epochs_s1):
        model.train()
        for bx, bmeta, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt_s1.zero_grad()
            logits = model(bx, mode="stage1")
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_s1.step()

    # --- STAGE 2A: Freeze CNN, align metadata branch ---
    for p in model.cnn.parameters():
        p.requires_grad = False

    meta_params = [p for p in model.parameters() if p.requires_grad]
    opt_s2a = torch.optim.Adam(meta_params, lr=lr_s2a, weight_decay=1e-4)
    for epoch in range(epochs_s2a):
        model.train()
        for bx, bmeta, by in train_loader:
            bx, bmeta, by = bx.to(device), bmeta.to(device), by.to(device)
            opt_s2a.zero_grad()
            logits = model(bx, bmeta, mode="joint")
            loss = criterion(logits, by)
            loss.backward()
            opt_s2a.step()

    # --- STAGE 2B: Global joint fine-tuning ---
    for p in model.cnn.parameters():
        p.requires_grad = True

    opt_s2b = torch.optim.Adam(model.parameters(), lr=lr_s2b, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_s2b, mode="max", factor=0.5, patience=4)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs_s2b):
        model.train()
        for bx, bmeta, by in train_loader:
            bx, bmeta, by = bx.to(device), bmeta.to(device), by.to(device)
            opt_s2b.zero_grad()
            logits = model(bx, bmeta, mode="joint")
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_s2b.step()

        # Validation checkpointing
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for bx, bmeta, by in val_loader:
                bx, bmeta = bx.to(device), bmeta.to(device)
                logits = model(bx, bmeta, mode="joint")
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(by.numpy())

        current_f1 = f1_score(val_targets, val_preds, zero_division=0)
        scheduler.step(current_f1)

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_weights, save_path)

    return model, best_val_f1
