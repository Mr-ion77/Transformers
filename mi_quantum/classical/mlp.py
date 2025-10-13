import torch
import sklearn
import numpy as np
import math

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, n_layers = 3, n_classes = 1, dropout = 0.0, red_factor = 1.4):
        super().__init__()

        assert n_layers >= 2, "MLP must have at least 2 layers"

        self.first_layer = torch.nn.Linear(input_dim, input_dim)
        self.first_activation = torch.nn.ReLU()
        self.first_dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        # create per-layer dropout instances (if dropout>0) so each layer has its own module
        # compute progressively reduced dimensions for each layer
        # guard against degenerate red_factor values
        if red_factor <= 1.0:
            # enforce a small reduction factor if user mistakenly provides <=1
            red_factor = 1.01

        dimensions = []
        for i in range(n_layers):
            dim = int(math.floor(input_dim / (red_factor ** i)))
            dim = max(1, dim)  # ensure at least 1 unit per layer
            dimensions.append(dim)

        print("MLP layer dimensions:", dimensions)
        layers = []
        for i in range(n_layers - 1):
            layers.append(torch.nn.Sequential(
                torch.nn.Linear(dimensions[i], dimensions[i+1]),
                torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity(),
                torch.nn.ReLU()
            ))
        self.fc = torch.nn.ModuleList(layers)
        self.fc_last = torch.nn.Linear( dimensions[-1] , n_classes)
        
 
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input except batch dim
        x = self.first_layer(x)
        x = self.first_activation(x)
        x = self.first_dropout(x)
        # apply each middle layer in sequence
        for layer in self.fc:
            x = layer(x)
        x = self.fc_last(x)
        return x

def train_mlp_classifier(MLP_model, train_dl, val_dl, test_dl, epochs, lr, weight_decay, device):
    MLP_model.to(device)
    optimizer = torch.optim.Adam(MLP_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()    # labels should be long integers, shape (batch,)

    train_auc = []
    val_auc = []
    test_auc = []
    train_acc = []
    val_acc = []
    test_acc = []

    n_params = sum(p.numel() for p in MLP_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in MLP: {n_params}")

    # helper: safe import of sklearn metrics (fall back if not installed)
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
        _have_sklearn = True
    except Exception:
        _have_sklearn = False

    def _compute_metrics(dl):
        # returns (auc, acc)
        y_trues = []
        y_probs = []
        MLP_model.eval()
        with torch.no_grad():
            for inputs, labels, index in dl:
                inputs = inputs.to(device)
                labels = labels.to(device).long().squeeze(1) if labels.dim() > 1 else labels.to(device).long()   # squeeze removes extra dim if present
                outputs = MLP_model(inputs)
                # for multiclass outputs (N, C) use softmax to get class probabilities
                if outputs.dim() == 2 and outputs.shape[1] > 1:
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                else:
                    # fallback to sigmoid for single-logit outputs
                    probs = torch.sigmoid(outputs)
                y_trues.append(labels.cpu())
                y_probs.append(probs.cpu())

        if len(y_trues) == 0:
            return float('nan'), float('nan')

        y_true = torch.cat(y_trues).numpy().ravel()
        y_prob = torch.cat(y_probs).numpy()  # keep shape (N, C) for multiclass
        # predicted class = argmax over class axis for multiclass
        if y_prob.ndim == 1:
            # single-probability case (binary/logit reduced to 1 dim)
            preds = (y_prob >= 0.5).astype(int)
        else:
            preds = y_prob.argmax(axis=-1)

        # accuracy (safe to compute)
        try:
            acc = float((preds == y_true).astype(int).mean())
        except Exception:
            acc = float('nan')

        # AUC (may be undefined if only one class present)
        auc = float('nan')
        if _have_sklearn:
            try:
                # if multiclass, binarize labels and use multi_class option
                from sklearn.preprocessing import label_binarize
                if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                    auc = float(roc_auc_score(y_true, y_prob))
                else:
                    y_true_bin = label_binarize(y_true, classes=range(y_prob.shape[1]))
                    auc = float(roc_auc_score(y_true_bin, y_prob, multi_class='ovr'))
            except Exception:
                auc = float('nan')

        return auc, acc

    for epoch in range(epochs):
        MLP_model.train()
        for inputs, labels, index in train_dl:
            inputs, labels = inputs.to(device), labels.to(device).long().squeeze(1) if labels.dim() > 1 else labels.to(device).long()   # squeeze removes extra dim if present
            outputs = MLP_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # compute validation loss, AUC, and accuracy
        MLP_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, index in val_dl:
                inputs, labels = inputs.to(device), labels.to(device).long().squeeze(1) if labels.dim() > 1 else labels.to(device).long()
                outputs = MLP_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_dl)

        # compute metrics for train and val sets
        # Training metrics: compute on the training dataloader (could be expensive for large datasets)
        train_auc_epoch, train_acc_epoch = _compute_metrics(train_dl)
        val_auc_epoch, val_acc_epoch = _compute_metrics(val_dl)

        train_auc.append(train_auc_epoch)
        train_acc.append(train_acc_epoch)
        val_auc.append(val_auc_epoch)
        val_acc.append(val_acc_epoch)

        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc_epoch:.4f}, Train Acc: {train_acc_epoch:.4f}, "
              f"Val AUC: {val_auc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")

    test_loss = 0.0
    MLP_model.eval()
    with torch.no_grad():
        for inputs, labels, index in test_dl:
            inputs, labels = inputs.to(device), labels.to(device).long().squeeze(1) if labels.dim() > 1 else labels.to(device).long()
            outputs =  MLP_model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_dl)
    print(f'Test Loss: {test_loss:.4f}')

    # compute test metrics
    test_auc, test_acc = _compute_metrics(test_dl)

    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')

    # return model, final test loss, and metrics history
    metrics = {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return test_auc, test_acc, val_auc[-1], val_acc[-1], train_auc[-1], n_params