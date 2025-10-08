import torch


class MLP(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Sequential(torch.nn.LazyLinear(hidden_size), torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU())
        self.fc3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def train_mlp_classifier(MLP_model, train_dl, val_dl, test_dl, epochs, lr, weight_decay, device):
    MLP_model.to(device)
    optimizer = torch.optim.Adam(MLP_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_auc = []
    val_auc = []
    test_auc = []
    train_acc = []
    val_acc = []
    test_acc = []

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
            for inputs, labels in dl:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = MLP_model(inputs)
                probs = torch.sigmoid(outputs)
                y_trues.append(labels.cpu())
                y_probs.append(probs.cpu())

        if len(y_trues) == 0:
            return float('nan'), float('nan')

        y_true = torch.cat(y_trues).numpy().ravel()
        y_prob = torch.cat(y_probs).numpy().ravel()
        preds = (y_prob >= 0.5).astype(int)

        # accuracy (safe to compute)
        try:
            acc = float((preds == y_true).astype(int).mean())
        except Exception:
            acc = float('nan')

        # AUC (may be undefined if only one class present)
        auc = float('nan')
        if _have_sklearn:
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc = float('nan')

        return auc, acc

    for epoch in range(epochs):
        MLP_model.train()
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = MLP_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # compute validation loss, AUC, and accuracy
        MLP_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
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
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = MLP_model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_dl)
    print(f'Test Loss: {test_loss:.4f}')

    # compute test metrics
    test_auc_val, test_acc_val = _compute_metrics(test_dl)
    test_auc.append(test_auc_val)
    test_acc.append(test_acc_val)

    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc_val:.4f}, Test Acc: {test_acc_val:.4f}')

    # return model, final test loss, and metrics history
    metrics = {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return MLP_model, test_auc, test_acc, val_auc, val_acc, train_auc