import time
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import shutil
import pandas as pd
import random
from PIL import Image 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import kornia.augmentation as K
import kornia.geometry.transform as T
import kornia.constants as C
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomRotation(nn.Module):
    """
    A custom wrapper for kornia's rotate function that accepts padding_mode.
    This mimics K.RandomRotation but exposes the padding_mode argument.
    """
    def __init__(self, degrees, same_on_batch=False, padding_mode='zeros', prob = 0.5):
        super().__init__()
        # Store degrees as a float tensor (will be moved to the input device in forward)
        # Use as_tensor so lists/tuples also work and we keep a numeric dtype.
        self.degrees = torch.as_tensor(degrees, dtype=torch.float32)
        self.same_on_batch = same_on_batch
        self.padding_mode = padding_mode
        # store provided probability (use the passed value)
        self.prob = prob
        
    def forward(self, x, prob=None):
        # Avoid using `self` in default args; resolve the effective prob here
        if prob is None:
            prob = self.prob
        # 1. Decide whether to apply the rotation.
        # random.random() returns a float in [0,1). We apply rotation with probability `prob`.
        if random.random() >= prob:
            return x
        
        # Get a random angle between -degrees and +degrees
        # (torch.rand * 2 - 1) creates a random number in [-1, 1]
        
        # Move stored degrees to the same device as the input to avoid device-mismatch errors
        deg = self.degrees.to(x.device)
        if self.same_on_batch:
            # Generate ONE angle for the entire batch
            angle =  random.choice( list(deg) ).clone().detach()
            # Repeat this angle for every item in the batch
            angle = angle.repeat(x.shape[0])
        else:
            # Generate a different angle for each item in the batch
            angle = (torch.rand(x.shape[0], device=x.device) * 2 - 1) * deg[0]

        # 2. Apply the functional rotation
        # T.rotate is the function that does the actual work
        output = T.rotate(
            x, 
            angle, 
            mode='bilinear', 
            padding_mode=self.padding_mode
        )

        assert output.shape == x.shape, f"Output shape {output.shape} does not match input shape {x.shape}"
        return output


def save_attention(output,image,dir,patch_size):
    attentions = output.attentions[-1] # we are only interested in the attention maps of the last layer
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    threshold = 0.6
    w_featmap = image.shape[-2] // patch_size
    h_featmap = image.shape[-1] // patch_size #model.config.patch_size

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
    attentions = attentions.detach().numpy()

    # show attentions heatmaps
    image = np.transpose(image[0].cpu(), (1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min())
    
    for j in range(nh):
        # Crear una figura
        fig, ax = plt.subplots()

        # Mostrar la imagen de fondo
        ax.imshow(image, cmap='gray')

        # Superponer la imagen con transparencia alpha
        attentions_normalized = (attentions[j] - attentions[j].min())/ (attentions[j].max()-attentions[j].min())
        # Overlay the attention image with transparency
        im = ax.imshow(attentions_normalized, alpha=0.4, cmap='coolwarm')

        # Add a colorbar for the attention map using the imshow object
        cbar = plt.colorbar(im, ax=ax)  # Use the im object as the mappable
        cbar.set_label('Attention Values')  # Label for the colorbar

        # Hide the axes
        ax.axis('off')

        # Guardar la imagen resultante
        plt.savefig(dir+"_attn-head_"+str(j)+".jpg", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    return


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Implementación de Focal Loss.

        Args:
            alpha (float, opcional): Factor de equilibrio para clases desbalanceadas. Default es 1.
            gamma (float, opcional): Foco para dar mayor peso a las muestras mal clasificadas. Default es 2.
            reduction (str, opcional): Especifica la reducción aplicada a la salida: 'none', 'mean' o 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calcula la Focal Loss.

        Args:
            inputs (Tensor): Logits de tamaño [batch_size, num_classes].
            targets (Tensor): Índices de clases verdaderas de tamaño [batch_size].

        Returns:
            Tensor: Pérdida calculada de acuerdo con el tipo de reducción especificado.
        """
        # Calculamos la entropía cruzada
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculamos las probabilidades de cada clase
        pt = torch.exp(-ce_loss)
        
        # Calculamos la Focal Loss
        focal_loss = self.alpha * abs(1 - pt) ** self.gamma * ce_loss # Aquí no debería haber un signo menos? En caso contrario la pérdida sería negativa

        # Aplicamos la reducción especificada
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def label_smoothing_loss(predictions, targets, smoothing=0.1):
    """
    Calcula la pérdida de entropía cruzada con suavizado de etiquetas.

    Args:
        predictions (Tensor): Salida del modelo antes de softmax (logits) de tamaño [batch_size, num_classes].
        targets (Tensor): Índices de clases verdaderas de tamaño [batch_size].
        smoothing (float): Factor de suavizado (entre 0 y 1). 0 significa sin suavizado.

    Returns:
        Tensor: Pérdida promedio de entropía cruzada con suavizado de etiquetas.
    """
    # Número de clases
    num_classes = predictions.size(1)
    
    # One-hot encoding de las etiquetas verdaderas
    with torch.no_grad():
        true_dist = torch.zeros_like(predictions)                           # Crear un tensor de ceros con la misma forma que predictions
        true_dist.fill_(smoothing / (num_classes - 1))                      # Llenar el tensor con el valor de suavizado dividido por el número de clases menos 1
        true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)   # Reemplazar los valores en la columna correspondiente a las etiquetas verdaderas con 1.0 - suavizado
    
    # Calcular la pérdida con entropía cruzada suave
    loss = torch.mean(torch.sum(-true_dist * F.log_softmax(predictions, dim=-1), dim=-1))
    return loss



def train_and_evaluate(
    model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, valid_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader, num_classes: int, num_epochs: int, device: torch.device, mapping: bool = False, 
    learning_rate: float = 1e-4, res_folder: str = "results_cc", hidden_size: int = 12, patch_size: int = 4, num_heads: int = 1, 
    dropout: float = 0.0225, num_transf: int = 1, mlp: int = 1, wd: float = 0.1, verbose: bool = False, patience : int = -1, scheduler_factor : float = 0.98, 
    autoencoder : bool = False, save_reconstructed_images : bool = False, augmentation_prob: float = 0.5, val_train_pond = 1) -> None:
    """Trains the given model on the given dataloaders for the given parameters"""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    model = model.to(device)
    model.trainlosslist = []
    model.vallosslist = []
    model.auclist = []
    model.acclist = []
    best_y_true, best_y_pred, best_y_pred_prob = [], [], []

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    # Definir el scheduler StepLR
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_factor)  # Reduce LR cada 5 épocas por un factor de 0.1

    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {number_of_parameters}")

    augmentation = augmentation_prob > 0

    if augmentation:
        aug_pipeline = nn.Sequential(
            K.RandomHorizontalFlip(p=augmentation_prob, same_on_batch=True),
            K.RandomVerticalFlip(p=augmentation_prob, same_on_batch=True)
        ).to(device)

    if autoencoder:
        criterion = nn.MSELoss()
        best_val_mse, best_epoch = float('inf'), 0

    else:
        if num_classes == 2:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
      
        best_val_auc, best_val_acc, best_epoch = 0.0, 0, 0
        best_tr_auc, best_tr_acc, best_epoch_tr = 0.0, 0, 0

    start_time = time.time()
    for epoch in range(num_epochs):
        step = 0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1:3}/{num_epochs}", unit="batch", bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}') as progress_bar:
            model.train()
            train_loss = 0.0
            y_trueTr, y_predTr = [], []
            
            for inputs, labels, *ind in train_dataloader:

                nonzerodim = True
                for d in inputs.shape:
                    if d == 0:
                        nonzerodim = False
                        break
                assert nonzerodim, f"Input has zero dimension in shape {inputs.shape}"

                inputs, labels = inputs.to(device), labels.to(device) 
                if not autoencoder:               
                    if num_classes == 2:
                        labels = labels.float()
                    else:
                        labels = labels.long()

                    labels = labels.squeeze(1) if labels.dim() > 1 else labels

                operation_start_time = time.time()

                optimizer.zero_grad()

                if verbose:
                    print(f" Zero grad ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                # Inputs shape is either (B, C, H, W) or (B, S, C, P, P)    S= seq_len, P=patch_size
                if augmentation:

                    D = inputs.ndim
                    if D == 5:
                        B, S, C, H, W = inputs.shape
                        inputs = inputs.view( (B*S, C, H, W) )
                    
                    inputs = aug_pipeline(inputs)

                    if D == 5:
                        inputs = inputs.view( (B, S, C, H, W) )
        
                
                outputs = model(inputs)

                outputs = outputs[0] if isinstance(outputs, tuple) or isinstance(outputs, list) else outputs  # Get only the outputs, not the attentions

                if num_classes == 2 and outputs.shape[1] == 2:
                    outputs = outputs[:, 1]

                if verbose:
                    print(f" Forward ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()
                if not autoencoder:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, inputs)

                if verbose:
                    print(f" Loss ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                loss.backward()

                if verbose:
                    print(f" Backward ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                optimizer.step()

                if verbose:
                    print(f" Optimizer step ({time.time()-operation_start_time:.2f}s)")

                step += 1
                progress_bar.update(1)

                train_loss += loss.item()
                if not autoencoder:
                    probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)

                    y_trueTr.extend(labels.tolist())
                    y_predTr.extend(probabilities.tolist())
                
            train_loss /= len(train_dataloader) 
            if not autoencoder:
                tr_auc = 100.0 * roc_auc_score(y_trueTr, y_predTr, multi_class='ovr')
                y_predTr_array = [value >= 0.5 for value in y_predTr] if num_classes == 2 else np.argmax(y_predTr, axis=1)
                tr_acc = 100.0 * accuracy_score(y_trueTr, y_predTr_array)

            ## Validation
            model.eval()
            y_trueVal, y_predVal = [], []
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels, *ind in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if num_classes == 2:
                        labels = labels.float()
                    else:
                        labels = labels.long()
                    labels = labels.squeeze(1) if labels.dim() > 1 else labels

                    outputs = model(inputs)  # Get only the outputs, not the attentions
                    outputs = outputs[0] if isinstance(outputs, tuple) or isinstance(outputs, list)  else outputs  # Get only the outputs, not the attentions

                    if num_classes == 2 and outputs.shape[1] == 2:
                        outputs = outputs[:, 1]
                    if not autoencoder:
                        val_loss += criterion(outputs, labels).item()
                    else:
                        val_loss += criterion(outputs, inputs).item()

                    
                    if not autoencoder:
                        probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)
                        # Guardar las predicciones y etiquetas verdaderas para AUC y precisión
                        y_trueVal.extend(labels.tolist())
                        y_predVal.extend(probabilities.tolist())

            val_loss /= len(valid_dataloader)
            if not autoencoder:
                val_auc = (100.0 * roc_auc_score(y_trueVal, y_predVal, multi_class='ovr'))* val_train_pond + tr_auc * (1-val_train_pond)

            # Actualiza el learning rate con StepLR
            scheduler.step()  # Reduce LR a intervalos regulares

            best_threshold = 0.1
            best_val_acc = 0

            if not autoencoder:

                thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] if num_classes == 2 else [0.5]

                for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    y_predVal_array = [value >= threshold for value in y_predVal] if num_classes == 2 else np.argmax(y_predVal, axis=1)
                    val_acc = 100.0 * accuracy_score(y_trueVal, y_predVal_array)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_threshold = threshold

                extra_string_for_binary = f", with threshold={best_threshold:.2f}" if num_classes == 2 else ""
                progress_bar.set_postfix_str(f"Val Loss={val_loss:.3f}, Valid AUC={val_auc:.2f}%, Train AUC={tr_auc:.2f} ||| Valid ACC={best_val_acc:.2f}%, Train ACC={tr_acc:.2f}%")

            else:
                progress_bar.set_postfix_str(f"Val Loss={val_loss:.3f} ||| Train Loss={train_loss:.3f}")

            model.trainlosslist.append(train_loss) 
            model.vallosslist.append(val_loss) 

            if not autoencoder:
                model.trauclist.append(tr_auc)
                model.tracclist.append(tr_acc)
                model.auclist.append(val_auc) 
                model.acclist.append(val_acc) 

            if not autoencoder:
            
                if val_auc > best_val_auc:
                    epochs_no_improve = 0
                    best_val_acc = val_acc
                    best_val_auc = val_auc
                    best_epoch = epoch + 1
                    best_y_pred = y_predVal_array
                    best_y_pred_prob = y_predVal
                    best_y_true = y_trueVal
                    state_dict = model.state_dict()  # get the original state_dict
                    torch.save(state_dict, f'{res_folder}/model_weights_val_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.pth')

                else:
                    if patience > 0:
                        epochs_no_improve += 1
                        print(f"\nSin mejora durante {epochs_no_improve} épocas")

                        if epochs_no_improve >= patience:
                            print("\nEarly Stopping activado. Deteniendo entrenamiento.")
                            break

                if tr_auc > best_tr_auc:
                    best_tr_auc = tr_auc
                    best_tr_acc = tr_acc
                    best_epoch_tr = epoch + 1
                    state_dict = model.state_dict()  # get the original state_dict
                    torch.save(state_dict, f'{res_folder}/model_weights_tr_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.pth')

            else:
                if val_loss < best_val_mse:
                    best_val_mse = val_loss
                    best_epoch = epoch + 1
                    state_dict = model.state_dict()  # get the original state_dict
                    torch.save(state_dict, f'{res_folder}/model_weights_val_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.pth')
    
    # Save predictions to CSV
    if not autoencoder:
        df = pd.DataFrame({
            'trainlosslist': model.trainlosslist,
            'trauclist': model.trauclist,
            'tracclist': model.tracclist,
            'vallosslist': model.vallosslist,
            'auclist': model.auclist,
            'acclist': model.acclist
        })

    else:
        df = pd.DataFrame({
            'trainlosslist': model.trainlosslist,
            'vallosslist': model.vallosslist
        })

    df.to_csv(f'{res_folder}/Training_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.csv', index=False)

    print(f"TOTAL TIME = {time.time()-start_time:.2f}s")
    if not autoencoder:
        print(f"BEST AUC TRAIN = {best_tr_auc:.2f}% AT EPOCH {best_epoch_tr}")
        print(f"BEST AUC VAL = {best_val_auc:.2f}% AT EPOCH {best_epoch}")

    else:
        print(f"BEST MSE TRAIN = {train_loss:.4f} AT EPOCH {num_epochs}")
        print(f"BEST MSE VAL = {best_val_mse:.4f} AT EPOCH {best_epoch}")
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    model.load_state_dict(torch.load(f'{res_folder}/model_weights_val_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.pth', weights_only=True))

    # Evaluación en el conjunto de prueba
    model.eval()
    y_true, y_pred, images = [], [], []
    with torch.no_grad():
        for inputs, labels, ind in test_dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float() if num_classes == 2 else labels.long()
            labels = labels.squeeze(1) if labels.dim() > 1 else labels

            outputs = model(inputs)  # Get only the outputs, not the attentions
            outputs = outputs[0] if isinstance(outputs, tuple) or isinstance(outputs, list) else outputs  # Get only the outputs, not the attentions

            if save_reconstructed_images and autoencoder and epoch == num_epochs - 1:

                os.makedirs(f'{res_folder}/autoencoder_images', exist_ok=True)

                # Guardar las imágenes reconstruidas
                reconstructed_images = outputs.cpu()
                original_images = inputs.cpu()
                batch_size = reconstructed_images.size(0)
                for i in range(batch_size):
                    recon_img = reconstructed_images[i]
                    orig_img = original_images[i]

                    # Normalizar las imágenes al rango [0, 1]
                    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
                    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())

                    # Convertir a formato PIL y guardar
                    recon_pil = Image.fromarray((recon_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    orig_pil = Image.fromarray((orig_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

                    recon_pil.save(f'{res_folder}/autoencoder_images/reconstructed_image_{i}.png')
                    orig_pil.save(f'{res_folder}/autoencoder_images/original_image_{i}.png')

            if num_classes == 2 and outputs.shape[1] == 2:
                outputs = outputs[:, 1]
    
            if not autoencoder:
                probabilities = torch.sigmoid(outputs) if num_classes == 2 else torch.softmax(outputs, dim=1)
        
                y_true.extend(labels.tolist())
                y_pred.extend(probabilities.tolist())

            images.extend(ind.tolist() if isinstance(ind, torch.Tensor) else ind)

    
    if not autoencoder:
        # Cálculo de AUC y precisión
        test_auc = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')
        y_pred_array = [value >= best_threshold for value in y_pred] if num_classes == 2 else np.argmax(y_pred, axis=1)
        test_acc = 100.0 * accuracy_score(y_true, y_pred_array)
    
        # Guardar resultados
        df = pd.DataFrame({
            'Images': images,
            'True Labels': y_true,
            'Predictions': y_pred_array,
            'Predicted Probabilities': y_pred
        })

        print(f"TEST AUC: {test_auc:.2f}%, TEST ACC: {test_acc:.2f}%")

    else:
        mse = np.mean((np.array(inputs.cpu()) - np.array(outputs.cpu()))**2, )
        df = pd.DataFrame({
            'Images': images,
            'mse': mse
        })

        print(f"TEST MSE: {mse:.4f}")

    df.to_csv(f'{res_folder}/predictions_test_{learning_rate}_{hidden_size}_{dropout}_{num_heads}_{num_transf}_{mlp}_{patch_size}_{wd}.csv', index=False)
    
        

    ## Saving attention maps   
    if mapping:
        # Create directories for misclassified images
        train_misclassified_dir = f'{res_folder}/train_misclassified'
        test_misclassified_dir = f'{res_folder}/test_misclassified'
        
        try:
            os.makedirs(f'{res_folder}/train')
            os.makedirs(f'{res_folder}/test')
            for i in range(num_classes):
                os.makedirs(f'{res_folder}/train/class{i}')
                os.makedirs(f'{res_folder}/test/class{i}')

            os.makedirs(train_misclassified_dir)
            os.makedirs(test_misclassified_dir)
            for i in range(num_classes):
                os.makedirs(f'{train_misclassified_dir}/class{i}')
                os.makedirs(f'{test_misclassified_dir}/class{i}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        valid_dataloader_single = torch.utils.data.DataLoader(valid_dataloader.dataset, batch_size=1)
        for images, labels, names in valid_dataloader_single:
            images, labels = images.to(device), labels.to(device)
            dir = f'{res_folder}/test/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            dir_misclassified_test = f'{test_misclassified_dir}/class'+str(int(labels))+'/image'+str(int(names)).zfill(4)
            with torch.no_grad():
                pred = model(images)[0]  # Get both outputs and attentions
                if res_folder == "results_qc" or res_folder == "results_qq":
                    images = images.permute(0, 3, 1, 2)
                    images = images[:, 0:1, :, :]
                save_attention(pred, images, dir,patch_size)  # Pass attentions instead of pred
                # Find misclassified samples
                if (num_classes == 2 and pred.shape[1]==2):
                    pred2 = pred[:,1]

                else:
                    pred2 = pred
                probs = torch.sigmoid(pred2) if num_classes == 2 else torch.softmax(pred2, dim=1)
                pred2 = [value >= 0.5 for value in probs] if num_classes == 2 else torch.argmax(probs, dim=1)
                misclassified = pred2[0] != labels.squeeze().bool()
                if misclassified:
                    save_attention(pred, images, dir_misclassified_test,patch_size)  # Use attention for misclassified

    if not autoencoder:
        return test_auc, test_acc, best_val_auc, best_val_acc, best_tr_auc, best_tr_acc, number_of_parameters
    else:
        return mse, best_val_mse, number_of_parameters
