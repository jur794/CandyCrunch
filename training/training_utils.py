import copy
import json
import numpy as np
import time
import torch
import torch.nn as nn
import wandb
import logging
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.exceptions import UndefinedMetricWarning
from candycrunch.analysis import glycan_to_graph_monos, mono_graph_to_nx, enumerate_k_graphs, mono_frag_to_string
from glycowork.ml.model_training import EarlyStopping, disable_running_stats, enable_running_stats, training_setup, \
    Poly1CrossEntropyLoss
from glycowork.ml.models import init_weights

from torchmetrics.functional import accuracy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category = UndefinedMetricWarning)
warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def calculate_class_distribution(dataloader, num_classes):
    """Calculate class distribution in a dataloader"""
    class_counts = np.zeros(num_classes)
    total_samples = 0

    for data in dataloader:
        y = data[-1].squeeze()
        for label in y.cpu().numpy():
            if label < num_classes:
                class_counts[int(label)] += 1
                total_samples += 1

    return class_counts, total_samples


def prepare_batch(data, model_type):

    if model_type == "CNN":
        (
            mz_list,
            peak_list,
            mz_remainder,
            precursor,
            glycan_type,
            rt,
            mode_in,
            lc,
            modification,
            trap,
            y,
        ) = data

        mz_features = torch.stack([mz_list, mz_remainder], dim=1).to(device)

        inputs = [
            mz_features,
            precursor.to(device),
            glycan_type.to(device),
            rt.to(device),
            mode_in.to(device),
            lc.to(device),
            modification.to(device),
            trap.to(device),
        ]

        y = y.squeeze().to(device)

    elif model_type in {"Transformer", "CNN_Decoder"}:
        (
            mz_list,
            peak_list,
            mz_remainder,
            precursor,
            glycan_type,
            rt,
            mode_in,
            lc,
            modification,
            trap,
            y_class,
            decoder_input_ids,
            tgt_key_padding_mask,
            target_labels,
        ) = data

        spectrum = (torch.stack([mz_list, mz_remainder], dim=1)
                    if model_type == "CNN_Decoder" else peak_list)
        inputs = [
            spectrum.to(device),
            precursor.to(device),
            glycan_type.to(device),
            rt.to(device),
            mode_in.to(device),
            lc.to(device),
            modification.to(device),
            trap.to(device),
            decoder_input_ids.to(device),
            tgt_key_padding_mask.to(device),
        ]

        y = (target_labels.to(device), y_class.squeeze().to(device))

    else:
        raise ValueError(
            f"Unknown model_type={model_type!r}. Expected 'CNN', 'Transformer', or 'CNN_Decoder'."
        )

    return inputs, y


def calculate_metrics_sklearn(pred, y, num_classes, batch_size_small=True, present_labels_only=False):
    pred_classes = torch.argmax(pred, dim=1)

    pred_classes_np = pred_classes.cpu().numpy()
    y_np = y.cpu().numpy()

    if present_labels_only:
        labels = np.unique(y_np)
    else:
        labels = range(num_classes)

    try:
        if batch_size_small or len(np.unique(y_np)) < 2:
            f1_micro = f1_score(y_np, pred_classes_np, average='micro', zero_division=0)
            f1_macro = f1_micro
            f1_weighted = f1_micro
        else:
            f1_macro = f1_score(
                y_np, pred_classes_np,
                average='macro',
                zero_division=0,
                labels=labels
            )
            f1_weighted = f1_score(
                y_np, pred_classes_np,
                average='weighted',
                zero_division=0,
                labels=labels
            )
            f1_micro = f1_score(y_np, pred_classes_np, average='micro', zero_division=0)
    except Exception as e:
        f1_macro = f1_weighted = f1_micro = 0.0

    # Calculate MCC (returns 0 if only one class present)
    try:
        if len(np.unique(y_np)) > 1:
            mcc = matthews_corrcoef(y_np, pred_classes_np)
        else:
            mcc = 0.0
    except Exception as e:
        mcc = 0.0

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'mcc': mcc
    }


class custom_loss(torch.nn.Module):
    def __init__(self, primary_loss, dist_sim, dist_comp, logit_norm = False, t = 1.0):
        super(custom_loss, self).__init__()
        self.primary_loss = primary_loss
        self.dist_sim = dist_sim
        self.dist_comp = dist_comp
        self.logit_norm = logit_norm
        self.t = t

    def forward(self, output, target):
        if self.logit_norm:
            norms = torch.norm(output, p = 2, dim = -1, keepdim = True) + 1e-7
            output = torch.div(output, norms) / self.t
        loss2 = self.primary_loss(output, target)
        output = torch.nn.functional.softmax(output, dim = 1)
        target_sim = self.dist_sim[target]
        loss_sim = output * target_sim
        target_comp = self.dist_comp[target]
        loss_comp = output * target_comp
        loss = loss_comp.mean() + loss_sim.mean() + loss2
        return loss


def train_model(model, dataloaders, criterion, optimizer,
                scheduler, glycans, num_epochs = None, patience = None, log_to_wandb = True, num_classes = None,
                model_type = None, setting_name = None, moe_aux_loss_weight=None,checkpoint_metadata=None, log_prefix = ""):
    """trains a deep learning model on predicting glycan properties

    Arguments:
    :-
    model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
    dataloaders (PyTorch object): dictionary of dataloader objects with keys 'train' and 'val'
    criterion (PyTorch object): PyTorch loss function
    optimizer (PyTorch object): PyTorch optimizer
    scheduler (PyTorch object): PyTorch learning rate decay
    num_epochs (int): number of epochs for training; default:25
    patience (int): number of epochs without improvement until early stop; default:50
    log_to_wandb (bool): whether to log metrics to wandb; default:True
    num_classes (int): number of classes; default:None (uses len(glycans))

    Returns:
    :-
    Returns the best model seen during training
    """
    since = time.time()
    _wlog = lambda metrics: wandb.log({log_prefix + k: v for k, v in metrics.items()})
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    best_acc = 0.0
    val_losses = []
    val_acc = []
    train_losses = []
    train_acc = []

    metric_names = [
        "loss",
        "accuracy",
        "mcc",
        "f1_macro",
        "f1_weighted",
        "top5_accuracy",
        "top10_accuracy",
    ]

    metrics_dict = {
        "train": {name: [] for name in metric_names},
        "val": {name: [] for name in metric_names},
        "time_seconds": [],
    }

    start = time.time_ns()

    if num_classes is None:
        num_classes = len(glycans)

    # Check class distribution in validation set
    val_class_counts, val_total = calculate_class_distribution(dataloaders['val'], num_classes)
    classes_in_val = np.sum(val_class_counts > 0)

    print(f"Validation set: {classes_in_val}/{num_classes} classes present ({val_total} total samples)")
    print(
        f"Training set: {np.sum(calculate_class_distribution(dataloaders['train'], num_classes)[0] > 0)}/{num_classes} classes present")

    # Log class distribution summary to wandb
    if log_to_wandb:
        train_class_counts, train_total = calculate_class_distribution(dataloaders['train'], num_classes)

        _wlog({
            "dataset_stats/train_samples": train_total,
            "dataset_stats/val_samples": val_total,
            "dataset_stats/train_classes_present": np.sum(train_class_counts > 0),
            "dataset_stats/val_classes_present": classes_in_val,
            "dataset_stats/total_classes": num_classes,
        })

        # Log top 20 most frequent classes in training
        top_classes_idx = np.argsort(train_class_counts)[-20:][::-1]
        top_classes_data = []
        for idx in top_classes_idx:
            if train_class_counts[idx] > 0:
                class_name = glycans[idx] if idx < len(glycans) else f'Class_{idx}'
                class_name = class_name[:40]  # Truncate
                top_classes_data.append([class_name, int(train_class_counts[idx])])

        if top_classes_data:
            class_table = wandb.Table(data = top_classes_data, columns = ["Class", "Train Count"])
            _wlog({"top_20_classes": class_table})

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = []
            running_acc = []
            running_mcc = []
            running_f1_macro = []
            running_f1_weighted = []
            running_topk = []
            running_topk10 = []

            # For accumulating predictions for end-of-epoch metrics
            all_preds_epoch = []
            all_labels_epoch = []

            for data in dataloaders[phase]:
                inputs, y = prepare_batch(data, model_type)
                optimizer.zero_grad(set_to_none = True)

                with torch.set_grad_enabled(phase == 'train'):
                    # first forward pass
                    enable_running_stats(model)

                    if moe_aux_loss_weight == None:
                        pred = model(*inputs)
                        loss = criterion(pred, y)
                        if phase == 'train':
                            loss.backward()
                            optimizer.first_step(zero_grad = True)
                            # second forward pass
                            disable_running_stats(model)
                            criterion(model(*inputs),y).backward()
                            optimizer.second_step(zero_grad = True)

                    elif moe_aux_loss_weight != None:
                        pred = model(*inputs)
                        loss = criterion(pred, y)

                        model_for_aux = model.module if hasattr(model, "module") else model

                        if phase == "train" and hasattr(model_for_aux, "get_aux_loss"):
                            aux_loss = model_for_aux.get_aux_loss()
                            if aux_loss is not None:
                                loss = loss + moe_aux_loss_weight * aux_loss

                        if phase == 'train':
                            loss.backward()
                            optimizer.first_step(zero_grad = True)

                            # second forward pass for SAM
                            disable_running_stats(model)

                            pred_second = model(*inputs)
                            loss_second = criterion(pred_second, y)

                            if hasattr(model_for_aux, "get_aux_loss"):
                                aux_loss_second = model_for_aux.get_aux_loss()
                                if aux_loss_second is not None:
                                    loss_second = loss_second + moe_aux_loss_weight * aux_loss_second

                            loss_second.backward()
                            optimizer.second_step(zero_grad = True)

                # Collect predictions for end-of-epoch metrics (for validation)
                if phase == 'val':
                    all_preds_epoch.append(pred.detach().cpu())
                    all_labels_epoch.append(y.detach().cpu())

                # Collect batch metrics
                running_loss.append(loss.item())
                running_acc.append(accuracy(pred, y, task = "multiclass", num_classes = num_classes))
                running_topk.append(accuracy(pred, y, task = "multiclass", num_classes = num_classes, top_k = 5))
                running_topk10.append(accuracy(pred, y, task = "multiclass", num_classes = num_classes, top_k = 10))

                # Calculate sklearn metrics for this batch
                sklearn_metrics = calculate_metrics_sklearn(pred, y, num_classes, batch_size_small = True)
                running_mcc.append(sklearn_metrics['mcc'])
                running_f1_macro.append(sklearn_metrics['f1_macro'])
                running_f1_weighted.append(sklearn_metrics['f1_weighted'])

            # Average metrics at end of epoch
            epoch_loss = np.mean(running_loss)
            epoch_acc = torch.mean(torch.stack(running_acc))
            epoch_topk = torch.mean(torch.stack(running_topk))
            epoch_topk10 = torch.mean(torch.stack(running_topk10))
            epoch_mcc = np.mean(running_mcc)
            epoch_f1_macro = np.mean(running_f1_macro)
            epoch_f1_weighted = np.mean(running_f1_weighted)

            # For validation, compute more accurate metrics on all predictions
            if phase == 'val' and len(all_preds_epoch) > 0:
                all_preds_cat = torch.cat(all_preds_epoch, dim = 0)
                all_labels_cat = torch.cat(all_labels_epoch, dim = 0)
                final_metrics = calculate_metrics_sklearn(
                    all_preds_cat,
                    all_labels_cat,
                    num_classes,
                    batch_size_small = False,
                    present_labels_only = True
                )
                epoch_f1_macro_final = final_metrics['f1_macro']
                epoch_f1_weighted_final = final_metrics['f1_weighted']
                epoch_mcc_final = final_metrics['mcc']
            else:
                epoch_f1_macro_final = epoch_f1_macro
                epoch_f1_weighted_final = epoch_f1_weighted
                epoch_mcc_final = epoch_mcc

            logging.info(
                '{} Loss: {:.4f} Acc: {:.4f} MCC: {:.4f} F1-macro: {:.4f} F1-weighted: {:.4f} Top-5: {:.4f} Top-10: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc_final, epoch_f1_macro_final, epoch_f1_weighted_final,
                    epoch_topk, epoch_topk10))
            print(
                '{} Loss: {:.4f} Acc: {:.4f} MCC: {:.4f} F1-macro: {:.4f} F1-weighted: {:.4f} Top-5: {:.4f} Top-10: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc_final, epoch_f1_macro_final, epoch_f1_weighted_final,
                    epoch_topk, epoch_topk10))

            # Log to wandb
            if log_to_wandb:
                _wlog({
                    f'{phase}/loss': epoch_loss,
                    f'{phase}/accuracy': epoch_acc,
                    f'{phase}/mcc': epoch_mcc_final,
                    f'{phase}/f1_score_macro': epoch_f1_macro_final,
                    f'{phase}/f1_score_weighted': epoch_f1_weighted_final,
                    f'{phase}/top5_accuracy': epoch_topk,
                    f'{phase}/top10_accuracy': epoch_topk10,
                    'epoch': epoch
                })

            metrics_dict[phase]["loss"].append(float(epoch_loss))
            metrics_dict[phase]["accuracy"].append(float(epoch_acc.item() if hasattr(epoch_acc, "item") else epoch_acc))
            metrics_dict[phase]["mcc"].append(float(epoch_mcc_final))
            metrics_dict[phase]["f1_macro"].append(float(epoch_f1_macro_final))
            metrics_dict[phase]["f1_weighted"].append(float(epoch_f1_weighted_final))
            metrics_dict[phase]["top5_accuracy"].append(
                float(epoch_topk.item() if hasattr(epoch_topk, "item") else epoch_topk))
            metrics_dict[phase]["top10_accuracy"].append(
                float(epoch_topk10.item() if hasattr(epoch_topk10, "item") else epoch_topk10))

            # keep best model state_dict
            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'val':
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc.item())
                # check Early Stopping & adjust learning rate if needed
                early_stopping(epoch_loss, model)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc.item())

            torch.cuda.empty_cache()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print()
        print(f"Time since start: {(time.time_ns() - start) / 1e9:.2f} seconds")
        metrics_dict["time_seconds"].append(float((time.time_ns() - start) / 1e9))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))

    os.makedirs("./models", exist_ok = True)

    metrics_path = f'./models/CandyCrunch_metrics_{setting_name}.json'
    plot_path = f'./models/CandyCrunch_metric_{setting_name}.png'
    metrics_dict["best"] = {
        "val_loss": float(best_loss),
        "val_accuracy": float(best_acc.item() if hasattr(best_acc, "item") else best_acc),
    }

    metrics_dict["completed_epochs"] = len(metrics_dict["val"]["loss"])

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent = 2)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))

    # Plot loss & score over the course of training
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))

    ax1.plot(range(len(val_losses)), val_losses, label = 'Validation')
    ax1.plot(range(len(train_losses)), train_losses, label = 'Training')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Training - Loss')
    ax1.legend()
    ax1.grid(True, alpha = 0.3)

    ax2.plot(range(len(val_acc)), val_acc, label = 'Validation')
    ax2.plot(range(len(train_acc)), train_acc, label = 'Training')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Training - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Save best model weights
    best_model_path = f'./models/CandyCrunch_{setting_name}.pt'

    checkpoint = dict(checkpoint_metadata or {})
    checkpoint["state_dict"] = best_model_wts
    checkpoint["best_val_loss"] = float(best_loss)
    checkpoint["best_val_accuracy"] = (float(best_acc.item()) if hasattr(best_acc, "item") else float(best_acc))
    torch.save(checkpoint, best_model_path)

    # Log final plots/model to wandb
    if log_to_wandb:
        _wlog({
            'training_plots/loss_curves': wandb.Image(plot_path),
            'best_metrics/best_val_loss': best_loss,
            'best_metrics/best_val_accuracy': best_acc.item() if hasattr(best_acc, 'item') else best_acc,
        })
        wandb.save(best_model_path)
        wandb.save(metrics_path)

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)

    return model


def train_decoder_model(model, dataloaders, optimizer, scheduler, pad_token_id, vocab_size,
                        class_criterion=None, class_loss_weight=0.0, num_epochs=None, patience=None,
                        log_to_wandb=True, model_type="CNN_Decoder", setting_name=None,
                        checkpoint_metadata=None, label_smoothing=0.1, moe_aux_loss_weight=None,
                        log_prefix=""):
    """Trains a CNN or peak-list Transformer decoder to generate IUPAC tokens.

    Mirrors train_model's epoch/early-stopping/checkpoint/plotting scaffolding (same SAM
    two-forward-pass optimizer pattern, same output file conventions), but loss/metrics operate
    on token sequences (teacher-forced next-token prediction) plus an optional auxiliary
    classification loss, instead of a single class label per sample.

    Arguments:
    :-
    model (PyTorch object): CandyCrunch_CNN_Decoder or CandyCrunch_Transformer
    dataloaders (PyTorch object): dictionary of dataloader objects with keys 'train' and 'val'
    optimizer (PyTorch object): SAM-style optimizer with first_step/second_step
    scheduler (PyTorch object): PyTorch learning rate decay
    pad_token_id (int): tokenizer's pad token id, for ignore_index and masking
    vocab_size (int): tokenizer's vocab size
    class_criterion (PyTorch object): optional auxiliary classification loss (e.g. custom_loss(...))
    class_loss_weight (float): weight on the auxiliary class_CE term; 0 disables it entirely
    num_epochs (int): number of epochs for training
    patience (int): number of epochs without improvement until early stop

    Returns:
    :-
    Returns the best model seen during training
    """
    since = time.time()
    _wlog = lambda metrics: wandb.log({log_prefix + k: v for k, v in metrics.items()})
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    best_token_acc = 0.0
    val_losses, val_token_acc = [], []
    train_losses, train_token_acc = [], []

    token_criterion = nn.CrossEntropyLoss(ignore_index = pad_token_id, label_smoothing = label_smoothing)
    model_for_aux = model.module if hasattr(model, "module") else model

    def unpack_output(output):
        return output if isinstance(output, tuple) else (output, None)

    def compute_loss(token_logits, class_logits, target_labels, y_class):
        loss = token_criterion(token_logits.reshape(-1, vocab_size), target_labels.reshape(-1))
        class_loss = None
        if class_loss_weight > 0 and class_criterion is not None:
            class_loss = class_criterion(class_logits, y_class)
            loss = loss + class_loss_weight * class_loss
        return loss, class_loss

    metric_names = ["loss", "token_accuracy", "sequence_accuracy", "token_perplexity", "class_accuracy"]
    metrics_dict = {
        "train": {name: [] for name in metric_names},
        "val": {name: [] for name in metric_names},
        "time_seconds": [],
    }

    start = time.time_ns()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = []
            running_token_correct = 0
            running_token_total = 0
            running_seq_correct = 0
            running_seq_total = 0
            running_class_correct = 0
            running_class_total = 0

            for data in dataloaders[phase]:
                inputs, (target_labels, y_class) = prepare_batch(data, model_type)
                optimizer.zero_grad(set_to_none = True)

                with torch.set_grad_enabled(phase == 'train'):
                    enable_running_stats(model)
                    token_logits, class_logits = unpack_output(model(*inputs))
                    loss, _ = compute_loss(token_logits, class_logits, target_labels, y_class)
                    if moe_aux_loss_weight is not None and hasattr(model_for_aux, "get_aux_loss"):
                        loss = loss + moe_aux_loss_weight * model_for_aux.get_aux_loss()

                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step(zero_grad = True)

                        # second forward pass for SAM
                        disable_running_stats(model)
                        token_logits2, class_logits2 = unpack_output(model(*inputs))
                        loss2, _ = compute_loss(token_logits2, class_logits2, target_labels, y_class)
                        if moe_aux_loss_weight is not None and hasattr(model_for_aux, "get_aux_loss"):
                            loss2 = loss2 + moe_aux_loss_weight * model_for_aux.get_aux_loss()
                        loss2.backward()
                        optimizer.second_step(zero_grad = True)

                # Token-level accuracy (ignoring pad positions) and exact full-sequence match.
                mask = (target_labels != pad_token_id)
                pred_tokens = token_logits.argmax(dim = -1)
                running_token_correct += int(((pred_tokens == target_labels) & mask).sum().item())
                running_token_total += int(mask.sum().item())

                seq_correct = ((pred_tokens == target_labels) | ~mask).all(dim = 1)
                running_seq_correct += int(seq_correct.sum().item())
                running_seq_total += seq_correct.size(0)

                if class_loss_weight > 0 and class_criterion is not None:
                    class_pred = class_logits.argmax(dim = -1)
                    running_class_correct += int((class_pred == y_class).sum().item())
                    running_class_total += y_class.size(0)

                running_loss.append(loss.item())

            epoch_loss = np.mean(running_loss)
            epoch_token_acc = running_token_correct / max(running_token_total, 1)
            epoch_seq_acc = running_seq_correct / max(running_seq_total, 1)
            epoch_class_acc = running_class_correct / max(running_class_total, 1) if running_class_total else None
            epoch_perplexity = float(np.exp(min(epoch_loss, 20.0)))

            log_line = '{} Loss: {:.4f} Token-Acc: {:.4f} Seq-Acc: {:.4f} PPL: {:.2f}'.format(
                phase, epoch_loss, epoch_token_acc, epoch_seq_acc, epoch_perplexity)
            if epoch_class_acc is not None:
                log_line += ' Class-Acc: {:.4f}'.format(epoch_class_acc)
            print(log_line)

            if log_to_wandb:
                wandb_log = {
                    f'{phase}/loss': epoch_loss,
                    f'{phase}/token_accuracy': epoch_token_acc,
                    f'{phase}/sequence_accuracy': epoch_seq_acc,
                    f'{phase}/token_perplexity': epoch_perplexity,
                    'epoch': epoch,
                }
                if epoch_class_acc is not None:
                    wandb_log[f'{phase}/class_accuracy'] = epoch_class_acc
                _wlog(wandb_log)

            metrics_dict[phase]["loss"].append(float(epoch_loss))
            metrics_dict[phase]["token_accuracy"].append(float(epoch_token_acc))
            metrics_dict[phase]["sequence_accuracy"].append(float(epoch_seq_acc))
            metrics_dict[phase]["token_perplexity"].append(float(epoch_perplexity))
            if epoch_class_acc is not None:
                metrics_dict[phase]["class_accuracy"].append(float(epoch_class_acc))

            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_seq_acc > best_token_acc:
                best_token_acc = epoch_seq_acc
            if phase == 'val':
                val_losses.append(epoch_loss)
                val_token_acc.append(epoch_seq_acc)
                early_stopping(epoch_loss, model)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_token_acc.append(epoch_seq_acc)

            torch.cuda.empty_cache()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print()
        print(f"Time since start: {(time.time_ns() - start) / 1e9:.2f} seconds")
        metrics_dict["time_seconds"].append(float((time.time_ns() - start) / 1e9))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:.4f}, best sequence accuracy: {:.4f}'.format(best_loss, best_token_acc))

    os.makedirs("./models", exist_ok = True)

    metrics_path = f'./models/CandyCrunch_metrics_{setting_name}.json'
    plot_path = f'./models/CandyCrunch_metric_{setting_name}.png'
    metrics_dict["best"] = {
        "val_loss": float(best_loss),
        "val_sequence_accuracy": float(best_token_acc),
    }
    metrics_dict["completed_epochs"] = len(metrics_dict["val"]["loss"])

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent = 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))

    ax1.plot(range(len(val_losses)), val_losses, label = 'Validation')
    ax1.plot(range(len(train_losses)), train_losses, label = 'Training')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Training - Loss')
    ax1.legend()
    ax1.grid(True, alpha = 0.3)

    ax2.plot(range(len(val_token_acc)), val_token_acc, label = 'Validation')
    ax2.plot(range(len(train_token_acc)), train_token_acc, label = 'Training')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Sequence Accuracy')
    ax2.set_title('Model Training - Sequence Accuracy')
    ax2.legend()
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

    best_model_path = f'./models/CandyCrunch_{setting_name}.pt'

    checkpoint = dict(checkpoint_metadata or {})
    checkpoint["state_dict"] = best_model_wts
    checkpoint["best_val_loss"] = float(best_loss)
    checkpoint["best_val_sequence_accuracy"] = float(best_token_acc)
    torch.save(checkpoint, best_model_path)

    if log_to_wandb:
        _wlog({
            'training_plots/loss_curves': wandb.Image(plot_path),
            'best_metrics/best_val_loss': best_loss,
            'best_metrics/best_val_sequence_accuracy': best_token_acc,
        })
        wandb.save(best_model_path)
        wandb.save(metrics_path)

    model.load_state_dict(best_model_wts)

    return model


def fit_beam_temperature(beam_scores, target_ranks, grid=None):
    """fits a single scalar temperature to calibrate beam-search confidence scores

    Grid search (not gradient descent) over a scalar T minimizing the mean negative
    log-likelihood of softmax(scores / T) at the target rank -- the beam-search analogue of
    Platt scaling. Rows where the true glycan wasn't found in any beam (target_ranks[i] is None)
    are dropped from the fit; the coverage rate (rows kept / rows total) is returned too, since
    low coverage is itself informative (a calibration run can't fix a decoder that isn't finding
    the right structure at all).

    Arguments:
    :-
    beam_scores (list[np.ndarray]): length-normalized log-probs per spectrum, one array per row
    target_ranks (list[Optional[int]]): index into that row's array matching the true glycan, or
                                        None if the truth wasn't present in any beam
    grid (np.ndarray): candidate temperature values; default np.logspace(-1, 1, 81)

    Returns:
    :-
    (T, coverage_rate): the fitted scalar temperature and the fraction of rows with a match
    """
    if grid is None:
        grid = np.logspace(-1, 1, 81)

    kept = [(scores, rank) for scores, rank in zip(beam_scores, target_ranks) if rank is not None]
    coverage_rate = len(kept) / max(len(beam_scores), 1)
    if not kept:
        return 1.0, coverage_rate

    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nlls = []
        for scores, rank in kept:
            log_probs = torch.log_softmax(torch.as_tensor(scores, dtype = torch.float32) / T, dim = 0)
            nlls.append(-log_probs[rank].item())
        mean_nll = float(np.mean(nlls))
        if mean_nll < best_nll:
            best_nll = mean_nll
            best_T = float(T)

    return best_T, coverage_rate


def _calibrated_beam_probs(row_conf, temperature):
    """Recomputes softmax(scores/T) from beam_search_topk's T=1 probabilities, without needing
    the raw pre-softmax scores: since softmax is invariant to a constant per-row shift, and
    log(T=1 probs) is the true scores shifted by exactly such a constant (their logsumexp),
    softmax(log(T=1 probs) / T) == softmax(true_scores / T) for any T. So there's no need for a
    separate raw-scores return path on beam_search_topk -- this reconstructs it exactly."""
    scores = np.log(np.clip(np.asarray(row_conf, dtype = np.float64), 1e-12, None))
    shifted = (scores - scores.max()) / temperature
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def calibrate_decoder_checkpoint(checkpoint_path, val_dataloader, val_true_glycans, glycan_class,
                                 tokenizer, k=25, length_penalty=0.7,
                                 pred_thresh_grid=None, extra_thresh_grid=None):
    """Calibrates a trained generative checkpoint on a held-out validation set.

    Runs beam search once (temperature=1.0) on the val set, fits a single scalar temperature via
    fit_beam_temperature, and sweeps a small pred_thresh x extra_thresh grid mirroring
    wrap_inference's actual filter predicate (enforce_class(...) and confidence > pred_thresh) to
    report top-1 accuracy and mean surviving-candidate count per combination. Writes the fitted
    temperature/pred_thresh/extra_thresh/beam_width/length_penalty into the checkpoint's
    inference_defaults dict in place -- wrap_inference already reads
    checkpoint.get("inference_defaults", ...), so no consumption-side code changes are needed.

    Arguments:
    :-
    checkpoint_path (str): path to a CNN decoder or Transformer .pt checkpoint (updated in place)
    val_dataloader (PyTorch): dataloader from process_for_inference over the held-out val split
    val_true_glycans (list[str]): ground-truth IUPAC string per row of val_dataloader, same order
    glycan_class (str): glycan class as used by glycowork's enforce_class ("O", "N", "lipid", "free")
    tokenizer (BPETokenizer): tokenizer the model's decoder was trained with
    k (int): beam width; default:25
    length_penalty (float): GNMT-style length normalization exponent; default:0.7
    pred_thresh_grid, extra_thresh_grid (list[float]): threshold values to sweep; sensible
                                                       defaults provided

    Returns:
    :-
    A dict summarizing the fitted temperature, coverage@k, beam discard rate, and the full
    pred_thresh/extra_thresh grid results (also what gets printed).
    """
    from candycrunch.prediction import beam_search_topk, get_model, _safe_canonicalize
    from glycowork.motif.processing import enforce_class

    model = get_model(checkpoint_path)
    preds, confs, discard_rate = beam_search_topk(
        val_dataloader, model, tokenizer = tokenizer, k = k,
        max_length = getattr(model, "_candycrunch_target_len", None),
        length_penalty = length_penalty, temperature = 1.0, return_discard_rate = True)

    canon_truth = [_safe_canonicalize(g) for g in val_true_glycans]

    beam_scores, target_ranks = [], []
    for row_preds, row_conf, truth in zip(preds, confs, canon_truth):
        scores = np.log(np.clip(np.asarray(row_conf, dtype = np.float64), 1e-12, None))
        beam_scores.append(scores)
        target_ranks.append(row_preds.index(truth) if truth in row_preds else None)

    T, coverage = fit_beam_temperature(beam_scores, target_ranks)
    print(f"Fitted temperature T={T:.3f}, coverage@{k}={coverage:.3f}, beam discard rate={discard_rate:.3f}")

    if pred_thresh_grid is None:
        pred_thresh_grid = [0.005, 0.01, 0.02, 0.05]
    if extra_thresh_grid is None:
        extra_thresh_grid = [0.1, 0.2, 0.3, 0.5]

    grid_results = []
    for pred_thresh in pred_thresh_grid:
        for extra_thresh in extra_thresh_grid:
            n_correct, n_total, survivor_counts = 0, 0, []
            for row_preds, row_conf, truth in zip(preds, confs, canon_truth):
                if truth is None or not row_preds:
                    continue
                n_total += 1
                calibrated = _calibrated_beam_probs(row_conf, T)
                survivors = [g for g, c in zip(row_preds, calibrated)
                            if c > pred_thresh and enforce_class(g, glycan_class, c, extra_thresh = extra_thresh)]
                survivor_counts.append(len(survivors))
                if survivors and survivors[0] == truth:
                    n_correct += 1
            grid_results.append({
                "pred_thresh": pred_thresh, "extra_thresh": extra_thresh,
                "top1_accuracy": n_correct / max(n_total, 1),
                "mean_survivors": float(np.mean(survivor_counts)) if survivor_counts else 0.0,
            })
            print(f"  pred_thresh={pred_thresh} extra_thresh={extra_thresh}: "
                 f"top1_acc={grid_results[-1]['top1_accuracy']:.3f} "
                 f"mean_survivors={grid_results[-1]['mean_survivors']:.2f}")

    best = max(grid_results, key = lambda r: r["top1_accuracy"])

    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = False)
    checkpoint.setdefault("inference_defaults", {}).update({
        "temperature": float(T), "pred_thresh": best["pred_thresh"], "extra_thresh": best["extra_thresh"],
        "beam_width": k, "length_penalty": length_penalty,
    })
    checkpoint["calibration"] = {
        "coverage_at_k": coverage, "beam_discard_rate": discard_rate,
        "n_val_rows": len(val_true_glycans), "grid_results": grid_results,
    }
    torch.save(checkpoint, checkpoint_path)

    return {
        "temperature": T, "coverage_at_k": coverage, "beam_discard_rate": discard_rate,
        "best_pred_thresh": best["pred_thresh"], "best_extra_thresh": best["extra_thresh"],
        "grid_results": grid_results,
    }
