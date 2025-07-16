import os.path

import numpy as np

from collections import Counter

import pandas as pd
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import logging
import torch


# def __balance_val_split(dataset, val_split=0.):
#     targets = np.array(dataset.targets)
#     train_indices, val_indices = train_test_split(
#         np.arange(targets.shape[0]),
#         test_size=val_split,
#         stratify=targets
#     )
#
#     train_dataset = Subset(dataset, indices=train_indices)
#     val_dataset = Subset(dataset, indices=val_indices)
#
#     return train_dataset, val_dataset
#
#
# def __split_of_train_sequence(subset: Subset, train_split=1.0):
#     if train_split == 1:
#         return subset
#
#     targets = np.array([subset.dataset.targets[i] for i in subset.indices])
#     train_indices, _ = train_test_split(
#         np.arange(targets.shape[0]),
#         test_size=1 - train_split,
#         stratify=targets
#     )
#
#     train_dataset = Subset(subset.dataset, indices=[subset.indices[i] for i in train_indices])
#
#     return train_dataset
#
#
# def __log_class_statistics(subset: Subset):
#     train_classes = [subset.dataset.targets[i] for i in subset.indices]
#     print(dict(Counter(train_classes)))


def count_success(preds, labels, calc_stats=False):
    counter_success = 0
    stats = {i: [0, 0] for i in range(0, 100)}
    for i_bs in range(preds.shape[0]):
        if preds[i_bs] == labels[i_bs]:
            counter_success += 1
            if bool(calc_stats):
                stats[int(labels[i_bs])][0] += 1  # correct predictions in class dim 0
        if bool(calc_stats):
            stats[int(labels[i_bs])][1] += 1  # total samples in class in dim 1
    return counter_success, stats


def train_epoch_batch(model, dataloader, loss_fn, optimizer, device, scheduler=None, batch_size=1, clip_gradients=False, clip_weights=False):
    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        batch, labels = data

        batch = batch.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(batch)
        outs_squeeze = outputs.squeeze(1)  # remove the temporal dimension

        loss = loss_fn(outs_squeeze, labels)  # loss = criterion(outputs[0], labels[0])
        loss.backward()

        # Clip gradients to prevent them from exploding.
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Weight clipping
        if clip_weights:
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-0.5, 0.5)

        # Step optimizer and scheduler
        optimizer.step()
        if scheduler:
            scheduler.step()



        # # Visualize gradients
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.norm())

        running_loss += loss.item()

        # Statistics
        preds = torch.argmax(outs_squeeze, dim=1)  # [bs,1]
        pred_correct += count_success(preds, labels)[0]
        pred_all += preds.shape[0]  # it should be equal to batch size

    average_running_loss = running_loss / len(dataloader)
    average_running_acc = pred_correct / pred_all

    print("\n>>> Total Train: ", str(pred_all))
    return average_running_loss, average_running_acc



# def evaluate_batch(model, loss_fn, dataloader, device, print_stats=False):
#     pred_correct, pred_all = 0, 0
#     stats = {i: [0, 0] for i in range(100)}  # Assuming 100 classes
#     val_loss = 0.0
#     for i, data in enumerate(dataloader):
#         batch, labels = data
#
#         batch = batch.to(device)
#         labels = labels.to(device, dtype=torch.long)
#
#         outputs = model(batch)
#         outs_squeeze = outputs.squeeze(1)  # remove the temporal dimension
#
#         loss = loss_fn(outs_squeeze, labels)  # loss = criterion(outputs[0], labels[0])
#         val_loss += loss.item()
#
#         # Statistics
#         preds = torch.argmax(outs_squeeze, dim=1)  # [bs,1]
#         # print('predictions:', preds)
#         # print('labels:', labels)
#         pred_correct_i, batch_stats = count_success(preds, labels, calc_stats=True)
#
#         # Accumulate stats
#         for k in batch_stats:
#             stats[k][0] += batch_stats[k][0]  # correct
#             stats[k][1] += batch_stats[k][1]  # total
#
#         pred_correct += pred_correct_i
#         pred_all += preds.shape[0]  # it should be equal to batch size
#
#     average_val_loss = val_loss / len(dataloader)
#
#     print(">>> Total Validation: ", str(pred_all))
#     if print_stats:
#         stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
#         print("Validation accuracies statistics:")
#         print(str(stats) + "\n")
#         logging.info("Validation accuracies statistics:")
#         logging.info(str(stats) + "\n")
#
#
#     average_val_acc = pred_correct / pred_all
#     return average_val_loss, average_val_acc, stats

def evaluate_batch(model, loss_fn, dataloader, device, print_stats=False):
    pred_correct, pred_all = 0, 0
    num_classes = 100  # adjust if needed
    val_loss = 0.0

    # Track TP, FP, FN for each class
    class_metrics = {i: {"TP": 0, "FP": 0, "FN": 0} for i in range(num_classes)}

    for i, data in enumerate(dataloader):
        batch, labels = data

        batch = batch.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(batch)
        outs_squeeze = outputs.squeeze(1)

        loss = loss_fn(outs_squeeze, labels)
        val_loss += loss.item()

        preds = torch.argmax(outs_squeeze, dim=1)

        for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            if true == pred:
                class_metrics[true]["TP"] += 1
            else:
                class_metrics[true]["FN"] += 1
                class_metrics[pred]["FP"] += 1

        pred_correct += torch.sum(preds == labels).item()
        pred_all += labels.size(0)

    average_val_loss = val_loss / len(dataloader)
    average_val_acc = pred_correct / pred_all

    if print_stats:
        print("Validation class-wise metrics:")
        for k, v in class_metrics.items():
            print(f"Class {k}: TP={v['TP']} FP={v['FP']} FN={v['FN']}")
            logging.info(f"Class {k}: TP={v['TP']} FP={v['FP']} FN={v['FN']}")

    return average_val_loss, average_val_acc, class_metrics


def evaluate_batch_savePred(model, dataloader, device, save_path, print_stats=False, n_classes=100):
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(n_classes+1)}
    list_preds = []
    list_labels = []
    df_preds = pd.DataFrame([], columns=["index", "pred", "label"])
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs) #.expand(1, -1, -1)
        outsSqueeze = outputs.squeeze(1) # remove the temporal dimension

        # Statistics
        posteriors = np.array(torch.softmax(outputs, dim=-1).squeeze(1).tolist()[0])
        preds = torch.argmax(outsSqueeze, dim=1)
        # save posteriors:
        np.save(os.path.join(save_path, str(i)+".npy"), posteriors)
        df_preds = pd.concat([df_preds, pd.DataFrame([[i, preds.item(), labels.item()]], columns=["index", "pred", "label"])])

        pred_correct_i, stats = count_success(preds, labels, stats)

        pred_correct += pred_correct_i
        pred_all += preds.shape[0]  # it should be equal to batch size
        #print("PRED:", str(pred), " - LABEL:", str(int(labels[0][0])))

    #print("-- Query EVOL (END EPOCH): ", model.class_query, " --")
    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")
    df_preds.to_csv(os.path.join(save_path, "summary_preds.csv"), index=False)
    print(">> Total files Validation: ", str(pred_all))
    return pred_correct, pred_all, (pred_correct / pred_all), stats



# # withoutbatchs > batch = 1
# def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
#     pred_correct, pred_all = 0, 0
#     running_loss = 0.0
#
#     for i, data in enumerate(dataloader):
#         inputs, labels = data
#         inputs = inputs.squeeze(0).to(device)
#         labels = labels.to(device, dtype=torch.long)
#
#         optimizer.zero_grad()
#         outputs = model(inputs).expand(1, -1, -1)
#
#
#         loss = criterion(outputs[0], labels[0])
#         loss.backward()
#         # clip_grad_norm_ to prevent gradients from exploding.
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
#         running_loss += loss.item()
#
#         # Statistics
#         if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
#             pred_correct += 1
#         pred_all += 1
#
#     if scheduler:
#         scheduler.step(running_loss.item() / len(dataloader))
#     print("Total files Train: ", str(i))
#     return running_loss, pred_correct, pred_all, (pred_correct / pred_all)


# def evaluate(model, dataloader, device, print_stats=False):
#     pred_correct, pred_all = 0, 0
#     stats = {i: [0, 0] for i in range(101)}
#
#     for i, data in enumerate(dataloader):
#         inputs, labels = data
#         inputs = inputs.squeeze(0).to(device)
#         labels = labels.to(device, dtype=torch.long)
#         outputs = model(inputs).expand(1, -1, -1)
#
#         # Statistics
#         pred = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))
#         #print("PRED:", str(pred), " - LABEL:", str(int(labels[0][0])))
#         if pred == int(labels[0][0]):
#             stats[int(labels[0][0])][0] += 1
#             pred_correct += 1
#
#         stats[int(labels[0][0])][1] += 1
#         pred_all += 1
#         #print("POS EVOL: ", model.pos)
#     if print_stats:
#         stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
#         print("Label accuracies statistics:")
#         print(str(stats) + "\n")
#         logging.info("Label accuracies statistics:")
#         logging.info(str(stats) + "\n")
#     print("Total files Validation: ", str(i+1))
#     return pred_correct, pred_all, (pred_correct / pred_all)
#
#
# def evaluate_top_k(model, dataloader, device, k=5):
#     pred_correct, pred_all = 0, 0
#     for i, data in enumerate(dataloader):
#         inputs, labels = data
#         inputs = inputs.squeeze(0).to(device)
#         labels = labels.to(device, dtype=torch.long)
#         outputs = model(inputs).expand(1, -1, -1)
#         if int(labels[0][0]) in torch.topk(outputs, k).indices.tolist():
#             pred_correct += 1
#         pred_all += 1
#     return pred_correct, pred_all, (pred_correct / pred_all)
