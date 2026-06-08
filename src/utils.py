import torch


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


def train_epoch_batch(model, dataloader, loss_fn, optimizer, device, scheduler=None, clip_gradients=False, clip_weights=False):
    model.train()   # important for RNNs like LSTM for activating dropout layer

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

        # # Weight clipping
        # if clip_weights:
        #     with torch.no_grad():
        #         for param in model.parameters():
        #             param.clamp_(-0.5, 0.5)

        # Step optimizer and scheduler
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item()

        # Statistics
        preds = torch.argmax(outs_squeeze, dim=1)  # [bs,1]
        pred_correct += count_success(preds, labels)[0]
        pred_all += preds.shape[0]  # it should be equal to batch size

    average_running_loss = running_loss / len(dataloader)
    average_running_acc = pred_correct / pred_all

    return average_running_loss, average_running_acc



def evaluate_batch(model, loss_fn, dataloader, device, return_preds=False):
    model.eval()  # Set model to evaluation mode, important for RNNs like LSTM for activating dropout layer
    pred_correct, pred_all = 0, 0
    val_loss = 0.0

    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            batch, labels = data

            batch = batch.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(batch)
            outs_squeeze = outputs.squeeze(1)
            loss = loss_fn(outs_squeeze, labels)
            val_loss += loss.item()

            preds = torch.argmax(outs_squeeze, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pred_correct += torch.sum(preds == labels).item()
            pred_all += labels.size(0)

    average_val_loss = val_loss / len(dataloader)
    average_val_acc = pred_correct / pred_all

    if return_preds:
        return average_val_loss, average_val_acc, (all_labels, all_preds)
    return average_val_loss, average_val_acc, None

