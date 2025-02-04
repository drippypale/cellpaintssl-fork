import os
import time
from functools import partial

import numpy as np
import pandas
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.metrics import roc_auc_score, f1_score

from metrics import accuracy, F1
from pyll.base import TorchModel, AverageMeter
from pyll.session import PyLL
from pyll.utils.workspace import Workspace


def main():
    session = PyLL()
    datasets = session.datasets
    model = session.model
    config = session.config
    batchsize_eval = config.training.batchsize
    
    loader_val = torch.utils.data.DataLoader(datasets["test"],
                                                 batch_size=batchsize_eval, shuffle=False,
                                                 num_workers=config.workers, pin_memory=False, drop_last=False)
 
    # checkpoint = torch.load("/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/dino_split3/fnn_dino3/2023-11-21T20-50-00/checkpoints/model_best.pth.tar")
    # checkpoint = torch.load("/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/dino_split2/fnn_dino2/2023-11-21T20-49-51/checkpoints/model_best.pth.tar")
    checkpoint = torch.load("/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/dino_split1/fnn_dino1/2023-11-21T20-49-44/checkpoints/model_best.pth.tar")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    class_aucs, class_f1s= validate(loader_val, "test", model, config, 0)
    # np.save('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_aucs3.npy', class_aucs)
    # np.save('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_f1s3.npy', class_f1s)
    # np.save('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_aucs2.npy', class_aucs)
    # np.save('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_f1s2.npy', class_f1s)
    np.save('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_aucs1.npy', class_aucs)
    np.save('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_f1s1.npy', class_f1s)
    print("Done")


def validate(loader, split_name, model: TorchModel, config, samples_seen):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    batchsize = loader.batch_size
    print('batchsize')
    print(batchsize)
    n_samples = len(loader.dataset)

    n_tasks = loader.dataset.num_classes    
    predictions = np.zeros(shape=(n_samples, n_tasks))
    targets = np.zeros(shape=(n_samples, n_tasks))
    sample_keys = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, batch in enumerate(loader):
        with torch.no_grad():
            input = batch["input"]
            target = batch["target"]
            sample_keys.extend(batch["ID"])
            target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.sigmoid(output)
            loss = model.module.loss(output.cuda(non_blocking=True), target.cuda(non_blocking=True))

        # store predictions and labels
        target = target.cpu().numpy()
        pred = output.cpu().data.numpy()
        pred_tasks = pred
        target_tasks = target

        # store
        predictions[i * batchsize:(i + 1) * batchsize, :] = pred_tasks
        targets[i * batchsize:(i + 1) * batchsize, :] = target_tasks / 2 + 0.5

        # measure accuracy and record loss
        acc = accuracy(pred_tasks, target_tasks)
        losses.update(loss.item(), input.size(0))
        accuracies.update(acc, input.size(0))
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('{split}: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, loss=losses, acc=accuracies, split=split_name))
            
    # AUC
    class_aucs = []
    class_f1s = []
    over9 =0
    over8 =0
    over7 =0
    for i in range(n_tasks):
        # try:
        if np.any(targets[:, i] == 0) and np.any(targets[:, i] == 1):
            samples = list(np.where(targets[:, i] == 0)[0]) + list(np.where(targets[:, i] == 1)[0])
            class_auc = roc_auc_score(y_true=targets[samples, i], y_score=predictions[samples, i])
            class_f = f1_score(y_true=targets[samples, i], y_pred=predictions[samples, i].round())
            if class_auc >= 0.9:
                over9+=1
            if class_auc >= 0.8:
                over8+=1
            if class_auc >= 0.7:
                over7+=1
        else:
            class_auc = 0.5
            class_f = 0.0
        class_aucs.append(class_auc)
        class_f1s.append(class_f)

    print('class_aucs tabel values')
    for index, val in np.ndenumerate(class_aucs):
        print ('){}, {}'.format(index,val))

    print('------------------------------')
    
    print('class_f1 tabel values')
    for index, val in np.ndenumerate(class_f1s):
        print ('){}, {}'.format(index,val))

    mean_auc = float(np.mean(class_aucs))
    mean_new_f = float(np.mean(class_f1s))

    # write statistics

    print(' * Accuracy {acc.avg:.3f}\tAUC {auc:.3f}\tAUC>0.9 {auc9:.3f}\tAUC>0.8 {auc8:.3f}\tAUC>0.7 {auc7:.3f}\tF1 {f1:.3f}'.format(acc=accuracies, auc=mean_auc, auc9=over9, auc8=over8, auc7=over7, f1=mean_new_f))

    return class_aucs, class_f1s


if __name__ == '__main__':
    main()
