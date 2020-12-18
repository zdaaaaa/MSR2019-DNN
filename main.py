from model.dnn import Model as Model
import re
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from model.layers.regularization import L2Regularization
from dataset.nn_dataset import DatasetFactory
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from utils.util import weights_init

seed = 777
device = 0
div_min = 0.00001
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DataConfiguration:
    labels = ['CVSS2_Conf', 'CVSS2_Integrity', 'CVSS2_Avail', 'CVSS2_AccessVect',
          'CVSS2_AccessComp', 'CVSS2_Auth', 'CVSS2_Severity']
    configs = list(range(1, 9))


class TrainConfiguration:
    learning_rate = 0.01
    milestones = {
        1: 0.01,
        2000: 0.001,
        4000: 0.0008,
        6000: 0.0005,
        8000: 0.0001
        # 5: 0.00005,
        # 100: 0.00005
    }
    batch_size = 64

def run(tconf, dconf):
    for config in dconf.configs:
        start_word_ngram = 1
        end_word_ngram = 1

        if config == 1:
            start_word_ngram = 1
            end_word_ngram = 1
        elif config == 2:
            start_word_ngram = 1
            end_word_ngram = 1
        elif config <= 5:
            if config == 3:
                start_word_ngram = 1
                end_word_ngram = 2
            elif config == 4:
                start_word_ngram = 1
                end_word_ngram = 3
            elif config == 5:
                start_word_ngram = 1
                end_word_ngram = 4
        else:
            if config == 6:
                start_word_ngram = 1
                end_word_ngram = 2
            elif config == 7:
                start_word_ngram = 1
                end_word_ngram = 3
            elif config == 8:
                start_word_ngram = 1
                end_word_ngram = 4

        for label in dconf.labels:
            k_fold = 5
            split_year = 2016
            start_year = split_year - k_fold

            for year in range(start_year, split_year):
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                ds_factory = DatasetFactory(label, config, start_word_ngram, end_word_ngram, year)
                train_ds = ds_factory.get_train_dataset()
                valid_ds = ds_factory.get_valid_dataset()

                train_loader = DataLoader(
                    dataset=train_ds,
                    batch_size=tconf.batch_size,
                    shuffle=True,
                    drop_last=True
                )

                valid_loader = DataLoader(
                    dataset=valid_ds,
                    batch_size=tconf.batch_size,
                    shuffle=False,
                    drop_last=True
                )

                x_shape = 0
                for _, (x, y) in enumerate(valid_loader, 0):
                    x_shape = x.shape[1]
                    break

                model = Model(x_shape, 3, 512, 1)

                model = model.cuda(device)
                model.apply(weights_init)
                if tconf.l2_weight_decay > 0:
                    reg_loss = L2Regularization(model, tconf.l2_weight_decay).cuda(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=tconf.learning_rate)
                step = 1
                for epoch in range(tconf.max_epoch):
                    loss_item = 0
                    target_num = torch.zeros((1, 3))
                    predict_num = torch.zeros((1, 3))
                    acc_num = torch.zeros((1, 3))

                    for _, (x, y) in enumerate(train_loader, 0):
                        if step in tconf.milestones:
                            print('Set lr=', tconf.milestones[step])
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = tconf.milestones[step]
                        model.train()
                        x = x.cuda(device)
                        target = y.cuda(device)
                        output = model(x)
                        _, predict = torch.max(output.data, 1)
                        pre_mask = torch.zeros(output.size()).scatter_(1, predict.cpu().view(-1, 1).long(), 1.)
                        predict_num += pre_mask.sum(0)
                        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1).long(), 1.)
                        # tar_mask = target.cpu()
                        target_num += tar_mask.sum(0)
                        acc_mask = pre_mask * tar_mask
                        acc_num += acc_mask.sum(0)
                        # (n,3), (n,3)
                        if tconf.weight_flag:
                            class_weight = torch.FloatTensor(tconf.weight)
                            class_weight = class_weight.cuda(device)
                            loss = F.nll_loss(output, target.long(), weight=class_weight)
                        else:
                            loss = F.nll_loss(output, target.long())
                        loss.backward()
                        loss_item += loss.item()
                        if step == 0:
                            optimizer.zero_grad()
                        elif step % tconf.batch_size == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                        if step % 100 == 0:
                            recall = acc_num / (target_num + div_min)
                            precision = acc_num / (predict_num + div_min)
                            F1 = 2 * recall * precision / (recall + precision + div_min)
                            accuracy = acc_num.sum(1) / (target_num.sum(1) + div_min)
                            recall = (recall.numpy()[0] * 100).round(3)
                            precision = (precision.numpy()[0] * 100).round(3)
                            F1 = (F1.numpy()[0] * 100).round(3)
                            accuracy = (accuracy.numpy()[0] * 100).round(3)
                            if step % 1000 == 0:
                                print('[Train step:{}]'.format(step))
                                print('loss:', loss_item)
                                print('recall', " ".join('%s' % id for id in recall))
                                print('precision', " ".join('%s' % id for id in precision))
                                print('F1', " ".join('%s' % id for id in F1))
                                print('accuracy', accuracy)
                            loss_item = 0
                        step += 1
                    recall = acc_num / (target_num + div_min)
                    precision = acc_num / (predict_num + div_min)
                    F1 = 2 * recall * precision / (recall + precision + div_min)
                    accuracy = acc_num.sum(1) / (target_num.sum(1) + div_min)
                    recall = (recall.numpy()[0] * 100).round(3)
                    precision = (precision.numpy()[0] * 100).round(3)
                    F1 = (F1.numpy()[0] * 100).round(3)
                    accuracy = (accuracy.numpy()[0] * 100).round(3)
                    print('total train:')
                    print('recall', " ".join('%s' % id for id in recall))
                    print('precision', " ".join('%s' % id for id in precision))
                    print('F1', " ".join('%s' % id for id in F1))
                    print('accuracy', accuracy)
                    # valid
                    with torch.no_grad():
                        target_num = torch.zeros((1, 3))
                        predict_num = torch.zeros((1, 3))
                        acc_num = torch.zeros((1, 3))
                        for _, (x, y) in enumerate(
                                valid_loader, 0):
                            x = x.cuda(device)
                            target = y.cuda(device)
                            output = model(x)
                            _, predict = torch.max(output.data, 1)
                            pre_mask = torch.zeros(output.size()).scatter_(1, predict.cpu().view(-1, 1).long(), 1.)
                            predict_num += pre_mask.sum(0)
                            tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1).long(), 1.)
                            target_num += tar_mask.sum(0)
                            acc_mask = pre_mask * tar_mask
                            acc_num += acc_mask.sum(0)
                        recall = acc_num / (target_num + div_min)
                        precision = acc_num / (predict_num + div_min)
                        F1 = 2 * recall * precision / (recall + precision + div_min)
                        accuracy = acc_num.sum(1) / (target_num.sum(1) + div_min)
                        recall = (recall.numpy()[0] * 100).round(3)
                        precision = (precision.numpy()[0] * 100).round(3)
                        F1 = (F1.numpy()[0] * 100).round(3)
                        accuracy = (accuracy.numpy()[0] * 100).round(3)
                        print('total valid:')
                        print('recall', " ".join('%s' % id for id in recall))
                        print('precision', " ".join('%s' % id for id in precision))
                        print('F1', " ".join('%s' % id for id in F1))
                        print('accuracy', accuracy)

if __name__ == '__main__':
    start_t = time.time()
    tconf = TrainConfiguration()
    dconf = DataConfiguration()
    run(tconf, dconf)
    end_t = time.time()
    print('run time: ', int((end_t - start_t) / 60), 'minutes', int((end_t - start_t) % 60), 'seconds')
