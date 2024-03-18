import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from dataset.dataset import AVDataset, CAVDataset, M3AEDataset, TVDataset, Modal3Dataset, CLIPDataset
from models.basic_model import AVClassifier, CAVClassifier, M3AEClassifier, Modal3Classifier, CLIPClassifier
from utils.utils import setup_seed, weight_init, GSPlugin, History
import datetime

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="CREMA-D", type=str,
                        help='Currently, we only support Food-101, MVSA, CREMA-D')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', "QMF"])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default = 0.3, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default = "ckpt/", type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1, 2', type=str, help='GPU ids')
    parser.add_argument('--lorb', default="m3ae", type=str, help='model_select in [large, base, m3ae]')
    parser.add_argument('--gs_flag', action='store_true')
    parser.add_argument('--av_alpha', default=0.5, type=float, help='2 modal fusion alpha in GS')
    parser.add_argument('--cav_opti', action='store_true')
    parser.add_argument('--cav_lrs', action='store_true')
    parser.add_argument('--cav_augnois', action='store_true')
    parser.add_argument('--modal3', action='store_true', help='3 modality fusion flag')
    parser.add_argument('--dynamic', action='store_true', help='if dynamic fusion in GS')
    parser.add_argument('--a_alpha', default=0.35, type=float, help='audio alpha in 3 modal GS')
    parser.add_argument('--v_alpha', default=0.25, type=float, help='visual alpha in 3 modal GS')
    parser.add_argument('--t_alpha', default=0.4, type=float, help='textual alpha in 3 modal GS')
    parser.add_argument('--clip', action='store_true', help='run using clip pre-trained feature')
    parser.add_argument('--ckpt_load_path_train', default = None, type=str, help='loaded path when training')
    

    return parser.parse_args()

def calculate_entropy(output):
    probabilities = F.softmax(output, dim=0)
    # probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities)
    return entropy

def calculate_gating_weights(encoder_output_1, encoder_output_2):
    
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)
    
    max_entropy = max(entropy_1, entropy_2)
    
    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)
    
    sum_weights = gating_weight_1 + gating_weight_2
    
    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights
    
    return gating_weight_1, gating_weight_2

def calculate_gating_weights3(encoder_output_1, encoder_output_2, encoder_output_3):
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)
    entropy_3 = calculate_entropy(encoder_output_3)
    
    max_entropy = max(entropy_1, entropy_2, entropy_3)
    
    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)
    gating_weight_3 = torch.exp(max_entropy - entropy_3)
    
    sum_weights = gating_weight_1 + gating_weight_2 + gating_weight_3
    
    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights
    gating_weight_3 /= sum_weights
    
    return gating_weight_1, gating_weight_2, gating_weight_3

def rank_loss(confidence, idx, history):
    # make input pair
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + (rank_margin / rank_target_nonzero).reshape((-1,1))

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                        rank_input2,
                                        -rank_target.reshape(-1,1))

    return ranking_loss

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, 
                gs_plugin = None, writer=None, gs_flag = False, av_alpha = 0.5,
                txt_history = None, img_history = None, audio_history = None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _loss_t = 0
    len_dataloader = len(dataloader)
    for batch_step, data_packet in enumerate(dataloader):
        if args.lorb == "m3ae":
            if args.modal3:
                token, padding_mask, image, spec, label, idx = data_packet
                token = token.to(device)
                padding_mask = padding_mask.to(device)
                image = image.to(device)
                spec = spec.to(device)
                label = label.to(device)
            else:
                token, padding_mask, image, label, idx = data_packet
                token = token.to(device)
                padding_mask = padding_mask.to(device)
                image = image.to(device)
                label = label.to(device)
        else:
            spec, image, label, idx = data_packet
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

        optimizer.zero_grad()
        if not gs_flag:
            if args.lorb == "large":               
                a, v = model(spec, image)
                _,_,out = model.module.fusion_module(a, v)
            elif args.lorb == "m3ae":                
                if args.modulation == "QMF":
                    if args.modal3:
                        out_a, out_v, out_t = model(token, padding_mask, image, spec)
                        audio_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
                        img_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))
                        txt_energy = torch.log(torch.sum(torch.exp(out_t), dim=1))

                        txt_conf = txt_energy / 10
                        img_conf = img_energy / 10
                        audio_conf = audio_energy / 10
                        txt_conf = torch.reshape(txt_conf, (-1, 1))
                        img_conf = torch.reshape(img_conf, (-1, 1))
                        audio_conf = torch.reshape(audio_conf, (-1, 1))
                        out = (out_a * audio_conf.detach() + out_v * img_conf.detach() + out_t * txt_conf.detach())

                        audio_clf_loss = nn.CrossEntropyLoss()(out_a, label)
                        img_clf_loss = nn.CrossEntropyLoss()(out_v, label)
                        txt_clf_loss = nn.CrossEntropyLoss()(out_t, label)
                        clf_loss = txt_clf_loss + img_clf_loss + audio_clf_loss

                        audio_loss = nn.CrossEntropyLoss(reduction='none')(out_a, label).detach()
                        img_loss = nn.CrossEntropyLoss(reduction='none')(out_v, label).detach()
                        txt_loss = nn.CrossEntropyLoss(reduction='none')(out_t, label).detach()

                        txt_history.correctness_update(idx, txt_loss, txt_conf.squeeze())
                        img_history.correctness_update(idx, img_loss, img_conf.squeeze())
                        audio_history.correctness_update(idx, audio_loss, audio_conf.squeeze())

                        txt_rank_loss = rank_loss(txt_conf, idx, txt_history)
                        img_rank_loss = rank_loss(img_conf, idx, img_history)
                        audio_rank_loss = rank_loss(audio_conf, idx, audio_history)

                        crl_loss = txt_rank_loss + img_rank_loss + audio_rank_loss
                        loss = torch.mean(clf_loss + crl_loss)
                    else:
                        out_a, out_v = model(token, padding_mask, image)
                        txt_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
                        img_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))

                        txt_conf = txt_energy / 10
                        img_conf = img_energy / 10
                        txt_conf = torch.reshape(txt_conf, (-1, 1))
                        img_conf = torch.reshape(img_conf, (-1, 1))
                        out = (out_a * txt_conf.detach() + out_v * img_conf.detach())

                        txt_clf_loss = nn.CrossEntropyLoss()(out_a, label)
                        img_clf_loss = nn.CrossEntropyLoss()(out_v, label)
                        clf_loss = txt_clf_loss + img_clf_loss

                        txt_loss = nn.CrossEntropyLoss(reduction='none')(out_a, label).detach()
                        img_loss = nn.CrossEntropyLoss(reduction='none')(out_v, label).detach()

                        txt_history.correctness_update(idx, txt_loss, txt_conf.squeeze())
                        img_history.correctness_update(idx, img_loss, img_conf.squeeze())

                        txt_rank_loss = rank_loss(txt_conf, idx, txt_history)
                        img_rank_loss = rank_loss(img_conf, idx, img_history)

                        crl_loss = txt_rank_loss + img_rank_loss
                        loss = torch.mean(clf_loss + crl_loss)

                else:
                    if args.modal3:
                        a, v, t = model(token, padding_mask, image, spec)
                        _,_,_,out = model.module.fusion_module(a, v, t)
                    else:
                        a, v = model(token, padding_mask, image)
                        _,_,out = model.module.fusion_module(a, v)
            else:
                if args.modulation == "QMF":
                    out_a, out_v = model(spec.unsqueeze(1).float(), image.float())

                    txt_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
                    img_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))

                    txt_conf = txt_energy / 10
                    img_conf = img_energy / 10
                    txt_conf = torch.reshape(txt_conf, (-1, 1))
                    img_conf = torch.reshape(img_conf, (-1, 1))
                    out = (out_a * txt_conf.detach() + out_v * img_conf.detach())

                    txt_clf_loss = nn.CrossEntropyLoss()(out_a, label)
                    img_clf_loss = nn.CrossEntropyLoss()(out_v, label)
                    clf_loss = txt_clf_loss + img_clf_loss

                    txt_loss = nn.CrossEntropyLoss(reduction='none')(out_a, label).detach()
                    img_loss = nn.CrossEntropyLoss(reduction='none')(out_v, label).detach()

                    txt_history.correctness_update(idx, txt_loss, txt_conf.squeeze())
                    img_history.correctness_update(idx, img_loss, img_conf.squeeze())

                    txt_rank_loss = rank_loss(txt_conf, idx, txt_history)
                    img_rank_loss = rank_loss(img_conf, idx, img_history)

                    crl_loss = txt_rank_loss + img_rank_loss
                    cml_loss = nn.CrossEntropyLoss()(out, label)
                    
                    # loss = 0.8 * cml_loss + 0.2 * torch.mean(clf_loss + crl_loss)
                    loss = cml_loss + clf_loss + 0.1 * crl_loss
                else:
                    if args.clip:
                        a, v, out = model(spec, image)
                    else:
                        a, v, out = model(spec.unsqueeze(1).float(), image.float())
            if args.modulation != "QMF":
                if args.modal3:
                    if args.fusion_method == 'sum':
                        out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                                model.module.fusion_module.fc_y.bias)
                        out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                                model.module.fusion_module.fc_x.bias)
                    else:
                        weight_size = model.module.fusion_module.fc_out.weight.size(1)
                        out_t = (torch.mm(t, torch.transpose(model.module.fusion_module.fc_out.weight[:, 2 * weight_size // 3:], 0, 1))
                                + model.module.fusion_module.fc_out.bias / 3)
                        out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 3:2 * weight_size // 3], 0, 1))
                                + model.module.fusion_module.fc_out.bias / 3)

                        out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 3], 0, 1))
                                + model.module.fusion_module.fc_out.bias / 3)
                else:
                    if args.fusion_method == 'sum':
                        out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                                model.module.fusion_module.fc_y.bias)
                        out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                                model.module.fusion_module.fc_x.bias)
                    else:
                        weight_size = model.module.fusion_module.fc_out.weight.size(1)
                        out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                                + model.module.fusion_module.fc_out.bias / 2)

                        out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                                + model.module.fusion_module.fc_out.bias / 2)

            if args.modulation != "QMF":
                loss = criterion(out, label)
            if args.modal3:
                loss_t = criterion(out_t, label)
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)
            loss.backward()

            if args.modulation == 'Normal' or args.modulation == "QMF":
                # no modulation, regular optimization
                pass
            else:
                if args.modal3:
                    # Modulation starts here !
                    score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
                    score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
                    score_t = sum([softmax(out_t)[i][label[i]] for i in range(out_t.size(0))])

                    ratio_v = score_v / (score_a + score_t)
                    ratio_a = score_a / (score_v + score_t)
                    ratio_t = score_t / (score_v + score_a)

                    if ratio_v > 1:
                        coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                        coeff_a = 1
                        coeff_t = 1
                    elif ratio_t > 1:
                        coeff_t = 1 - tanh(args.alpha * relu(ratio_t))
                        coeff_a = 1
                        coeff_v = 1
                    else:
                        coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                        coeff_v = 1
                        coeff_t = 1

                    if args.use_tensorboard:
                        iteration = epoch * len(dataloader) + batch_step
                        writer.add_scalar('data/ratio v', ratio_v, iteration)
                        writer.add_scalar('data/coefficient v', coeff_v, iteration)
                        writer.add_scalar('data/coefficient a', coeff_a, iteration)
                        writer.add_scalar('data/coefficient t', coeff_t, iteration)

                    if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
                        for name, parms in model.named_parameters():
                            if parms.grad is None:
                                continue
                            layer = str(name).split('.')[1]
                            if 'mae_a' in layer and len(parms.grad.size()) == 4:
                                if args.modulation == 'OGM_GE':
                                    parms.grad = parms.grad * coeff_a + \
                                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                                elif args.modulation == 'OGM':
                                    parms.grad *= coeff_a

                            if 'mae_v' in layer and len(parms.grad.size()) == 4:
                                if args.modulation == 'OGM_GE':
                                    parms.grad = parms.grad * coeff_v + \
                                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                                elif args.modulation == 'OGM':
                                    parms.grad *= coeff_v
                            if 'mae_t' in layer and len(parms.grad.size()) == 4:
                                if args.modulation == 'OGM_GE':
                                    parms.grad = parms.grad * coeff_t + \
                                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                                elif args.modulation == 'OGM':
                                    parms.grad *= coeff_t
                    else:
                        pass
                else:
                    score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
                    score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

                    ratio_v = score_v / score_a
                    ratio_a = 1 / ratio_v

                    if ratio_v > 1:
                        coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                        coeff_a = 1
                    else:
                        coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                        coeff_v = 1

                    if args.use_tensorboard:
                        iteration = epoch * len(dataloader) + batch_step
                        writer.add_scalar('data/ratio v', ratio_v, iteration)
                        writer.add_scalar('data/coefficient v', coeff_v, iteration)
                        writer.add_scalar('data/coefficient a', coeff_a, iteration)

                    if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
                        for name, parms in model.named_parameters():
                            layer = str(name).split('.')[1]

                            if 'audio' in layer and len(parms.grad.size()) == 4:
                                if args.modulation == 'OGM_GE':
                                    parms.grad = parms.grad * coeff_a + \
                                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                                elif args.modulation == 'OGM':
                                    parms.grad *= coeff_a

                            if 'visual' in layer and len(parms.grad.size()) == 4:
                                if args.modulation == 'OGM_GE':
                                    parms.grad = parms.grad * coeff_v + \
                                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                                elif args.modulation == 'OGM':
                                    parms.grad *= coeff_v
                    else:
                        pass

            optimizer.step()

            _loss += loss.item()
            _loss_a += loss_a.item()
            _loss_v += loss_v.item()
            if args.modal3:
                _loss_t += loss_t.item()
        elif gs_flag:
            if args.lorb == "large":
                a, v = model(spec, image)
            elif args.lorb == "m3ae":
                if args.modal3:
                    a, v, t = model(token, padding_mask, image, spec)
                else:
                    a, v = model(token, padding_mask, image)
            else:
                if args.clip:
                    a, v = model(spec, image)
                else:
                    a, v = model(spec.unsqueeze(1).float(), image.float())
            out_a = model.module.fusion_module.fc_out(a)
            
            loss_a = criterion(out_a, label)
            loss_a.backward()

            gs_plugin.before_update(model.module.fusion_module.fc_out, a, 
                                    batch_step, len_dataloader, gs_plugin.exp_count)
            optimizer.step()
            optimizer.zero_grad()

            gs_plugin.exp_count += 1
            
            out_v = model.module.fusion_module.fc_out(v)
            
            loss_v = criterion(out_v, label)
            loss_v.backward()

            gs_plugin.before_update(model.module.fusion_module.fc_out, v, 
                                    batch_step, len_dataloader, gs_plugin.exp_count)
            optimizer.step()
            optimizer.zero_grad()

            gs_plugin.exp_count += 1
            if args.modal3:
                out_t = model.module.fusion_module.fc_out(t)
                
                loss_t = criterion(out_t, label)
                loss_t.backward()

                gs_plugin.before_update(model.module.fusion_module.fc_out, t, 
                                        batch_step, len_dataloader, gs_plugin.exp_count)
                optimizer.step()
                optimizer.zero_grad()

                gs_plugin.exp_count += 1

            for n, p in model.named_parameters():
                if p.grad != None:
                    del p.grad

            _loss += (loss_a * av_alpha + loss_v * (1 - av_alpha)).item()
            _loss_a += loss_a.item()
            _loss_v += loss_v.item()
            if args.modal3:
                _loss_t += loss_t.item()

        else:
            print("MLA do not support this mode")
            exit(0)
    scheduler.step()
    if args.modal3:
        return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _loss_t / len(dataloader)    
    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)

def valid(args, model, device, dataloader, 
          gs_flag = False, av_alpha = 0.5, 
          a_alpha = 0.35, v_alpha = 0.25, t_alpha = 0.4):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'MVSA':
        n_classes = 3
    elif args.dataset == 'KineticSound':
        # Incomplete
        pass
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'Food101':
        n_classes = 101
    elif args.dataset == 'AVE':
        # Incomplete
        pass
    elif args.dataset == "CUB":
        # Incomplete
        pass
    elif args.dataset == "IEMOCAP":
        n_classes = 4
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]
        pred_result = []
        for step, data_packet in enumerate(dataloader):
            if args.lorb == "m3ae":
                if args.modal3:
                    token, padding_mask, image, spec, label, idx = data_packet
                    token = token.to(device)
                    padding_mask = padding_mask.to(device)
                    image = image.to(device)
                    spec = spec.to(device)
                    label = label.to(device)
                else:
                    token, padding_mask, image, label, idx = data_packet
                    token = token.to(device)
                    padding_mask = padding_mask.to(device)
                    image = image.to(device)
                    label = label.to(device)
            else:
                spec, image, label, idx = data_packet
                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)
            
            if not gs_flag:
                if args.lorb == "large":
                    a, v = model(spec, image)
                    _, _, out = model.module.fusion_module(a, v)
                elif args.lorb == "m3ae":
                    if args.modulation == "QMF":
                        if args.modal3:
                            out_a, out_v, out_t = model(token, padding_mask, image, spec)
                            audio_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
                            img_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))
                            txt_energy = torch.log(torch.sum(torch.exp(out_t), dim=1))

                            txt_conf = txt_energy / 10
                            img_conf = img_energy / 10
                            audio_conf = audio_energy / 10
                            txt_conf = torch.reshape(txt_conf, (-1, 1))
                            img_conf = torch.reshape(img_conf, (-1, 1))
                            audio_conf = torch.reshape(audio_conf, (-1, 1))
                            out = (out_a * audio_conf.detach() + out_v * img_conf.detach() + out_t * txt_conf.detach())
                        else:
                            out_a, out_v = model(token, padding_mask, image)
                            txt_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
                            img_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))

                            txt_conf = txt_energy / 10
                            img_conf = img_energy / 10
                            txt_conf = torch.reshape(txt_conf, (-1, 1))
                            img_conf = torch.reshape(img_conf, (-1, 1))
                            out = (out_a * txt_conf.detach() + out_v * img_conf.detach())
                    else:
                        if args.modal3:
                            a, v, t = model(token, padding_mask, image, spec)
                            _,_,_,out = model.module.fusion_module(a, v, t)
                        else:
                            a, v = model(token, padding_mask, image)
                            _, _, out = model.module.fusion_module(a, v)
                else:
                    if args.modulation == "QMF":
                        out_a, out_v = model(spec.unsqueeze(1).float(), image.float())

                        txt_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
                        img_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))

                        txt_conf = txt_energy / 10
                        img_conf = img_energy / 10
                        txt_conf = torch.reshape(txt_conf, (-1, 1))
                        img_conf = torch.reshape(img_conf, (-1, 1))
                        out = (out_a * txt_conf.detach() + out_v * img_conf.detach())
                    else:
                        if args.clip:
                            a, v, out = model(spec, image)
                        else:    
                            a, v, out = model(spec.unsqueeze(1).float(), image.float())

                if args.modulation != "QMF":
                    if args.modal3:
                        if args.fusion_method == 'sum':
                            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                                    model.module.fusion_module.fc_y.bias / 2)
                            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                                    model.module.fusion_module.fc_x.bias / 2)
                        else:
                            weight_size = model.module.fusion_module.fc_out.weight.size(1)
                            out_t = (torch.mm(t, torch.transpose(model.module.fusion_module.fc_out.weight[:, 2 * weight_size // 3:], 0, 1))
                                    + model.module.fusion_module.fc_out.bias / 3)
                            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 3:2 * weight_size // 3], 0, 1))
                                    + model.module.fusion_module.fc_out.bias / 3)

                            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 3], 0, 1))
                                    + model.module.fusion_module.fc_out.bias / 3)
                    else:
                        if args.fusion_method == 'sum':
                            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                                    model.module.fusion_module.fc_y.bias / 2)
                            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                                    model.module.fusion_module.fc_x.bias / 2)
                        else:
                            weight_size = model.module.fusion_module.fc_out.weight.size(1)
                            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, int(weight_size//2):], 0, 1)) +
                                    model.module.fusion_module.fc_out.bias / 2)
                            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :int(weight_size//2)], 0, 1)) +
                                    model.module.fusion_module.fc_out.bias / 2)
            
            elif gs_flag:
                if args.lorb == "large":
                    a, v = model(spec, image)
                elif args.lorb == "m3ae":
                    if args.modal3:
                        a, v, t = model(token, padding_mask, image, spec)
                    else:
                        a, v = model(token, padding_mask, image)
                else:
                    if args.clip:
                        a, v = model(spec, image)
                    else:
                        a, v= model(spec.unsqueeze(1).float(), image.float())
                    
                out_a = model.module.fusion_module.fc_out(a)
                out_v = model.module.fusion_module.fc_out(v)
                if args.modal3:
                    out_t = model.module.fusion_module.fc_out(t)
                if args.dynamic:
                    if args.modal3:
                        audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)
                        out = (out_a * audio_conf + out_v * img_conf + out_t * txt_conf)
                    else:
                        txt_conf, img_conf = calculate_gating_weights(out_a, out_v)
                        out = (out_a * txt_conf + out_v * img_conf)
                else:
                    if args.modal3:
                        out = a_alpha * out_a + v_alpha * out_v + t_alpha * out_t
                    else:
                        out = av_alpha * out_a + (1-av_alpha) * out_v

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)
            if args.modal3:
                pred_t = softmax(out_t)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                if args.modal3:
                    t = np.argmax(pred_t[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0
                if args.modal3:
                    if np.asarray(label[i].cpu()) == t:
                        acc_t[label[i]] += 1.0
    if args.modal3:
        return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), sum(acc_t) / sum(num)    
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
def wa_model(exp_dir):
    all_ckpts = os.listdir(exp_dir)
    sdA = torch.load(os.path.join(exp_dir, all_ckpts[0]), map_location='cpu')["model"]
    model_cnt = 1
    for epoch in range(1, len(all_ckpts)):
        sdB = torch.load(os.path.join(exp_dir, all_ckpts[epoch]), map_location='cpu')["model"]
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, 1, len(all_ckpts)))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA


def main(av_alpha = 0.5):
    args = get_arguments()
    # print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    
    if args.lorb == "large":
        model = CAVClassifier(args)
    elif args.lorb == "m3ae":
        if args.modal3:
            model = Modal3Classifier(args)
        else:
            model = M3AEClassifier(args)
    else:
        if args.clip:
            model = CLIPClassifier(args)
        else:
            model = AVClassifier(args)
            model.apply(weight_init)
    
    if args.ckpt_load_path_train:
        loaded_dict = torch.load(args.ckpt_load_path_train)
        state_dict = loaded_dict['model']
        state_dict = {key[7:]: state_dict[key] for key in state_dict}
        del state_dict["fusion_module.fc_out.weight"]
        del state_dict["fusion_module.fc_out.bias"]
        missing, unexcepted = model.load_state_dict(state_dict, strict = False)
        print('Trained model loaded!')
    
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    if args.lorb == "large" and args.cav_opti:
        # optimizer = optim.SGD(model.module.fusion_module.fc_out.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        mlp_list = ['fusion_module.fc_out.weight', 'module.fusion_module.fc_out.bias']
        mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.module.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.module.named_parameters()))
        mlp_params = [i[1] for i in mlp_params]
        base_params = [i[1] for i in base_params]
        optimizer = optim.Adam([{'params': base_params, 'lr': args.learning_rate / 10}, 
                                {'params': mlp_params, 'lr': args.learning_rate}],
                                weight_decay=5e-7, 
                                betas=(0.95, 0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 0.0, betas=(0.9, 0.999))
    if args.lorb == "large" and args.cav_lrs:
        args.lrscheduler_start = 2
        args.lrscheduler_step = 1
        args.lrscheduler_decay = 0.5
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
                                                    gamma = args.lrscheduler_decay)    

    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'KineticSound':
        # Incomplete
        pass
    elif args.dataset == 'MVSA':
        if args.lorb == "large":
            train_dataset = CAVDataset(args, mode='train')
            test_dataset = CAVDataset(args, mode='test')
        elif args.lorb == "m3ae":
            train_dataset = M3AEDataset(args, mode='train')
            test_dataset = M3AEDataset(args, mode='test')
        else:
            train_dataset = TVDataset(args, mode='train')
            test_dataset = TVDataset(args, mode='test')
    elif args.dataset == 'CUB':
        # Incomplete
        pass
    elif args.dataset == 'CREMAD':
        if args.lorb == "large":
            train_dataset = CAVDataset(args, mode='train')
            test_dataset = CAVDataset(args, mode='test')
        elif args.lorb == "m3ae":
            train_dataset = M3AEDataset(args, mode='train')
            test_dataset = M3AEDataset(args, mode='test')
        else:
            train_dataset = AVDataset(args, mode='train')
            test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'AVE':
        # Incomplete
        pass
    elif args.dataset == 'IEMOCAP':
        train_dataset = Modal3Dataset(args, mode='train')
        test_dataset = Modal3Dataset(args, mode='test')
    elif args.dataset == 'Food101':
        if args.clip:
            train_dataset = CLIPDataset(args, mode="train")
            test_dataset = CLIPDataset(args, mode="test")
        else:
            if args.lorb == "large":
                train_dataset = CAVDataset(args, mode='train')
                test_dataset = CAVDataset(args, mode='test')
            elif args.lorb == "m3ae":
                train_dataset = M3AEDataset(args, mode='train')
                test_dataset = M3AEDataset(args, mode='test')
            else:
                train_dataset = AVDataset(args, mode='train')
                test_dataset = AVDataset(args, mode='test')
    
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support Food-101, MVSA, and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)
    # GS Plugin
    gs = GSPlugin() if args.gs_flag else None

    if args.modulation == "QMF":
        txt_history = History(len(train_dataloader.dataset))
        img_history = History(len(train_dataloader.dataset))
        audio_history = History(len(train_dataloader.dataset))
    else:
        txt_history = None
        img_history = None
        audio_history = None
    
    if args.train:

        best_acc = 0.0
        if args.gs_flag:
            log_name = '{}_{}_{}'.format(args.fusion_method, "GS", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            log_name = '{}_{}_{}'.format(args.fusion_method, args.modulation, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset, log_name)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                writer = SummaryWriter(writer_path)

                if args.modal3:
                    batch_loss, batch_loss_a, batch_loss_v, batch_loss_t = train_epoch(args, epoch, model, device, 
                                                                     train_dataloader, optimizer,
                                                                     scheduler, gs_plugin = gs, 
                                                                     writer = writer, 
                                                                     gs_flag = args.gs_flag, 
                                                                     av_alpha = av_alpha,
                                                                     txt_history = txt_history,
                                                                     img_history = img_history,
                                                                     audio_history=audio_history)
                    acc, acc_a, acc_v, acc_t = valid(args, model, device, test_dataloader, 
                                            av_alpha= av_alpha, 
                                            gs_flag= args.gs_flag,
                                            a_alpha= args.a_alpha,
                                            v_alpha= args.v_alpha,
                                            t_alpha= args.t_alpha)

                    writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                                'Audio Loss': batch_loss_a,
                                                'Visual Loss': batch_loss_v,
                                                'Text Loss': batch_loss_t}, epoch)

                    writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                    'Audio Accuracy': acc_a,
                                                    'Visual Accuracy': acc_v,
                                                    'Text Accuracy': acc_t}, epoch)
                else:
                    batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device, 
                                                                        train_dataloader, optimizer,
                                                                        scheduler, gs_plugin = gs, 
                                                                        writer = writer, 
                                                                        gs_flag = args.gs_flag, 
                                                                        av_alpha = av_alpha,
                                                                        txt_history = txt_history,
                                                                        img_history = img_history)
                    acc, acc_a, acc_v = valid(args, model, device, test_dataloader, 
                                            av_alpha= av_alpha, 
                                            gs_flag= args.gs_flag)

                    writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                                'Audio Loss': batch_loss_a,
                                                'Visual Loss': batch_loss_v}, epoch)

                    writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                    'Audio Accuracy': acc_a,
                                                    'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.alpha,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                if args.modal3:
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f} ".format(acc_a, acc_v, acc_t))
                else:    
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                if args.modal3:
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f} ".format(acc_a, acc_v, acc_t))
                else:    
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))

    else:
        # if args.lorb == "large":
        #     state_dict = wa_model("ckpt/")
        # else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']

        missing, unexcepted = model.load_state_dict(state_dict)
        print('Trained model loaded!')
        
        if not args.modal3:
            acc, acc_a, acc_v = valid(args, model, device, 
                                      test_dataloader, args.ewc_flag, args.gs_flag, args.av_alpha)
            print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
        else:
            acc, acc_a, acc_v, acc_t = valid(args, model, device, test_dataloader, 
                                             args.ewc_flag, args.gs_flag, args.av_alpha,
                                             a_alpha= args.a_alpha, v_alpha= args.v_alpha, t_alpha= args.t_alpha)
            print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}, accuracy_t: {}'.format(acc, acc_a, acc_v, acc_t))


if __name__ == "__main__":
    main(av_alpha = 0.55)
