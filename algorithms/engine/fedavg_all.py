import copy
import numpy as np
import time, math
import torch
from torch.utils.data import DataLoader

from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from utils.mask_help import *
from test import test_img
try:
    from test import test_img_setapgd
    _HAS_SETAPGD_TEST = True
except Exception:
    _HAS_SETAPGD_TEST = False

from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average

from ..defense.byzantine_robust_aggregation import multi_krum, bulyan, tr_mean, geomed
from ..defense.sparsefed import sparsefed

from ..defense.lasa import lasa
from ..defense.signguard import signguard
from ..defense.dnc import dnc
from ..defense.rlr import robust_aggregation

import time

from ..attack import attack

# [新增辅助函数] 计算针对翻转标签的代理损失指导梯度
def compute_surrogate_guidance(net_glob, dataloader, device, num_of_label):
    net_glob.eval() # 使用 eval 模式提取精确梯度
    net_glob.zero_grad()
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    
    # 1. 抽取恶意客户端的一个 Batch 的真实数据
    images, labels = next(iter(dataloader))
    images = images.to(device)
    # 模拟 Label Flip 攻击（与你 local_solver 中的逻辑保持绝对一致）
    target_labels = (num_of_label - labels).to(device) 
    
    outputs = net_glob(images)
    
    # -----------------------------------------
    # 代理目标 1：Cross Entropy (CE) 梯度
    # -----------------------------------------
    loss_ce = criterion_ce(outputs, target_labels)
    loss_ce.backward(retain_graph=True)
    
    # 提取梯度，注意要和 state_dict 的 keys 严格对齐 (跳过诸如 running_mean 等无需梯度的 buffer)
    param_grads_ce = {name: param.grad.clone() for name, param in net_glob.named_parameters() if param.grad is not None}
    g_ce_list = []
    for k in net_glob.state_dict().keys():
        if k in param_grads_ce:
            g_ce_list.append(param_grads_ce[k].flatten())
        else:
            g_ce_list.append(torch.zeros_like(net_glob.state_dict()[k]).flatten())
            
    # 因为我们想要 *最小化* 翻转标签的 CE Loss，所以梯度更新方向应该是负梯度 (-gradient)
    g_ce = -torch.cat(g_ce_list) 
    
    net_glob.zero_grad()
    
    # -----------------------------------------
    # 代理目标 2：Margin Loss (CW) 梯度
    # -----------------------------------------
    correct_logits = torch.gather(outputs, 1, target_labels.unsqueeze(1)).squeeze(1)
    outputs_clone = outputs.clone()
    outputs_clone.scatter_(1, target_labels.unsqueeze(1), -1e4)
    max_other_logits, _ = torch.max(outputs_clone, dim=1)
    
    # CW Loss: 强迫目标类别的得分远超第二名
    loss_cw = torch.mean(torch.relu(max_other_logits - correct_logits + 50.0))
    loss_cw.backward()
    
    param_grads_cw = {name: param.grad.clone() for name, param in net_glob.named_parameters() if param.grad is not None}
    g_cw_list = []
    for k in net_glob.state_dict().keys():
        if k in param_grads_cw:
            g_cw_list.append(param_grads_cw[k].flatten())
        else:
            g_cw_list.append(torch.zeros_like(net_glob.state_dict()[k]).flatten())
            
    # 同理，想要 *最小化* CW Loss，更新方向为负梯度
    g_cw = -torch.cat(g_cw_list) 
    
    net_glob.zero_grad()
    
    return g_ce, g_cw

def fedavg_all(args):
    ################################### hyperparameter setup ########################################
    print("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60])
    # args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users = load_partition(args)
    args, dataset_train, dataset_test, dataset_val, _, dict_users = load_partition(args)
    print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test) + len(dataset_val)))
    print('num. of training data:{}'.format(len(dataset_train)))
    print('num. of testing data:{}'.format(len(dataset_test)))
    print('num. of validation data:{}'.format(len(dataset_val)))
    # print('num. of public data:{}'.format(len(dataset_public)))
    print('num. of users:{}'.format(len(dict_users)))

    sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users)) # max 525, min 3


    print('average num. of samples per user:{}'.format(sample_per_users))
    

    
    print("{:<50}".format("-" * 15 + " model setup " + "-" * 50)[0:60])
    args, net_glob, global_model, args.dim = model_setup(args)

    print('model dim:', args.dim)

    ###################################### model initialization ###########################
    t1 = time.time()
    train_loss, test_acc = [], []
    print("{:<50}".format("-" * 15 + " training... " + "-" * 50)[0:60])
    # initialize data loader for training and/or public dataset
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i])
        ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        data_loader_list.append(ldr_train)

    net_glob.train()

    best_test_accuracy = 0

    nsr = 0.0

    attack_flag = False
    defend_flag = False
    if hasattr(args, 'attack'):
        if args.attack != 'None':
            attack_flag = True
        else:
            args.attack = None
            args.num_attackers = 0
    else:
        args.attack = None
        args.num_attackers = 0
    
    if hasattr(args, 'defend'):
        if args.defend != 'None':
            defend_flag = True
        else:
            args.defend = None
    else:
        args.defend = None

    # sampling attackers' id
    if args.attack:
        attacked_idxs = list(np.random.choice(range(args.num_users), int(args.num_attackers/args.num_selected_users*args.num_users), replace=False))
    overall_attack_ratio = []

    if args.attack != 'dynamic':
        attack_method = attack(args.attack)

    for t in range(args.round):
        if args.attack:
            gt_attack_cnt = 0

        ## learning rate decaying
        if args.dataset == 'shakespeare' or args.dataset == 'femnist':
            if (t+1) % 10 == 0:
                args.local_lr = args.local_lr * args.decay_weight
        else:
            args.local_lr = args.local_lr * args.decay_weight

        if args.num_attackers == 99:
            upper = int(25 * args.num_selected_users / 100)
            args.num_attackers = np.random.randint(10, upper+1)
            attacked_idxs = list(np.random.choice(range(args.num_users), int(args.num_attackers/args.num_selected_users*args.num_users), replace=False))

            print('At this round, attack ratio is %s' % args.num_attackers)

        ############################################################# FedAvg ##########################################
        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), args.num_selected_users, replace=False))

        local_models, local_losses, local_updates, malicious_updates, delta_norms= [], [], [], [], []
        
        if args.dataset == 'shakespeare':
            num_of_label = 89
        elif args.dataset == 'femnist':
            num_of_label = 61
        else:
            num_of_label = 9

        local_solver = LocalUpdate(args=args)

        for i in selected_idxs:
            start = time.time()

            ################## <<< Attack Point 1: train with poisoned data
            net_glob.load_state_dict(global_model)
            
            if attack_flag and i in attacked_idxs:
                gt_attack_cnt += 1
                local_model, local_loss = local_solver.local_sgd_mome(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i], attack_flag=attack_flag, attack_method=args.attack, num_of_label=num_of_label)
            else:
                local_model, local_loss = local_solver.local_sgd_mome(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
            
            

            local_losses.append(local_loss)
            # compute model update
            model_update = {k: local_model[k] - global_model[k] for k in global_model.keys()}


            # compute model update norm
            end = time.time()

            # clipping local model 
            if defend_flag:
                if args.defend in ['sparsefed', 'tr_mean', 'krum', 'bulyan', 'fedavg', 'geomed'] and 'cifar' not in args.dataset:
                    delta_norm = torch.norm(torch.cat([torch.flatten(model_update[k]) for k in model_update.keys()]))
                    delta_norms.append(delta_norm)
                    threshold = delta_norm / args.clip
                    if threshold > 1.0:
                        for k in model_update.keys():
                            model_update[k] = model_update[k] / threshold
            # collecting local models
            # 32 bits * args.dim, {(index, param)}: k*32+log2(d); 32->4; 
            if attack_flag and i in attacked_idxs:
                malicious_updates.append(model_update)
            else:
                local_updates.append(model_update)

            #
        # calculate_sparsity(local_model)
        # add malicious update to the start of local updates
        malicious_attackers_this_round = len(malicious_updates)
        args.malicious_attackers_this_round = malicious_attackers_this_round
        if args.attack == 'non_attack':
            malicious_attackers_this_round = 0
        
        print('attack numbers = ' + str(malicious_attackers_this_round))
        local_updates = malicious_updates + local_updates
        # gt attack ratio
        if args.num_attackers > 0:
            gt_attack_ratio = gt_attack_cnt / args.num_selected_users
            print('current iteration attack ratio: '+str(gt_attack_ratio))
            overall_attack_ratio.append(gt_attack_ratio)

        train_loss = sum(local_losses) / args.num_selected_users

        

        ################## <<< Attack Point 2: local model poisoning attacks
        ################## <<< Attack Point 2: local model poisoning attacks
        if malicious_attackers_this_round != 0:
            if args.attack == 'mos_attack' or 'mos' in args.attack: # 请根据你实际传的 args.attack 名字修改
                # 随便找一个参与了本轮攻击的恶意客户端，拿他的数据生成指导梯度
                malicious_client_idx = [idx for idx in selected_idxs if idx in attacked_idxs][0]
                ldr_malicious = data_loader_list[malicious_client_idx]
                
                # 提取语义破坏的指导梯度
                g_ce, g_cw = compute_surrogate_guidance(net_glob, ldr_malicious, args.device, num_of_label)
                
                # 传给 MOS-Attack
                local_updates = attack_method(local_updates, args, malicious_attackers_this_round, g_ce=g_ce, g_cw=g_cw)
            else:
                local_updates = attack_method(local_updates, args, malicious_attackers_this_round)
        
        ## robust/non-robust global aggregation
        if args.attack:
            print('attack:' + args.attack)
        else:
            print('attack: None')

        if args.defend:
            print('defend:' + args.defend)
        else:
            print('defend: None')

        if args.defend == 'multi_krum':
            aggregate_model, _ = multi_krum(local_updates, multi_k=True)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'krum':
            aggregate_model, _ = multi_krum(local_updates, multi_k=False)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'bulyan':
            aggregate_model, _ = bulyan(local_updates)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'tr_mean':
            aggregate_model = tr_mean(local_updates)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'sparsefed':
            if t > 0:
                global_model, momentum, error = sparsefed(local_updates, global_model, args, momentum, error)
            else:
                global_model, momentum, error = sparsefed(local_updates, global_model, args)

        elif args.defend == 'signguard':
            global_model = signguard(local_updates, global_model, args)
        
        elif args.defend == 'dnc':
            global_model = dnc(local_updates, global_model, args)

        elif args.defend == 'lasa':
            global_model = lasa(local_updates, global_model, args)

        elif args.defend == 'geomed':
            global_model = geomed(local_updates, global_model, args)
            
        elif args.defend == 'rlr':
            global_model = robust_aggregation(local_updates, global_model, args)

        elif args.defend == 'fedavg':
            global_model = average(global_model, local_updates) # just fedavg

        ## test global model on server side
        net_glob.load_state_dict(global_model)

        # Clean accuracy (no gradients needed)
        with torch.no_grad():
            test_acc, _ = test_img(net_glob, dataset_test, args)

        # Optional: adversarial (SetAPGD) robust accuracy (needs gradients)
        robust_acc = None
        if getattr(args, 'eval_setapgd', 0) and _HAS_SETAPGD_TEST:
            robust_acc = test_img_setapgd(
                net_glob, dataset_test, args,
                eps=getattr(args, 'setapgd_eps', 8/255),
                steps=getattr(args, 'setapgd_steps', 50),
                K=getattr(args, 'setapgd_K', 5),
                norm=getattr(args, 'setapgd_norm', 'Linf'),
                loss_num=getattr(args, 'setapgd_loss_num', 8),
                n_restarts=getattr(args, 'setapgd_restarts', 1),
            )

        with open(args.exp_record, 'a') as f:
            msg = 'At round %d: the global model accuracy is %.5f' % (t, test_acc)
            if robust_acc is not None:
                msg += ' | SetAPGD robust acc: %.5f' % (robust_acc)
            f.write(msg + '\n')

            if t == args.round - 1:
                f.write('-----' + '\n')
        if robust_acc is None:
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                  format(t, train_loss, test_acc))
        else:
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f} | SetAPGD robust_acc = {:.3f}'.
                  format(t, train_loss, test_acc, robust_acc))
        
        if best_test_accuracy < test_acc:
            best_test_accuracy = test_acc

        if math.isnan(train_loss) or train_loss > 1e8 or t == args.round - 1:
            t2 = time.time()
            hours, rem = divmod(t2-t1, 3600)
            minutes, seconds = divmod(rem, 60)
            print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            print("best test accuracy ", best_test_accuracy)
            if len(overall_attack_ratio) > 0:
                print("overall poisoned ratio ", str(np.average(overall_attack_ratio)))
                return best_test_accuracy, np.average(overall_attack_ratio)
            else:
                return best_test_accuracy, 0
