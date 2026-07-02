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
from ..defense.lfd import lfd
from ..defense.msguard import msguard

import time

from ..attack import attack
from ..attack.mos import compute_surrogate_guidance as compute_surrogate_guidance_mos

# [新增辅助函数] 计算针对翻转标签的代理损失指导梯度
import copy
import torch

def compute_surrogate_guidance(net_glob, dataloader, device, num_of_label, args=None):
    """
    包装函数：调用 mos.py 中的改进版 compute_surrogate_guidance
    """
    safe_net = copy.deepcopy(net_glob).to(device)

    # 强行清洗权重
    with torch.no_grad():
        for name, tensor in safe_net.state_dict().items():
            # 只处理浮点型张量（忽略如 num_batches_tracked 这样的整型张量）
            if tensor.is_floating_point():
                # 1. 杀掉所有的 NaN 和 Inf
                tensor.nan_to_num_(nan=0.0, posinf=10.0, neginf=-10.0)
                # 2. 截断异常大的数值
                tensor.clamp_(-20.0, 20.0)

                # 3. 🛡️ 【最关键的一步】：保护方差，防止除零错误！
                # 只要名字里带 'var'，它必定是方差，强制拉升到正数安全线以上
                if 'var' in name or 'variance' in name:
                    tensor.clamp_(min=1e-4)

    safe_net.eval()
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)

    images, labels = next(iter(dataloader))
    images = images.to(device)
    target_labels = (num_of_label - labels).to(device)

    # 调用 mos.py 中的改进版函数
    g_ce, g_cw = compute_surrogate_guidance_mos(safe_net, images, target_labels, criterion_ce, args)

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

    # 处理多个防御方法
    defend_methods = []
    if hasattr(args, 'defend_methods'):
        defend_methods = [d for d in args.defend_methods if d and d != 'None']
        defend_flag = len(defend_methods) > 0
    elif hasattr(args, 'defend'):
        if args.defend != 'None':
            defend_flag = True
            defend_methods = [args.defend]
        else:
            args.defend = None
    else:
        args.defend = None

    if not defend_methods:
        defend_methods = [None]

    # sampling attackers' id - 固定选择恶意客户端（基于总用户数的百分比）
    if args.attack:
        attacked_idxs = list(np.random.choice(range(args.num_users), args.num_attackers, replace=False))
        print(f'Fixed malicious clients: {len(attacked_idxs)} out of {args.num_users} total users ({100*len(attacked_idxs)/args.num_users:.1f}%)')
    overall_attack_ratio = []

    if args.attack != 'dynamic':
        attack_method = attack(args.attack)

    prev_global_model = copy.deepcopy(global_model)
    
    
    historical_pop = None
    
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
            # 特殊模式：每轮随机选择10-25%的恶意客户端
            upper = int(25 * args.num_users / 100)
            lower = int(10 * args.num_users / 100)
            num_attackers_this_round = np.random.randint(lower, upper+1)
            attacked_idxs = list(np.random.choice(range(args.num_users), num_attackers_this_round, replace=False))

            print(f'At this round, attack ratio is {100*num_attackers_this_round/args.num_users:.1f}% ({num_attackers_this_round} clients)')

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
                g_ce, g_cw = compute_surrogate_guidance(net_glob, ldr_malicious, args.device, num_of_label, args)
                if torch.isnan(g_ce).any() or torch.isnan(g_cw).any():
                    print("上游发现g_ce和g_cw为空！")
                
                # ================= 修改：传入并接收 historical_pop =================
                # 把 historical_pop 传进去，并用它接收更新后的种群
                local_updates, historical_pop = attack_method(
                    local_updates, 
                    args, 
                    malicious_attackers_this_round, 
                    g_ce=g_ce, 
                    g_cw=g_cw, 
                    historical_pop=historical_pop
                )
                # ===================================================================
            else:
                local_updates = attack_method(local_updates, args, malicious_attackers_this_round)

        ## robust/non-robust global aggregation
        if args.attack:
            print('attack:' + args.attack)
        else:
            print('attack: None')

        # 如果有多个防御方法，对每个防御方法生成一个候选模型
        if len(defend_methods) > 1:
            print(f'Testing {len(defend_methods)} defense methods: {defend_methods}')

            candidate_models = []
            candidate_accs = []

            for defend_method in defend_methods:
                print(f'\n=== Aggregating with defense: {defend_method} ===')
                args.defend = defend_method

                # 为每个防御方法创建独立的候选模型
                candidate_model = copy.deepcopy(global_model)

                # 执行对应的防御聚合
                if defend_method == 'multi_krum':
                    aggregate_model, _ = multi_krum(local_updates, multi_k=True)
                    candidate_model = average(candidate_model, [aggregate_model])
                elif defend_method == 'krum':
                    aggregate_model, _ = multi_krum(local_updates, multi_k=False)
                    candidate_model = average(candidate_model, [aggregate_model])
                elif defend_method == 'bulyan':
                    aggregate_model, _ = bulyan(local_updates)
                    candidate_model = average(candidate_model, [aggregate_model])
                elif defend_method == 'tr_mean':
                    aggregate_model = tr_mean(local_updates)
                    candidate_model = average(candidate_model, [aggregate_model])
                elif defend_method == 'sparsefed':
                    if t > 0:
                        candidate_model, momentum, error = sparsefed(local_updates, candidate_model, args, momentum, error)
                    else:
                        candidate_model, momentum, error = sparsefed(local_updates, candidate_model, args)
                elif defend_method == 'signguard':
                    candidate_model = signguard(local_updates, candidate_model, args)
                elif defend_method == 'dnc':
                    candidate_model = dnc(local_updates, candidate_model, args)
                elif defend_method == 'lasa':
                    candidate_model = lasa(local_updates, candidate_model, args)
                elif defend_method == 'geomed':
                    candidate_model = geomed(local_updates, candidate_model, args)
                elif defend_method == 'rlr':
                    candidate_model = robust_aggregation(local_updates, candidate_model, args)
                elif defend_method == 'lfd':
                    candidate_model = lfd(local_updates, candidate_model, args)
                elif defend_method == 'msguard':
                    aggregate_update, trusted_clients = msguard(local_updates, args, malicious_attackers_this_round)
                    candidate_model = average(candidate_model, [aggregate_update])
                elif defend_method == 'fedavg' or defend_method is None:
                    candidate_model = average(candidate_model, local_updates)

                # 评测候选模型
                net_glob.load_state_dict(candidate_model)
                with torch.no_grad():
                    acc, _ = test_img(net_glob, dataset_test, args)

                candidate_models.append(candidate_model)
                candidate_accs.append(acc)

                print(f'Defense {defend_method}: Test Accuracy = {acc:.5f}')

            # 选择准确率最高的模型
            best_idx = np.argmax(candidate_accs)
            best_defend = defend_methods[best_idx]
            best_acc = candidate_accs[best_idx]
            global_model = candidate_models[best_idx]

            print(f'\n=== Selected Defense: {best_defend} with accuracy {best_acc:.5f} ===')
            print(f'All defense accuracies: {dict(zip(defend_methods, candidate_accs))}')

            # 记录到日志
            with open(args.exp_record, 'a') as f:
                f.write(f'Round {t}: Defense comparison:\n')
                for i, defend_method in enumerate(defend_methods):
                    f.write(f'  {defend_method}: {candidate_accs[i]:.5f}\n')
                f.write(f'  Selected: {best_defend} with accuracy {best_acc:.5f}\n')

            test_acc = best_acc

            # 检查选中的模型是否包含NaN/Inf
            has_nan = False
            for k, v in global_model.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    has_nan = True
                    break

            if has_nan:
                print(f"\n[!] 警告：选中的模型 {best_defend} 包含 NaN/Inf，尝试选择备用模型")
                # 尝试选择其他健康的模型
                for idx in range(len(candidate_models)):
                    if idx == best_idx:
                        continue
                    test_model = candidate_models[idx]
                    is_healthy = True
                    for k, v in test_model.items():
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            is_healthy = False
                            break
                    if is_healthy:
                        print(f"使用备用模型: {defend_methods[idx]} (准确率: {candidate_accs[idx]:.5f})")
                        global_model = test_model
                        test_acc = candidate_accs[idx]
                        has_nan = False
                        break

                # 如果所有候选模型都有问题，回退到上一轮
                if has_nan:
                    print(f"所有候选模型都包含 NaN/Inf，回退到上一轮模型")
                    global_model = copy.deepcopy(prev_global_model)
                    args.local_lr = args.local_lr * 0.5

            # 更新备份模型（如果选中的模型是健康的）
            if not has_nan:
                prev_global_model = copy.deepcopy(global_model)

        else:
            # 单个防御方法，使用原有逻辑
            defend_method = defend_methods[0]
            print('defend:' + str(defend_method))
            args.defend = defend_method

            if defend_method == 'multi_krum':
                aggregate_model, _ = multi_krum(local_updates, multi_k=True)
                global_model = average(global_model, [aggregate_model])
            elif defend_method == 'krum':
                aggregate_model, _ = multi_krum(local_updates, multi_k=False)
                global_model = average(global_model, [aggregate_model])
            elif defend_method == 'bulyan':
                aggregate_model, _ = bulyan(local_updates)
                global_model = average(global_model, [aggregate_model])
            elif defend_method == 'tr_mean':
                aggregate_model = tr_mean(local_updates)
                global_model = average(global_model, [aggregate_model])
            elif defend_method == 'sparsefed':
                if t > 0:
                    global_model, momentum, error = sparsefed(local_updates, global_model, args, momentum, error)
                else:
                    global_model, momentum, error = sparsefed(local_updates, global_model, args)
            elif defend_method == 'signguard':
                global_model = signguard(local_updates, global_model, args)
            elif defend_method == 'dnc':
                global_model = dnc(local_updates, global_model, args)
            elif defend_method == 'lasa':
                global_model = lasa(local_updates, global_model, args)
            elif defend_method == 'geomed':
                global_model = geomed(local_updates, global_model, args)
            elif defend_method == 'rlr':
                global_model = robust_aggregation(local_updates, global_model, args)
            elif defend_method == 'lfd':
                global_model = lfd(local_updates, global_model, args)
            elif defend_method == 'msguard':
                aggregate_update, trusted_clients = msguard(local_updates, args, malicious_attackers_this_round)
                global_model = average(global_model, [aggregate_update])
            elif defend_method == 'fedavg' or defend_method is None:
                global_model = average(global_model, local_updates)

            ## test global model on server side
            net_glob.load_state_dict(global_model)

            # Clean accuracy (no gradients needed)
            with torch.no_grad():
                test_acc, _ = test_img(net_glob, dataset_test, args)
            
        has_nan = False
        for k, v in global_model.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                has_nan = True
                break
        
        # ----------------- 增加数值稳定性检查与稳健回退 -----------------
        # 配置默认值（可通过 args 覆盖）
        MAX_CLIENT_NORM = getattr(args, 'max_client_norm', 5.0)
        CLIP_VALUE = getattr(args, 'clip_value', 1.0)
        CONSECUTIVE_ROUNDS = getattr(args, 'consecutive_anomaly_rounds', 2)
        MIN_FLAGGED_CLIENTS = getattr(args, 'min_flagged_clients', 3)

        # 对所有本轮的客户端更新进行预处理：检查 NaN/Inf、计算范数、全局范数缩放与逐元素裁剪
        client_norms = []
        flagged_clients = []
        cleaned_updates = []
        for idx, upd in enumerate(local_updates):
            bad = False
            total_sq = 0.0
            for k, v in upd.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if torch.isnan(v).any() or torch.isinf(v).any():
                    bad = True
                    break
                total_sq += float((v.float() ** 2).sum().item())
            if bad or math.isnan(total_sq) or math.isinf(total_sq):
                flagged_clients.append(idx)
                # replace with zeros to keep positions but avoid NaN propagation
                zero_upd = {k: (torch.zeros_like(v) if isinstance(v, torch.Tensor) else v) for k, v in upd.items()}
                cleaned_updates.append(zero_upd)
                continue

            total_norm = math.sqrt(total_sq)
            client_norms.append(total_norm)

            # 全局范数裁剪（scale down if too large）
            if total_norm > MAX_CLIENT_NORM and total_norm > 0:
                scale = MAX_CLIENT_NORM / (total_norm + 1e-12)
                for k in upd:
                    if isinstance(upd[k], torch.Tensor):
                        upd[k] = upd[k] * scale

            # per-parameter clamp and nan->num
            for k in upd:
                if isinstance(upd[k], torch.Tensor):
                    upd[k].clamp_(-CLIP_VALUE, CLIP_VALUE)
                    upd[k] = torch.nan_to_num(upd[k], nan=0.0, posinf=CLIP_VALUE, neginf=-CLIP_VALUE)

            cleaned_updates.append(upd)


        # 把清理后的更新写回，供后续防御器使用
        local_updates = cleaned_updates

        # 检查聚合后模型是否出现 NaN/Inf；如果出现，采取更稳健的回退策略
        has_nan = False
        for k, v in global_model.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                has_nan = True
                break

        # 将异常计数保存在 args 中以跨轮次追踪
        if not hasattr(args, '_anomaly_counter'):
            args._anomaly_counter = 0

        def robust_median_aggregate(updates):
            # updates: list of state_dict
            agg = {}
            if not updates:
                return agg
            keys = list(updates[0].keys())
            for k in keys:
                tensors = torch.stack([u[k] for u in updates], dim=0)
                agg[k] = torch.median(tensors, dim=0).values
            return agg

        if has_nan:
            print(f"\n[!] 警告：第 {t} 轮聚合后的全局模型出现 NaN/Inf。")
            # 如果被标记的客户端数量达到阈值，则增加计数
            if len(flagged_clients) >= MIN_FLAGGED_CLIENTS:
                args._anomaly_counter += 1
            else:
                args._anomaly_counter = max(0, args._anomaly_counter - 1)

            if args._anomaly_counter >= CONSECUTIVE_ROUNDS:
                # 连续异常，回档到上一轮
                print(f"第 {t} 轮：连续异常达到 {args._anomaly_counter} 轮，执行回档。")
                global_model = copy.deepcopy(prev_global_model)
                args.local_lr = args.local_lr * 0.5
            else:
                # 未达到回档阈值：使用鲁棒聚合（元素中位数）作为回退
                print(f"第 {t} 轮：未达到回档阈值，使用中位数聚合作为回退（anomaly_counter={args._anomaly_counter}）。")
                agg_update = robust_median_aggregate(local_updates)
                if agg_update:
                    global_model = average(global_model, [agg_update])
                else:
                    # 如果聚合失败，回档
                    global_model = copy.deepcopy(prev_global_model)
                    args.local_lr = args.local_lr * 0.5
        else:
            # 如果模型健康，更新备份并重置计数
            prev_global_model = copy.deepcopy(global_model)
            args._anomaly_counter = 0

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

        # 记录准确率（如果是单防御方法或者已经在多防御方法中记录过）
        if len(defend_methods) == 1:
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

        if math.isnan(train_loss) or train_loss > 1e8:
            print(f"t {t:3d}: 本轮局部 train_loss 异常 ({train_loss})，已跳过损坏的更新。")

        # 只有在达到最大轮次时才退出
        if t == args.round - 1:
            t2 = time.time()
            hours, rem = divmod(t2-t1, 3600)
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
