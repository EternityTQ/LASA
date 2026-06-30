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

# [新增辅助函数] 计算针对翻转标签的代理损失指导梯度
import copy
import torch

def compute_surrogate_guidance(net_glob, dataloader, device, num_of_label):
    #print("\n" + "="*60)
    #print("[MOS DEBUG] 🕵️‍♂️ 开始提取代理梯度，进入侦探模式...")
    safe_net = copy.deepcopy(net_glob).to(device)
    
    # ================= 探针 1：检查传入模型的初始健康度 =================
    has_nan_weight = any(torch.isnan(p).any().item() for p in safe_net.parameters())
    #print(f"[MOS DEBUG] 1. 刚拷贝来的模型权重是否已包含 NaN/Inf? : {has_nan_weight}")

    # 强行清洗权重
# ================= 终极沙箱清洗：包含 Weights 和 Buffers =================
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
    # =====================================================================

    safe_net.eval()
    safe_net.zero_grad()
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    # ================= 探针 2：检查输入数据 (很多人会忽略这里) =================

    target_labels = (num_of_label - labels).to(device) 
    
    # 1. 前向传播
    outputs = safe_net(images)
    
    # ================= 探针 3：检查前向传播结果 =================
    has_nan_out = torch.isnan(outputs).any().item()

    
    # 软截断保底
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)
    
    # -----------------------------------------
    # 代理目标 1：Cross Entropy (CE) 梯度
    # -----------------------------------------
    loss_ce = criterion_ce(outputs, target_labels)
    
    # ================= 探针 4：检查 CE Loss =================
    #print(f"[MOS DEBUG] 4. 计算得出的 loss_ce 数值: {loss_ce.item()}")
    
    loss_ce.backward(retain_graph=True)
    
    param_grads_ce = {name: param.grad.clone() for name, param in safe_net.named_parameters() if param.grad is not None}
    
    # ================= 探针 5：检查反向传播后的原始梯度 =================
    has_nan_gce_raw = any(torch.isnan(g).any().item() for g in param_grads_ce.values())
    #print(f"[MOS DEBUG] 5. backward() 之后，参数字典中的原生梯度是否含 NaN? : {has_nan_gce_raw}")
    
    g_ce_list = [param_grads_ce[k].flatten() if k in param_grads_ce else torch.zeros_like(safe_net.state_dict()[k]).flatten() for k in safe_net.state_dict().keys()]
    g_ce = -torch.cat(g_ce_list) 
    #print(f"[MOS DEBUG] 6. 最终拼接好的 g_ce 向量是否含 NaN? : {torch.isnan(g_ce).any().item()}")
    
    safe_net.zero_grad()
    
    # -----------------------------------------
    # 代理目标 2：Margin Loss (CW) 梯度
    # -----------------------------------------
    correct_logits = torch.gather(outputs, 1, target_labels.unsqueeze(1)).squeeze(1)
    outputs_clone = outputs.clone()
    outputs_clone.scatter_(1, target_labels.unsqueeze(1), -1e4)
    max_other_logits, _ = torch.max(outputs_clone, dim=1)
    
    loss_cw = torch.mean(torch.relu(max_other_logits - correct_logits + 10.0))
    #print(f"[MOS DEBUG] 7. 计算得出的 loss_cw 数值: {loss_cw.item()}")
    
    loss_cw.backward()
    
    param_grads_cw = {name: param.grad.clone() for name, param in safe_net.named_parameters() if param.grad is not None}
    g_cw_list = [param_grads_cw[k].flatten() if k in param_grads_cw else torch.zeros_like(safe_net.state_dict()[k]).flatten() for k in safe_net.state_dict().keys()]
    g_cw = -torch.cat(g_cw_list) 
    
    #print(f"[MOS DEBUG] 8. 最终拼接好的 g_cw 向量是否含 NaN? : {torch.isnan(g_cw).any().item()}")
    #print("="*60 + "\n")
    
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
            
        elif args.defend == 'lfd':
            global_model = lfd(local_updates, global_model, args)
            
        elif args.defend == 'msguard':
            aggregate_update, trusted_clients = msguard(local_updates, args, malicious_attackers_this_round)
            global_model = average(global_model, [aggregate_update])
            
        

        elif args.defend == 'fedavg':
            global_model = average(global_model, local_updates) # just fedavg
            
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
