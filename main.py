import torch, random, argparse, os, copy
import numpy as np
from algorithms.engine.fedavg_all import fedavg_all
from mmengine.config import Config
import os
# 允许显存分配器使用扩展段来减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def merge_config(config, args):
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=int,
                        default=2,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="seed")
    parser.add_argument('--repeat', type=int, default=3, help='repeat index')
    parser.add_argument('--freeze_datasplit', type=int, default=0, help='freeze to save dict_users.pik or not')
    parser.add_argument('--sparsity', type=float, default=0.3, help='pre-defined sparsity')
    parser.add_argument('--num_attackers', type=int, default=20, help='Bayzatine attckers')
    parser.add_argument('--beta', type=float, default=0, help='ema')
    parser.add_argument('--attack', type=str, default='agrTailoredTrmean', help='attack method', choices=['agrTailoredTrmean', 'agrAgnosticMinMax', 'agrAgnosticMinSum', 'signflip_attack', 'noise_attack', \
                'random_attack', 'lie_attack', 'byzmean_attack', 'non_attack','mos_attack', 'skew_attack'])
    parser.add_argument('--defend1', type=str, default='lasa', help='primary defend method', choices=['fedavg', 'signguard', 'dnc', 'lasa', 'bulyan', 'tr_mean', 'multi_krum', 'sparsefed', 'geomed','rlr', 'lfd'])
    parser.add_argument('--defend2', type=str, default=None, help='secondary defend method (optional)', choices=[None, 'fedavg', 'signguard', 'dnc', 'lasa', 'bulyan', 'tr_mean', 'multi_krum', 'sparsefed', 'geomed','rlr', 'lfd'])
    parser.add_argument('--defend3', type=str, default=None, help='tertiary defend method (optional)', choices=[None, 'fedavg', 'signguard', 'dnc', 'lasa', 'bulyan', 'tr_mean', 'multi_krum', 'sparsefed', 'geomed','rlr', 'lfd'])
    parser.add_argument('--loss_mask', type=str, default='1111111', help='Binary string to select active losses in mos_attack, 7 bits: [l_ce, l_cw, l_magnitude, l_group, l_sign, l_pca, l_subspace]')
    parser.add_argument('--dataset', type=str, default='cifar', help='dataset')
    parser.add_argument('--lambda_n', type=float, default=1.0, help='reserver sparsity')
    parser.add_argument('--lambda_s', type=float, default=1.0, help='reserver sparsity')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet distribution alpha for data heterogeneity (smaller = more non-IID)')

    # MOS attack specific parameters
    parser.add_argument('--mos_conv_sparsity', type=float, default=0.3, help='Sparsity ratio for conv layers in MOS attack guidance gradient (0.0-1.0)')
    parser.add_argument('--mos_classifier_sparsity', type=float, default=1.0, help='Sparsity ratio for classifier layers in MOS attack guidance gradient (0.0-1.0)')
    parser.add_argument('--use_dnc_aware_mask', type=int, default=1, help='Enable DNC-aware mask when DNC defense is active (0: disabled, 1: enabled)')
    parser.add_argument('--enable_subspace_constraint', type=int, default=1, help='Enable subspace robustness constraint in MOS attack (0: disabled, 1: enabled)')
    parser.add_argument('--mos_inject_attack_ray_diagnostics', type=int, default=0,
                        help='Inject and track small-alpha g_attack ray candidates (0: disabled, 1: enabled)')
    parser.add_argument('--mos_adaptive_guided_init', type=int, default=0,
                        help='Scale MOS guidance seeds to the current constraint-feasible ray boundary')

    # NEW: Scoring system parameters (打分系统参数)
    parser.add_argument('--score_mode', type=str, default='sigmoid', choices=['sigmoid', 'relu', 'linear'],
                        help='Scoring function mapping mode for constraint evaluation (sigmoid: smooth, relu: fast, linear: piecewise)')
    parser.add_argument('--constraint_k_sigma', type=float, default=2.0,
                        help='Threshold coefficient for benign gradient constraint (threshold = mean + k * std, k=2.0 covers 95%% benign gradients)')

    meta_args = parser.parse_args()

    # 收集所有防御方法
    defend_methods = [meta_args.defend1]
    if meta_args.defend2:
        defend_methods.append(meta_args.defend2)
    if meta_args.defend3:
        defend_methods.append(meta_args.defend3)
    meta_args.defend_methods = defend_methods

    # 为了向后兼容，保留defend属性（使用第一个防御方法）
    meta_args.defend = meta_args.defend1

    if meta_args.dataset == 'sha':
        meta_args.dataset = 'shakespeare'

    if meta_args.dataset != 'noniidcifar' and meta_args.dataset != 'noniidcifar100':
        meta_args.alpha = -1

    meta_args.config_name = 'attack/%s/basee.yaml' % meta_args.dataset

    config_path = os.path.join('config/', meta_args.config_name)
    config = Config.fromfile(config_path)
    meta_args = merge_config(config, meta_args)

    # 使用所有防御方法来命名结果目录
    defend_str = '_'.join(defend_methods)
    meta_args.results_dir = './exp_results/%s/Attack_%s_Raito_%d/Defense_%s/' % (meta_args.dataset, str(meta_args.attack), meta_args.num_attackers, defend_str)

    # num_attackers 是百分比，转换为基于总用户数的实际数量
    meta_args.num_attackers = int(meta_args.num_attackers * meta_args.num_users / 100)

    if meta_args.defend == 'sparsefed':
        meta_args.com_p = 1 - meta_args.sparsity

    if meta_args.gpu != -1:
        meta_args.device = torch.device(f'cuda:{meta_args.gpu}')
    else:
        meta_args.device = torch.device('cpu')

    if not os.path.exists(meta_args.results_dir):
        os.makedirs(meta_args.results_dir)
    
    meta_args.exp_record = '%s/results.txt' % (meta_args.results_dir)

    # for reproducibility
    score_box = []
    poisoned_ratio_box = []
    for r in range(meta_args.repeat):
        args = copy.deepcopy(meta_args)
        print('############ Case '+ str(r) + ' ############')
        random.seed(args.seed+r)
        torch.manual_seed(args.seed+r)
        # torch.cuda.manual_seed(args.seed+args.repeat) # avoid
        np.random.seed(args.seed+r)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        best_result, poisoned_ratio = fedavg_all(args)
        score_box.append(best_result)
        poisoned_ratio_box.append(poisoned_ratio)
    print('repeated scores: ' + str(score_box))
    avg_score = np.average(score_box)
    print('avg of the scores ' + str(avg_score))
    
    print('repeated poisoned ratio: ' + str(poisoned_ratio_box))
    avg_poisoned_ratio = np.average(poisoned_ratio_box)
    print('avg of the poisoned ratios ' + str(avg_poisoned_ratio))

    if 'maskfeddp' in str(meta_args.config_name):
        sparsity = str(meta_args.sparsity)

    # String to write
    my_string = '---' * 10 + '\n' +\
                'dataset is: ' + str(meta_args.dataset) + ', ' + '\n' +\
                'attack is: ' + str(meta_args.attack) + ', ' + '\n' +\
                'defend methods: ' + str(meta_args.defend_methods) + ', ' + '\n' +\
                'DP: ' + str(getattr(args, 'use_dp', False)) + ', ' + '\n' +\
                'num_attackers is: ' + str(meta_args.num_attackers) + ', ' + '\n' +\
                'sparsity is: ' + str(meta_args.sparsity) + ', ' + '\n' +\
                'repeated scores: ' + str(score_box) + ', ' + '\n' +\
                'avg of the scores ' + str(avg_score) + ', ' + '\n' +\
                'repeated poisoned ratios ' + str(poisoned_ratio_box) + ', ' + '\n' +\
                'avg of the poisoned ratios ' + str(avg_poisoned_ratio) + ', ' + '\n' +\
                '---'* 10 + '\n'

    # 'config name is: ' + str(meta_args.config_name) + ', ' + '\n' +\

    # Open the file in write mode
    with open(meta_args.exp_record, 'a') as f:
        # Write the string to the file
        f.write(my_string)
