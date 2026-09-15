
import torch
import numpy as np
from utils.mask_help import *



def topk(vector, args):
    '''
    return the mask for topk of vector
    '''
    k_dim = int(args.com_p * args.dim)
    
    mask = torch.zeros_like(vector)
    # flat_abs = abs(flat)
    _, indices = torch.topk(vector**2, k_dim)
    # generate a mask, set topk as 1, otherwise 0
    mask[indices] = 1
    # mask = {k: mask[s:d].reshape(model[k].shape) for k, (s, d) in zip(model.keys(), idx)}
    return mask, mask*vector


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        # if key.split('.')[-1] == 'num_batches_tracked':
        #     continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_net_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the net parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict


def _clone_updates(local_updates):
    return [
        {key: value.detach().clone() for key, value in update.items()}
        for update in local_updates
    ]


def _lasa_layer_audit(local_updates, args, candidate_index=None):
    """Run LASA's real preprocessing/statistics and audit one client.

    This helper intentionally mirrors ``lasa`` (global norm clipping, per-client
    top-k sparsification, then layer-wise median/std norm and sign filters).  It
    operates on private copies and returns the processed updates as well as the
    exact layer membership decisions needed by shadow diagnostics.
    """
    updates = _clone_updates(local_updates)
    finite_original_indices = []
    finite_updates = []
    for original_index, update in enumerate(updates):
        vector = parameters_dict_to_vector_flt(update)
        if not vector.isnan().any():
            finite_original_indices.append(original_index)
            finite_updates.append(update)
    if not finite_updates:
        raise ValueError("LASA received no finite client updates")
    audited_index = None
    if candidate_index is not None:
        try:
            audited_index = finite_original_indices.index(candidate_index)
        except ValueError as exc:
            raise ValueError("audited LASA candidate is non-finite") from exc

    flat = torch.stack([parameters_dict_to_vector_flt(update) for update in finite_updates])
    grad_norm = torch.norm(flat, dim=1).reshape((-1, 1))
    norm_clip = grad_norm.median(dim=0)[0].item()
    clipped_norm = torch.clamp(grad_norm, 0, norm_clip)
    clipped = (flat / grad_norm) * clipped_norm
    clipped_updates = [vector_to_net_dict(clipped[i], finite_updates[i])
                       for i in range(len(finite_updates))]

    for update in finite_updates:
        mask = generate_init_mask(update)
        mask = update_mask(update, mask, args.sparsity)
        apply_mask(update, mask)

    layer_audit = []
    all_set = set(range(len(finite_updates)))
    for key in finite_updates[0].keys():
        if 'num_batches_tracked' in key:
            continue
        grads = torch.stack([update[key].flatten() for update in finite_updates])
        raw_norm = torch.norm(grads.float(), dim=1).cpu().numpy()
        norm_med, norm_std = np.median(raw_norm), np.std(raw_norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_mz = np.abs((raw_norm - norm_med) / norm_std)
        norm_pass = set(int(i) for i in np.argwhere(norm_mz < args.lambda_n).reshape(-1))

        raw_sign = []
        for update in finite_updates:
            signs = torch.sign(update[key])
            denominator = torch.sum(torch.abs(signs))
            raw_sign.append((0.5 * (1 + torch.sum(signs) / denominator
                                    * (1 - args.sparsity))).item())
        sign_med, sign_std = np.median(raw_sign), np.std(raw_sign)
        with np.errstate(divide='ignore', invalid='ignore'):
            sign_mz = np.abs((np.asarray(raw_sign) - sign_med) / sign_std)
        sign_pass = set(int(i) for i in np.argwhere(sign_mz < args.lambda_s).reshape(-1))
        joint = norm_pass.intersection(sign_pass)
        aggregation_indices = joint if joint else all_set
        layer_audit.append({
            'layer': key,
            'norm_mz': None if audited_index is None else float(norm_mz[audited_index]),
            'sign_mz': None if audited_index is None else float(sign_mz[audited_index]),
            'norm_pass': None if audited_index is None else audited_index in norm_pass,
            'sign_pass': None if audited_index is None else audited_index in sign_pass,
            'joint_pass': None if audited_index is None else audited_index in joint,
            'aggregation_includes_candidate': (
                None if audited_index is None else audited_index in aggregation_indices),
            'benign_indices': sorted(aggregation_indices),
        })
    return finite_updates, clipped_updates, layer_audit


def audit_candidate_under_lasa(candidate, local_updates, global_model, args,
                               candidate_index=0):
    """Read-only audit of a candidate under the implementation's LASA rules."""
    del global_model  # Included in the stable public signature; LASA filters updates only.
    updates = _clone_updates(local_updates)
    updates[candidate_index] = {
        key: value.detach().clone() for key, value in candidate.items()
    }
    _, _, layers = _lasa_layer_audit(updates, args, candidate_index)
    count = max(len(layers), 1)
    sign_values = np.asarray([layer['sign_mz'] for layer in layers], dtype=float)
    finite_sign = sign_values[np.isfinite(sign_values)]
    stats = {
        'lasa_sign_pass_fraction': sum(layer['sign_pass'] for layer in layers) / count,
        'lasa_norm_pass_fraction': sum(layer['norm_pass'] for layer in layers) / count,
        'lasa_joint_pass_fraction': sum(layer['joint_pass'] for layer in layers) / count,
        'lasa_aggregation_include_fraction': sum(
            layer['aggregation_includes_candidate'] for layer in layers) / count,
        'lasa_mean_sign_mz': float(np.mean(finite_sign)) if finite_sign.size else float('nan'),
        'lasa_p90_sign_mz': float(np.quantile(finite_sign, 0.9)) if finite_sign.size else float('nan'),
        'lasa_max_sign_mz': float(np.max(finite_sign)) if finite_sign.size else float('nan'),
        'lasa_layer_audit': layers,
    }
    return stats


def _lasa_legacy(local_updates, global_model, args):
    ###########################
    ########## local ##########
    ###########################

    local_updates_ = []
    for i in range(len(local_updates)):
        vector = parameters_dict_to_vector_flt(local_updates[i])
        if vector.isnan().any():
            continue
        local_updates_.append(local_updates[i])

    local_updates = local_updates_

    flat_local_updates = []

    for i in range(len(local_updates)):
        vector = parameters_dict_to_vector_flt(local_updates[i])
        if vector.isnan().any():
            continue
        flat_local_updates.append(vector)

    flat_all_grads = torch.stack(flat_local_updates, dim=0)
    grad_norm = torch.norm(flat_all_grads, dim=1).reshape((-1, 1))
    norm_clip = grad_norm.median(dim=0)[0].item()
    grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
    grads_clip = (flat_all_grads/grad_norm)*grad_norm_clipped

    del grad_norm, norm_clip, grad_norm_clipped

    clipped_local_updates = []

    for i in range(len(local_updates)):
        net = vector_to_net_dict(grads_clip[i], local_updates[i])
        clipped_local_updates.append(net)

    # Pre-aggregation sparsification
    for i in range(len(local_updates)):
        global_mask = generate_init_mask(local_updates[i])
        global_mask = update_mask(local_updates[i], global_mask, args.sparsity)
        local_updates[i] = apply_mask(local_updates[i], global_mask)


    key_mean_weight = {}
    for key in local_updates[0].keys():
        if 'num_batches_tracked' in key:
            continue
        key_flat_para = []
        all_set = set([i for i in range(args.num_selected_users)])
        for param in local_updates:
            flat_param = param[key].flatten()
            # print(flat_param.numel())
            key_flat_para.append(flat_param)
        grads = torch.stack(key_flat_para, dim=0)


        # Norm check
        grad_l2norm = torch.norm(grads.float(), dim=1).cpu().numpy()
        norm_med = np.median(grad_l2norm)
        norm_std = np.std(grad_l2norm)

        # Calcualte MZ-score
        for i in range(len(grad_l2norm)):
            grad_l2norm[i] = np.abs((grad_l2norm[i] - norm_med) / norm_std)

        benign_idx1 = all_set.copy()
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < args.lambda_n)]))


        ##################
        # Sign check
        layer_sign = []
        for i in range(len(local_updates)):
            layer_sign.append(0.5 * (1 + torch.sum(torch.sign(local_updates[i][key])) / torch.sum(torch.abs(torch.sign(local_updates[i][key]))) * (1 - args.sparsity)).item())

        benign_idx2 = all_set.copy()
        if len(layer_sign) > 0:
            median = np.median(layer_sign)
            std = np.std(layer_sign)

            # Calcualte MZ-score
            for i in range(len(layer_sign)):
                layer_sign[i] = np.abs((layer_sign[i] - median) / std)
            benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(torch.tensor(layer_sign).cpu().numpy() < args.lambda_s)]))

        
        benign_idx = list(benign_idx2.intersection(benign_idx1))
        
        if len(benign_idx) == 0:
            benign_idx = list(all_set)
        #print(f"DEBUG: 幸存的客户端索引: {benign_idx}")
        # Layer-wise adaptive aggregation
        key_mean_weight[key] = torch.mean(torch.stack([clipped_local_updates[i][key] for i in benign_idx], dim=0), dim=0)

    for key in key_mean_weight.keys():
        if 'num_batches_tracked' in key:
            continue
        global_model[key].data += key_mean_weight[key].data
    

    return global_model


def lasa(local_updates, global_model, args):
    """LASA aggregation using the same layer decisions exposed by the audit."""
    _, clipped_updates, layers = _lasa_layer_audit(
        local_updates, args, candidate_index=None)
    for layer in layers:
        key = layer['layer']
        indices = layer['benign_indices']
        mean_update = torch.mean(torch.stack(
            [clipped_updates[i][key] for i in indices], dim=0), dim=0)
        global_model[key].data += mean_update.data
    return global_model
