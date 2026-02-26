"""
BFA (Bit-Flip Attack) for quantized models.
This module is specifically designed for models using pth.quantization_nm.
"""

import random
import torch
# Import from pth.quantization_nm instead of models.quantization
from pth.quantization_nm import quan_Conv2d, quan_Linear, quantize
import operator
from attack.data_conversion import *


class BFA(object):
    def __init__(self, criterion, model, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        
        # attributes for random attack: detect quantized layers by duck-typing
        self.module_list = []
        for name, m in model.named_modules():
            if hasattr(m, 'b_w') and hasattr(m, 'N_bits') and hasattr(m, 'weight'):
                self.module_list.append(name)

    def flip_bit(self, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        if self.k_top is None:
            k_top = m.weight.detach().flatten().__len__()
        else:
            k_top = self.k_top
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,k_top).short()) \
        // m.b_w.abs().repeat(1,k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        # only allow flipping up to self.n_bits2flip bits (typically 1)
        n_to_flip = max(1, int(self.n_bits2flip))
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(n_to_flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
            ^ w_bin_topk

        # 7. update the weight in the original weight tensor
        # keep a copy of the bin before modification for logging
        w_bin_before = w_bin.clone()
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        param_flipped = bin2int(w_bin,
                    m.N_bits).view(m.weight.data.size()).float()

        # return both the flipped parameter and binary info for profiling
        return param_flipped, {
            'w_bin_before': w_bin_before,
            'w_bin_after': w_bin,
            'w_idx_topk': w_idx_topk
        }

    def progressive_bit_search(self, model, data, target):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data)
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target)
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # collect quantized modules to attack; if none exist, provide clear error
        quant_modules = [(name, module) for name, module in model.named_modules()
                 if hasattr(module, 'b_w') and hasattr(module, 'N_bits') and hasattr(module, 'weight')]
        if len(quant_modules) == 0:
            raise RuntimeError("No quantized Conv/Linear modules found in model for BFA attack")

        # 3. For PBS we only flip one bit per iteration to ensure hamming_dist == 1.
        # Set number of bits to flip to 1 for this progressive search.
        self.n_bits2flip = 1

        # iterate all the quantized conv and linear layer once, computing candidate
        # single-bit flips and the resulting loss; choose the module that maximizes loss.
        for name, module in quant_modules:
            clean_weight = module.weight.data.detach()
            # flip_bit now returns (param, bin_info)
            attack_ret = self.flip_bit(module)
            if isinstance(attack_ret, tuple):
                attack_weight, _bin_info = attack_ret
            else:
                attack_weight = attack_ret
                _bin_info = None

            # change the weight to attacked weight and get loss
            module.weight.data = attack_weight
            output = model(data)
            self.loss_dict[name] = self.criterion(output, target).item()
            # change the weight back to the clean weight
            module.weight.data = clean_weight

        if len(self.loss_dict) == 0:
            grad_stats = {}
            for name, module in quant_modules:
                if module.weight.grad is None:
                    grad_stats[name] = 0.0
                else:
                    grad_stats[name] = float(module.weight.grad.abs().sum().item())
            raise RuntimeError(f"BFA failed: no loss entries computed for any quantized module. Grad sums: {grad_stats}")

        # pick the module that resulted in the max loss when a single-bit flip is applied
        max_loss_module = max(self.loss_dict.items(), key=operator.itemgetter(1))[0]
        # update loss_max to the maximum loss after attack
        self.loss_max = self.loss_dict[max_loss_module]

        # 4. apply the chosen single-bit flip and produce detailed INT8 profiling
        attack_log = []
        for module_idx, (name, module) in enumerate(model.named_modules()):
            if name == max_loss_module:
                clean_weight = module.weight.data.detach()
                attack_weight, bin_info = self.flip_bit(module)

                # bin_info contains bin before/after and indices of topk weights
                w_bin_before = bin_info.get('w_bin_before')
                w_bin_after = bin_info.get('w_bin_after')
                w_idx_topk = bin_info.get('w_idx_topk')

                # compute where the integer representation changed (indices into flattened weights)
                changed_idx_flat = torch.nonzero((w_bin_before != w_bin_after)).view(-1)

                # For each changed weight (should be small; typical PBS: 1)
                print('attacked module:', max_loss_module)
                for weight_flat_idx_tensor in changed_idx_flat:
                    weight_flat_idx = int(weight_flat_idx_tensor.item())

                    # integer representation before/after (signed via two's complement)
                    int_before_signed = int(bin2int(w_bin_before[weight_flat_idx], module.N_bits).item())
                    int_after_signed = int(bin2int(w_bin_after[weight_flat_idx], module.N_bits).item())
                    int_diff = int_after_signed - int_before_signed

                    # compute hamming distance via xor popcount on unsigned representation
                    unsigned_before = int(w_bin_before[weight_flat_idx].item())
                    unsigned_after = int(w_bin_after[weight_flat_idx].item())
                    xor = unsigned_before ^ unsigned_after
                    try:
                        hamming_dist = xor.bit_count()
                    except AttributeError:
                        # fallback for older Python: use bin()
                        hamming_dist = bin(xor).count('1')

                    # get multi-dimensional weight index for printing
                    weight_shape = list(module.weight.data.size())
                    idx_tuple = []
                    rem = weight_flat_idx
                    for dim in reversed(weight_shape):
                        idx_tuple.append(rem % dim)
                        rem = rem // dim
                    idx_tuple = tuple(reversed(idx_tuple))

                    # compute float values according to module's quantization step
                    quant_step = float(module.step_size.item()) if hasattr(module, 'step_size') else None
                    weight_prior_int = int_before_signed
                    weight_post_int = int_after_signed
                    weight_prior_float_by_step = weight_prior_int * quant_step if quant_step is not None else clean_weight[idx_tuple].item()
                    weight_post_float_by_step = weight_post_int * quant_step if quant_step is not None else attack_weight[idx_tuple].item()

                    print('attacked weight index:', idx_tuple)
                    print('INT before (signed):', weight_prior_int)
                    print('INT after  (signed):', weight_post_int)
                    print('weight before (float, int*step):', weight_prior_float_by_step)
                    print('weight after  (float, int*step):', weight_post_float_by_step)
                    print('quant step (module.step_size):', quant_step)
                    print('int diff (int_after - int_before):', int_diff)
                    print('hamming distance:', hamming_dist)

                    tmp_list = [module_idx,
                                self.bit_counter + 1,
                                max_loss_module,
                                idx_tuple,
                                weight_prior_float_by_step,
                                weight_post_float_by_step,
                                weight_prior_int,
                                weight_post_int,
                                int_diff,
                                hamming_dist,
                                quant_step]
                    attack_log.append(tmp_list)

                # finally set the attacked weight into the model
                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return attack_log


    def random_flip_one_bit(self, model):
        """
        Note that, the random bit-flip may not support on binary weight quantization.
        """
        chosen_module = random.choice(self.module_list)
        for name, m in model.named_modules():
            if name == chosen_module:
                flatten_weight = m.weight.detach().view(-1)
                chosen_idx = random.choice(range(flatten_weight.__len__()))
                # convert the chosen weight to 2's complement
                bin_w = int2bin(flatten_weight[chosen_idx], m.N_bits).short()
                # randomly select one bit
                bit_idx = random.choice(range(m.N_bits))
                mask = (bin_w.clone().zero_() + 1) * (2**bit_idx)
                bin_w = bin_w ^ mask
                int_w = bin2int(bin_w, m.N_bits).float()
                
                ##############################################
                ###   attack profiling
                ###############################################
                
                weight_mismatch = flatten_weight[chosen_idx] - int_w
                attack_weight_idx = chosen_idx
                
                print('attacked module:', chosen_module)
                
                attack_log = [] # init an empty list for profile
                
                
                weight_idx = chosen_idx
                weight_prior = flatten_weight[chosen_idx]
                weight_post = int_w

                print('attacked weight index:', weight_idx)
                print('weight before random attack:', weight_prior)
                print('weight after random attack:', weight_post)  
                
                tmp_list = ["module_idx", # module index in the net
                            self.bit_counter + 1, # current bit-flip index
                            "loss", # current bit-flip module
                            weight_idx, # attacked weight index in weight tensor
                            weight_prior, # weight magnitude before attack
                            weight_post # weight magnitude after attack
                            ] 
                attack_log.append(tmp_list)                            
                
                self.bit_counter += 1
                #################################
                
                flatten_weight[chosen_idx] = int_w
                m.weight.data = flatten_weight.view(m.weight.data.size())
                
            
                
                
        return attack_log
