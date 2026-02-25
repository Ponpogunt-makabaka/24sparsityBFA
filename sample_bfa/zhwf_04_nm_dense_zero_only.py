import os
import copy
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet20_nm import resnet20_nm
from sparse_ops import SparseConv, SparseLinear
import torch.nn as nn
import argparse

def forward_and_compute_gradients(
	model,
	data_loader,
	criterion,
	attack_sample_size: int = 128,
	attack_batch_size: int = 128,
	device: str = "cuda",
):
	device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
	model = model.to(device)
	model.eval() 
	model.zero_grad()

	all_imgs = []
	all_tgts = []
	processed = 0
	for images, targets in data_loader:
		if processed >= attack_sample_size:
			break
		rem = attack_sample_size - processed
		all_imgs.append(images[:rem])
		all_tgts.append(targets[:rem])
		processed += all_imgs[-1].size(0)
	
	full_images = torch.cat(all_imgs).to(device)
	full_targets = torch.cat(all_tgts).to(device)

	outputs = model(full_images)
	loss = criterion(outputs, full_targets)
	loss.backward()
	
	acc_loss = loss.item()

	grads = {}
	for name, p in model.named_parameters():
		if p.grad is None:
			continue
		if "step_size" in name:
			continue
		grads[name] = p.grad.detach().cpu().clone()

	return acc_loss, grads

def load_and_quantize_ckpt(ckpt_path="1_sparse_finetune.pth", data_root="~/data/cifar10", batch_size=128, device="cuda"):
	device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
	data_root = os.path.expanduser(data_root)

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])
	loader = DataLoader(datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

	model = resnet20_nm()
	if not os.path.isfile(ckpt_path):
		ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)

	ckpt = torch.load(ckpt_path, map_location="cpu")
	if isinstance(ckpt, dict):
		for key in ("state_dict", "model_state_dict", "model", "model_state"):
			if key in ckpt:
				state_dict = ckpt[key]
				break
		else:
			state_dict = ckpt
	else:
		state_dict = ckpt

	state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
	model.load_state_dict(state_dict)

	model_fp = model.to(device).eval()
	correct = total = 0
	with torch.no_grad():
		for images, targets in loader:
			images, targets = images.to(device), targets.to(device)
			preds = model_fp(images).argmax(1)
			correct += preds.eq(targets).sum().item()
			total += targets.size(0)
	fp32_top1 = 100.0 * correct / total if total else 0.0

	model_q = resnet20_nm()
	model_q.load_state_dict(state_dict)

	layer_info = {}
	quant_types = (torch.nn.Conv2d, torch.nn.Linear, SparseConv, SparseLinear)
	for name, module in model_q.named_modules():
		if isinstance(module, quant_types) and hasattr(module, 'weight') and module.weight is not None:
			w = module.weight.data.cpu()
			max_abs = float(w.abs().max().item())
			step = max_abs / 127.0 if max_abs > 0 else 1e-8
			q = torch.clamp(torch.round(w / step), -127, 127).to(torch.int8)
			module.weight.data = (q.float() * step).to(module.weight.data.device)
			layer_name = name if name else module.__class__.__name__
			layer_info[layer_name] = {'step_size': float(step), 'quantized_int8': q.cpu()}

	model_q = model_q.to(device).eval()
	correct_q = total_q = 0
	with torch.no_grad():
		for images, targets in loader:
			images, targets = images.to(device), targets.to(device)
			preds = model_q(images).argmax(1)
			correct_q += preds.eq(targets).sum().item()
			total_q += targets.size(0)
	int8_top1 = 100.0 * correct_q / total_q if total_q else 0.0

	print(f"Loaded checkpoint: {ckpt_path}\nDevice: {device}\nSamples: {total}\nFP32 Top-1: {fp32_top1:.2f}%\nINT8 Top-1: {int8_top1:.2f}%")

	return fp32_top1, int8_top1, layer_info

if __name__ == "__main__":
	fp, iq, info = load_and_quantize_ckpt()

	# report quantized sparsity for all attackable layers (layer1/2/3, excluding downsample)
	attack_layers = [k for k in info.keys() if (k.startswith('layer1.') or k.startswith('layer2.') or k.startswith('layer3.')) and ('downsample' not in k)]
	if len(attack_layers) == 0:
		attack_layers = [k for k in info.keys() if k.startswith('layer1.')] if any(k.startswith('layer1.') for k in info.keys()) else list(info.keys())
	print('Quantized sparsity for attack layers:')
	for lm in attack_layers:
		q = info[lm]['quantized_int8']
		zeros = int((q == 0).sum().item())
		total = int(q.numel())
		sparsity = 100.0 * zeros / total if total > 0 else 0.0
		print(f" - {lm}: {sparsity:.2f}% ({zeros}/{total} zeros)")

	attack_sample_size = 128
	attack_batch_size = 128

	train_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32, padding=4),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])
	trainset = datasets.CIFAR10(root=os.path.expanduser("~/data/cifar10"), train=True, download=False, transform=train_transform)
	attack_loader = DataLoader(trainset, batch_size=attack_batch_size, shuffle=True, num_workers=4, pin_memory=True)

	model_attack = resnet20_nm()
	ckpt_path = os.path.join(os.path.dirname(__file__), "1_sparse_finetune.pth") if not os.path.isfile("1_sparse_finetune.pth") else "1_sparse_finetune.pth"
	ckpt = torch.load(ckpt_path, map_location="cpu")
	if isinstance(ckpt, dict):
		for key in ("state_dict", "model_state_dict", "model", "model_state"):
			if key in ckpt:
				state = ckpt[key]
				break
		else:
			state = ckpt
	else:
		state = ckpt

	state = {k.replace("module.", ""): v for k, v in state.items()}
	model_attack.load_state_dict(state)

	criterion = nn.CrossEntropyLoss()

	# parse args for iterative attacks
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_iter', type=int, default=10, help='number of attack iterations')
	parser.add_argument('--attack_sample_size', type=int, default=128, help='number of samples used to evaluate flips (attack sample size)')
	parser.add_argument('--topk', type=int, default=20, help='top-K gradient indices to consider per layer')
	args = parser.parse_args()
	n_iter = args.n_iter
	attack_sample_size = args.attack_sample_size
	topk_arg = args.topk

	# find all layer modules under layer1, layer2, layer3 (exclude downsample modules)
	layer_modules = [k for k in info.keys() if (k.startswith('layer1.') or k.startswith('layer2.') or k.startswith('layer3.')) and ('downsample' not in k)]
	if len(layer_modules) == 0:
		# fallback to original single target if nothing found
		layer_modules = ["layer1.0.conv1"]

	# prepare small attack batch (used for quick flip eval)
	all_imgs = []
	all_targets = []
	processed = 0
	for img_b, tgt_b in attack_loader:
		if processed >= attack_sample_size: break
		rem = attack_sample_size - processed
		all_imgs.append(img_b[:rem])
		all_targets.append(tgt_b[:rem])
		processed += all_imgs[-1].size(0)
	test_imgs = torch.cat(all_imgs).to(next(model_attack.parameters()).device)
	test_tgts = torch.cat(all_targets).to(next(model_attack.parameters()).device)

	# full test loader and evaluator
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])
	test_loader = DataLoader(datasets.CIFAR10(root=os.path.expanduser("~/data/cifar10"), train=False, download=False, transform=test_transform), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

	def evaluate_top1(model, loader, device):
		model = model.to(device).eval()
		correct = total = 0
		with torch.no_grad():
			for images, targets in loader:
				images, targets = images.to(device), targets.to(device)
				preds = model(images).argmax(1)
				correct += preds.eq(targets).sum().item()
				total += targets.size(0)
		return 100.0 * correct / total if total else 0.0

	# iterative attack loop
	applied_history = []
	applied_set = set()  # set of (module_name, multi_idx) already flipped
	for it in range(n_iter):
		print(f"\n=== Iteration {it+1}/{n_iter} ===")
		loss_val, grads = forward_and_compute_gradients(model_attack, attack_loader, criterion, attack_sample_size, attack_batch_size)

		# ensure test batch is on same device as model (forward_and_compute_gradients may have moved model)
		param_dev = next(model_attack.parameters()).device
		test_imgs = test_imgs.to(param_dev)
		test_tgts = test_tgts.to(param_dev)

		global_entries = []
		for module_name in layer_modules:
			target_name = module_name + '.weight'
			if target_name in grads and module_name in info:
				g = grads[target_name]
				q_weights = info[module_name]['quantized_int8']
				step_size = info[module_name]['step_size']
				g_flat = g.flatten()
				abs_sorted, indices = torch.sort(g_flat.abs(), descending=True)

				layer_entries = []
				topk = min(topk_arg, indices.numel())
				for i in range(topk):
					idx = indices[i].item()
					grad_val = g_flat[idx].item()
					# compute multi-dimensional index
					dims = g.shape
					multi_idx = []
					temp_idx = idx
					for d in reversed(dims):
						multi_idx.append(temp_idx % d)
						temp_idx //= d
					multi_idx = tuple(reversed(multi_idx))

					orig_q = q_weights[multi_idx].item()
					flipped_q = orig_q - 128 if orig_q >= 0 else orig_q + 128

					with torch.no_grad():
						new_val = float(flipped_q) * step_size
						param = None
						for name, p in model_attack.named_parameters():
							if name == target_name:
								param = p
								break

						if param is not None:
							orig_val = param.data[multi_idx].item()
							param.data[multi_idx] = new_val
							new_output = model_attack(test_imgs)
							new_loss = criterion(new_output, test_tgts).item()
							param.data[multi_idx] = orig_val
						else:
							orig_val = float('nan')
							new_loss = float('nan')

						delta = new_loss - loss_val

						# skip candidate if already flipped
						key = (module_name, tuple(multi_idx))
						if key in applied_set:
							continue

						# only consider flips that increase loss (positive delta)
						if delta <= 0:
							continue

						layer_entries.append({
						'rank': i+1,
						'module_name': module_name,
						'target_name': target_name,
						'multi_idx': multi_idx,
						'grad_val': grad_val,
						'orig_q': int(orig_q) if not isinstance(orig_q, float) or not (orig_q != orig_q) else orig_q,
						'flipped_q': int(flipped_q) if not isinstance(flipped_q, float) or not (flipped_q != flipped_q) else flipped_q,
						'orig_val': orig_val,
						'loss_val': loss_val,
						'new_loss': new_loss,
						'delta': delta,
					})

				# end topk loop
				# append all evaluated top-k candidates from this layer so global selection
				# can choose from every candidate rather than only the single layer max
				if len(layer_entries) > 0:
					global_entries.extend(layer_entries)
			# end if module has grads

		# choose global max (only positive deltas were collected)
		if len(global_entries) == 0:
			print('No positive-delta candidate flips found this iteration; stopping.')
			break

		# sort candidates by delta descending and pick the first whose original INT8 is ZERO
		sorted_entries = sorted(global_entries, key=lambda e: e['delta'], reverse=True)
		chosen = None
		for rank, entry in enumerate(sorted_entries):
			module_nm_tmp = entry['module_name']
			idx_tmp = tuple(entry['multi_idx']) if isinstance(entry['multi_idx'], (list, tuple)) else entry['multi_idx']
			qcpu_tmp = info[module_nm_tmp]['quantized_int8']
			orig_int8_tmp = int(qcpu_tmp[idx_tmp].item())
			# if top candidate is non-zero, inform the user
			if rank == 0 and orig_int8_tmp != 0:
				print('Top candidate was non-zero; zero clip not the best in this iter.')
			if orig_int8_tmp == 0:
				chosen = entry
				break

		if chosen is None:
			print('No zero INT8 candidate found this iteration; stopping.')
			break

		global_max = chosen
		# print global best zero flip and int8 value before/after candidate flip
		module_nm = global_max['module_name']
		idx = tuple(global_max['multi_idx']) if isinstance(global_max['multi_idx'], (list, tuple)) else global_max['multi_idx']
		qcpu = info[module_nm]['quantized_int8']
		orig_int8 = int(qcpu[idx].item())
		cand_int8 = int(global_max['flipped_q'])
		print(f"Global best zero flip: module={global_max['module_name']} target={global_max['target_name']} index={global_max['multi_idx']} delta={global_max['delta']:+.6f}")
		print(f"INT8 before: {orig_int8} | INT8 (after flip): {cand_int8}")
		# compute and print quantized sparsity for the attacked layer
		qcpu_layer = info[module_nm]['quantized_int8']
		zeros = int((qcpu_layer == 0).sum().item())
		total = int(qcpu_layer.numel())
		sparsity_pct = 100.0 * zeros / total if total > 0 else 0.0
		print(f"Layer quantized sparsity: {sparsity_pct:.2f}% ({zeros}/{total} zeros)")

		# evaluate baseline top1 before applying flip
		baseline_top1 = evaluate_top1(model_attack, test_loader, device)

		# apply flip permanently to model_attack and update quantized info
		applied_module = global_max['module_name']
		applied_target = global_max['target_name']
		multi_idx = tuple(global_max['multi_idx']) if isinstance(global_max['multi_idx'], (list, tuple)) else global_max['multi_idx']
		flipped_q = global_max['flipped_q']
		step_size = info[applied_module]['step_size']
		new_val = float(flipped_q) * step_size
		param = None
		for name, p in model_attack.named_parameters():
			if name == applied_target:
				param = p
				break
		if param is not None:
			with torch.no_grad():
				param.data[multi_idx] = new_val
			# update quantized_int8 in info (CPU tensor)
			qcpu = info[applied_module]['quantized_int8']
			qcpu = qcpu.clone()
			qcpu[multi_idx] = torch.tensor(flipped_q, dtype=torch.int8)
			info[applied_module]['quantized_int8'] = qcpu
			# print int8 after applying flip
			# print(f"INT8 after apply: {int(info[applied_module]['quantized_int8'][multi_idx].item())}")
			applied_history.append(global_max)
			# evaluate after flip
			flipped_top1 = evaluate_top1(model_attack, test_loader, device)
			print(f"Applied flip. Baseline Top-1: {baseline_top1:.2f}%, After flip: {flipped_top1:.2f}%, Drop: {baseline_top1 - flipped_top1:.2f}%")
		else:
			print('Failed to apply flip: parameter not found in model_attack')

	# finished iterations
	# print('\nAttack finished. Summary of applied flips:')
	# for i, a in enumerate(applied_history):
	# 	print(f"{i+1:2d}: module={a['module_name']} target={a['target_name']} index={a['multi_idx']} delta={a['delta']:+.6f} flipped_q={a['flipped_q']}")
