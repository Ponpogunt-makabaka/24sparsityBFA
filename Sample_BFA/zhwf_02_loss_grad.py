import os
import copy
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
"""Loader + INT8 quantization evaluation for resnet20_nm on CIFAR-10.

Functions:
- load_and_quantize_ckpt(): loads checkpoint, runs FP32 inference,
  applies per-tensor symmetric INT8 quantization on weights, runs quantized
  inference, and returns (fp32_top1, int8_top1, layer_info).

layer_info maps layer name -> {'step_size': float, 'quantized_int8': torch.IntTensor}
"""

from resnet20_nm import resnet20_nm
from sparse_ops import SparseConv, SparseLinear
import torch.nn as nn


def forward_and_compute_gradients(
	model,
	data_loader,
	criterion,
	attack_sample_size: int = 128,
	attack_batch_size: int = 128,
	device: str = "cuda",
):
	"""Compute loss and gradients exactly like BFA.py does.
	
	It uses model.eval() and computes gradient on a single combined batch 
	of attack_sample_size to match the 'data' tensor passed in BFA.
	"""
	device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
	model = model.to(device)
	model.eval() # Aligned with BFA.py:96
	model.zero_grad()

	# Prepare a single batch of 'attack_sample_size' samples
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

	# Single forward pass like BFA.py:99
	outputs = model(full_images)
	loss = criterion(outputs, full_targets)
	
	# Backward directly without manual scaling, like BFA.py:108
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

	# build model and load checkpoint
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

	# FP32 evaluation
	model_fp = model.to(device).eval()
	correct = total = 0
	with torch.no_grad():
		for images, targets in loader:
			images, targets = images.to(device), targets.to(device)
			preds = model_fp(images).argmax(1)
			correct += preds.eq(targets).sum().item()
			total += targets.size(0)
	fp32_top1 = 100.0 * correct / total if total else 0.0

	# INT8 quantization (per-tensor symmetric) on weights only
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
			# replace weight with dequantized float for inference
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

	# Print per-layer quantization info: step size, quantized shape, min/max
	# print("\nPer-layer quantization summary:")
	# for name in sorted(info.keys()):
	# 	entry = info[name]
	# 	q = entry['quantized_int8']
	# 	try:
	# 		q_min = int(q.min().item())
	# 		q_max = int(q.max().item())
	# 	except Exception:
	# 		q_min = None
	# 		q_max = None
	# 	print(f"- {name}: step={entry['step_size']:.6e}, shape={tuple(q.shape)}, min={q_min}, max={q_max}")

	# Unit test for forward_and_compute_gradients: use train data and print
	# CrossEntropy loss and gradients for layer `layer1.0.conv1.weight`.
	attack_sample_size = 128
	attack_batch_size = 128

	# build a train loader for attack samples
	train_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32, padding=4),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])
	trainset = datasets.CIFAR10(root=os.path.expanduser("~/data/cifar10"), train=True, download=False, transform=train_transform)
	attack_loader = DataLoader(trainset, batch_size=attack_batch_size, shuffle=True, num_workers=4, pin_memory=True)

	# load model (FP32) with same checkpoint
	model_attack = resnet20_nm()
	# reuse checkpoint loading logic
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
	loss_val, grads = forward_and_compute_gradients(model_attack, attack_loader, criterion, attack_sample_size, attack_batch_size)

	print(f"\nUnit test (layer1.0.conv1): accumulated_loss={loss_val:.6f}")
	target_name = "layer1.0.conv1.weight"
	if target_name in grads:
		g = grads[target_name].flatten()
		abs_sorted, idx = torch.sort(g.abs(), descending=True)
		print(f"gradient stats: mean={g.mean():.6e}, std={g.std():.6e}, min={g.min():.6e}, max={g.max():.6e}")
		print("Top 20 abs gradient values:")
		for i, val in enumerate(abs_sorted[:20]):
			print(f"{i+1:2d}: {float(val):.6e}")
	else:
		print(f"Parameter {target_name} not found in gradients.")