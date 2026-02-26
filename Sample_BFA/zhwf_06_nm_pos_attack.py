import os
import torch
import numpy as np
from resnet20_nm import resnet20_nm


def load_state(ckpt_path: str):
	if not os.path.isfile(ckpt_path):
		ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
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
	return state


def group_layer1_conv1(ckpt_path: str = "1_sparse_finetune.pth", pos_flag: bool = False, mask_flag: bool = False):
	"""Load checkpoint and perform 2:4 grouping for layer1.0.conv1.
	Returns a list of groups; each group is a list of 9 positions, each position is dict with 'vals' and 'mask' and 'posenc'.

	Args:
		ckpt_path: path to checkpoint file
		pos_flag: if True, printing will include position-encoding alongside weights
		mask_flag: if True, printing will include mask alongside weights
	Note:
		If both `pos` and `mask` are False, the function prints only the weights (used by `--group-only`).
	"""
	state = load_state(ckpt_path)
	key = "layer1.0.conv1.weight"
	if key in state:
		w = state[key].cpu().numpy()
	else:
		# try to instantiate model and read param
		model = resnet20_nm()
		model.load_state_dict(state)
		for name, p in model.named_parameters():
			if name == key:
				w = p.data.cpu().numpy()
				break
		else:
			raise KeyError(f"{key} not found in checkpoint or model")

	# expect shape (out, in, kH, kW)
	out_ch, in_ch, kH, kW = w.shape
	assert in_ch == 16 and out_ch >= 1 and kH * kW == 9, f"unexpected shape {w.shape}"

	# take out[0]
	out0 = w[0].reshape(in_ch, -1)  # (16,9)
	eps = 1e-8

	def _frac8_is_zero(x: float) -> bool:
		"""Return True if the first 8 digits after the decimal point are all '0'.

		Uses fixed-point formatting with 12 fractional digits to inspect the
		fractional part precisely (format(..., '.12f')).
		"""
		s = format(abs(float(x)), '.12f')
		if '.' not in s:
			return False
		frac = s.split('.')[-1]
		return frac[:8] == '00000000'

	groups = []
	for g in range(4):
		s = g * 4
		e = s + 4
		group_positions = []
		for pos in range(9):
			vals = out0[s:e, pos].tolist()
			mask = [0 if _frac8_is_zero(v) else 1 for v in vals]
			# Ensure exactly two ones per 2:4 group. If detection yields !=2, pick top-2 by abs value.
			orig_mask = mask.copy()
			if sum(mask) != 2:
				abs_vals = [abs(float(v)) for v in vals]
				top2 = sorted(range(len(vals)), key=lambda i: abs_vals[i], reverse=True)[:2]
				mask = [1 if i in top2 else 0 for i in range(len(vals))]
			# pos encoding: two nonzero indices (sorted) -> 4-bit
			idxs = [i for i, m in enumerate(mask) if m == 1]
			a, b = sorted(idxs)[:2]
			posenc = f"{a:02b}{b:02b}"
			group_positions.append({"vals": vals, "mask": mask, "posenc": posenc})
		groups.append(group_positions)

	# print summary
	print(f"Loaded {ckpt_path}; layer {key} shape {w.shape}")
	print("2:4 grouping for out[0]: 4 groups x 9 positions => 36 entries")
	for gi, group in enumerate(groups):
		print(f"-- Group {gi} (input channels {gi*4}-{gi*4+3}) --")
		for posi, entry in enumerate(group):
			vals_str = " ".join([f"{float(x):.6e}" for x in entry['vals']])
			if pos_flag:
				print(f"pos{posi}: [posenc={entry['posenc']}] vals=[{vals_str}]")
			elif mask_flag:
				mask_str = "".join([str(m) for m in entry['mask']])
				print(f"pos{posi}: [mask={mask_str}] vals=[{vals_str}]")
			else:
				print(f"pos{posi}: vals=[{vals_str}]")

	# default sanity check: verify posenc matches mask-derived encoding
	try:
		ok = verify_pos_mask_equivalence(groups)
		if not ok:
			print('[WARNING] posenc and mask verification failed')
	except Exception as e:
		print(f'[WARNING] verification raised exception: {e}')

	return groups


def verify_pos_mask_equivalence(groups) -> bool:
	"""Verify that for every group-position, `posenc` matches the indices of ones in `mask`.

	Returns True if all entries match; prints mismatches and returns False otherwise.
	"""
	errors = []
	for gi, group in enumerate(groups):
		for pi, entry in enumerate(group):
			mask = entry.get('mask', [])
			idxs = [i for i, m in enumerate(mask) if m == 1]
			if len(idxs) != 2:
				errors.append((gi, pi, 'mask_not_two_ones', mask, entry.get('vals')))
				continue
			a, b = sorted(idxs)[:2]
			expected = f"{a:02b}{b:02b}"
			if entry.get('posenc') != expected:
				errors.append((gi, pi, entry.get('posenc'), expected, mask, entry.get('vals')))
	if errors:
		print("POS/MASK mismatches:")
		for e in errors:
			print(e)
		return False
	print('POSENC OK: all entries match mask-derived encoding')
	return True


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--ckpt', type=str, default='1_sparse_finetune.pth')
	parser.add_argument('--group-only', action='store_true')
	parser.add_argument('--pos', action='store_true', help='If set, print weight and position-encoding')
	parser.add_argument('--mask', action='store_true', help='If set, print weight and mask encoding')
	args = parser.parse_args()

	if args.group_only:
		group_layer1_conv1(args.ckpt, pos_flag=args.pos, mask_flag=args.mask)
	else:
		print('Run with --group-only to perform 2:4 grouping for layer1.0.conv1')

