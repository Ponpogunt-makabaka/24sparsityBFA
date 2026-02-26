#!/usr/bin/env python3
import os
import re
import sys
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def run_and_collect(script: str, n_iter: int, topG: int, topK: int, extra_args=None):
    # Build command targeted at zhwf_10_nm_pos_attack.py
    # Map this script's --n_iter and --topk to the pos-attack flags: --n-iter and --proxy-topG
    cmd = [sys.executable, script, '--nca-proxy-best-only', '--n-iter', str(n_iter), '--proxy-topG', str(topG), '--per-group-K', str(topK)]
    if extra_args:
        cmd += extra_args

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    after_vals = []
    before_int8 = []
    saved_plot_path = None
    # pos-attack prints lines like:
    # "...\nacc_before\n0.921600\nacc_after\n0.922300\n"
    next_is_after = False

    # stream output and parse
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='')
        # detect acc_after label; the next non-empty line is the numeric accuracy (0-1)
        if re.match(r'^\s*acc_after\s*$', line):
            next_is_after = True
            continue
        if next_is_after:
            m = re.search(r'([0-9]+\.[0-9]+)', line)
            if m:
                try:
                    val = float(m.group(1)) * 100.0
                    after_vals.append(val)
                    # keep placeholder for before_int8 to preserve return shape
                    before_int8.append(None)
                except Exception:
                    pass
            next_is_after = False

    proc.wait()
    return after_vals, before_int8, saved_plot_path


def plot_and_save(after_vals, before_int8, out_dir, n_iter, topG, topK, script):
    if not after_vals:
        return None
    plt.figure()
    x = list(range(1, len(after_vals)+1))
    # draw line
    plt.plot(x, after_vals, linestyle='-', color='C0')
    # split points into red-star (INT8-before == 0) and others
    red_x, red_y = [], []
    other_x, other_y = [], []
    for i, val in enumerate(after_vals):
        b = None
        if i < len(before_int8):
            b = before_int8[i]
        if b is not None and b == 0:
            red_x.append(x[i])
            red_y.append(val)
        else:
            other_x.append(x[i])
            other_y.append(val)
    # other points: no outline, default circle marker, slightly smaller
    if other_x:
        plt.scatter(other_x, other_y, c='C0', s=50, marker='o')
    # red star points: larger and star marker
    if red_x:
        plt.scatter(red_x, red_y, c='red', s=90, marker='*')
    plt.xlabel('Iteration')
    plt.ylabel('Top-1 Accuracy After Flip (%)')
    plt.title(f'After-Flip Top-1 vs Iteration (n={n_iter}, topG={topG}, topK={topK})')
    plt.grid(True)
    # use the script base name (without path or .py) at start of filename
    script_base = os.path.splitext(os.path.basename(script))[0]
    fname = f'{script_base}_n{n_iter}_topG{topG}_topK{topK}.png'
    out_path = os.path.join(out_dir, fname)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    # y-axis: ticks every 10, fixed range 0-100
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_ylim(0, 100)
    plt.savefig(out_path)
    print(f'Wrote plot: {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Run zhwf_04_nm_dense.py and plot after-flip Top-1 vs iter')
    parser.add_argument('--n_iter', type=int, required=True)
    parser.add_argument('--topG', type=int, required=True, help='Number of groups to keep after group-level ranking (maps to --proxy-topG)')
    parser.add_argument('--topK', type=int, required=True, help='Number of candidates per selected group to evaluate (maps to --per-group-K)')
    parser.add_argument('--workdir', type=str, default='.', help='workspace directory containing the dense script')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, help='extra args to pass to dense script')
    parser.add_argument('--script', type=str, default='zhwf_04_nm_dense.py', help='python script to run')
    args = parser.parse_args()

    cwd = os.path.abspath(args.workdir)
    os.chdir(cwd)

    after_vals, before_int8, saved_plot = run_and_collect(args.script, args.n_iter, args.topG, args.topK, args.extra)

    out = None
    out = plot_and_save(after_vals, before_int8, cwd, args.n_iter, args.topG, args.topK, args.script)

    # if no after_vals found but dense script saved a plot, rename it to include args
    if out is None and saved_plot and os.path.exists(saved_plot):
        new_name = os.path.join(cwd, f'zhwf_04_dense_n{args.n_iter}_k{args.topK}.png')
        try:
            os.replace(saved_plot, new_name)
            print(f'Renamed dense-produced plot to: {new_name}')
            out = new_name
        except Exception as e:
            print(f'Failed to rename saved plot: {e}')

    if out is None:
        print('No data found to plot.')
        sys.exit(2)

if __name__ == "__main__":
    main()
