#!/usr/bin/env python3
import os
import re
import sys
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def run_and_collect(script: str, n_iter: int, topk: int, extra_args=None):
    cmd = [sys.executable, script, '--n_iter', str(n_iter), '--topk', str(topk)]
    if extra_args:
        cmd += extra_args

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    after_vals = []
    before_int8 = []
    saved_plot_path = None
    after_re = re.compile(r'After flip:\s*([0-9]+\.?[0-9]*)%')
    before_re = re.compile(r'INT8 before:\s*(-?\d+)')
    saved_re = re.compile(r'Saved plot:\s*(\S+)')
    last_before = None

    # stream output and parse
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='')
        m_before = before_re.search(line)
        if m_before:
            try:
                last_before = int(m_before.group(1))
            except Exception:
                last_before = None

        m = after_re.search(line)
        if m:
            try:
                after_vals.append(float(m.group(1)))
                # pair the most recent before value (may be None)
                before_int8.append(last_before)
            except Exception:
                pass
            finally:
                last_before = None
        m2 = saved_re.search(line)
        if m2:
            saved_plot_path = m2.group(1)

    proc.wait()
    return after_vals, before_int8, saved_plot_path


def plot_and_save(after_vals, before_int8, out_dir, n_iter, topk, script):
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
    plt.title(f'After-Flip Top-1 vs Iteration (dense, n={n_iter}, k={topk})')
    plt.grid(True)
    # use the script base name (without path or .py) at start of filename
    script_base = os.path.splitext(os.path.basename(script))[0]
    fname = f'{script_base}_n{n_iter}_topk{topk}.png'
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
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--workdir', type=str, default='.', help='workspace directory containing the dense script')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, help='extra args to pass to dense script')
    parser.add_argument('--script', type=str, default='zhwf_04_nm_dense.py', help='python script to run')
    args = parser.parse_args()

    cwd = os.path.abspath(args.workdir)
    os.chdir(cwd)

    after_vals, before_int8, saved_plot = run_and_collect(args.script, args.n_iter, args.topk, args.extra)

    out = None
    out = plot_and_save(after_vals, before_int8, cwd, args.n_iter, args.topk, args.script)

    # if no after_vals found but dense script saved a plot, rename it to include args
    if out is None and saved_plot and os.path.exists(saved_plot):
        new_name = os.path.join(cwd, f'zhwf_04_dense_n{args.n_iter}_k{args.topk}.png')
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
