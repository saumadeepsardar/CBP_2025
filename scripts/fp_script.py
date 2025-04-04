import os
import csv
import pandas as pd
import re
import datetime
import time
import subprocess
import multiprocessing as mp
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--trace_dir', help='path to trace directory', required=True)
parser.add_argument('--results_dir', help='path to results directory', required=True)
args = parser.parse_args()
trace_dir = Path(args.trace_dir)
results_dir = Path(args.results_dir)

def get_fp_traces(trace_dir):
    fp_dir = os.path.join(trace_dir, "fp")
    if not os.path.exists(fp_dir):
        raise FileNotFoundError(f"Directory {fp_dir} does not exist")
    traces = [os.path.join(fp_dir, f) for f in os.listdir(fp_dir) if f.endswith('_trace.gz')]
    return traces

my_traces = get_fp_traces(trace_dir)

def process_run_op(pass_status, my_trace_path, my_run_name, op_file):
    run_name_split = re.split(r"/", my_run_name)
    wl_name = run_name_split[0]  # e.g., "fp"
    run_name = run_name_split[1]  # e.g., "fp_0_trace"
    print(f'Extracting data from: {op_file} | WL:{wl_name} | Run:{run_name}')
    
    exec_time = 0
    _NumUncondBr = 0
    _MispUncondBr = 0
    _MR = 0
    _MPKI = 0
    
    trace_size = os.path.getsize(my_trace_path) / (1024 * 1024)  # Size in MB
    pass_status_str = 'Fail'

    uncond_section_header = 'UNCONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation)'
    process_uncond_section = False
    found_uncond_line_to_process = False

    if pass_status:
        pass_status_str = 'Pass'
        with open(op_file, "r") as text_file:
            for line in text_file:
                if not line.strip():
                    continue

                if 'ExecTime' in line:
                    exec_time = float(line.strip().split()[-1])

                if not process_uncond_section and uncond_section_header in line:
                    process_uncond_section = True

                if process_uncond_section:
                    if found_uncond_line_to_process:
                        curr_split_line = line.split()
                        _NumUncondBr = int(curr_split_line[0])
                        _MispUncondBr = int(curr_split_line[1])
                        _MR = float(curr_split_line[2])
                        _MPKI = float(curr_split_line[3])
                        process_uncond_section = False
                        found_uncond_line_to_process = False

                    if all(x in line for x in ['NumBr', 'MispBr', 'MR', 'MPKI']):
                        found_uncond_line_to_process = True

    if _NumUncondBr > 0:
        _MR = (_MispUncondBr / _NumUncondBr) * 100
        _MPKI = (_MispUncondBr / (_NumUncondBr / 1000))

    retval = {
        'Workload': wl_name,
        'Run': run_name,
        'TraceSize': trace_size,
        'Status': pass_status_str,
        'ExecTime': exec_time,
        'NumUncondBr': _NumUncondBr,
        'MispUncondBr': _MispUncondBr,
        'MR': _MR,
        'MPKI': _MPKI
    }
    return retval

def execute_trace(my_trace_path):
    assert os.path.exists(my_trace_path)
    
    run_split = re.split(r"/", my_trace_path)
    my_wl = run_split[-2]  # e.g., "fp"
    run_name = run_split[-1].split(".")[0]  # e.g., "fp_0_trace"
    os.makedirs(f'{results_dir}/{my_wl}', exist_ok=True)

    my_run_name = f'{my_wl}/{run_name}'
    exec_cmd = f'../cbp {my_trace_path}'  # Assuming cbp is in parent directory
    op_file = f'{results_dir}/{my_wl}/{run_name}.log'

    do_process = not os.path.exists(op_file)
    pass_status = True

    if do_process:
        print(f'Begin processing run: {my_run_name}')
        try:
            begin_time = time.time()
            run_op = subprocess.check_output(exec_cmd, shell=True, text=True)
            end_time = time.time()
            exec_time = end_time - begin_time
            with open(op_file, "w") as text_file:
                print(f"CMD: {exec_cmd}", file=text_file)
                print(f"{run_op}", file=text_file)
                print(f"ExecTime = {exec_time}", file=text_file)
        except subprocess.CalledProcessError as e:
            # Check for segmentation fault (exit code 139)
            if e.returncode == 139:
                print(f'Run: {my_run_name} failed with segmentation fault - skipping')
                return (False, my_trace_path, op_file, my_run_name)
            else:
                print(f'Run: {my_run_name} failed with error code {e.returncode}')
                pass_status = False

    return (pass_status, my_trace_path, op_file, my_run_name)

if __name__ == '__main__':
    print(f'Running {len(my_traces)} traces from fp folder: {my_traces}')

    os.makedirs(results_dir, exist_ok=True)

    with mp.Pool() as pool:
        results = pool.map(execute_trace, my_traces)

    df = pd.DataFrame(columns=['Workload', 'Run', 'TraceSize', 'Status', 'ExecTime', 'NumUncondBr', 'MispUncondBr', 'MR', 'MPKI'])
    for my_result in results:
        pass_status, trace_path, op_file, my_run_name = my_result
        if pass_status:  # Only process successful runs
            run_dict = process_run_op(pass_status, trace_path, my_run_name, op_file)
            my_df = pd.DataFrame([run_dict])
            df = pd.concat([df, my_df], ignore_index=True) if not df.empty else my_df.copy()

    print(df)
    timestamp = datetime.datetime.now().strftime("%m_%d_%H-%M-%S")
    df.to_csv(f'{results_dir}/uncond_branch_fp_results_{timestamp}.csv', index=False)

    print('\n--- Aggregate Metrics for Unconditional Branches (fp) ---')
    for wl in df['Workload'].unique():
        wl_df = df[df['Workload'] == wl]
        mr_mean = wl_df['MR'].astype(float).mean()
        mpki_mean = wl_df['MPKI'].astype(float).mean()
        print(f'WL: {wl:<10} MR Mean: {mr_mean:.2f}% | MPKI Mean: {mpki_mean:.2f}')
    overall_mr = df['MR'].astype(float).mean()
    overall_mpki = df['MPKI'].astype(float).mean()
    print(f'Overall MR: {overall_mr:.2f}% | Overall MPKI: {overall_mpki:.2f}')
