## Run on Dias
# # Import all what we need.
# encoding: utf-8
from datetime import datetime
import os
import bz2
import glob
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import numexpr as ne
import astropy.constants as ac
from tqdm import tqdm
from io import StringIO
from itertools import chain
# ThreadPoolExecutor is limited by the GIL in Python and is not efficient for CPU-bound tasks
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from multiprocessing import Pool, freeze_support, Manager, shared_memory
import json
from collections import OrderedDict
import threading
import sys
import getpass
import pickle
import math
from collections import deque
import itertools
import traceback
import multiprocessing as mp
import gc
from queue import Queue, Empty

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)
pd.options.mode.chained_assignment = None

# Initialize multi-process support
freeze_support()
num_cpus = mp.cpu_count()

# The input file path
def parse_args():
    parse = argparse.ArgumentParser(description='PyExoCross Program')
    parse.add_argument('-p', '--path', type=str, metavar='', required=True, help='Input file path')
    args = parse.parse_args()
    inp_filepath = args.path
    return inp_filepath
inp_filepath = parse_args()

# # Report time
class Timer:    
    def start(self):
        self.start_CPU = time.process_time()
        self.start_sys = time.time()
        return self

    def end(self, *args):
        self.end_CPU = time.process_time()
        self.end_sys = time.time()
        self.interval_CPU = self.end_CPU - self.start_CPU
        self.interval_sys = self.end_sys - self.start_sys
        print('{:25s} : {}'.format('Running time on CPU', self.interval_CPU), 's')
        print('{:25s} : {}'.format('Running time on system', self.interval_sys), 's')
        
    def cal(self, *args):
        self.end_CPU = time.process_time()
        self.end_sys = time.time()
        self.interval_CPU = self.end_CPU - self.start_CPU
        self.interval_sys = self.end_sys - self.start_sys
        return(self.interval_CPU, self.interval_sys) 


global_cpu_samples = []
memory_samples = []
global_memory_samples = []

def start_main_cpu_sampling(interval=2):
    stop_event = threading.Event()
    def loop():
        while not stop_event.is_set():
            try:
                usage = psutil.cpu_percent(interval=None)
                # usage = subprocess.check_output(
                #     "ps -u $USER -o pcpu= | paste -sd+ - | bc",
                #     shell=True, executable='/bin/bash'
                # ).decode().strip()
                global_cpu_samples.append(float(usage))  # Record once each time
            except Exception:
                pass
            time.sleep(interval)
    thread = threading.Thread(target=loop)
    thread.start()
    return stop_event, thread

def start_memory_sampling(memory_samples, interval=2):
    mem_stop_event = threading.Event()
    main_pid = os.getpid()

    def loop():
        # Use current process start time as baseline
        t0 = time.time()
        while not mem_stop_event.is_set():
            try:
                # Count memory of current process and all its child processes
                total_memory_mb = 0
                
                # Get current process
                main_proc = psutil.Process(main_pid)
                total_memory_mb += main_proc.memory_info().rss
                
                # Get all child processes (including recursive child processes)
                try:
                    children = main_proc.children(recursive=True)
                    for child in children:
                        try:
                            total_memory_mb += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                total_memory_mb = total_memory_mb / 1024 / 1024  # Convert to MB
                timestamp = time.time() - t0  # Use relative time
                memory_samples.append((timestamp, total_memory_mb))
            except Exception as e:
                print(f" File memory sampling error: {e}")
                pass
            time.sleep(interval)

    thread = threading.Thread(target=loop)
    thread.start()
    return mem_stop_event, thread

def start_main_memory_sampling(interval=2):
    stop_event = threading.Event()
    user = getpass.getuser()
    main_pid = os.getpid()  # Record main process ID
    
    def loop():
        while not stop_event.is_set():
            try:
                # Count all Python processes related to the molecular calculation task
                total_rss_mb = 0
                python_processes = []
                
                # Get main process
                main_proc = psutil.Process(main_pid)
                
                for p in psutil.process_iter(['pid', 'name', 'username', 'memory_info', 'ppid', 'cmdline']):
                    try:
                        # Check if it's a Python process
                        if not any(python_name in p.info['name'].lower() 
                                  for python_name in ['python', 'python3', 'python3.11', 'python3.12']):
                            continue
                            
                        # Check if it's the current user's process
                        if p.info['username'] != user:
                            continue
                        
                        # Check if it's related to current molecular calculation
                        is_related = False
                        relation_type = "unknown"
                        
                        # Method 1: Check if it's the main process
                        if p.info['pid'] == main_pid:
                            is_related = True
                            relation_type = "main"
                        
                        # Method 2: Check if it's a direct child process of main process (file processing process)
                        elif p.info['ppid'] == main_pid:
                            is_related = True
                            relation_type = "file_worker"
                        
                        # Method 3: Check if it's a recursive child process of main process (chunk processing process)
                        else:
                            try:
                                # Check if it's a recursive child process of main process
                                current_pid = p.info['pid']
                                parent_pid = p.info['ppid']
                                
                                # Trace up parent processes until finding main process or root process
                                while parent_pid != 1 and parent_pid != main_pid:
                                    try:
                                        parent_proc = psutil.Process(parent_pid)
                                        parent_pid = parent_proc.ppid()
                                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                                        break
                                
                                if parent_pid == main_pid:
                                    is_related = True
                                    relation_type = "chunk_worker"
                                    
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        # Method 4: Check command line arguments (backup method)
                        if not is_related and p.info['cmdline']:
                            cmdline = ' '.join(p.info['cmdline']).lower()
                            # Check if it contains molecular calculation related keywords
                            if any(keyword in cmdline for keyword in ['dias', 'exomol', 'cooling', 'pyexocross']):
                                is_related = True
                                relation_type = "cmdline_match"
                        
                        if is_related:
                            total_rss_mb += p.info['memory_info'].rss
                            python_processes.append({
                                'pid': p.info['pid'],
                                'name': p.info['name'],
                                'memory_mb': p.info['memory_info'].rss / 1024 / 1024,
                                'relation': relation_type,
                                'ppid': p.info['ppid']
                            })
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                total_rss_mb = total_rss_mb / 1024 / 1024  # Convert to MB
                timestamp = time.time() - global_start_time
                global_memory_samples.append((timestamp, total_rss_mb))

            except Exception as e:
                print(f" Global memory sampling error: {e}")
                pass
            time.sleep(interval)
    thread = threading.Thread(target=loop)
    thread.start()
    return stop_event, thread

    
# # Read Information from Input File
class InputWarning(UserWarning):
    pass


def inp_para(inp_filepath):
    # Find the maximum column for all the rows.
    with open(inp_filepath, 'r') as temp_f:
        col_count = max([len([x for x in l.split(" ") if x.strip()]) for l in temp_f.readlines()])
    # Generate column names (names will be 0, 1, 2, ..., maximum columns - 1).
    column_names = [i for i in range(col_count)] 
    inp_df = pd.read_csv(inp_filepath, sep='\\s+', header = None, names=column_names, usecols=column_names)
    col0 = inp_df[0]
    
    # Database
    database = inp_df[col0.isin(['Database'])][1].values[0].upper().replace('EXOMOL','ExoMol')
    
    # Basic information
    molecule = inp_df[col0.isin(['Molecule'])][1].values[0]
    isotopologue = inp_df[col0.isin(['Isotopologue'])][1].values[0]
    dataset = inp_df[col0.isin(['Dataset'])][1].values[0]
    mol_iso_id = int(inp_df[col0.isin(['MolIsoID'])][1].iloc[0])
    
    # File path
    read_path = inp_df[col0.isin(['ReadPath'])][1].values[0]
    save_path = inp_df[col0.isin(['SavePath'])][1].values[0]
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path, exist_ok=True)
        
    # Functions
    CoolingFunctions = int(inp_df[col0.isin(['CoolingFunctions'])][1].iloc[0])
   
    # Cores and chunks
    ncputrans = int(inp_df[col0.isin(['NCPUtrans'])][1].iloc[0])
    ncpufiles = int(inp_df[col0.isin(['NCPUfiles'])][1].iloc[0])
    chunk_size = int(inp_df[col0.isin(['ChunkSize'])][1].iloc[0])
    
        
    # Calculate cooling functions
    if CoolingFunctions == 1:
        Ntemp = int(inp_df[col0.isin(['Ntemp'])][1].iloc[0])
        Tmax = int(inp_df[col0.isin(['Tmax'])][1].iloc[0])
    else:
        Ntemp = 0
        Tmax = 0
    
    # Molecule and isotopologue ID, abundance, mass uncertainty, lifetime and g-factor
    molecule_id = int(mol_iso_id/10)
    isotopologue_id = mol_iso_id - molecule_id * 10
    if database == 'ExoMol':
        # Read ExoMol definition file (.def) to get the mass.
        # deffile_path = (read_path+'/'+molecule+'/'+isotopologue+'/'+dataset+'/'+isotopologue+'__'+dataset+'.def')
        # def_df = pd.read_csv(deffile_path,sep='\\s+',usecols=[0,1,2,3,4],names=['0','1','2','3','4'],header=None)
        # abundance = 1
        # mass = float(def_df[def_df['4'].isin(['mass'])]['0'].values[0]) # ExoMol mass (Dalton)
        # if def_df.to_string().find('Uncertainty') != -1:
        # check_uncertainty = int(def_df[def_df['2'].isin(['Uncertainty'])]['0'].values[0])
        # else:
        # check_uncertainty = 0
        deffile_path = (read_path+'/'+molecule+'/'+isotopologue+'/'+dataset+'/'+isotopologue+'__'+dataset+'.def')

        if os.path.exists(deffile_path):
            def_df = pd.read_csv(deffile_path, sep='\\s+', usecols=[0,1,2,3,4], names=['0','1','2','3','4'], header=None)
            abundance = 1
            # mass = float(def_df[def_df['4'].isin(['mass'])]['0'].values[0])
            mass_rows = def_df[def_df['4'].isin(['mass'])]
            if not mass_rows.empty:
                try:
                    mass = float(mass_rows['0'].values[0])
                except ValueError:
                    print(f" Failed to parse mass value in {deffile_path}, using default mass=1.0")
                    mass = 1.0
            else:
                print(f" 'mass' not found in .def file: {deffile_path}, using default mass=1.0")
                mass = 1.0

            if def_df.to_string().find('Uncertainty') != -1:
                check_uncertainty = int(def_df[def_df['2'].isin(['Uncertainty'])]['0'].values[0])
            else:
                check_uncertainty = 0
        else:
            print(f" .def file not found: {deffile_path}")
            abundance = 1
            mass = -1.0
            check_uncertainty = 0
    else:
        raise ImportError("Please add the name of the database 'ExoMol' into the input file.")

      
    return (database, molecule, isotopologue, dataset, read_path, save_path, CoolingFunctions,
            ncputrans, ncpufiles, chunk_size, Ntemp, Tmax, molecule_id, isotopologue_id, abundance, mass,
            check_uncertainty)


# # Constants and Parameters
# Parameters for calculating
Tref = 296.0                        # Reference temperature is 296 K
Pref = 1.0                          # Reference pressure is 1 bar
N_A = ac.N_A.value                  # Avogadro number (1/mol)
h = ac.h.to('erg s').value          # Planck's const (erg s)
c = ac.c.to('cm/s').value           # Velocity of light (cm/s)
kB = ac.k_B.to('erg/K').value       # Boltzmann's const (erg/K)
R = ac.R.to('J / (K mol)').value    # Molar gas constant (J/(K mol))
c2 = h * c / kB                     # Second radiation constant (cm K)

(database, molecule, isotopologue, dataset, read_path, save_path, CoolingFunctions,
 ncputrans, ncpufiles, chunk_size, Ntemp, Tmax, molecule_id, isotopologue_id, abundance, mass, 
 check_uncertainty) = inp_para(inp_filepath)
# pandarallel.initialize(nb_workers=ncputrans,progress_bar=False) # Initialize.

cooling_output_root = "/share/data1/xucapjix/shuchen/cooling"

runtime_record = {
    "start_time": time.time(),
    "trans_times": OrderedDict()  # preserve file order
}

# Constants
c2InvTref = c2 / Tref                 # c2 / T_ref (cm)
PI = np.pi
hc = h * c                            # erg cm
ln22 = np.log(2)*2
sinPI = np.sin(np.pi)
SqrtPI = np.sqrt(np.pi)
Sqrtln2 = np.sqrt(np.log(2))
OneminSqrtPIln2 = 1 - np.sqrt(np.pi * np.log(2))
Negln2 = -np.log(2)
PI4c = np.pi * 4 * c
Inv8Pic = 1 / (8 * np.pi * c)         # 8 * pi * c (s/cm)
Inv4Pi = 1 / (4 * np.pi)
Inv2ln2 = 1 / (2 * np.log(2))
InvSqrt2 = 1 / np.sqrt(2)
InvSqrtPi= 1 / np.sqrt(np.pi)
InvSprtln2 = 1 / np.sqrt(np.log(2))
InvSqrt2Pi = 1 / np.sqrt(2 * np.pi)
InvSqrt2ln2 = 1 / np.sqrt(2 * np.log(2))
TwoSqrt2ln2 = 2 * np.sqrt(2 * np.log(2))
Sqrtln2InvPi = np.sqrt(np.log(2) / np.pi)
Sqrt2NAkBln2mInvc = np.sqrt(2 * N_A * kB * np.log(2) / mass) / c

# # Convert frequency, upper and lower energy and J
# Calculate frequency
def cal_v(Ep, Epp):
    v = ne.evaluate('Ep - Epp')
    return(v)

# Calculate upper state energy with ExoMol database
def cal_Ep(Epp, v):
    Ep = ne.evaluate('Epp + v')
    return(Ep)

# Calculate upper J
def cal_Jp(Fp, Fpp, Jpp):
    Jp = ne.evaluate('Fp + Fpp - Jpp')
    return(Jp)

# Calculate F
def cal_F(g):
    F = ne.evaluate('(g - 1) * 0.5')
    return(F)

# Read Input Files
# # Read ExoMol Database Files
# ## Read States File
def read_all_states(read_path):
    t = Timer()
    t.start()
    print('Reading states ...')
    
    states_filename = (read_path + molecule + '/' + isotopologue + '/' + dataset 
                       + '/' + isotopologue + '__' + dataset + '.states.bz2')
    
    # only read required columns; keep original precision
    if check_uncertainty == 1:
        usecols = [0, 1, 2, 3, 4]  # id, E, g, J, unc
        names = ['id', 'E', 'g', 'J', 'unc']
    else:
        usecols = [0, 1, 2, 3]  # id, E, g, J
        names = ['id', 'E', 'g', 'J']
    
    if os.path.exists(states_filename):
        states_df = pd.read_csv(
            states_filename, 
            compression='bz2', 
            sep=r'\s+', 
            header=None,
            usecols=usecols,
            names=names,
            engine="python"
        )
    elif os.path.exists(states_filename.replace('.bz2', '')):
        states_df = pd.read_csv(
            states_filename.replace('.bz2', ''), 
            sep=r'\s+', 
            header=None,
            usecols=usecols,
            names=names,
            engine="python"
        )
    else:
        raise ImportError("No such states file, please check the read path and states filename format!")

    before = len(states_df)
    for col in ['E', 'g', 'J']:
        if col in states_df.columns:
            states_df[col] = pd.to_numeric(states_df[col], errors='coerce')

    states_df = states_df.dropna(subset=[c for c in ['E','g','J'] if c in states_df.columns])
    dropped = before - len(states_df)
    if dropped > 0:
        print(f"Cleaned states: dropped {dropped} row(s) with invalid E/g/J")

    # set index to improve merge efficiency
    states_df.set_index('id', inplace=True)
    
    
    t.end()     
    print(f'Finished reading states! Shape: {states_df.shape}, Memory: {states_df.memory_usage(deep=True).sum() / 1024**2:.1f}MB\n')       
    print('* * * * * - - - - - * * * * * - - - - - * * * * * - - - - - * * * * *\n')                
    return states_df


# ## Decompress Large .trans.bz2 Files
def command_decompress(trans_filename):
    # Directory where the decompressed .trans files will be saved
    # trans_dir = read_path+molecule+'/'+isotopologue+'/'+dataset+'/decompressed/'
    trans_dir = f"/share/data1/xucapjix/shuchen/tmp_decompressed/{molecule}/{isotopologue}/{dataset}/"
    if os.path.exists(trans_dir):
        pass
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(trans_dir, exist_ok=True)
    trans_file = os.path.join(trans_dir, trans_filename.split('/')[-1].replace('.bz2', ''))
    if os.path.exists(trans_file):
        num = 0
    else:
        command = f'bunzip2 < {trans_filename} > {trans_file}'
        print('Decompressing file:', trans_filename)
        subprocess.run(command, shell=True)
        num = 1
    return(trans_file, num)


# ## Get transitions File
def get_transfiles(read_path):
    # Get all the transitions files from the folder including the older version files which are named by vn(version number).
    trans_filepaths_all = glob.glob(read_path + molecule + '/' + isotopologue + '/' + dataset + '/' + '*.trans.bz2')
    if trans_filepaths_all == []:
        trans_filepaths_all = glob.glob(read_path + molecule + '/' + isotopologue + '/' + dataset + '/' + '*.trans')
        print(" Fallback to decompressed .trans files")
    num_transfiles_all = len(trans_filepaths_all)    # The number of all transitions files including the older version files.
    trans_filepaths = []    # The list of the lastest transitions files.
    all_decompress_num = 0
    decompress_num = 0
    for i in range(num_transfiles_all):
        filepath = trans_filepaths_all[i]
        # skip file if corresponding partial result exists
        partial_path = get_partial_path(filepath)
        if os.path.exists(partial_path):
            print(f"[Skip ] Partial exists, skip file: {os.path.basename(filepath)}")
            continue
        split_version = filepath.split('__')[-1].split('.')[0].split('_')    # Split the filenames.
        num = len(split_version)
        # There are four format filenames.
        # The lastest transitions files named in four formats:
        # 1. Filenames are named with the name of isotopologue and dataset.
        # End with .trans.bz2.
        # e.g. 14N-16O__XABC.trans.bz2'
        # 2. Filenames are named with the name of isotopologue and dataset.
        # Also have the range of wavenumbers xxxxx-yyyyy.
        # End with .trans.bz2.
        # e.g. 1H2-16O__POKAZATEL__00000-00100.trans.bz2
        # 3. The older version transitions files are named with vn(version number) based on the first format of the lastest files.
        # e.g. 14N-16O__XABC_v2.trans.bz2
        # 4. The older version transitions files are named with updated date (yyyymmdd).
        # e.g. 1H3_p__MiZATeP__20170330.trans.bz2
        # After split the filenames:
        # The first format filenames only leave the dataset name, e.g. XABC.
        # The second format filenames only leave the range of the wavenumber, e.g. 00000-00100.
        # The third format filenames leave two parts(dataset name and version number), e.g. XABC and v2.
        # The fourth format filenames only leave the updated date, e.g. 20170330.
        # This program only process the lastest data, so extract the filenames named by the first two format.
        if num == 1:     
            if split_version[0] == dataset:        
                trans_filepaths.append(filepath)
                print(f" Using decompressed .trans file: {filepath}")
            elif len(split_version[0].split('-')) == 2:
                file_size_bytes = os.path.getsize(filepath)
                if file_size_bytes/1024**3 > 1:
                    (trans_filepath, num) = command_decompress(filepath)
                    all_decompress_num += 1
                    decompress_num += num
                    print(f" Using decompressed file from bz2: {trans_filepath}")
                else:
                    trans_filepath = filepath
                    print(f" Using small bz2 file directly: {trans_filepath}")
                trans_filepaths.append(trans_filepath)
            else:
                pass
        else:
            pass
    print('Number of all transitions files \t\t:', num_transfiles_all)
    print('Number of all decompressed transitions files \t:', all_decompress_num)
    print('Number of new decompressed transitions files \t:', decompress_num)
    return trans_filepaths    

# Read partition function with local partition function file
def read_exomol_pf(read_path, T):
    pf_filename = (read_path + molecule + '/' + isotopologue + '/' + dataset 
                   + '/' + isotopologue + '__' + dataset + '.pf')
    pf_col_name = ['T', 'Q']
    pf_df = pd.read_csv(pf_filename, sep='\\s+', names=pf_col_name, header=None)
    # Q = pf_df[pf_df['T'].isin([T])]['Q']

    # interpolate Q(T); returns reasonable value even if T is not exactly in the table
    Ts = pf_df['T'].astype(float).values
    Qs = pf_df['Q'].astype(float).values

    if T < Ts.min() or T > Ts.max():
        raise ValueError(f"T = {T} is out of range of the PF file ({Ts.min()} ~ {Ts.max()})")

    Q = float(np.interp(T, Ts, Qs))  # interpolate Q(T)

    return(Q)

def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def ProcessCoolingFunction(states_df, Ts, trans_df, Q_dict, temp_block=None):
    if trans_df.empty:
        return np.zeros(len(Ts))

    if states_df.index.has_duplicates:
        states_df = states_df[~states_df.index.duplicated(keep='first')]

    try:
        idx = states_df.index
        td = trans_df.loc[trans_df['u'].isin(idx) & trans_df['l'].isin(idx)]
        if td.empty:
            return np.zeros(len(Ts))

        u = td['u'].to_numpy()
        l = td['l'].to_numpy()

        A  = td['A'].to_numpy()
        Eu = states_df.loc[u, 'E'].to_numpy()
        gu = states_df.loc[u, 'g'].to_numpy()
        El = states_df.loc[l, 'E'].to_numpy()
        gl = states_df.loc[l, 'g'].to_numpy()

        swap = El > Eu
        Ep = np.where(swap, El, Eu)
        gp = np.where(swap, gl, gu)
        v  = np.abs(cal_v(Eu, El))

        del td, u, l, Eu, El, gu, gl
        gc.collect()

    except Exception:
        return np.zeros(len(Ts))

    try:
        Ts_arr = np.asarray(Ts)
        Q_arr  = np.array([Q_dict[T] for T in Ts_arr])

        const = (A * hc) * v * gp
        del A, v, gp

        if temp_block is None:
            exp_factor = np.exp(-c2 * Ep[:, None] / Ts_arr[None, :])   # (N_lines × N_T)
            summed = const @ exp_factor                                 # (N_T,)
            cooling = summed / (4 * np.pi * Q_arr)
            return cooling
        else:
            nT = Ts_arr.size
            out = np.empty(nT)
            for s in range(0, nT, temp_block):
                e = min(s + temp_block, nT)
                exp_blk = np.exp(-c2 * Ep[:, None] / Ts_arr[None, s:e])
                out[s:e] = const @ exp_blk
                del exp_blk
            cooling_func = out / (4 * np.pi * Q_arr)
            del out, const, Q_arr, Ts_arr, Ep
            gc.collect()

    except Exception:
        return np.zeros(len(Ts))

    return cooling_func

def get_partial_path(trans_filepath):
    trans_filename = os.path.basename(trans_filepath).replace(".bz2", "")
    cf_folder = os.path.join(cooling_output_root, molecule, isotopologue)
    partial_folder = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    os.makedirs(partial_folder, exist_ok=True)
    return os.path.join(partial_folder, trans_filename + ".cf.partial")

def save_partial_result(partial_path, Ts, cooling_func):
    np.savetxt(partial_path, np.column_stack((Ts, cooling_func)), fmt="%8.1f %20.8E")

# function to merge partial files
def merge_all_partials(molecule, isotopologue, dataset, Ts, output_cf_path):
    partial_folder = os.path.join(cooling_output_root, molecule,isotopologue,
                                   f"{isotopologue}__{dataset}__partials")
    partial_files = sorted(glob.glob(os.path.join(partial_folder, "*.cf.partial")))
    if not partial_files:
        raise RuntimeError("No .cf.partial files found, cannot generate final cooling file.")

    print(f" Merging {len(partial_files)} partials...")
    merged = np.loadtxt(partial_files[0])
    for pf in partial_files[1:]:
        merged[:, 1] += np.loadtxt(pf)[:, 1]

    np.savetxt(output_cf_path, merged, fmt="%8.1f %20.8E")
    print(f" Merged cooling function saved: {output_cf_path}")


def generate_memory_log_path(molecule, isotopologue, trans_filepath):
    trans_filename = os.path.basename(trans_filepath).replace(".bz2", "").replace("/", "_")
    # memory_log_dir = os.path.join(cooling_output_root, molecule, "memory_logs")
    memory_log_dir = os.path.join(cooling_output_root, molecule, isotopologue, "memory_logs")
    os.makedirs(memory_log_dir, exist_ok=True)
    memory_log_path = os.path.join(memory_log_dir, f"{trans_filename}__memory.csv")
    return memory_log_path


class ChunkGenerator:
    def __init__(self, filepath, compression, chunk_size):
        self.filepath = filepath
        self.compression = compression
        self.chunk_size = chunk_size
        self.total_lines = 0
        self.total_chunks = 0

    def __iter__(self):
        # reset instance state
        self.total_lines = 0
        self.total_chunks = 0
        
        try:
            print(f" Starting chunk iteration with engine='c'")
            chunk_iter = pd.read_csv(
                self.filepath,
                compression=self.compression,
                sep=r'\s+',
                header=None,
                names=["u", "l", "A"],
                usecols=[0, 1, 2],  # only read required columns
                chunksize=self.chunk_size,
                iterator=True,
                engine="c",
                low_memory=False
            )
            
            for chunk in chunk_iter:
                chunk["u"] = pd.to_numeric(chunk["u"], errors="coerce")
                chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
                chunk["A"] = pd.to_numeric(chunk["A"], errors="coerce")

                valid_mask = chunk["A"].notna() & (chunk["A"] >= 0)
                chunk = chunk[valid_mask]
                chunk = chunk.dropna(subset=["u", "l"])

                if not chunk.empty:
                    self.total_lines += len(chunk)
                    self.total_chunks += 1
                    yield chunk, self.total_lines

        except Exception as e:
            print(f" Warning: engine='c' failed with error: {e}. Switching to engine='python'")
            # reset state
            self.total_lines = 0
            self.total_chunks = 0
            
            chunk_iter = pd.read_csv(
                self.filepath,
                compression=self.compression,
                sep=r'\s+',
                header=None,
                names=["u", "l", "A"],
                usecols=[0, 1, 2],
                chunksize=self.chunk_size,
                iterator=True,
                engine="python"
            )
            
            for chunk in chunk_iter:
                chunk["u"] = pd.to_numeric(chunk["u"], errors="coerce")
                chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
                chunk["A"] = pd.to_numeric(chunk["A"], errors="coerce")

                valid_mask = chunk["A"].notna() & (chunk["A"] >= 0)
                chunk = chunk[valid_mask]
                chunk = chunk.dropna(subset=["u", "l"])
                
                if not chunk.empty:
                    self.total_lines += len(chunk)
                    self.total_chunks += 1
                    yield chunk, self.total_lines

    def count_chunks(self):
        """
        Count valid chunks and total lines without changing instance state.
        Return (total_chunks, total_lines)
        """
        total_lines = 0
        total_chunks = 0
        
        try:
            chunk_iter = pd.read_csv(
                self.filepath,
                compression=self.compression,
                sep=r'\s+',
                header=None,
                names=["u", "l", "A"],
                usecols=[0, 1, 2],
                chunksize=self.chunk_size,
                iterator=True,
                engine="c",
                low_memory=False
            )
            
            for chunk in chunk_iter:
                chunk["u"] = pd.to_numeric(chunk["u"], errors="coerce")
                chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
                chunk["A"] = pd.to_numeric(chunk["A"], errors="coerce")

                valid_mask = chunk["A"].notna() & (chunk["A"] >= 0)
                chunk = chunk[valid_mask]
                chunk = chunk.dropna(subset=["u", "l"])

                if not chunk.empty:
                    total_lines += len(chunk)
                    total_chunks += 1

        except Exception as e:
            print(f" Warning: engine='c' failed with error: {e}. Switching to engine='python'")
            
            chunk_iter = pd.read_csv(
                self.filepath,
                compression=self.compression,
                sep=r'\s+',
                header=None,
                names=["u", "l", "A"],
                usecols=[0, 1, 2],
                chunksize=self.chunk_size,
                iterator=True,
                engine="python"
            )
            
            for chunk in chunk_iter:
                chunk["u"] = pd.to_numeric(chunk["u"], errors="coerce")
                chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
                chunk["A"] = pd.to_numeric(chunk["A"], errors="coerce")
                
                valid_mask = chunk["A"].notna() & (chunk["A"] >= 0)
                chunk = chunk[valid_mask]
                chunk = chunk.dropna(subset=["u", "l"])

                if not chunk.empty:
                    total_lines += len(chunk)
                    total_chunks += 1
        
        return total_chunks, total_lines



# Worker function - defined at top level to avoid pickle issues
def calculator_worker(chunk, states_df_local, Ts, Q_dict):
    """Worker process function; no shared memory used"""
    pid = os.getpid()
    cpu_t0 = time.process_time()
    wall_t0 = time.time()
    
    try:
        # states_dfshared memory
        res = ProcessCoolingFunction(states_df_local, Ts, chunk, Q_dict)
        
        cpu_t1 = time.process_time()
        wall_t1 = time.time()
        cpu_time = float(cpu_t1 - cpu_t0)
        wall_time = float(wall_t1 - wall_t0)
        
        return np.array(res), cpu_time
        
    except Exception as e:
        print(f" PID {pid}: Calculation failed: {e}")
        return np.zeros(len(Ts)), 0.0

def calculate_cooling_func(states_df, Ts, trans_filepath, ncpufiles, ncputrans, chunk_size, Q_dict):
    trans_filename = os.path.basename(trans_filepath)
    print(f" Start processing {trans_filename} in PID {os.getpid()}")
    
    # initialize result dict
    result = {
        "filename": trans_filename,
        "wall_time_s": 0.0,
        "cpu_time_s": 0.0,
        "lines": 0,
        "size_GB": 0.0,
        "wall_trans_rate": None,
        "cpu_trans_rate": None,
        "wall_time_per_GB": None,
        "cpu_time_per_GB": None,
        "estimated_average_cores_used": 0,
        "ncpufiles_used": ncpufiles,
        "ncputrans_used": ncputrans,
        "chunksize_used": chunk_size
    }
    

    if os.path.getsize(trans_filepath) < 100:
        print(f" Skipping empty file: {trans_filename}")
        partial_path = get_partial_path(trans_filepath)
        cooling_func = np.zeros(len(Ts))
        save_partial_result(partial_path, Ts, cooling_func)
        return result

    wall_t0 = time.time()
    memory_samples = []
    mem_stop_event, mem_thread = start_memory_sampling(memory_samples, interval=2)

    cooling_func = np.zeros(len(Ts))
    total_cpu_time = 0.0
    compression = 'bz2' if trans_filepath.endswith('.bz2') else None

    cg = ChunkGenerator(trans_filepath, compression, chunk_size)
    num_chunks, num_lines = cg.count_chunks()
    print(f" File: {trans_filename}, chunks: {num_chunks}, lines: {num_lines}")
    
    # use the provided parameters; no inner adjustments
    safe_workers = ncputrans
    safe_chunk_size = chunk_size
    
    # update result with actual parameters used
    result["ncputrans_used"] = safe_workers
    result["chunksize_used"] = safe_chunk_size
    
    # initialize progress bar
    proc_pbar = tqdm(total=num_chunks, desc="Processing chunks")
    
    
    with ProcessPoolExecutor(max_workers=safe_workers) as executor:        
        # submit all chunk tasks; let the pool manage
        futures = []
        chunk_iter = iter(cg)
        
        # submit the initial batch of tasks
        for _ in range(min(safe_workers * 2, num_chunks)):
            try:
                chunk_data = next(chunk_iter)
                chunk, _ = chunk_data
                future = executor.submit(calculator_worker, chunk, states_df, Ts, Q_dict)
                futures.append(future)
            except StopIteration:
                break
        
        completed_chunks = 0
        
        # handle completed tasks and submit new ones
        while futures:
            # wait for any task to complete
            done, not_done = wait(futures, return_when=FIRST_COMPLETED)
            futures = list(not_done)  # convert to list to allow append
            
            for future in done:
                try:
                    res, cpu = future.result()
                    cooling_func += res
                    total_cpu_time += cpu
                    completed_chunks += 1
                    proc_pbar.update(1)
                    
                except Exception as e:
                    print(f" Error in future result: {e}")
                    completed_chunks += 1
                    proc_pbar.update(1)
            
            # submit new tasks
            while len(futures) < safe_workers and completed_chunks + len(futures) < num_chunks:
                try:
                    chunk_data = next(chunk_iter)
                    chunk, _ = chunk_data
                    future = executor.submit(calculator_worker, chunk, states_df, Ts, Q_dict)
                    futures.append(future)
                except StopIteration:
                    break
    
    # close progress bar
    proc_pbar.close()
    
    # stop memory monitoring
    mem_stop_event.set()
    mem_thread.join()
    
    wall_time = time.time() - wall_t0

    partial_path = get_partial_path(trans_filepath)
    save_partial_result(partial_path, Ts, cooling_func)

    file_to_check = trans_filepath.replace('.bz2', '') if trans_filepath.endswith('.bz2') else trans_filepath
    if not os.path.exists(file_to_check):
        file_to_check = trans_filepath
    size_GB = os.path.getsize(file_to_check) / 1024**3
    lines = num_lines
    est_cores_used_by_true_cpu = round(total_cpu_time / wall_time, 2) if wall_time > 0 else None

    # result
    result.update({
        "filename": trans_filename,
        "wall_time_s": round(wall_time, 3),
        "cpu_time_s": round(total_cpu_time, 3),
        "lines": lines,
        "size_GB": round(size_GB, 3),
        "wall_trans_rate": round(lines / wall_time, 2),
        "cpu_trans_rate": round(lines / total_cpu_time, 2),
        "wall_time_per_GB": round(wall_time / size_GB, 2),
        "cpu_time_per_GB": round(total_cpu_time / size_GB, 2),
        "estimated_average_cores_used": est_cores_used_by_true_cpu,
        # ncpufiles_used, ncputrans_used, chunksize_used
    })

    single_runtime_dir = os.path.join(cooling_output_root, molecule, isotopologue, f"{isotopologue}__{dataset}__runtime_split")
    os.makedirs(single_runtime_dir, exist_ok=True)
    single_runtime_path = os.path.join(single_runtime_dir, trans_filename.replace(".bz2", "") + "__runtime.json")
    with open(single_runtime_path, "w") as f:
        json.dump({trans_filename: result}, f, indent=2, default=convert_np)
    print(f" Single runtime written to: {single_runtime_path}")

    memory_log_path = generate_memory_log_path(molecule, isotopologue, trans_filepath)
    with open(memory_log_path, "w") as f:
        f.write("timestamp,MB\n")
        for t, mem in memory_samples:
            f.write(f"{t},{mem:.2f}\n")
    print(f" Memory log saved to: {memory_log_path}")

    if "tmp_decompressed" in trans_filepath and trans_filepath.endswith(".trans"):
        try:
            os.remove(trans_filepath)
            print(f" Deleted decompressed trans file: {trans_filepath}")
        except Exception as e:
            print(f" Failed to delete temp trans file: {trans_filepath}\n{e}")

    return result




def classify_files_by_size(trans_filepaths):
    """
    classify files by size
    """
    small_files = []    # < 2GB
    medium_files = []   # 2-10GB
    large_files = []    # 10-30GB
    huge_files = []     # > 30GB
    
    for filepath in trans_filepaths:
        file_size_gb = os.path.getsize(filepath) / 1024**3
        
        if file_size_gb < 2:
            small_files.append((filepath, file_size_gb))
        elif file_size_gb < 10:
            medium_files.append((filepath, file_size_gb))
        elif file_size_gb < 30:
            large_files.append((filepath, file_size_gb))
        else:
            huge_files.append((filepath, file_size_gb))
    
    print(f" File classification:")
    print(f"   - Small files (<2GB): {len(small_files)}")
    print(f"   - Medium files (2-10GB): {len(medium_files)}")
    print(f"   - Large files (10-30GB): {len(large_files)}")
    print(f"   - Huge files (>30GB): {len(huge_files)}")
    
    return small_files, medium_files, large_files, huge_files

# Adaptive parameter selection function
def get_adaptive_parameters_by_file_size(file_size_gb, base_ncputrans, base_ncpufiles, base_chunk_size, total_files=None):
    """
    Adapt parameters based on file size and number of files
    """
    
    # use default strategy if total_files is not provided
    if total_files is None:
        total_files = 50  # moderate number of files
    
    # adjust based on file count and size
    if total_files < 50:  # few files
        if file_size_gb < 2:  # small files
            adaptive_ncputrans = min(base_ncputrans, 24)
            adaptive_ncpufiles = min(base_ncpufiles, 4)
            adaptive_chunk_size = max(base_chunk_size, 100000)
            strategy = "small_file_balanced"
            print(f" Small file strategy (few files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
            
        elif file_size_gb < 10:  # medium files
            adaptive_ncputrans = min(base_ncputrans, 32)
            adaptive_ncpufiles = min(base_ncpufiles, 3)
            adaptive_chunk_size = max(base_chunk_size, 50000)
            strategy = "medium_file_balanced"
            print(f" Medium file strategy (few files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
            
        elif file_size_gb < 30:  # large files
            adaptive_ncputrans = min(base_ncputrans, 32)
            adaptive_ncpufiles = min(base_ncpufiles, 2)
            adaptive_chunk_size = max(base_chunk_size, 20000)
            strategy = "large_file_conservative"
            print(f" Large file strategy (few files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
            
        else:  # large files
            adaptive_ncputrans = min(base_ncputrans, 48)
            adaptive_ncpufiles = min(base_ncpufiles, 1)
            adaptive_chunk_size = max(base_chunk_size, 10000)
            strategy = "huge_file_minimal"
            print(f" Huge file strategy (few files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
            
    else:  # English comment
        if file_size_gb < 2:  # small files
            # small filesH2O
            adaptive_ncputrans = min(base_ncputrans, 4)
            adaptive_ncpufiles = min(base_ncpufiles, 16)
            adaptive_chunk_size = max(base_chunk_size, 50000)
            strategy = "small_file_many_files"
            print(f" Small file strategy (many files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
            
        elif file_size_gb < 10:  # medium files
            adaptive_ncputrans = min(base_ncputrans, 6)
            adaptive_ncpufiles = min(base_ncpufiles, 12)
            adaptive_chunk_size = max(base_chunk_size, 30000)
            strategy = "medium_file_many_files"
            print(f" Medium file strategy (many files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
            
        elif file_size_gb < 30:  # large files
            adaptive_ncputrans = min(base_ncputrans, 8)
            adaptive_ncpufiles = min(base_ncpufiles, 6)
            adaptive_chunk_size = max(base_chunk_size, 10000)
            strategy = "large_file_many_files"
            print(f" Large file strategy (many files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
            
        else:  # large files
            adaptive_ncputrans = min(base_ncputrans, 12)
            adaptive_ncpufiles = min(base_ncpufiles, 2)
            adaptive_chunk_size = max(base_chunk_size, 100000)
            strategy = "huge_file_many_files"
            print(f" Huge file strategy (many files): Workers={adaptive_ncputrans}, Files={adaptive_ncpufiles}, Chunk={adaptive_chunk_size}")
    
    return {
        'ncputrans': adaptive_ncputrans,
        'ncpufiles': adaptive_ncpufiles,
        'chunk_size': adaptive_chunk_size,
        'strategy': strategy
    }

def get_priority_processing_order(trans_filepaths):
    """
    Get prioritized processing order: small files first
    """
    small_files, medium_files, large_files, huge_files = classify_files_by_size(trans_filepaths)
    
    # small files -> medium files -> large files -> large files
    priority_order = []
    
    if small_files:
        priority_order.append(('small', small_files))
    if medium_files:
        priority_order.append(('medium', medium_files))
    if large_files:
        priority_order.append(('large', large_files))
    if huge_files:
        priority_order.append(('huge', huge_files))
    
    return priority_order

# Cooling function for ExoMol database
def exomol_cooling(states_df, Ntemp, Tmax):
    print('Calculate cooling functions.') 
    print('Running on ', ncputrans, 'cores.')

    global runtime_record

    global global_start_time
    global_start_time = time.time()
    runtime_record["start_time"] = global_start_time

    # start main CPU and memory sampling
    stop_sampler, sampler_thread = start_main_cpu_sampling(interval=2)
    mem_stop_event, mem_thread = start_main_memory_sampling(interval=2)

    t = Timer()
    t.start()
    Ts = np.array(range(Ntemp, Tmax + 1, Ntemp)) 
    Q_dict = {T: read_exomol_pf(read_path, T) for T in Ts}

    print('Reading all transitions and calculating cooling functions ...')

    cf_folder = os.path.join(cooling_output_root, molecule, isotopologue)
    cf_path = os.path.join(cf_folder, f'{isotopologue}__{dataset}.cf')
    runtime_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__runtime.json")
    os.makedirs(cf_folder, exist_ok=True)

    # if cf already exists, skip computation but keep original runtime
    if os.path.exists(cf_path):
        print(f"[Skip ] Final .cf file already exists: {cf_path}")
        if os.path.exists(runtime_path):
            with open(runtime_path, 'r') as f:
                runtime_record = json.load(f, object_pairs_hook=OrderedDict)
            print("[] Loaded existing runtime record.")
        else:
            runtime_record = OrderedDict()
            runtime_record["note"] = ".cf exists but runtime.json is missing"
            runtime_record["trans_times"] = OrderedDict()
        
        # must stop CPU/MEM sampling threads in main thread
        stop_sampler.set()
        sampler_thread.join()
        mem_stop_event.set()
        mem_thread.join()

        return runtime_record  # English comment


    # if runtime.json exists, load recorded times
    if os.path.exists(runtime_path):
        with open(runtime_path, 'r') as f:
            existing_runtime = json.load(f)
    else:
        existing_runtime = {}
    old_trans_times = OrderedDict(existing_runtime.get("trans_times", {}))
    runtime_record["trans_times"] = OrderedDict({**old_trans_times})  # English comment

    runtime_record["large_trans_files_ncpufiles_1"] = []

    trans_filepaths = get_transfiles(read_path)
    
    # new: intelligent file classification and prioritized processing
    print(" Analyzing file sizes and setting adaptive parameters...")
    priority_order = get_priority_processing_order(trans_filepaths)
    
    # English comment
    total_files = len(trans_filepaths)
    total_size_gb = sum(os.path.getsize(fp) / 1024**3 for fp in trans_filepaths)
    print(f" Total: {total_files} files, {total_size_gb:.1f}GB")
    
    def process_file_with_adaptive_parameters(trans_filepath, base_ncputrans, base_ncpufiles, base_chunk_size, total_files):
        """Process a single file using adaptive parameters"""
        trans_filename = os.path.basename(trans_filepath).replace(".bz2", "")
        partial_path = get_partial_path(trans_filepath)
        
        if os.path.exists(partial_path):
            print(f"[Skip ] Already exists: {trans_filename}")
            if trans_filename in old_trans_times:
                runtime_record["trans_times"][trans_filename] = old_trans_times[trans_filename]
            return None
        
        # English comment
        file_size_gb = os.path.getsize(trans_filepath) / 1024**3
        adaptive_params = get_adaptive_parameters_by_file_size(file_size_gb, base_ncputrans, base_ncpufiles, base_chunk_size, total_files)
        
        # English comment
        safe_workers = adaptive_params['ncputrans']
        safe_chunk_size = adaptive_params['chunk_size']
        
        print(f" {trans_filename}: Adaptive strategy={adaptive_params['strategy']}")
        print(f" {trans_filename}: Workers={safe_workers}, ChunkSize={safe_chunk_size}")
        
        return (trans_filepath, safe_workers, safe_chunk_size, adaptive_params['strategy'])

    # ProcessPoolExecutor
    print(f" Creating single ProcessPoolExecutor with {ncpufiles} workers...")
    print(f" Main process PID: {os.getpid()}")
    print(f" Main process memory: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
    
    # dynamically choose file-level parallelism by file type
    def get_file_level_parallelism(file_type, file_list, total_files):
        """Get suitable file-level parallelism by file type"""
        if not file_list:
            return ncpufiles
        
        # English comment
        sample_file_size = file_list[0][1]
        adaptive_params = get_adaptive_parameters_by_file_size(sample_file_size, ncputrans, ncpufiles, chunk_size, total_files)
        strategy_ncpufiles = adaptive_params['ncpufiles']
        
        print(f" {file_type.upper()} files: Using {strategy_ncpufiles} file-level workers (strategy: {adaptive_params['strategy']})")
        return strategy_ncpufiles
    
    # use ncpufiles as file-level parallelism
    print(f" Using {ncpufiles} workers for file-level parallelism")
    
    # process per file type with different parallelism
    all_futures = []
    
    # small files -> medium files -> large files -> large files
    for file_type, file_list in priority_order:
        if not file_list:
            continue
            
        print(f"\n Processing {len(file_list)} {file_type.upper()} files...")
        
        # use strategy-chosen number of workers for each file type
        file_level_workers = get_file_level_parallelism(file_type, file_list, total_files)
        
        print(f" Creating ProcessPoolExecutor with {file_level_workers} workers for {file_type.upper()} files")
        
        with ProcessPoolExecutor(max_workers=file_level_workers) as executor:
            file_futures = []
            
            for trans_filepath, file_size_gb in file_list:
                # English comment
                file_params = process_file_with_adaptive_parameters(trans_filepath, ncputrans, ncpufiles, chunk_size, total_files)
                
                if file_params is None:  # English comment
                    continue
                
                trans_filepath, safe_workers, safe_chunk_size, strategy = file_params
                trans_filename = os.path.basename(trans_filepath).replace(".bz2", "")
                
                print(f" Submitting {trans_filename} with strategy: {strategy}")
                future = executor.submit(calculate_cooling_func, states_df, Ts, trans_filepath, ncpufiles, safe_workers, safe_chunk_size, Q_dict)
                file_futures.append((future, trans_filename))
            
            # wait for all tasks of the current file type to complete
            print(f" Waiting for {len(file_futures)} {file_type.upper()} files to complete...")
            
            for i, (future, filename) in enumerate(tqdm(file_futures, desc=f"Processing {file_type} files")):
                try:
                    print(f"\n  Main: Waiting for {file_type} task {i+1}/{len(file_futures)}: {filename}")
                    result = future.result()
                    filename_key = result.pop("filename")
                    runtime_record["trans_times"][filename_key] = result
                    print(f" Main: {file_type} task {filename} completed successfully")
                    print(f" Main process memory: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
                except Exception as e:
                    print(f" Error in {file_type} task {filename}: {e}")
                    print(f" Main process memory after error: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
            
            # futurefor statistics
            all_futures.extend(file_futures)

    t.end()
    print('Finished reading all transitions and calculating cooling functions!\n')

    print('Saving cooling functions into file ...')   
    ts = Timer()    
    ts.start()     

    partial_folder = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    partial_files = glob.glob(os.path.join(partial_folder, "*.cf.partial"))


    # === get current partial filenames (without path) ===
    existing_partials = set(os.path.basename(pf).replace(".cf.partial", "") for pf in partial_files)
    expected_partials = set(
        os.path.basename(tp).replace(".bz2", "") for tp in trans_filepaths
    )

    missing_partials = sorted(expected_partials - existing_partials)
    print(f" Missing partials: {len(missing_partials)} / {len(expected_partials)}")

    # === if missing, run another pass ===
    if missing_partials:
        print(f" Re-running {len(missing_partials)} missing trans files ...")
        with ProcessPoolExecutor(max_workers=ncpufiles) as executor:
            futures = []
            missing_filepaths = [
                tp for tp in trans_filepaths
                if os.path.basename(tp).replace(".bz2", "").replace(".trans", "") in missing_partials
            ]

            for trans_filepath in tqdm(missing_filepaths):
                futures.append(executor.submit(calculate_cooling_func, states_df, Ts, trans_filepath, ncpufiles, ncputrans, chunk_size, Q_dict))

            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    filename = result.pop("filename")
                    runtime_record["trans_times"][filename] = result
                except Exception as e:
                    print(f" Error in retry task: {e}")

    # check again whether all partial files exist
    partial_files = glob.glob(os.path.join(partial_folder, "*.cf.partial"))
    existing_partials_after_retry = set(os.path.basename(pf).replace(".cf.partial", "") for pf in partial_files)
    
    if existing_partials_after_retry >= expected_partials:
        merge_all_partials(molecule, isotopologue, dataset, Ts, cf_path)
        print(' Used all .cf.partial files to create final cooling function.')
    else:
        missing_after_retry = expected_partials - existing_partials_after_retry
        print(f" Still incomplete after retry: {len(existing_partials_after_retry)} / {len(expected_partials)}")
        print(f" Missing files: {missing_after_retry}")
        # Continue execution even if incomplete, return current runtime_record
        print(" Continuing with incomplete results...")


    ts.end()
    print('Cooling functions file has been saved:', cf_path, '\n')

    # CPU
    stop_sampler.set()
    sampler_thread.join()
    # stop main memory sampling thread
    mem_stop_event.set()
    mem_thread.join()

    # save overall memory usage log
    mem_log_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__memory_log.csv")
    with open(mem_log_path, "w") as f:
        f.write("timestamp,MB\n")
        for t, mem in global_memory_samples:
            f.write(f"{t:.3f},{mem:.2f}\n")
    print(f" Global memory log saved: {mem_log_path}")

    runtime_record["end_time"] = time.time()
    runtime_record["total_cpu_time_s"] = round(
        sum(v["cpu_time_s"] for v in runtime_record["trans_times"].values()), 3
    )

    # ---- .cf.partial ----
    partials_dir = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    partial_files = glob.glob(os.path.join(partials_dir, "*.cf.partial"))

    # .trans
    total_size_bytes = 0
    for trans_filepath in trans_filepaths:
        try:
            total_size_bytes += os.path.getsize(trans_filepath)
        except Exception:
            pass
        
    total_lines = sum(entry.get("lines", 0) for entry in runtime_record.get("trans_times", {}).values())
    total_size_GB = round(total_size_bytes / (1024**3), 3)
    runtime_record["total_lines"] = total_lines
    runtime_record["total_size_GB"] = total_size_GB

    # overall estimated_by_sampling (from global_cpu_samples)
    if global_cpu_samples:
        avg_sample_percent = sum(global_cpu_samples) / len(global_cpu_samples)
        estimated_by_sampling = round(avg_sample_percent / 100, 2)
        runtime_record["estimated_by_sampling"] = estimated_by_sampling
    else:
        estimated_by_sampling = None

    runtime_record.update({
        "molecule": molecule,
        "isotopologue": isotopologue,
        "dataset": dataset,
        "NCPUfiles": ncpufiles,
        "NCPUtrans": ncputrans,
        "ChunkSize": chunk_size,
        "total_lines": total_lines,
        "total_size_GB": total_size_GB,
        "estimated_by_sampling": estimated_by_sampling if global_cpu_samples else None
    })

    # with open(runtime_path, "w") as f:
    # json.dump(runtime_record, f, indent=2, default=convert_np)
    return runtime_record

# Get Results
def get_results(read_path): 
    runtime_record = OrderedDict()  

    t_tot = Timer()
    t_tot.start()  
    # ExoMol or HITRAN
    if database == 'ExoMol':
        print('ExoMol database')
        print('Molecule\t\t:', molecule, '\nIsotopologue\t:', isotopologue, '\nDataset\t\t\t:', dataset)
        # All functions need whole states.
        states_df = read_all_states(read_path)  
        print('Finished reading all transitions')      

        # Functions
        if CoolingFunctions == 1:
            result = exomol_cooling(states_df, Ntemp, Tmax)
            if result is not None:
                runtime_record = result
            else:
                # If exomol_cooling returns None, create a basic runtime_record
                runtime_record = OrderedDict()
                runtime_record["trans_times"] = OrderedDict()
                runtime_record["note"] = "exomol_cooling returned None due to incomplete processing"
    print('\nThe program total running time:')    
    t_tot.end()
    cpu_time_main, wall_time_main = t_tot.cal()

    # define runtime_path same as in exomol_cooling
    cf_folder = os.path.join(cooling_output_root, molecule, isotopologue)
    runtime_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__runtime.json")

    # append main program timing to runtime_record
    runtime_record["main_cpu_time_s"] = float(round(cpu_time_main, 3))
    runtime_record["main_wall_time_s"] = float(round(wall_time_main, 3))

    # only compute cores used when trans tasks complete (total_cpu_time_s present)
    if "total_cpu_time_s" in runtime_record and runtime_record["main_wall_time_s"] > 0:
        runtime_record["estimated_average_cores_used"] = round(
            runtime_record["total_cpu_time_s"] / runtime_record["main_wall_time_s"], 2
        )

    # ===== write runtime.json in fixed order to keep field and nesting order consistent =====
    ordered_trans_times = OrderedDict()
    for fname, data in runtime_record.get("trans_times", {}).items():
        ordered_trans_times[fname] = OrderedDict([
            ("wall_time_s", data["wall_time_s"]),
            ("cpu_time_s", data["cpu_time_s"]),
            ("lines", data["lines"]),
            ("size_GB", data["size_GB"]),
            ("wall_trans_rate", data["wall_trans_rate"]),
            ("cpu_trans_rate", data["cpu_trans_rate"]),
            ("wall_time_per_GB", data["wall_time_per_GB"]),
            ("cpu_time_per_GB", data["cpu_time_per_GB"]),
            ("estimated_average_cores_used", data["estimated_average_cores_used"]),
            ("ncpufiles_used", data.get("ncpufiles_used", None)),
            ("ncputrans_used", data.get("ncputrans_used", None)),
            ("chunksize_used", data.get("chunksize_used", None))
        ])

    ordered_runtime = OrderedDict()
    ordered_runtime["molecule"] = runtime_record["molecule"]
    ordered_runtime["dataset"] = runtime_record["dataset"]
    ordered_runtime["isotopologue"] = runtime_record["isotopologue"]
    ordered_runtime["total_lines"] = runtime_record["total_lines"]
    ordered_runtime["total_size_GB"] = runtime_record["total_size_GB"]
    ordered_runtime["NCPUfiles"] = runtime_record["NCPUfiles"]
    ordered_runtime["NCPUtrans"] = runtime_record["NCPUtrans"]
    ordered_runtime["ChunkSize"] = runtime_record["ChunkSize"]
    ordered_runtime["start_time"] = runtime_record["start_time"]
    ordered_runtime["trans_times"] = ordered_trans_times
    ordered_runtime["end_time"] = runtime_record["end_time"]
    ordered_runtime["total_cpu_time_s"] = runtime_record["total_cpu_time_s"]
    ordered_runtime["main_cpu_time_s"] = runtime_record["main_cpu_time_s"]
    ordered_runtime["main_wall_time_s"] = runtime_record["main_wall_time_s"]
    ordered_runtime["estimated_average_cores_used"] = runtime_record["estimated_average_cores_used"]
    ordered_runtime["estimated_by_sampling"] = runtime_record["estimated_by_sampling"]
    if "large_trans_files_ncpufiles_1" in runtime_record:
        ordered_runtime["large_trans_files_ncpufiles_1"] = runtime_record["large_trans_files_ncpufiles_1"]

    with open(runtime_path, "w") as f:
        json.dump(ordered_runtime, f, indent=2, default=convert_np)

    print(f" Runtime record saved: {runtime_path}")
    print('Cooling functions have been saved!\n')  
    print('* * * * * - - - - - * * * * * - - - - - * * * * * - - - - - * * * * *\n')


    print('\nFinished!')
    pass

def main():
    get_results(read_path)

if __name__ == '__main__':
    # Use the stable 'spawn' method
    mp.set_start_method("spawn", force=True)
    print(" Using 'spawn' start method for stability")
    main()
