## Import all what we need.
# encoding: utf-8
import os
import bz2
import glob
import time
import requests
import argparse
import subprocess
import numpy as np
import pandas as pd
import numexpr as ne
import dask.dataframe as dd
import astropy.constants as ac
import matplotlib.pyplot as plt
import numexpr as ne
import dask.dataframe as dd
from tqdm import tqdm
from io import StringIO
from itertools import chain
from pandarallel import pandarallel
from matplotlib.collections import LineCollection
# ThreadPoolExecutor in Python is limited by the GIL and cannot efficiently parallelize CPU-bound tasks
from concurrent.futures import ThreadPoolExecutor
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, freeze_support
import json
from collections import OrderedDict
import threading
import sys


import warnings
warnings.simplefilter("ignore", np.ComplexWarning)
pd.options.mode.chained_assignment = None

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

import multiprocessing as mp
freeze_support()
num_cpus = mp.cpu_count()
print('Number of CPU: ', num_cpus)


# The input file path
def parse_args():
    parse = argparse.ArgumentParser(description='PyExoCross Program')
    parse.add_argument('-p', '--path', type=str, metavar='', required=True, help='Input file path')
    args = parse.parse_args()
    inp_filepath = args.path
    return inp_filepath
inp_filepath = parse_args()


## Report time
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
                usage = subprocess.check_output(
                    "ps -u $USER -o pcpu= | paste -sd+ - | bc",
                    shell=True, executable='/bin/bash'
                ).decode().strip()
                global_cpu_samples.append(float(usage))  # record a sample each time
            except Exception:
                pass
            time.sleep(interval)
    thread = threading.Thread(target=loop)
    thread.start()
    return stop_event, thread

def start_memory_sampling(memory_samples, interval=2):
    mem_stop_event = threading.Event()

    def loop():
        proc = psutil.Process(os.getpid())
        t0 = time.time()
        while not mem_stop_event.is_set():
            mem = proc.memory_info().rss / 1024 / 1024  # in MB
            timestamp = time.time() - t0
            memory_samples.append((timestamp, mem))
            time.sleep(interval)

    thread = threading.Thread(target=loop)
    thread.start()
    return mem_stop_event, thread

def start_main_memory_sampling(interval=2):
    stop_event = threading.Event()
    def loop():
        while not stop_event.is_set():
            try:
                mem = subprocess.check_output(
                    "ps -u a123 -o rss= | awk '{sum+=$1} END {print sum/1024}'",
                    shell=True, executable='/bin/bash'
                ).decode().strip()
                timestamp =  time.time() - global_start_time
                global_memory_samples.append((timestamp, float(mem)))
            except Exception:
                pass
            time.sleep(interval)
    thread = threading.Thread(target=loop)
    thread.start()
    return stop_event, thread

    
## Read Information from Input File
class InputWarning(UserWarning):
    pass


def inp_para(inp_filepath):
    # Find the maximum column for all the rows.
    with open(inp_filepath, 'r') as temp_f:
        col_count = max([len([x for x in l.split(" ") if x.strip()]) for l in temp_f.readlines()])
    # Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1).
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
                    print(f"Failed to parse mass value in {deffile_path}, using default mass=1.0")
                    mass = 1.0
            else:
                print(f"'mass' not found in .def file: {deffile_path}, using default mass=1.0")
                mass = 1.0

            if def_df.to_string().find('Uncertainty') != -1:
                check_uncertainty = int(def_df[def_df['2'].isin(['Uncertainty'])]['0'].values[0])
            else:
                check_uncertainty = 0
        else:
            print(f".def file not found: {deffile_path}")
            abundance = 1
            mass = -1.0
            check_uncertainty = 0
    else:
        raise ImportError("Please add the name of the database 'ExoMol' into the input file.")

      
    return (database, molecule, isotopologue, dataset, read_path, save_path, CoolingFunctions,
            ncputrans, ncpufiles, chunk_size, Ntemp, Tmax, molecule_id, isotopologue_id, abundance, mass,
            check_uncertainty)


## Constants and Parameters
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
pandarallel.initialize(nb_workers=ncputrans,progress_bar=False)    # Initialize.

cooling_output_root = "/Users/a123/Desktop/light_tasks/cooling"

runtime_record = {
    "start_time": time.time(),
    "trans_times": OrderedDict()  # keep file order
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

## Convert frequency, upper and lower energy and J
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
## Read ExoMol Database Files
### Read States File
def read_all_states(read_path):
    t = Timer()
    t.start()
    print('Reading states ...')
    states_df = pd.DataFrame()
    states_filename = (read_path + molecule + '/' + isotopologue + '/' + dataset 
                       + '/' + isotopologue + '__' + dataset + '.states.bz2')
    if os.path.exists(states_filename):    
        chunks = pd.read_csv(states_filename, compression='bz2', sep=r'\s+', header=None,
                            chunksize=100_000, iterator=True, dtype=object, engine="python")

    elif os.path.exists(states_filename.replace('.bz2','')):
        chunks = pd.read_csv(states_filename, compression='bz2', sep=r'\s+', header=None,
                            chunksize=100_000, iterator=True, dtype=object, engine="python")
    else:
        raise ImportError("No such states file, please check the read path and states filename format!")

    for chunk in chunks:
        states_df = pd.concat([states_df, chunk])
    if check_uncertainty == 1:
        states_df = states_df.rename(columns={0:'id',1:'E',2:'g',3:'J',4:'unc'})
        convert_dict = {'id':np.int32,'E':np.float64,'g':np.int32,'J':np.float16,'unc':np.float32}
        states_df = states_df.astype(convert_dict)
    else:      
        states_df = states_df.rename(columns={0:'id',1:'E',2:'g',3:'J'})  
        convert_dict = {'id':np.int32,'E':np.float64,'g':np.int32,'J':np.float16}
        states_df = states_df.astype(convert_dict)
    t.end()     
    print('Finished reading states!\n')       
    print('* * * * * - - - - - * * * * * - - - - - * * * * * - - - - - * * * * *\n')                
    return(states_df)


### Decompress Large .trans.bz2 Files
def command_decompress(trans_filename):
    # Directory where the decompressed .trans files will be saved
    # trans_dir = read_path+molecule+'/'+isotopologue+'/'+dataset+'/decompressed/'
    trans_dir = f"/Users/a123/Desktop/light_tasks/tmp_decompressed/{molecule}/{isotopologue}/{dataset}/"
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

### Get transitions File
def get_transfiles(read_path):
    # Get all the transitions files from the folder including the older version files which are named by vn(version number).
    trans_filepaths_all = glob.glob(read_path + molecule + '/' + isotopologue + '/' + dataset + '/' + '*.trans.bz2')
    if trans_filepaths_all == []:
        trans_filepaths_all = glob.glob(read_path + molecule + '/' + isotopologue + '/' + dataset + '/' + '*.trans')
    num_transfiles_all = len(trans_filepaths_all)    # The number of all transitions files including the older version files.
    trans_filepaths = []    # The list of the lastest transitions files.
    all_decompress_num = 0
    decompress_num = 0
    for i in range(num_transfiles_all):
        split_version = trans_filepaths_all[i].split('__')[-1].split('.')[0].split('_')    # Split the filenames.
        num = len(split_version)
        # There are four format filenames.
        # The lastest transitions files named in four formats:
        # 1. Filenames are named with the name of isotopologue and dataset. 
        #    End with .trans.bz2.
        #    e.g. 14N-16O__XABC.trans.bz2'
        # 2. Filenames are named with the name of isotopologue and dataset. 
        #    Also have the range of wavenumbers xxxxx-yyyyy.
        #    End with .trans.bz2.
        #    e.g. 1H2-16O__POKAZATEL__00000-00100.trans.bz2
        # 3. The older version transitions files are named with vn(version number) based on the first format of the lastest files.
        #    e.g. 14N-16O__XABC_v2.trans.bz2
        # 4. The older version transitions files are named with updated date (yyyymmdd).
        #    e.g. 1H3_p__MiZATeP__20170330.trans.bz2
        # After split the filenames:
        # The first format filenames only leave the dataset name, e.g. XABC.
        # The second format filenames only leave the range of the wavenumber, e.g. 00000-00100.
        # The third format filenames leave two parts(dataset name and version number), e.g. XABC and v2.
        # The fourth format filenames only leave the updated date, e.g. 20170330.
        # This program only process the lastest data, so extract the filenames named by the first two format.
        if num == 1:     
            if split_version[0] == dataset:        
                trans_filepaths.append(trans_filepaths_all[i])
            elif len(split_version[0].split('-')) == 2:
                file_size_bytes = os.path.getsize(trans_filepaths_all[i])
                if file_size_bytes/1024**3 > 1:   
                    (trans_filepath, num) = command_decompress(trans_filepaths_all[i])
                    all_decompress_num += 1
                    decompress_num += num
                else:
                    trans_filepath = trans_filepaths_all[i]
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

    # Interpolate Q(T); returns a reasonable value even if T is not exactly a row value
    Ts = pf_df['T'].astype(float).values
    Qs = pf_df['Q'].astype(float).values

    if T < Ts.min() or T > Ts.max():
        raise ValueError(f"T = {T} is out of range of the PF file ({Ts.min()} ~ {Ts.max()})")

    Q = float(np.interp(T, Ts, Qs))  # Interpolate Q(T)

    return(Q)

def calculate_cooling(A, v, Ep, gp, T, Q):
    _sum = ne.evaluate(
        'sum(A * hc * v * gp * exp(-c2 * Ep / T))',
        local_dict={
            'A': A, 'hc': hc, 'v': v, 'gp': gp, 'Ep': Ep, 'T': T, 'c2': c2
        }
    )
    return ne.evaluate(
        '_sum / (4 * PI * Q)',
        local_dict={
            '_sum': _sum, 'PI': PI, 'Q': Q
        }
    )


def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def ProcessCoolingFunction(states_df, Ts, trans_df, Q_dict):
    merged_df = dd.merge(trans_df, states_df, left_on='u', right_on='id', 
                         how='inner').merge(states_df, left_on='l', right_on='id', how='inner', suffixes=("'", '"'))
    cooling_func_df = merged_df[['A',"E'",'E"',"g'"]]
    cooling_func_df['v'] = cal_v(cooling_func_df["E'"].values, cooling_func_df['E"'].values)

    if len(cooling_func_df) > 0:
        A = cooling_func_df['A'].values
        v = cooling_func_df['v'].values
        Ep = cooling_func_df["E'"].values
        gp = cooling_func_df["g'"].values
        cooling_func = [calculate_cooling(A, v, Ep, gp, T, Q_dict[T]) for T in Ts]
    else:
        cooling_func = np.zeros(len(Ts))

    return cooling_func


def get_partial_path(trans_filepath):
    trans_filename = os.path.basename(trans_filepath).replace(".bz2", "")
    cf_folder = os.path.join(cooling_output_root, molecule, isotopologue)
    partial_folder = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    os.makedirs(partial_folder, exist_ok=True)
    return os.path.join(partial_folder, trans_filename + ".cf.partial")


def save_partial_result(partial_path, Ts, cooling_func):
    np.savetxt(partial_path, np.column_stack((Ts, cooling_func)), fmt="%8.1f %20.8E")

# Function to merge all partial files
def merge_all_partials(molecule, isotopologue, dataset, Ts, output_cf_path):
    partial_folder = os.path.join(cooling_output_root, molecule,isotopologue,
                                   f"{isotopologue}__{dataset}__partials")
    partial_files = sorted(glob.glob(os.path.join(partial_folder, "*.cf.partial")))
    if not partial_files:
        raise RuntimeError("No .cf.partial files found, cannot generate final cooling file.")

    print(f"Merging {len(partial_files)} partials...")
    merged = np.loadtxt(partial_files[0])
    for pf in partial_files[1:]:
        merged[:, 1] += np.loadtxt(pf)[:, 1]

    np.savetxt(output_cf_path, merged, fmt="%8.1f %20.8E")
    print(f"Merged cooling function saved: {output_cf_path}")

def process_chunk_wrapper(args):
    states_df, Ts, chunk, Q_dict = args
    try:
        cpu_t0 = time.process_time()
        cooling_func = ProcessCoolingFunction(states_df, Ts, chunk, Q_dict)
        if isinstance(cooling_func, list):
            cooling_func = np.array(cooling_func)
        cpu_t1 = time.process_time()
        cpu_time = float(cpu_t1 - cpu_t0)
        return cooling_func, cpu_time
    except Exception as e:
        print(f"Error in process_chunk_wrapper: {e}")
        return None, 0.0  # or raise e if you want to terminate the program

def generate_memory_log_path(molecule, isotopologue, trans_filepath):
    trans_filename = os.path.basename(trans_filepath).replace(".bz2", "").replace("/", "_")
    # memory_log_dir = os.path.join(cooling_output_root, molecule, "memory_logs")
    memory_log_dir = os.path.join(cooling_output_root, molecule, isotopologue, "memory_logs")
    os.makedirs(memory_log_dir, exist_ok=True)
    memory_log_path = os.path.join(memory_log_dir, f"{trans_filename}__memory.csv")
    return memory_log_path


def get_available_memory_gb():
    return psutil.virtual_memory().available / (1024 ** 3)

def chunk_generator(trans_filepath, compression, chunk_size):
    chunk_iter = pd.read_csv(
        trans_filepath,
        compression=compression,
        sep=r"\s+",
        header=None,
        names=["u", "l", "A"],
        usecols=[0, 1, 2],
        chunksize=chunk_size,
        iterator=True,
        engine="python"
    )
    for chunk in chunk_iter:
        chunk = chunk[chunk["A"].notna() & (chunk["A"] >= 0)]
        if not chunk.empty:
            yield chunk


def calculate_cooling_func(states_df, Ts, trans_filepath, ncpufiles, ncputrans, Q_dict):
    trans_filename = os.path.basename(trans_filepath)
    print('Processing transitions file:', trans_filename)

    # Pre-check whether the file is empty (<100 bytes)
    if os.path.getsize(trans_filepath) < 100:
        print(f"Skipping empty file: {trans_filename}")
        # Save an empty .cf.partial file
        partial_path = get_partial_path(trans_filepath)
        cooling_func = np.zeros(len(Ts))
        save_partial_result(partial_path, Ts, cooling_func)

        return {
            "filename": trans_filename,
            "wall_time_s": 0.0,
            "cpu_time_s": 0.0,
            "lines": 0,
            "size_GB": 0.0,
            "wall_trans_rate": None,
            "cpu_trans_rate": None,
            "wall_time_per_GB": None,
            "cpu_time_per_GB": None,
            "estimated_average_cores_used": 0
        }

    
    wall_t0 = time.time()
    memory_samples = []
    mem_stop_event, mem_thread = start_memory_sampling(memory_samples, interval=2)

    cooling_func = np.zeros(len(Ts))
    total_cpu_time = 0.0

    # if trans_filepath.endswith('.bz2'):
    #     trans_df_chunk_list = safe_read_trans_file(trans_filepath, chunk_size, compression='bz2')
    # else:
    #     trans_df_chunk_list = safe_read_trans_file(trans_filepath, chunk_size)

    # args_list = [(states_df, Ts, chunk) for chunk in trans_df_chunk_list]

    compression = 'bz2' if trans_filepath.endswith('.bz2') else None

    # Execute in parallel
    cooling_results = []
    cpu_times = []

    # Always use ProcessPoolExecutor

    Q_dict = {T: read_exomol_pf(read_path, T) for T in Ts}

    # Build an argument list; each item is (states_df, Ts, chunk)
    # args_list = [(states_df, Ts, chunk) for chunk in chunk_gen]
    args_list = [(states_df, Ts, chunk, Q_dict) for chunk in chunk_generator(trans_filepath, compression, chunk_size)]

    cooling_results = []
    cpu_times = []

    with ProcessPoolExecutor(max_workers=ncputrans) as executor:
        futures = [executor.submit(process_chunk_wrapper, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {trans_filename}"):
            try:
                res, cpu = future.result()
                if res is not None:
                    cooling_results.append(res)
                    cpu_times.append(cpu)
            except Exception as e:
                print(f"Chunk failed: {e}")

            
    cooling_func = sum(cooling_results)
    total_cpu_time = sum(cpu_times)

    mem_stop_event.set()
    mem_thread.join()

    wall_t1 = time.time()
    wall_time = wall_t1 - wall_t0

    partial_path = get_partial_path(trans_filepath)
    save_partial_result(partial_path, Ts, cooling_func)

    file_to_check = trans_filepath.replace('.bz2', '') if trans_filepath.endswith('.bz2') else trans_filepath
    if not os.path.exists(file_to_check):
        file_to_check = trans_filepath
    size_GB = os.path.getsize(file_to_check) / 1024**3

    with (bz2.open(file_to_check, 'rt') if file_to_check.endswith('.bz2') else open(file_to_check)) as f:
        lines = sum(1 for _ in f)

    est_cores_used_by_true_cpu = round(total_cpu_time / wall_time, 2) if wall_time > 0 else None

    result = {
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
        "ncpufiles_used": ncpufiles,
        "ncputrans_used": ncputrans
    }

    # Save runtime for a single trans file
    single_runtime_dir = os.path.join(cooling_output_root, molecule, isotopologue, f"{isotopologue}__{dataset}__runtime_split")
    os.makedirs(single_runtime_dir, exist_ok=True)
    single_runtime_path = os.path.join(single_runtime_dir, trans_filename.replace(".bz2", "") + "__runtime.json")
    with open(single_runtime_path, "w") as f:
        json.dump({trans_filename: result}, f, indent=2, default=convert_np)

    print(f"Single runtime written to: {single_runtime_path}")

    # Save memory sampling log
    memory_log_path = generate_memory_log_path(molecule, isotopologue, trans_filepath)
    with open(memory_log_path, "w") as f:
        f.write("timestamp,MB\n")
        for t, mem in memory_samples:
            f.write(f"{t},{mem:.2f}\n")
    print(f"Memory log saved to: {memory_log_path}")

    # Auto-delete temporary trans files in tmp_decompressed
    if "tmp_decompressed" in trans_filepath and trans_filepath.endswith(".trans"):
        try:
            os.remove(trans_filepath)
            print(f"Deleted decompressed trans file: {trans_filepath}")
        except Exception as e:
            print(f"Failed to delete temp trans file: {trans_filepath}\n{e}")

    return result

def wait_until_memory_free(threshold_gb=40, check_interval=10):
    while True:
        available = psutil.virtual_memory().available / 1024**3
        if available >= threshold_gb:
            return
        print(f"Waiting: only {available:.2f} GB available < {threshold_gb} GB")
        time.sleep(check_interval)

# Split trans_filepaths
def split_trans_by_suffix(trans_filepaths):
    large_trans = []
    small_trans = []
    for f in trans_filepaths:
        if f.endswith('.trans'):  # decompressed, large file
            large_trans.append(f)
        elif f.endswith('.bz2'):  # small file, compressed
            small_trans.append(f)
    return large_trans, small_trans


# Cooling function for ExoMol database
def exomol_cooling(states_df, Ntemp, Tmax):
    print('Calculate cooling functions.') 
    print('Running on ', ncputrans, 'cores.')

    runtime_record = OrderedDict() 

    global global_start_time
    global_start_time = time.time()
    runtime_record["start_time"] = global_start_time

    # Start CPU and memory sampling in the main process
    stop_sampler, sampler_thread = start_main_cpu_sampling(interval=2)
    mem_stop_event, mem_thread = start_main_memory_sampling(interval=2)

    t = Timer()
    t.start()
    Ts = np.array(range(Ntemp, Tmax + 1, Ntemp)) 
    print('Reading all transitions and calculating cooling functions ...')

    cf_folder = os.path.join(cooling_output_root, molecule, isotopologue)
    cf_path = os.path.join(cf_folder, f'{isotopologue}__{dataset}.cf')
    runtime_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__runtime.json")
    os.makedirs(cf_folder, exist_ok=True)

    # If cf already exists, skip computation but keep the original runtime
    if os.path.exists(cf_path):
        print(f"[Skip] Final .cf file already exists: {cf_path}")
        if os.path.exists(runtime_path):
            with open(runtime_path, 'r') as f:
                runtime_record = json.load(f, object_pairs_hook=OrderedDict)
            print("Loaded existing runtime record.")
        else:
            runtime_record = OrderedDict()
            runtime_record["note"] = ".cf exists but runtime.json is missing"
            runtime_record["trans_times"] = OrderedDict()
        
        # Must stop CPU/MEM sampling threads in the main process
        stop_sampler.set()
        sampler_thread.join()
        mem_stop_event.set()
        mem_thread.join()

        return runtime_record  # Skip only; do not break the main program flow


    # If runtime.json exists, load recorded times
    if os.path.exists(runtime_path):
        with open(runtime_path, 'r') as f:
            existing_runtime = json.load(f)
    else:
        existing_runtime = {}
    old_trans_times = OrderedDict(existing_runtime.get("trans_times", {}))
    runtime_record["trans_times"] = OrderedDict({**old_trans_times})  # Merge old records first

    runtime_record["large_trans_files_ncpufiles_1&ncputrans_4"] = []

    trans_filepaths = get_transfiles(read_path)
    large_trans, small_trans = split_trans_by_suffix(trans_filepaths)
    Q_dict = {T: read_exomol_pf(read_path, T) for T in Ts}

    def run_trans_group(trans_group, ncpufiles_setting, ncputrans_setting, label, Q_dict):
            with ProcessPoolExecutor(max_workers=ncpufiles_setting) as executor:
                futures = []
                for trans_filepath in tqdm(trans_group, desc=f"Submitting {label} tasks"):
                    trans_filename = os.path.basename(trans_filepath).replace(".bz2", "")
                    partial_path = get_partial_path(trans_filepath)
                    if os.path.exists(partial_path):
                        print(f"[Skip] Already exists: {trans_filename}")
                        if trans_filename in old_trans_times:
                            runtime_record["trans_times"][trans_filename] = old_trans_times[trans_filename]
                        continue
                    if label == "LARGE" and ncpufiles_setting == 1:
                        runtime_record["large_trans_files_ncpufiles_1&ncputrans_4"].append(trans_filename)
                    futures.append(executor.submit(calculate_cooling_func, states_df, Ts, trans_filepath, ncpufiles_setting,ncputrans_setting, Q_dict))

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {label} tasks"):
                    try:
                        result = future.result()
                        filename = result.pop("filename")
                        runtime_record["trans_times"][filename] = result
                        if label == "LARGE":
                            runtime_record["large_trans_files_ncpufiles_1&ncputrans_4"].append(filename)
                    except Exception as e:
                        print(f"Error in one task ({label}): {e}")

    if large_trans:
        run_trans_group(large_trans, ncpufiles_setting=1, ncputrans_setting=4, label="LARGE", Q_dict=Q_dict)
    print("All large trans files finished. Now running small bz2 files...\n")
    if small_trans:
        run_trans_group(small_trans, ncpufiles_setting=ncpufiles,ncputrans_setting=ncputrans,label="SMALL", Q_dict=Q_dict)

    t.end()
    print('Finished reading all transitions and calculating cooling functions!\n')

    print('Saving cooling functions into file ...')   
    ts = Timer()    
    ts.start()     

    partial_folder = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    partial_files = glob.glob(os.path.join(partial_folder, "*.cf.partial"))


        # === Get existing partial filenames (without path) ===
    existing_partials = set(os.path.basename(pf).replace(".cf.partial", "") for pf in partial_files)
    expected_partials = set(
        os.path.basename(tp).replace(".bz2", "") for tp in trans_filepaths
    )

    missing_partials = sorted(expected_partials - existing_partials)
    print(f"Missing partials: {len(missing_partials)} / {len(expected_partials)}")

    # === If there are missing ones, run them again ===
    if missing_partials:
        print(f"Re-running {len(missing_partials)} missing trans files ...")
        with ProcessPoolExecutor(max_workers=ncpufiles) as executor:
            futures = []
            missing_filepaths = [
                tp for tp in trans_filepaths
                if os.path.basename(tp).replace(".bz2", "").replace(".trans", "") in missing_partials
            ]

            for trans_filepath in tqdm(missing_filepaths):
                futures.append(executor.submit(calculate_cooling_func, states_df, Ts, trans_filepath, ncpufiles, Q_dict))

            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    filename = result.pop("filename")
                    runtime_record["trans_times"][filename] = result
                except Exception as e:
                    print(f"Error in retry task: {e}")

    # Check again whether all partial files are present
    partial_files = glob.glob(os.path.join(partial_folder, "*.cf.partial"))
    if len(partial_files) == len(trans_filepaths):
        merge_all_partials(molecule, isotopologue, dataset, Ts, cf_path)
        print('Used all .cf.partial files to create final cooling function.')
    else:
        print(f"Still incomplete after retry: {len(partial_files)} / {len(trans_filepaths)}")
        return


    ts.end()
    print('Cooling functions file has been saved:', cf_path, '\n')

    # Stop CPU sampling in the main thread
    stop_sampler.set()
    sampler_thread.join()
    # Stop memory sampling thread in the main thread
    mem_stop_event.set()
    mem_thread.join()

    # Save overall memory usage log
    mem_log_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__memory_log.csv")
    with open(mem_log_path, "w") as f:
        f.write("timestamp,MB\n")
        for t, mem in global_memory_samples:
            f.write(f"{t:.3f},{mem:.2f}\n")
    print(f"Global memory log saved: {mem_log_path}")

    runtime_record["end_time"] = time.time()
    runtime_record["total_cpu_time_s"] = round(
        sum(v["cpu_time_s"] for v in runtime_record["trans_times"].values()), 3
    )

    # ---- More efficient computation of total lines and total size (based on .cf.partial files) ----
    partials_dir = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    partial_files = glob.glob(os.path.join(partials_dir, "*.cf.partial"))

    # Aggregate total size and line count of original .trans files
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

    # Overall estimated_by_sampling (from global_cpu_samples)
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
    #     json.dump(runtime_record, f, indent=2, default=convert_np)
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
            runtime_record = exomol_cooling(states_df, Ntemp, Tmax)
            # exomol_cooling(states_df, Ntemp, Tmax)   
    print('\nThe program total running time:')    
    t_tot.end()
    cpu_time_main, wall_time_main = t_tot.cal()

    # Define runtime_path the same as in exomol_cooling
    cf_folder = os.path.join(cooling_output_root, molecule, isotopologue)
    runtime_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__runtime.json")

    # Append main program timing to runtime_record
    runtime_record["main_cpu_time_s"] = float(round(cpu_time_main, 3))
    runtime_record["main_wall_time_s"] = float(round(wall_time_main, 3))

    # Compute cores used only when trans tasks are finished (total_cpu_time_s exists)
    if "total_cpu_time_s" in runtime_record and runtime_record["main_wall_time_s"] > 0:
        runtime_record["estimated_average_cores_used"] = round(
            runtime_record["total_cpu_time_s"] / runtime_record["main_wall_time_s"], 2
        )

    # ===== Write runtime.json in a consistent order to ensure field and nesting order =====
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
            ("ncputrans_used", data.get("ncputrans_used", None))
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
    if "large_trans_files_ncpufiles_1&ncputrans_4" in runtime_record:
        ordered_runtime["large_trans_files_ncpufiles_1&ncputrans_4"] = runtime_record["large_trans_files_ncpufiles_1&ncputrans_4"]

    with open(runtime_path, "w") as f:
        json.dump(ordered_runtime, f, indent=2, default=convert_np)

    print(f"Runtime record saved: {runtime_path}")
    print('Cooling functions have been saved!\n')  
    print('* * * * * - - - - - * * * * * - - - - - * * * * * - - - - - * * * * *\n')


    print('\nFinished!')
    pass

def main():
    get_results(read_path)

if __name__ == '__main__':
    main()



