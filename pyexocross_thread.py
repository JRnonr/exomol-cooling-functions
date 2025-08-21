## Optimize the original version of Jingxin’s code
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
from tqdm import tqdm
from io import StringIO
from itertools import chain
from pandarallel import pandarallel
from matplotlib.collections import LineCollection
# ThreadPoolExecutor in Python is limited by GIL and cannot efficiently parallelize CPU-intensive tasks
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, freeze_support
import json
from collections import OrderedDict

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
        # Read ExoMol definition file (.def) to get the mass.
        deffile_path = (read_path+'/'+molecule+'/'+isotopologue+'/'+dataset+'/'+isotopologue+'__'+dataset+'.def')
        def_df = pd.read_csv(deffile_path,sep='\\s+',usecols=[0,1,2,3,4],names=['0','1','2','3','4'],header=None)
        abundance = 1
        mass = float(def_df[def_df['4'].isin(['mass'])]['0'].values[0])     # ExoMol mass (Dalton)
        if def_df.to_string().find('Uncertainty') != -1:
            check_uncertainty = int(def_df[def_df['2'].isin(['Uncertainty'])]['0'].values[0])
        else:
            check_uncertainty = 0
    else:
        raise ImportError("Please add the name of the database 'ExoMol' into the input file.")

      
    return (database, molecule, isotopologue, dataset, read_path, save_path, CoolingFunctions,
            ncputrans, ncpufiles, chunk_size, Ntemp, Tmax, molecule_id, isotopologue_id, abundance, mass,
            check_uncertainty)

cooling_output_root = "/Users/a123/Desktop/light_tasks/cooling"

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

runtime_record = {
    "start_time": time.time(),
    "trans_times": OrderedDict()  # Maintain file order
}

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
        # chunks = pd.read_csv(states_filename, compression='bz2', sep='\s+', header=None,
        #                      chunksize=100_000, iterator=True, low_memory=False, dtype=object)
        chunks = pd.read_csv(states_filename, compression='bz2', sep=r'\s+', header=None,
                            chunksize=100_000, iterator=True, dtype=object, engine="python")

    elif os.path.exists(states_filename.replace('.bz2','')):
        # chunks = pd.read_csv(states_filename.replace('.bz2',''), sep='\s+', header=None,
        #                      chunksize=100_000, iterator=True, low_memory=False, dtype=object)
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

    # Interpolate to get Q(T), even if T is not exactly equal to a row, it can return a reasonable value
    Ts = pf_df['T'].astype(float).values
    Qs = pf_df['Q'].astype(float).values

    if T < Ts.min() or T > Ts.max():
        raise ValueError(f"T = {T} is out of range of the PF file ({Ts.min()} ~ {Ts.max()})")

    Q = float(np.interp(T, Ts, Qs))  # Interpolate to calculate Q(T)

    return(Q)


# Cooling Function
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


# Calculate cooling function
def ProcessCoolingFunction(states_df,Ts,trans_df):
    merged_df = dd.merge(trans_df, states_df, left_on='u', right_on='id', 
                         how='inner').merge(states_df, left_on='l', right_on='id', how='inner', suffixes=("'", '"'))
    cooling_func_df = merged_df[['A',"E'",'E"',"g'"]]
    cooling_func_df['v'] = cal_v(cooling_func_df["E'"].values, cooling_func_df['E"'].values)
    num = len(cooling_func_df)
    if num > 0:
        A = cooling_func_df['A'].values
        v = cooling_func_df['v'].values
        Ep = cooling_func_df["E'"].values
        gp = cooling_func_df["g'"].values
        cooling_func = [calculate_cooling(A, v, Ep, gp, Ts[i], read_exomol_pf(read_path, Ts[i])) 
                        for i in tqdm(range(len(Ts)), desc='Calculating')]  


    else:
        # cooling_func = np.zeros(Tmax)
        cooling_func = np.zeros(len(Ts))
    return cooling_func


def get_partial_path(trans_filepath):
    trans_filename = os.path.basename(trans_filepath).replace(".bz2", "")
    cf_folder = os.path.join(cooling_output_root, molecule)
    partial_folder = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    os.makedirs(partial_folder, exist_ok=True)
    return os.path.join(partial_folder, trans_filename + ".cf.partial")

def save_partial_result(partial_path, Ts, cooling_func):
    np.savetxt(partial_path, np.column_stack((Ts, cooling_func)), fmt="%8.1f %20.8E")

# 合并 partial 文件的函数
def merge_all_partials(molecule, isotopologue, dataset, Ts, output_cf_path):
    partial_folder = os.path.join(cooling_output_root, molecule,
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


def calculate_cooling_func(states_df, Ts, trans_filepath):
    trans_filename = trans_filepath.split('/')[-1]
    print('Processeing transitions file:', trans_filename)

    wall_t0 = time.time()
    cpu_t0 = time.process_time()

    if trans_filepath.split('.')[-1] == 'bz2':
        trans_df_chunk_list = pd.read_csv(trans_filepath, compression='bz2', sep='\s+', header=None,
                                          usecols=[0,1,2],names=['u','l','A'], chunksize=chunk_size, 
                                          iterator=True, low_memory=False, encoding='utf-8')
        # Process multiple files in parallel
        with ThreadPoolExecutor(max_workers=ncputrans) as trans_executor:
            futures = [trans_executor.submit(ProcessCoolingFunction,states_df,Ts,trans_df_chunk) 
                       for trans_df_chunk in tqdm(trans_df_chunk_list, desc='Processing '+trans_filename)]
            cooling_func = sum(np.array([future.result() for future in tqdm(futures)]))
    else:
        trans_dd = dd.read_csv(trans_filepath, sep='\s+', header=None, usecols=[0,1,2],names=['u','l','A'],encoding='utf-8')
        trans_dd_list = trans_dd.partitions
        # Process multiple files in parallel
        with ThreadPoolExecutor(max_workers=ncputrans) as trans_executor:
            futures = [trans_executor.submit(ProcessCoolingFunction,states_df,Ts,trans_dd_list[i].compute(scheduler='threads')) 
                       for i in tqdm(range(len(list(trans_dd_list))), desc='Processing '+trans_filename)]
            cooling_func = sum(np.array([future.result() for future in tqdm(futures)]))

    wall_t1 = time.time()
    cpu_t1 = time.process_time()

    partial_path = get_partial_path(trans_filepath)
    try:
        save_partial_result(partial_path, Ts, cooling_func)
        # Only record time after successful save
        # Count the size and number of lines of the trans file
        decompressed_path = trans_filepath.replace('.bz2','')
        if trans_filepath.endswith('.bz2'):
            decompressed_path = trans_filepath.replace('.bz2', '')

        if os.path.exists(decompressed_path):
            file_size_gb = os.path.getsize(decompressed_path) / 1024**3
        else:
            file_size_gb = os.path.getsize(trans_filepath) / 1024**3

        if os.path.exists(decompressed_path):
            with open(decompressed_path, 'r') as f:
                line_count = sum(1 for _ in f)
        else:
            # fallback: Estimate line count using bz2 file (may not be accurate, but prevents errors)
            with bz2.open(trans_filepath, 'rt') as f:
                line_count = sum(1 for _ in f)

        # Record time and performance metrics
        runtime_record["trans_times"][trans_filename] = {
            "wall_time_s": round(wall_t1 - wall_t0, 3),
            "cpu_time_s": round(cpu_t1 - cpu_t0, 3),
            "lines": line_count,
            "size_GB": round(file_size_gb, 3),
            "wall_trans_rate": round(line_count / (wall_t1 - wall_t0), 2),
            "cpu_trans_rate": round(line_count / (cpu_t1 - cpu_t0), 2),
            "wall_time_per_GB": round((wall_t1 - wall_t0) / file_size_gb, 2),
            "cpu_time_per_GB": round((cpu_t1 - cpu_t0) / file_size_gb, 2),
        }
    except Exception as e:
        print(f"Failed to save partial result for {trans_filename}: {e}")

    return cooling_func

# Cooling function for ExoMol database
def exomol_cooling(states_df, Ntemp, Tmax):
    print('Calculate cooling functions.') 
    print('Running on ', ncputrans, 'cores.')
    t = Timer()
    t.start()
    Ts = np.array(range(Ntemp, Tmax + 1, Ntemp)) 
    print('Reading all transitions and calculating cooling functions ...')

    cf_folder = os.path.join(cooling_output_root, molecule)
    cf_path = os.path.join(cf_folder, f'{isotopologue}__{dataset}.cf')
    runtime_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__runtime.json")

    # If cf already exists, skip
    if os.path.exists(cf_path):
        print(f"Skip] Final .cf file already exists: {cf_path}")
        t.end()
        return

    # If runtime.json already exists, load recorded time
    if os.path.exists(runtime_path):
        with open(runtime_path, 'r') as f:
            existing_runtime = json.load(f)
    else:
        existing_runtime = {}
    old_trans_times = OrderedDict(existing_runtime.get("trans_times", {}))
    # Merge old and current recorded trans_times
    runtime_record["trans_times"] = OrderedDict({**old_trans_times, **runtime_record["trans_times"]})
    
    trans_filepaths = get_transfiles(read_path)

    # Run remaining trans files in parallel
    with ThreadPoolExecutor(max_workers=ncpufiles) as executor:
        futures = []
        for trans_filepath in tqdm(trans_filepaths):
            trans_filename = os.path.basename(trans_filepath).replace(".bz2", "")
            partial_path = get_partial_path(trans_filepath)
            if os.path.exists(partial_path):
                print(f"Skip] Already exists: {trans_filename}")
                if trans_filename in old_trans_times:
                    runtime_record["trans_times"][trans_filename] = old_trans_times[trans_filename]
                continue
            futures.append(executor.submit(calculate_cooling_func, states_df, Ts, trans_filepath))

        _ = [future.result() for future in futures]

    t.end()
    print('Finished reading all transitions and calculating cooling functions!\n')

    print('Saving cooling functions into file ...')   
    ts = Timer()    
    ts.start()     
    os.makedirs(cf_folder, exist_ok=True)

    partial_folder = os.path.join(cf_folder, f"{isotopologue}__{dataset}__partials")
    partial_files = glob.glob(os.path.join(partial_folder, "*.cf.partial"))

    if len(partial_files) == len(trans_filepaths):
        merge_all_partials(molecule, isotopologue, dataset, Ts, cf_path)
        print(f"Used all .cf.partial files to create final cooling function.")
    else:
        print(f"Only {len(partial_files)} / {len(trans_filepaths)} partials found. Will retry next time.")
        return

    ts.end()
    print('Cooling functions file has been saved:', cf_path, '\n')

    # Accumulate total time in runtime_record
    runtime_record["end_time"] = time.time()
    runtime_record["total_wall_time_s"] = round(
        sum(v["wall_time_s"] for v in runtime_record["trans_times"].values()), 3
    )
    runtime_record["total_cpu_time_s"] = round(
        sum(v["cpu_time_s"] for v in runtime_record["trans_times"].values()), 3
    )

    with open(runtime_path, "w") as f:
        json.dump(runtime_record, f, indent=2)

    print(f"Runtime record saved: {runtime_path}")
    print('Cooling functions have been saved!\n')  
    print('* * * * * - - - - - * * * * * - - - - - * * * * * - - - - - * * * * *\n')

# Get Results
def get_results(read_path): 
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
            exomol_cooling(states_df, Ntemp, Tmax)   
    print('\nThe program total running time:')    
    t_tot.end()
    cpu_time_main, wall_time_main = t_tot.cal()
    # Define runtime_path the same as in exomol_cooling
    cf_folder = os.path.join(cooling_output_root, molecule)
    runtime_path = os.path.join(cf_folder, f"{isotopologue}__{dataset}__runtime.json")

    # Safely load runtime_record
    if os.path.exists(runtime_path):
        with open(runtime_path, "r") as f:
            runtime_record = json.load(f)
    else:
        runtime_record = {}
    # Append main program time to runtime_record
    runtime_record["main_cpu_time_s"] = round(cpu_time_main, 3)
    runtime_record["main_wall_time_s"] = round(wall_time_main, 3)
    # Write to runtime.json again (it's okay if it already exists)
    with open(runtime_path, "w") as f:
        json.dump(runtime_record, f, indent=2)
        
    print('\nFinished!')
    pass

def main():
    get_results(read_path)

if __name__ == '__main__':
    main()



