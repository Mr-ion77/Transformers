# --- This MUST be at the top ---
import sys, os
from pathlib import Path

# Path to the script itself
script_path = Path(__file__).resolve()

# Path to .../QTransformer
project_dir = script_path.parent.parent
# Path to .../home/carlosR
base_dir = project_dir.parent

# Add BOTH directories to sys.path
sys.path.append(str(project_dir))  # Adds /home/carlosR/QTransformer
sys.path.append(str(base_dir))     # Adds /home/carlosR

import torch, itertools
import json
import pandas as pd
import mi_quantum.quantum as quantum  
import mi_quantum.data as data
from mi_quantum.data import preprocess_and_save, cut_extra_channels_from_latents
import mi_quantum.training as training
# ... other imports ...
from TelegramBot import SendToTelegram


# Define a proper PyTorch class
class CosineEncoding(torch.nn.Module):
    def __init__(self, nesting = 1):
        super(CosineEncoding, self).__init__()
        self.nesting = nesting
    def forward(self, x):
        out = x
        for _ in range(self.nesting):
            out = torch.cos(out * torch.pi/2)
        return out

class SlicedDataLoader:
    """Wraps a DataLoader to slice the channel dimension dynamically."""
    def __init__(self, dataloader, slice_indices):
        self.dataloader = dataloader
        self.slice_indices = slice_indices

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, (tuple, list)):
                # batch[0] is images, batch[1:] contains labels, indices, etc.
                sliced_images = batch[0][:, self.slice_indices, ...]
                # Yield the sliced images followed by everything else exactly as it was
                yield (sliced_images, *batch[1:])
            else:
                yield batch[:, self.slice_indices, ...]

    def __len__(self):
        return len(self.dataloader)


# --- Global Constants ---
DEFAULT_COLUMNS = [
    'test_auc_sel', 'test_acc_sel' ,'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc', 'train_acc', '#params'
]

default_redundancies = [
    {'quantum': False, 'U3_layers' : [0, 1, 2], 'entangling_layers' : [0, 1, 2], 'invert_embedding': [False, True]},
    {'q_config' : 'none', 'channels_out' : [ [0], [1], [2], [3]]}
]

model_map = {
            'patch_size': 'patch_size',
            'num_head': 'num_heads',
            'num_transf': 'num_transformer_blocks',
            'mlp_size': 'mlp_hidden_size',
            'Attention_N': 'Attention_N',
            'quantum': 'quantum_mlp',
            'U3_layers': 'U3_layers',
            'entangling_layers': 'entangling_layers',
            'entangle_method': 'entangle_method',
            'invert_embedding': 'invert_embedding',
            'RD': 'RD',
            'dropout': 'dropout',
            'attention_selection': 'attention_selection',
            'selection_amount': 'selection_amount',
            'special_cls': 'special_cls',
            'parallel': 'parallel',
            'q_stride': 'q_stride',
            'connectivity': 'connectivity',
            'quantum_classification' : 'quantum_classification',
            'train_q' : 'train_q',
            'preprocessor' : 'preprocessor',
            'hidden_size' : 'hidden_size',
}

train_map = {
    'learning_rate' : 'learning_rate',
    'wd' : 'wd',
    'patience' : 'patience',
    'scheduler_factor' : 'scheduler_factor',
    'augmentation_prob' : 'augmentation_prob',
    'val_train_pond' : 'val_train_pond'
}

EXPERIMENTS_DIRECTORY = "/home/carlosR/QTransformer/ExperimentsForThesis/"

def report_progress(exp_id, idx, total, status="running"):
    """
    Saves the experiment state to the specific subdirectory.
    
    Args:
        exp_id (str): The full folder name (e.g., '8_Experiment_7_With_More_QVC')
        idx (int): Current experiment/epoch index
        total (int): Total number of experiments/epochs
        status (str): Current status string
    """
    state = {
        "idx": idx, 
        "total": total, 
        "status": status
    }
    
    # Path construction: /home/carlosR/QTransformer/ExperimentsForThesis/8_Experiment_.../state.json
    # We use os.path.join to handle the slashes correctly
    path = os.path.join(EXPERIMENTS_DIRECTORY, exp_id, "state.json")
    
    try:
        # Ensure the directory exists before writing (safety check)
        if not os.path.exists(os.path.dirname(path)):
            print(f"Warning: Directory {os.path.dirname(path)} does not exist.")
            return

        with open(path, "w") as f:
            json.dump(state, f, indent=4)
            
    except Exception as e:
        print(f"Failed to report progress to {path}: {e}")

def build_kwargs(source, mapping):
    """Only includes keys from source if they exist, mapped to the correct name."""
    return {target_key: source[p_key] for p_key, target_key in mapping.items() if p_key in source}

def build_prefixed_kwargs(params_dict, mapping, prefix="1_"):
        """
        Filters params_dict for keys starting with 'prefix', 
        strips the prefix, and matches against the mapping.
        """
        # Create a temporary de-prefixed dict for the builder
        if prefix:
            clean_dict = {k[len(prefix):]: v for k, v in params_dict.items() if k.startswith(prefix)}
        else:
            # For p2/data, we take everything that DOES NOT start with '1_'
            clean_dict = {k: v for k, v in params_dict.items() if not str(k).startswith('1_')}
        
        return build_kwargs(clean_dict, mapping)

def make_dropout(drop):
    return {'embedding_attn': drop, 'after_attn': drop, 'feedforward': drop, 'embedding_pos': drop}

def custom_update_dict(original, updates):
    for key, value in updates.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            custom_update_dict(original[key], value)
        else:
            original[key] = value

def make_directories_for_experiment(variant = 'selformer', exp_config = None, p1 = None, p2 = None , all_iter = None,
                                    m1_iter = None, data_iter = None, m2_iter = None, columns = DEFAULT_COLUMNS):
    # This function is kept as-is from your script, but 'exp_config' needs a default
    if exp_config is None:
        raise ValueError("exp_config cannot be None")
        
    base_dir = f"../QTransformer_Results_and_Datasets" 
    exp_dir = os.path.join(base_dir, exp_config['experiment_id'])
    
    os.makedirs(os.path.join(base_dir, variant + ('_results/current_results' if not exp_config['second_at_a_time'] else '_results/current_results2')), exist_ok=True)
    try:
        os.makedirs(exp_dir, exist_ok=False)
    except FileExistsError as e:
        print(f"Warning: {e}")
        print(f"Directory for experiment ID '{exp_config['experiment_id']}' already exists. Results may be overwritten.")

    # Save hyperparameters
    with open(os.path.join(exp_dir, 'hyperparameters.json'), 'w') as f:
        f.write(f"Experiment Name: {exp_config['experiment_name']}\nBatch Size: {exp_config['B']}\n")
        if p1:
             f.write(f"Number of Epochs Selector: {exp_config.get('N1' if variant == 'selformer' else 'N', 'N/A')}\n")
             f.write('\nHyperparameters for Selector\n')
             json.dump(p1, f, indent=4)
        if p2:
             f.write(f"Number of Epochs Classifier: {exp_config.get('N2' if variant == 'selformer' else 'N', 'N/A')}\n")
             f.write('\nHyperparameters for Classifier\n')
             json.dump(p2, f, indent=4)
        if exp_config:
            f.write('\nGeneral settings of the simulation:\n')
            json.dump(exp_config, f, indent=4, default = str)
        
        # Handle iter blocks
        iter_blocks = [
            ('All iter', all_iter), ('Model 1 iter', m1_iter),
            ('Data iter', data_iter), ('Model 2 iter', m2_iter)
        ]
        for header, block in iter_blocks:
            if block:
                f.write(f'\n{header}\n')
                json.dump(block, f, indent=4)

    csv_path = os.path.join(exp_dir, 'results_grid_search.csv')
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=columns).to_csv(csv_path, mode='a', header=True, index=False)

    return csv_path, exp_dir

def is_redundant(p, iterator, redundancies):
    """
    Checks if the current parameter set 'p' is a redundant combination 
    based on a list of redundancy rules.
    """
    for rule in redundancies:
        # 1. Identify the trigger (the key with a single value)
        # and the dependents (the keys with lists of redundant values)
        trigger_keys = [k for k, v in rule.items() if not isinstance(v, list)]
        dependent_keys = [k for k, v in rule.items() if isinstance(v, list) and k in iterator.keys()]
        
        if not (trigger_keys and dependent_keys):
            continue
            
        # Check if ALL triggers match the current 'p'
        # (This allows for complex triggers like {'quantum': False, 'parallel': 1})
        matches_trigger = all(p.get(k) == rule[k] for k in trigger_keys)
        
        if matches_trigger:
            # If the trigger matches, we only allow the 'canonical' trial.
            # The canonical trial is where every dependent param equals the FIRST item in its list.
            for d_key in dependent_keys:
                canonical_value = rule[d_key][0]
                if p.get(d_key) != canonical_value:
                    return True  # It is redundant, skip it!
                    
    return False # No redundancy rules triggered

def iterate_all_iter(root_id, base_exp_config, all_iter, p_bases):
    """
    A generator that handles the 'all_iter' loop logic.
    Yields: (current_exp_config, current_p_bases, current_params_all, all_iter_counter)
    """
    for all_iter_counter, pack_all in enumerate(itertools.product(*all_iter.values())):

        all_iter_counter += 1
        
        experiment_id = root_id
        if all_iter:
            experiment_id += '/all_iter_' + str(all_iter_counter) + '/'
        
        current_exp_config = base_exp_config.copy()
        current_exp_config['experiment_id'] = experiment_id
        
        current_p_bases = [p.copy() for p in p_bases]
        current_params_all = dict(zip(all_iter.keys(), pack_all))
        
        custom_update_dict(current_exp_config, current_params_all)
        for p in current_p_bases:
            custom_update_dict(p, current_params_all)
            
        print("---" * 10)
        print(f"Current overall params (all_iter {all_iter_counter}):", current_params_all)
        print("---" * 10)

        yield current_exp_config, current_p_bases, current_params_all, all_iter_counter

def load_data(data_flag='dermamnist', pixel=28, batch_size=32, **kwargs):
    """ Loads and returns the MedMNIST dataloaders. """
    print(f"Loading {data_flag} data with batch size {batch_size}...")
    
    loader_args = {
        'pixel': pixel, 'data_flag': data_flag, 'batch_size': batch_size,
        'num_workers': kwargs.get('num_workers', 4),
        'pin_memory': kwargs.get('pin_memory', True),
    }
    
    if kwargs.get('extra_tr_without_trans', False):
        loader_args['extra_tr_without_trans'] = True
        return data.get_medmnist_dataloaders(**loader_args) # (notrans_train, train, val, test, shape)
    else:
        return data.get_medmnist_dataloaders(**loader_args) # (train, val, test, shape)

def send_telegram_report(exp_config, csv_path=None, columns=None, title=None, error=None, progress=None):
    """Handles sending Telegram messages for progress, errors, or final reports."""
    if not exp_config.get('send_telegram', False):
        return

    try:
        if error:
            print(f"Sending error to Telegram: {error}")
            SendToTelegram(progress=progress, error_message=str(error))
        elif csv_path: # Final report
            print("Sending final report to Telegram...")
            SendToTelegram(csv_file=csv_path, columns=columns, title=title)
        elif progress is not None: # Progress update
             # Only print/send on specific milestones to avoid spam
             if progress in [0, 25, 50, 75, 100]: 
                print(f"Sending progress update: {progress}%")
                SendToTelegram(progress=progress)
    except Exception as e:
        print(f"Failed to send message to Telegram: {e}")

def log_experiment_row(csv_path, row_data, column_order):
    """ Appends a single row of experiment results to the CSV file. """
    df = pd.DataFrame([row_data], columns=column_order)
    df.to_csv(csv_path, mode='a', header=False, index=False)


def make_experiment_transformer(exp_config, p_base, all_iter={}, data_iter = {}, model_iter={}, redundancies = default_redundancies, graph_columns=['attention_selection', 'test_auc']):
    
    root_id = exp_config['experiment_id']
    experiment_name = exp_config['experiment_name']
    # Define columns for CSV logging
    columns = ['idx', 'all_iter_idx', 'q_config'] + list(all_iter.keys()) + list(data_iter.keys()) + list(model_iter.keys()) + DEFAULT_COLUMNS
    try:
        columns.remove('test_auc_sel')
        columns.remove('test_acc_sel')
    except:
        pass

    progress = 0
    report_progress(experiment_name, 0, exp_config['num_experiments'])
    
    # --- Use the 'all_iter' generator ---
    for current_exp_config, (p,), current_params_all, all_iter_counter in iterate_all_iter(
        root_id, exp_config, all_iter, [p_base]
    ):
        
        csv_path, save_path = None, None # Init for error handling
        
        #try:

        NoneBool, QuantumBool, CosBool = 'none' in exp_config['q_config'], 'quantum' in exp_config['q_config'], 'cosine' in exp_config['q_config']
        print(f"Current quantum configuration:\nNormal latent representations: {NoneBool}\nQuantum latent representations: {QuantumBool},\nCosine Encoding latent representations: {CosBool}\n")


        # --- Create directories and CSV file ---
        csv_path, save_path = make_directories_for_experiment(
            variant='transformer', # Specify variant
            exp_config=current_exp_config, 
            p1=p, # Use p1 slot for the single param dict
            all_iter=all_iter, 
            data_iter= data_iter,
            m2_iter=model_iter, # m2_iter for consistency, as it's the "model" iter
            columns=columns
        )

        # --- Load data ---
        notrans_train_dl, train_dl, val_dl, test_dl, shape = data.get_medmnist_dataloaders(
            pixel=exp_config['pixels'], 
            data_flag='dermamnist', 
            extra_tr_without_trans=True, 
            batch_size=exp_config['B'], 
            num_workers=4, 
            pin_memory=True
        )
        
        num_channels = shape[-1] if current_exp_config['channels_last'] else shape[0]
        processed_once = False
        # Send initial progress
        send_telegram_report(current_exp_config, progress=0)   

        # --- Extract channels_out to prevent redundant preprocessing ---
        has_channels_out = 'channels_out' in data_iter
        if has_channels_out:
            master_channels_out = sorted(list(set(c for sublist in data_iter['channels_out'] for c in sublist)))
            data_iter_base = {k: v for k, v in data_iter.items() if k != 'channels_out'}
            channels_out_list = data_iter['channels_out']
        else:
            master_channels_out = p.get('channels_out', [-1])
            data_iter_base = data_iter
            channels_out_list = [master_channels_out]

        # --- Calculate total iterations for global progress ---
        num_data_base = len(list(itertools.product(*data_iter_base.values()))) if data_iter_base else 1
        num_models = len(list(itertools.product(*model_iter.values()))) if model_iter else 1
        
        total_iterations = (
            num_data_base * len(channels_out_list) * num_models * len(exp_config['q_config']) * current_exp_config['num_experiments']
        )
        current_iteration = 0
        data_iteration = 0
        
        print(f"\nTotal planned iterations across all grid search parameters: {total_iterations}")

        # ==========================================================
        # 1. OUTERMOST LOOP: DATA GENERATION
        # ==========================================================

        for pack_data_base in itertools.product(*data_iter_base.values()) if data_iter_base else [()]:
            model_iteration = 0
            current_params_base = dict(zip(data_iter_base.keys(), pack_data_base))
            
            # Setup master parameters for preprocessing
            p_master = p.copy()
            custom_update_dict(p_master, current_params_base)
            p_master['channels_out'] = master_channels_out

            paddings = { 2 : { 'Up': 1, 'Down': 0, 'Left': 1, 'Right': 0 }, 3 : { 'Up': 1, 'Down': 1, 'Left': 1, 'Right': 1 } }

            Kernels = { 
                'none': torch.nn.Identity(),
                'quantum': quantum.quanvolution.QuantumConv2D(
                    kernel_size = p_master['quanv_kernel_size'],
                    stride = p_master['stride'],
                    padding = paddings[p_master['quanv_kernel_size']],
                    channels_out = p_master['channels_out'], 
                    ancilla = p_master['ancilla'],
                    graphs = p_master['connectivity'],
                    entangle_method = p_master['entangle_method'], 
                    invert_embedding = p_master['invert_embedding']
                ),
                'cosine': CosineEncoding(),
                'cosine2': CosineEncoding(),
            }

            Kernels = { k : v for k, v in Kernels.items() if ( (k == 'none') and NoneBool ) or ( (k == 'quantum') and QuantumBool ) or ( (k == 'cosine') and CosBool ) or ( (k == 'cosine2') and ('cosine2' in current_exp_config['q_config']) )}
            print(f"\n[DATA GENERATION] Kernels to be used: {list(Kernels.keys())} for params {pack_data_base}")
            
            # PROCESS ONCE per data configuration
            Latents_master = preprocess_and_save(
                B = exp_config['B'],
                DataLoaders = [notrans_train_dl, val_dl, test_dl],
                kernels = Kernels,
                save_path = f"../QTransformer_Results_and_Datasets/transformer_results/quantum_datasets",
                mode = 'standard',
                model1 = None,
                p1 = p_master,
                num_channels = num_channels,
                device = exp_config['device'],
                flatten_extra_channels = exp_config.get('augmenting', False),
                concatenate_original = exp_config.get('concatenate_original', False)
            )

            # ==========================================================
            # 2. CHANNELS_OUT Slicer Loop
            # ==========================================================
            channels_out_iteration = 0
            have_trained_none = False
            for current_channels_out in channels_out_list:
                
                current_params_data = current_params_base.copy()
                if has_channels_out:
                    current_params_data['channels_out'] = current_channels_out

                if current_params_data.get('channels_out', [None]) == [1] and current_params_data.get('entangle_method', [None]) == 'edges' or current_params_data.get('channels_out', [None]) == [3] and current_params_data.get('entangle_method', [None]) == 'CRX':
                    continue

                custom_update_dict(p, current_params_data)

                # Slicing logic
                Latents = {}
                if current_channels_out == master_channels_out:
                    Latents = Latents_master
                else:
                    slice_indices = []
                    for q in current_channels_out:
                        master_idx = master_channels_out.index(q)
                        slice_indices.extend(range(master_idx * num_channels, (master_idx + 1) * num_channels))

                    for q_config_key, (tr_dl, val_dl, te_dl, shape2) in Latents_master.items():
                        if q_config_key in ['quantum', 'cosine', 'cosine2']:
                            Latents[q_config_key] = (
                                SlicedDataLoader(tr_dl, slice_indices),
                                SlicedDataLoader(val_dl, slice_indices),
                                SlicedDataLoader(te_dl, slice_indices),
                                (len(slice_indices), shape2[1], shape2[2])
                            )
                        else:
                            Latents[q_config_key] = (tr_dl, val_dl, te_dl, shape2)

                # ==========================================================
                # 3. MODEL PARAMS LOOP
                # ==========================================================
                for pack_model in itertools.product(*model_iter.values()):

                    # ==========================================================
                    # 4. Q_CONFIG LOOP
                    # ==========================================================
                    for q_config in exp_config['q_config']:

                        if q_config == 'none' and have_trained_none:
                            print("Skipping redundant 'none' configuration")
                            continue

                        latent_train_dl, latent_val_dl, latent_test_dl, shape2 = Latents[q_config]

                        current_params_model = dict(zip(model_iter.keys(), pack_model))
                        if exp_config.get('square', True):
                            custom_update_dict( current_params_model, {'num_transf' : current_params_model.get('parallel',p['parallel']) } )
                        
                        custom_update_dict(p, current_params_model)

                        if is_redundant(p, model_iter, redundancies):
                            print("Skipping redundant parameter combination")
                            continue

                        # ==========================================================
                        # 5. INNERMOST LOOP: EXPERIMENT REPETITIONS
                        # ==========================================================
                        for idx in range(current_exp_config['num_experiments']):
                            
                            # Note: You might want to adjust how progress is reported now 
                            # that it tracks Grid Search combos * runs, rather than just runs.
                            progress = int( 100 * (data_iteration * current_exp_config['num_experiments']*model_iteration + idx) // total_iterations )
                            send_telegram_report(current_exp_config, progress=progress)
                            report_progress(experiment_name, idx + 1, exp_config['num_experiments'])
                            
                            print(f"\n===== Exp Run {idx + 1}/{current_exp_config['num_experiments']} | Q-Config: {q_config} =====")
                            print(f"Current model params: {current_params_model}")

                            oneortwo = 'current_results' if not current_exp_config['second_at_a_time'] else 'current_results2'
                            aux_save_path = Path(f"../QTransformer_Results_and_Datasets/transformer_results/" + oneortwo + f"/grid_search{idx}")
                            aux_save_path.mkdir(parents=True, exist_ok=True)
                            
                            hidden_size = len( p['channels_out'] ) * shape[0] * p['patch_size']**2
                            
                            model_kwargs = {
                                'img_size': shape[-1],
                                'num_channels': shape[0],
                                'num_classes': current_exp_config['num_classes'],
                                'hidden_size': hidden_size,
                                'channels_last': current_exp_config['channels_last'],
                                'quantum_classification': False,
                            }
                            model_kwargs.update(build_kwargs(p, model_map))

                            if 'dropout' in p:
                                model_kwargs['dropout'] = make_dropout(p['dropout'])

                            # FRESH MODEL INITIALIZATION PER RUN
                            model = quantum.vit.VisionTransformer(**model_kwargs)
                            
                            train_kwargs = {
                                'model': model,
                                'train_dataloader': latent_train_dl,
                                'valid_dataloader': latent_val_dl,
                                'test_dataloader': latent_test_dl,
                                'num_classes': current_exp_config['num_classes'],
                                'num_epochs': current_exp_config['N'],
                                'device': current_exp_config['device'],
                                'res_folder': str(aux_save_path),
                                'verbose': current_exp_config.get('verbose', False) ,
                                'mode' : 'transformer',
                                'optimizer' : current_exp_config.get('optimizer', 'Adam')
                            }
                            train_kwargs.update(build_kwargs(p, train_map))
                            train_kwargs['parameters'] = current_params_model  

                            results = training.train_and_evaluate(**train_kwargs)
                            test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params = results

                            # Log Results
                            row = {
                                'idx': idx + 1, 'q_config': q_config, 'all_iter_idx': all_iter_counter,
                                'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 
                                'val_acc': val_acc, 'train_auc': train_auc, 'train_acc': train_acc,
                                '#params': params,
                                **p 
                            }
                            log_experiment_row(csv_path, row, columns)

                    model_iteration += 1
                channels_out_iteration += 1
                have_trained_none = True if q_config == 'none' else have_trained_none
            data_iteration += 1


        # --- Send final report (using new helper) ---
        send_telegram_report(
            current_exp_config, 
            csv_path=csv_path, 
            columns=graph_columns,
            title=f"{current_exp_config['experiment_name']} (all_iter {all_iter_counter})"
        )

        # Mark as completed before finishing
        report_progress(experiment_name, exp_config['num_experiments'], exp_config['num_experiments'], status="COMPLETED ✅")

        #except Exception as e:
        #     # --- Send error report (using new helper) ---
        #     send_telegram_report(current_exp_config, error=e, progress=progress)
        #     print(f"Error encountered in all_iter {all_iter_counter}: {e}")
        #     # Continue to the next 'all_iter' loop if possible
        #     continue
        df_results = pd.read_csv(csv_path)
        return df_results


def make_experiment_selformer(exp_config, p1_base, data_base, p2_base, all_iter={}, m1_iter={}, data_iter={}, m2_iter={}, redundancies=default_redundancies, graph_columns=['q_config', 'test_auc']):
    """ 
    Improved version of the two-step Selformer pipeline.
    1. Trains a selector model.
    2. Uses selected patches to train a classifier model on latent representations.
    """
    
    root_id = exp_config['experiment_id']
    all_iter_counter = 0
    
    # Define columns for CSV logging
    columns = (['idx', 'all_iter_idx', 'q_config', 'channels_out', 'latent_shape', '2_selection_amount'] + 
               list(all_iter.keys()) + list(m1_iter.keys()) + list(data_iter.keys()) + 
               list(m2_iter.keys()) + DEFAULT_COLUMNS + ['#params_sel', '#params_class'])
    
    try:
        columns.remove('#params') 
    except ValueError:
        pass

    # --- Use the 'all_iter' generator style ---
    for pack_all in itertools.product(*all_iter.values()):
        all_iter_counter += 1
        current_params_all = dict(zip(all_iter.keys(), pack_all))
        
        # Create fresh copies and update with outer loop params
        p1 = p1_base.copy()
        p2 = p2_base.copy()
        data_ = data_base.copy()
        current_exp_config = exp_config.copy()
        
        custom_update_dict(p1, current_params_all)
        custom_update_dict(p2, current_params_all)
        custom_update_dict(current_exp_config, current_params_all)
        
        experiment_id = f"{root_id}/all_iter_{all_iter_counter}/"
        current_exp_config.update({'experiment_id': experiment_id, 'trained_selector_once': False})
        
        progress = 0
        csv_path, save_path = make_directories_for_experiment(
            variant='selformer', 
            exp_config=current_exp_config, 
            p1=p1, p2=p2, 
            all_iter=all_iter, m1_iter=m1_iter, data_iter=data_iter, m2_iter=m2_iter, 
            columns=columns
        )

        # --- Load data ---
        notrans_train_dl, train_dl, val_dl, test_dl, shape = data.get_medmnist_dataloaders(
            pixel=current_exp_config['pixels'], 
            data_flag='dermamnist', 
            extra_tr_without_trans=True, 
            batch_size=current_exp_config['B'], 
            num_workers=4, pin_memory=True
        )
        
        num_channels = shape[-1] if current_exp_config['channels_last'] else shape[0]
        send_telegram_report(current_exp_config, progress=0)

        # --- Experiment Repetition Loop ---
        for idx_ in range(current_exp_config['num_experiments']):
            idx = idx_ + 1
            progress = int(100 * idx // current_exp_config['num_experiments'])
            send_telegram_report(current_exp_config, progress=progress)
            
            print(f"\n===== Experiment Run {idx} / {current_exp_config['num_experiments']} =====")
            
            oneortwo = 'current_results' if not current_exp_config['second_at_a_time'] else 'current_results2'
            run_save_path = Path(f"../QTransformer_Results_and_Datasets/selformer_results/{oneortwo}/grid_search{idx}")
            selector_save_path = run_save_path / 'selector'
            classifier_save_path = run_save_path / 'classifier'
            selector_save_path.mkdir(parents=True, exist_ok=True)
            classifier_save_path.mkdir(parents=True, exist_ok=True)

            NoneBool, QuantumBool, CosBool = 'none' in current_exp_config['q_config'], 'patchwise' in current_exp_config['q_config'], 'cosine' in current_exp_config['q_config']

            # --- Selector Loop (m1_iter) ---
            for pack_sel in itertools.product(*m1_iter.values()):
                current_params_sel = dict(zip(m1_iter.keys(), pack_sel))
                custom_update_dict(p1, current_params_sel)
                
                if current_exp_config.get('square', True):
                    custom_update_dict(p1, {'1_num_transf': p1['1_parallel']})

                if (not current_exp_config['trained_selector_once']) or current_exp_config['repeat_selector']:
                    current_exp_config['trained_selector_once'] = True
                    
                    # Initialize Selector Model
                    model1_kwargs = {
                        'img_size': shape[-1],
                        'num_channels': shape[0],
                        'num_classes': current_exp_config['num_classes'],
                        'hidden_size': p1['1_hidden_size'],
                        'channels_last': current_exp_config['channels_last'],
                        'quantum_classification': False,
                    }
                    model1_kwargs.update(build_prefixed_kwargs(p1, model_map))
                    model1 = quantum.vit.VisionTransformer(**model1_kwargs)
                    
                    # Train Selector
                    train1_kwargs = {
                        'model': model1, 'train_dataloader': train_dl, 'valid_dataloader': val_dl, 'test_dataloader': test_dl,
                        'num_classes': current_exp_config['num_classes'], 'num_epochs': current_exp_config['N1'],
                        'device': current_exp_config['device'], 'res_folder': str(selector_save_path), 
                        'parameters': current_params_sel, 'verbose': current_exp_config.get('verbose', False), 'mode': 'transformer'
                    }
                    train1_kwargs.update(build_prefixed_kwargs(p1, train_map))
                    
                    results_sel = training.train_and_evaluate(**train1_kwargs)
                    test_auc_sel, test_acc_sel, val_auc_sel, val_acc_sel, train_auc_sel, _, params_sel = results_sel

                processed_data_once = False
                # --- Data Preprocessing Loop (data_iter) ---
                for pack_data in itertools.product(*data_iter.values()):
                    if pack_data or not processed_data_once:
                        processed_data_once = True
                        current_params_data = dict(zip(data_iter.keys(), pack_data))
                        custom_update_dict(data_, current_params_data)
                        custom_update_dict(p2, current_params_data)

                        # Kernel Setup
                        paddings = {2: {'Up': 1, 'Down': 0, 'Left': 1, 'Right': 0}, 3: {'Up': 1, 'Down': 1, 'Left': 1, 'Right': 1}}
                        Kernels = {
                            'none': torch.nn.Identity(),
                            'patchwise': quantum.quanvolution.QuantumConv2D(
                                kernel_size=data_['quanv_kernel_size'], stride=1, padding=paddings[data_['quanv_kernel_size']],
                                channels_out=data_['channels_out'], ancilla=data_['ancilla'], graphs=data_['connectivity'],
                                entangle_method=data_['entangle_method'], invert_embedding=data_['invert_embedding']
                            ),
                            'cosine': CosineEncoding()
                        }
                        Kernels = {k: v for k, v in Kernels.items() if (k == 'none' and NoneBool) or (k == 'patchwise' and QuantumBool) or (k == 'cosine' and CosBool)}

                        Latents = preprocess_and_save(
                            B=current_exp_config['B'], DataLoaders=[notrans_train_dl, val_dl, test_dl], kernels=Kernels,
                            save_path=f"../QTransformer_Results_and_Datasets/selformer_results/quantum_datasets",
                            mode='by_selected_patches', model1=model1, p1=p1, num_channels=num_channels, channels_last = current_exp_config['channels_last'],
                            flatten_extra_channels= data_['flatten_extra_channels'], device=current_exp_config['device'],
                            flatten=not current_exp_config['augmenting'], concatenate_original=current_exp_config.get('concatenate_original', False)
                        )

                        original_measured = data_['channels_out']
                        original_channels_out = len(data_['channels_out'])
                        TrainedFlattenedOnce = False
                        channel_iterator = range(original_channels_out * (current_exp_config['rewind_channels'] != 0), 0, -current_exp_config['rewind_channels']) if current_exp_config['rewind_channels'] > 0 else [original_channels_out]

                        # --- Classifier Loop (m2_iter) ---
                        for config_key in Kernels.keys():
                            for i in channel_iterator:
                                # Rewind logic
                                Current_Latents = Latents
                                if config_key == 'patchwise':
                                    if data_['flatten_extra_channels']:
                                        if TrainedFlattenedOnce: continue
                                        TrainedFlattenedOnce = True
                                    elif current_exp_config['rewind_channels'] > 0:
                                        data_.update({'channels_out': original_measured[:i]})
                                        Current_Latents = cut_extra_channels_from_latents(Latents, i, original_channels_out)

                                for pack_class in itertools.product(*m2_iter.values()):
                                    current_params_class = dict(zip(m2_iter.keys(), pack_class))
                                    
                                    # Redundancy Check
                                    if is_redundant(current_params_class, m2_iter, redundancies):
                                        continue

                                    custom_update_dict(p2, current_params_class)
                                    # Update selection amount based on channels
                                    sel_amt = p1['1_selection_amount'] + (1 - data_['flatten_extra_channels']) * p2['len_channels_scaler'] * original_channels_out
                                    p2.update({'selection_amount': sel_amt})
                                    
                                    if current_exp_config.get('square', True):
                                        custom_update_dict(p2, {'num_transf': p2['parallel']})

                                    latent_train, latent_val, latent_test, shape2 = Current_Latents[config_key]
                                    hidden_size2 = shape2[-1] if not current_exp_config['augmenting'] else shape2[-1] * shape2[-2] * shape2[-3]

                                    # Initialize Classifier Model
                                    model2_kwargs = {
                                        'img_size': shape2[-1], 'num_channels': shape[0], 'num_classes': current_exp_config['num_classes'],
                                        'hidden_size': hidden_size2, 'channels_last': current_exp_config['channels_last'], 'quantum_classification': False,
                                        'patch_embedding_required' : 'flatten' if current_exp_config['augmenting'] else 'none'
                                    }
                                    model2_kwargs.update(build_kwargs(p2, model_map))
                                    model2 = quantum.vit.VisionTransformer(**model2_kwargs)

                                    # Train Classifier
                                    train2_kwargs = {
                                        'model': model2, 'train_dataloader': latent_train, 'valid_dataloader': latent_val, 'test_dataloader': latent_test,
                                        'num_classes': current_exp_config['num_classes'], 'num_epochs': current_exp_config['N2'],
                                        'device': current_exp_config['device'], 'res_folder': str(classifier_save_path), 'mode': 'selected',
                                        'parameters': current_params_class, 'verbose': current_exp_config.get('verbose', False) 
                                    }
                                    train2_kwargs.update(build_kwargs(p2, train_map))

                                    results = training.train_and_evaluate(**train2_kwargs)
                                    test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params_class = results

                                    # Log results
                                    row = {
                                        'idx': idx, 'all_iter_idx': all_iter_counter, 'q_config': config_key, 
                                        'channels_out': len(data_['channels_out']) if config_key == 'patchwise' else 0,
                                        'latent_shape': [*shape2], '2_selection_amount': p2['selection_amount'],
                                        'test_auc_sel': test_auc_sel, 'test_acc_sel': test_acc_sel, '#params_sel': params_sel,
                                        'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc,
                                        'train_auc': train_auc, 'train_acc': train_acc, '#params_class': params_class,
                                        **p1, **p2
                                    }
                                    log_experiment_row(csv_path, row, columns)

        # Final Report
        send_telegram_report(
            current_exp_config, csv_path=csv_path, columns=graph_columns,
            title=f"{current_exp_config['experiment_name']} (all_iter {all_iter_counter})"
        )

    return pd.read_csv(csv_path)


class AutoencoderLatentWrapper(torch.nn.Module):
    """Wraps the AutoEnformer to output latents with (Batch, ...) shape."""
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        # x shape is (Batch, C, H, W)
        B = x.shape[0] 
        latents = self.autoencoder.get_latent_representation(x)

        # Ensure it has a channel dim for 2D processing if needed
        # Case A: (Batch, Hidden) -> (Batch, 1, Hidden)
        if len(latents.shape) == 2:
            return latents.unsqueeze(1)
            
        # Case B: (Batch, Parallel/Seq, Hidden) -> Keep as is (Channels = Parallel)
        return latents

def make_experiment_autoenformer(exp_config, p1_base, p2_base, data_kernels={}, all_iter={}, m1_iter={}, data_iter={}, m2_iter={}, redundancies=default_redundancies, graph_columns=['q_config', 'test_auc'], repeat_autoencoder = False):
    """
    Automated pipeline for AutoEnformer:
    1. Train AutoEnformer (Autoencoder).
    2. Extract Latents + Apply Quantum/Processing Kernels (passed via data_kernels).
    3. Train Classifier on processed latents.
    """

    root_id = exp_config['experiment_id']
    all_iter_counter = 0

    # Define columns for CSV logging
    columns = (['idx', 'all_iter_idx', 'q_config', 'latent_shape'] + 
               list(all_iter.keys()) + list(m1_iter.keys()) + list(data_iter.keys()) + 
               list(m2_iter.keys()) + DEFAULT_COLUMNS + ['#params_ae', '#params_class', 'test_mse', 'val_mse'])
    try:
        columns.remove('test_auc_sel')
        columns.remove('test_acc_sel')
        columns.remove('test_acc_sel')
    except:
        pass

    # --- Use the 'all_iter' generator style ---
    for pack_all in itertools.product(*all_iter.values()):
        all_iter_counter += 1
        current_params_all = dict(zip(all_iter.keys(), pack_all))
        autoencoded_once = False
        # Create fresh copies
        p1 = p1_base.copy() # Autoencoder params
        p2 = p2_base.copy() # Classifier params
        current_exp_config = exp_config.copy()

        custom_update_dict(p1, current_params_all)
        custom_update_dict(p2, current_params_all)
        custom_update_dict(current_exp_config, current_params_all)

        experiment_id = f"{root_id}/all_iter_{all_iter_counter}/"
        current_exp_config.update({'experiment_id': experiment_id, 'trained_autoencoder_once': False})

        progress = 0
        csv_path, save_path = make_directories_for_experiment(
            variant='autoenformer',
            exp_config=current_exp_config,
            p1=p1, p2=p2,
            all_iter=all_iter, m1_iter=m1_iter, data_iter=data_iter, m2_iter=m2_iter,
            columns=columns
        )

        # --- Load data ---
        # Note: Autoencoder usually needs raw images, so we use notrans_train_dl or train_dl depending on requirements
        notrans_train_dl, train_dl, val_dl, test_dl, shape = data.get_medmnist_dataloaders(
            pixel=current_exp_config['pixels'],
            data_flag='dermamnist',
            extra_tr_without_trans=True,
            batch_size=current_exp_config['B'],
            num_workers=4, pin_memory=True
        )

        num_channels = shape[-1] if current_exp_config['channels_last'] else shape[0]
        send_telegram_report(current_exp_config, progress=0)

        # --- Experiment Repetition Loop ---
        for idx_ in range(current_exp_config['num_experiments']):
            idx = idx_ + 1
            progress = int(100 * idx // current_exp_config['num_experiments'])
            send_telegram_report(current_exp_config, progress=progress)
            report_progress(current_exp_config['experiment_name'], idx, current_exp_config['num_experiments'])

            print(f"\n===== Experiment Run {idx} / {current_exp_config['num_experiments']} =====")

            oneortwo = 'current_results' if not current_exp_config['second_at_a_time'] else 'current_results2'
            run_save_path = Path(f"../QTransformer_Results_and_Datasets/autoenformer_results/{oneortwo}/grid_search{idx}")
            ae_save_path = run_save_path / 'autoencoder'
            classifier_save_path = run_save_path / 'classifier'
            ae_save_path.mkdir(parents=True, exist_ok=True)
            classifier_save_path.mkdir(parents=True, exist_ok=True)

            # --- Step 1: Autoencoder Loop (m1_iter) ---
            for pack_ae in itertools.product(*m1_iter.values()):
                current_params_ae = dict(zip(m1_iter.keys(), pack_ae))
                custom_update_dict(p1, current_params_ae)

                if (not current_exp_config['trained_autoencoder_once']) or current_exp_config.get('repeat_autoencoder', False):
                    current_exp_config['trained_autoencoder_once'] = True

                    # Initialize AutoEnformer
                    model1_kwargs = {
                        'img_size': shape[-1], # Assuming square
                        'num_channels': num_channels,
                        'patch_size': p1['patch_size'],
                        'hidden_size': p1['hidden_size'],
                        'num_heads': p1['num_head'],
                        'num_transformer_blocks': p1['num_transf'],
                        'attention_selection': p1['attention_selection'],
                        'mlp_hidden_size': p1['mlp_size'],
                        'Attention_N': p1['Attention_N'],
                        'dropout': p1.get('dropout', 0.0),
                        'parallel': p1['parallel'],
                        'channels_last': False, 
                        'q_stride': p1.get('q_stride', 1),
                        'special_cls' : p1.get('special_cls', False)
                    }
                    
                    model1 = quantum.vit.AutoEnformer(**model1_kwargs)

                    # Train Autoencoder
                    train1_kwargs = {
                        'model': model1,
                        'train_dataloader': train_dl,
                        'valid_dataloader': val_dl,
                        'test_dataloader': test_dl,
                        'num_classes': current_exp_config['num_classes'], 
                        'learning_rate': p1['learning_rate'],
                        'num_epochs': current_exp_config['N1'],
                        'device': current_exp_config['device'],
                        'res_folder': str(ae_save_path),
                        'wd': p1['weight_decay'],
                        'patience': p1['patience'],
                        'scheduler_factor': p1['scheduler_factor'],
                        'mode': 'autoencoder',
                        'save_reconstructed_images': (idx == 1),
                        'verbose': current_exp_config.get('verbose', True)
                    }
                    
                    test_mse, val_mse, params_ae = training.train_and_evaluate(**train1_kwargs)
                    print(f"Autoencoder Trained. MSE: {test_mse:.5f}")

                    # Wrap model for Latent Extraction
                    model1.eval()
                    # Ensure AutoencoderLatentWrapper is defined in your scope
                    latent_extractor = AutoencoderLatentWrapper(model1).to(current_exp_config['device'])

                processed_data_once = False
                
                # --- Step 2: Data Preprocessing (Latent -> Quantum -> Save) (data_iter) ---
                for pack_data in itertools.product(*data_iter.values()):
                    if pack_data or not processed_data_once:
                        processed_data_once = True
                        current_params_data = dict(zip(data_iter.keys(), pack_data))
                        custom_update_dict(p2, current_params_data) 

                        # Construct chained Kernels: Latent Extractor -> Data Kernel
                        ActiveKernels = {}
                        
                        # Filter keys based on current configuration and wrap them
                        for k, v in data_kernels.items():
                            if k in current_exp_config['q_config']:
                                # Ensure the kernel module is on the correct device
                                v = v.to(current_exp_config['device'])
                                ActiveKernels[k] = torch.nn.Sequential(latent_extractor, v)

                        # Generate Datasets
                        Latents = preprocess_and_save(
                            B=current_exp_config['B'],
                            DataLoaders=[ notrans_train_dl, val_dl,test_dl], 
                            kernels=ActiveKernels,
                            save_path=f"../QTransformer_Results_and_Datasets/autoenformer_results/quantum_datasets",
                            mode='standard', 
                            model1=None, 
                            p1=None,
                            num_channels=num_channels, 
                            device=current_exp_config['device'],
                            concatenate_original=False 
                        )

                    # --- Step 3: Classifier Loop (m2_iter) ---
                    for config_key in ActiveKernels.keys():
                        latent_train, latent_val, latent_test, shape2 = Latents[config_key]

                        print(f"DEBUG FOR SHAPES: supossed shape (shape2): {shape2}, actual shape: {next(iter(latent_train))[0].shape}")
                        
                        for pack_class in itertools.product(*m2_iter.values()):
                            current_params_class = dict(zip(m2_iter.keys(), pack_class))

                            if is_redundant(current_params_class, m2_iter, redundancies):
                                continue

                            custom_update_dict(p2, current_params_class)
                            
                            # Setup Classifier Model (DeViT)
                            model2 = quantum.vit.DeViT(
                                num_classes=current_exp_config['num_classes'],
                                p=p2,
                                shape=shape2, 
                                dim_latent=shape2[-1] 
                            )

                            train2_kwargs = {
                                'model': model2,
                                'train_dataloader': latent_train,
                                'valid_dataloader': latent_val,
                                'test_dataloader': latent_test,
                                'num_classes': current_exp_config['num_classes'],
                                'learning_rate': p2['learning_rate'],
                                'num_epochs': current_exp_config['N2'],
                                'device': current_exp_config['device'],
                                'res_folder': str(classifier_save_path),
                                'wd': p2['weight_decay'],
                                'patience': p2['patience'],
                                'scheduler_factor': p2['scheduler_factor'],
                                'mode': 'transformer',
                                'verbose': current_exp_config.get('verbose', True),
                                'augmentation_prob' : 0,
                            }

                            results = training.train_and_evaluate(**train2_kwargs)
                            test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params_class = results

                            # Log results
                            row = {
                                'idx': idx, 
                                'all_iter_idx': all_iter_counter, 
                                'q_config': config_key,
                                'channels_out': len(current_params_data.get('channels_out', [])),
                                'latent_shape': [*shape2],
                                'test_mse': test_mse, 
                                'val_mse': val_mse, 
                                '#params_ae': params_ae,
                                'test_auc': test_auc, 
                                'test_acc': test_acc, 
                                'val_auc': val_auc, 
                                'val_acc': val_acc,
                                'train_auc': train_auc, 
                                'train_acc': train_acc, 
                                '#params_class': params_class,
                                **p1, **p2
                            }
                            log_experiment_row(csv_path, row, columns)

        # Final Report
        send_telegram_report(
            current_exp_config, csv_path=csv_path, columns=graph_columns,
            title=f"{current_exp_config['experiment_name']} (all_iter {all_iter_counter})"
        )

    return pd.read_csv(csv_path)