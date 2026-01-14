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



# --- Global Constants ---
DEFAULT_COLUMNS = [
    'test_auc_sel', 'test_acc_sel' ,'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc', 'train_acc', '#params'
]

default_redundancies = [
    {'quantum': False, 'U3_layers': [False, True], 'entangling_layers': [False, True], 'invert_embedding': [False, True]},
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
            'attention_selection': 'attention_selection',
            'selection_amount': 'selection_amount',
            'special_cls': 'special_cls',
            'parallel': 'parallel',
            'q_stride': 'q_stride',
            'connectivity': 'connectivity'
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
    
    os.makedirs(os.path.join(base_dir, 'current_results' if not exp_config['second_at_a_time'] else 'current_results2'), exist_ok=True)
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

        # --- Experiment Repetition Loop ---
        for idx in range(current_exp_config['num_experiments']):
            progress = int( 100 * (idx + 1) // current_exp_config['num_experiments'] )
            # Send progress update (helper handles filtering)
            send_telegram_report(current_exp_config, progress=progress)
            report_progress(experiment_name, idx + 1, exp_config['num_experiments'])
            
            print(f"\n===== Experiment Run {idx + 1} / {current_exp_config['num_experiments']} =====")

            for pack_data in itertools.product(*data_iter.values()):

                if pack_data or not processed_once:

                    current_params_data = dict(zip(data_iter.keys(), pack_data))
                    if current_params_data.get('channels_out', [None]) == [1] and current_params_data.get('entangle_method', [None]) == 'edges' or current_params_data.get('channels_out', [None]) == [3] and current_params_data.get('entangle_method', [None]) == 'CRX':
                        continue

                    custom_update_dict(p, current_params_data)

                    DataLoaders = [notrans_train_dl, val_dl, test_dl]
                    paddings = { 2 : { 'Up': 1, 'Down': 0, 'Left': 1, 'Right': 0 }, 3 : { 'Up': 1, 'Down': 1, 'Left': 1, 'Right': 1 } }

                    Kernels = { 'none'      :   torch.nn.Identity(),
                                'quantum'   :   quantum.quanvolution.QuantumConv2D(
                                                    kernel_size = p['quanv_kernel_size'],
                                                    stride = p['stride'],
                                                    padding = paddings[p['quanv_kernel_size']],
                                                    channels_out = p['channels_out'],
                                                    ancilla= p['ancilla'],
                                                    graphs = p['connectivity'],
                                                    entangle_method = p['entangle_method'], 
                                                    invert_embedding = p['invert_embedding']
                                                ),
                                'cosine'    :   CosineEncoding(),
                                'cosine2'   :   CosineEncoding(),
                                
                                }

                    Kernels = { k : v for k, v in Kernels.items() if ( (k == 'none') and NoneBool ) or ( (k == 'quantum') and QuantumBool ) or ( (k == 'cosine') and CosBool ) or ( (k == 'cosine2') and ('cosine2' in current_exp_config['q_config']) )}
                    print("Kernels to be used:", list(Kernels.keys()) )
                    
                    Latents = preprocess_and_save(
                        B = exp_config['B'],
                        DataLoaders = DataLoaders,
                        kernels = Kernels,
                        save_path = f"../QTransformer_Results_and_Datasets/transformer_results/quantum_datasets",
                        mode = 'standard',
                        model1 = None,
                        p1 = p,
                        num_channels = num_channels,
                        device = exp_config['device'],
                        concatenate_original = exp_config.get('concatenate_original', False)
                    )
                    processed_once = True

                # --- Innermost Loop (model_iter) ---
                for pack_model in itertools.product(*model_iter.values()):

                    for q_config in exp_config['q_config']:

                        latent_train_dl, latent_val_dl, latent_test_dl, shape2 = Latents[q_config]

                        # Apply updates from model_iter
                        current_params_model = dict(zip(model_iter.keys(), pack_model))
                        if exp_config.get('square', True):
                            custom_update_dict( current_params_model, {'num_transf' : current_params_model.get('parallel',p['parallel']) } )

                        custom_update_dict(p, current_params_model)

                        print(f"\nCurrent params for model:", current_params_model)

                        oneortwo = 'current_results' if not current_exp_config['second_at_a_time'] else 'current_results2'
                        aux_save_path = Path(f"../QTransformer_Results_and_Datasets/transformer_results/" + oneortwo + f"/grid_search{idx}")
                        aux_save_path.mkdir(parents=True, exist_ok=True)
                        
                        hidden_size = len( p['channels_out'] ) * shape[0] * p['patch_size']**2
                        # --- Model Definition ---
                        # 1. Prepare base arguments from p (only those that exist)

                        if is_redundant(p, model_iter, redundancies):
                            print("Skipping redundant parameter combination")
                            continue

                        # 1. Define Model Arguments
                        # Fixed arguments (required or from current_exp_config)
                        model_kwargs = {
                            'img_size': shape[-1],
                            'num_channels': shape[0],
                            'num_classes': current_exp_config['num_classes'],
                            'hidden_size': hidden_size,
                            'channels_last': current_exp_config['channels_last'],
                            'quantum_classification': False,
                        }

                        # Dynamic arguments from p (maps 'p' key name -> VisionTransformer __init__ name)
                        model_kwargs.update(build_kwargs(p, model_map))

                        # Handle dropout separately since it uses a transformation function
                        if 'dropout' in p:
                            model_kwargs['dropout'] = make_dropout(p['dropout'])

                        # Initialize Model
                        model = quantum.vit.VisionTransformer(**model_kwargs)
                        print("Model initialized with parameters:", model_kwargs)
                        # 2. Define Training Arguments
                        train_kwargs = {
                            'model': model,
                            'train_dataloader': latent_train_dl,
                            'valid_dataloader': latent_val_dl,
                            'test_dataloader': latent_test_dl,
                            'num_classes': current_exp_config['num_classes'],
                            'num_epochs': current_exp_config['N'],
                            'device': current_exp_config['device'],
                            'res_folder': str(aux_save_path),
                            'mapping': False,
                            'autoencoder': False,
                            'verbose': current_exp_config.get('verbose', False) 
                        }

                        train_kwargs.update(build_kwargs(p, train_map))
                        train_kwargs['parameters'] = current_params_model  # Pass full p dict for reference

                        # --- Model Training Execution ---
                        print(f"Training model with attention_selection: {p.get('attention_selection', 'default')}")

                        results = training.train_and_evaluate(**train_kwargs)
                        test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params = results
                        print(f"\nPoint {idx+1} finished training. AUC: {test_auc:.5f}\n")

                        # --- Save Results (using new helper) ---
                        row = {
                            'idx': idx + 1, 'q_config': q_config, 'all_iter_idx': all_iter_counter,
                            'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 
                            'val_acc': val_acc, 'train_auc': train_auc, 'train_acc': train_acc,
                            '#params': params,
                            **p # Add all current hyperparameters
                        }
                        print("Logging results:", row)
                        log_experiment_row(csv_path, row, columns)

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
                        'mapping': False, 'autoencoder': False, 'parameters': current_params_sel, 'verbose': current_exp_config.get('verbose', False) 
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
                                        'patch_embedding_required' : 'false'
                                    }
                                    model2_kwargs.update(build_kwargs(p2, model_map))
                                    model2 = quantum.vit.VisionTransformer(**model2_kwargs)

                                    # Train Classifier
                                    train2_kwargs = {
                                        'model': model2, 'train_dataloader': latent_train, 'valid_dataloader': latent_val, 'test_dataloader': latent_test,
                                        'num_classes': current_exp_config['num_classes'], 'num_epochs': current_exp_config['N2'],
                                        'device': current_exp_config['device'], 'res_folder': str(classifier_save_path),
                                        'mapping': False, 'autoencoder': False, 'parameters': current_params_class, 'verbose': current_exp_config.get('verbose', False) 
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