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
from mi_quantum.data_to_qdata import preprocess_and_save, cut_extra_channels_from_latents
import mi_quantum.training as training
# ... other imports ...
from TelegramBot import SendToTelegram


# --- Global Constants ---
DEFAULT_COLUMNS = [
    'test_auc_sel', 'test_acc_sel' ,'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc', 'train_acc', '#params'
]

# --- Core Helper Functions (from your original script) ---

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
        
    base_dir = f"../QTransformer_Results_and_Datasets/{variant}_results"
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
             f.write(f"Number of Epochs Selector: {exp_config.get('N1', 'N/A')}\n")
             f.write('\nHyperparameters for Selector\n')
             json.dump(p1, f, indent=4)
        if p2:
             f.write(f"Number of Epochs Classifier: {exp_config.get('N2', 'N/A')}\n")
             f.write('\nHyperparameters for Classifier\n')
             json.dump(p2, f, indent=4)
        
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


def make_experiment_transformer(exp_config, p_base, all_iter={}, model_iter={}, graph_columns=['attention_selection', 'test_auc']):
    
    root_id = exp_config['experiment_id']
    
    # Define columns for CSV logging
    columns = ['idx', 'all_iter_idx'] + list(all_iter.keys()) + list(model_iter.keys()) + DEFAULT_COLUMNS
    try:
        columns.remove('test_auc_sel')
        columns.remove('test_acc_sel')
    except:
        pass

    progress = 0
    
    # --- Use the 'all_iter' generator ---
    for current_exp_config, (p,), current_params_all, all_iter_counter in iterate_all_iter(
        root_id, exp_config, all_iter, [p_base]
    ):
        
        csv_path, save_path = None, None # Init for error handling
        
        try:
            # --- Create directories and CSV file ---
            csv_path, save_path = make_directories_for_experiment(
                variant='transformer', # Specify variant
                exp_config=current_exp_config, 
                p1=p, # Use p1 slot for the single param dict
                all_iter=all_iter, 
                m2_iter=model_iter, # m2_iter for consistency, as it's the "model" iter
                columns=columns
            )

            # --- Load data ---
            train_dl, val_dl, test_dl, shape = load_data(
                data_flag='dermamnist',
                pixel=28,
                batch_size=current_exp_config['B'],
                num_workers=4,
                pin_memory=True
            )
            
            num_channels = shape[-1] if current_exp_config['channels_last'] else shape[0]
            
            # Send initial progress
            send_telegram_report(current_exp_config, progress=0)   

            # --- Experiment Repetition Loop ---
            for idx in range(current_exp_config['num_experiments']):
                progress = int( 100 * (idx + 1) // current_exp_config['num_experiments'] )
                # Send progress update (helper handles filtering)
                send_telegram_report(current_exp_config, progress=progress)
                
                print(f"\n===== Experiment Run {idx + 1} / {current_exp_config['num_experiments']} =====")

                # --- Innermost Loop (model_iter) ---
                for pack_model in itertools.product(*model_iter.values()):

                    
                    # Apply updates from model_iter
                    current_params_model = dict(zip(model_iter.keys(), pack_model))
                    if exp_config.get('square_arc', True):
                        custom_update_dict( current_params_model, {'num_transf' : p['paralel'] } )

                    custom_update_dict(p, current_params_model)

                    print(f"\nCurrent params for model:", current_params_model)

                    oneortwo = 'current_results' if not current_exp_config['second_at_a_time'] else 'current_results2'
                    aux_save_path = Path(f"../QTransformer_Results_and_Datasets/transformer_results/" + oneortwo + f"/grid_search{idx}")
                    aux_save_path.mkdir(parents=True, exist_ok=True)
                    
                    # --- Model Definition ---
                    model = quantum.vit.VisionTransformer(
                        img_size=shape[-1], num_channels=shape[0], num_classes=current_exp_config['num_classes'],
                        patch_size=p['patch_size'], hidden_size= shape[0]* p['patch_size']**2, num_heads=p['num_head'], Attention_N = p['Attention_N'],
                        num_transformer_blocks=p['num_transf'], attention_selection= p['attention_selection'], selection_amount= p['selection_amount'], special_cls = p['special_cls'], 
                        mlp_hidden_size=p['mlp_size'], quantum_mlp = p['quantum'], dropout = make_dropout(p['dropout']), channels_last=current_exp_config['channels_last'], quantum_classification = False,
                        paralel = p['paralel'], RD = p['RD'], q_stride = p['q_stride'], connectivity = p['connectivity']
                    )

                    # --- Model Training ---
                    print(f"Training model with attention_selection: {p['attention_selection']}")
                    test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params = training.train_and_evaluate(
                        model, train_dl, val_dl, test_dl, num_classes=current_exp_config['num_classes'],
                        learning_rate=p['learning_rate'], num_epochs=current_exp_config['N'], device=current_exp_config['device'], mapping=False,
                        res_folder=str(aux_save_path), hidden_size=p['hidden_size'], dropout=make_dropout(p['dropout']),
                        num_heads=p['num_head'], patch_size=p['patch_size'], num_transf=p['num_transf'],
                        mlp=p['mlp_size'], wd=p['weight_decay'], patience= p['patience'], scheduler_factor=p['scheduler_factor'], autoencoder=False,
                        val_train_pond = p['val_train_pond'], augmentation_prob = p['augmentation_prob']
                    )

                    print(f"\nPoint {idx+1} finished training. AUC: {test_auc:.5f}\n")

                    # --- Save Results (using new helper) ---
                    row = {
                        'idx': idx + 1, 'all_iter_idx': all_iter_counter,
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

        except Exception as e:
             # --- Send error report (using new helper) ---
             send_telegram_report(current_exp_config, error=e, progress=progress)
             print(f"Error encountered in all_iter {all_iter_counter}: {e}")
             # Continue to the next 'all_iter' loop if possible
             continue





def make_experiment_selformer(exp_config, p1_base, p2_base, all_iter={}, m1_iter={}, data_iter={}, m2_iter={}, graph_columns = [ 'q_config', 'test_auc']):
    """ Conducts a grid search over specified hyperparameters for a two-step quantum transformer model.
    1. Trains a selector model to select important patches from input data.
    2. Uses the selected patches to train a classifier model on the latent representations.
    Args:
        exp_config (dict): Experiment configuration parameters.
        p1_base (dict): Base hyperparameters for the selector model.
        p2_base (dict): Base hyperparameters for the classifier model.
        all_iter (dict): Hyperparameters to iterate over for both models.
        m1_iter (dict): Hyperparameters to iterate over for the selector model.
        data_iter (dict): Hyperparameters to iterate over for data preprocessing.
        m2_iter (dict): Hyperparameters to iterate over for the classifier model.
        graph_columns (list): Columns to include in Telegram report graphs.
    """
    
    root_id = exp_config['experiment_id']
    all_iter_counter = 1
    columns = ['idx', 'q_config', 'channels_out', 'latent_shape', '2_selection_amount'] + list(all_iter.keys()) + list(m1_iter.keys()) + list(data_iter.keys()) + list(m2_iter.keys()) + DEFAULT_COLUMNS + ['#params_sel', '#params_class']
    try:
        columns.remove('#params')         # Remove unused column (Used in transformer experiments)
        columns.remove('1_channels_out')  # Remove unused column (Used in transformer experiments)
    except:
        print("Columns '#params' or '1_channels_out' not in columns, skipping removal.")

    for pack_all in itertools.product(*all_iter.values()):

        experiment_id = root_id + '/all_iter_' + str(all_iter_counter) + '/'
        exp_config.update({'experiment_id' : experiment_id, 'trained_selector_once' : False})

        # Create fresh copies for each iteration
        p1 = p1_base.copy()
        p2 = p2_base.copy()
        
        # Modifications only affect the copies
        current_params = dict(zip(all_iter.keys(), pack_all))
        custom_update_dict( p1, current_params )
        custom_update_dict( p2, current_params )
        custom_update_dict( exp_config, current_params )
        print( "Current overall params:", current_params )


        
        try:
            # Save dictionary with all the hyperparameters and results in a json file
            progress = 0
            csv_path, save_path = make_directories_for_experiment( variant='selformer', exp_config = exp_config, p1 = p1, p2 = p2, 
                                                        all_iter = all_iter, m1_iter = m1_iter, data_iter = data_iter, m2_iter =m2_iter, columns = columns )
            # Load data
            notrans_train_dl, train_dl, val_dl, test_dl, shape = data.get_medmnist_dataloaders(
                pixel = exp_config['pixels'], data_flag='dermamnist', extra_tr_without_trans = True, batch_size=exp_config['B'], num_workers=4, pin_memory=True
            )
            

            # Obtain general settings regarding dataset shape
            num_channels = shape[-1] if exp_config['channels_last'] else shape[0]
            progress_levels = [0, 25, 50, 75, 100]

            # Grid search loop
            if exp_config['send_telegram']:
                    SendToTelegram(progress = 0)   

            for idx_ in range(exp_config['num_experiments']):
                idx = idx_ + 1
                progress = int( 100* (idx+1)//exp_config['num_experiments'] )
                if exp_config['send_telegram'] and progress in progress_levels:
                    SendToTelegram(progress = progress)                

                print(f"\n\nPoint {idx}")
                oneortwo = 'current_results' if not exp_config['second_at_a_time'] else 'current_results2'
                save_path = Path(f"../QTransformer_Results_and_Datasets/selformer_results/" + oneortwo + f"/grid_search{idx}")
                save_path.mkdir(parents=True, exist_ok=True)
                os.makedirs(save_path / 'autoencoder', exist_ok=True)

                # Determine if quantum processing is enabled based on exp_config['q_config']
                # exp_config['q_config'] can be a string or a list/tuple; handle both
                NoneBool, PatchBool = 'none' in exp_config['q_config'], 'patchwise' in exp_config['q_config']
                print(f"Current quantum configuration:\nNormal latent representations: {NoneBool}\nPatchwise Quantum latent representations: {PatchBool}")

                for pack_sel in itertools.product(*m1_iter.values()):

                    current_params = dict(zip(m1_iter.keys(), pack_sel))
                    custom_update_dict(p1, current_params)
                    custom_update_dict(p2, current_params)
                    if exp_config.get('square_arc', True):
                        custom_update_dict( p1, {'1_num_transf' : p1['1_paralel'] } )
                        custom_update_dict( p2, {'num_transf'   : p2['paralel']   } )
                    print("Current params for selector:", current_params)

                    if (not exp_config['trained_selector_once']) or exp_config['repeat_selector']:
                        exp_config['trained_selector_once'] = True

                        print(f"\nTraining first model: Selector\nOptiosn: Selector with Quantum Layers: {list(exp_config['q_config'])} and Learning Rate: {p1['1_learning_rate']}\n")
                        print(f"Shape of the data: {shape}\n")
                        
                        # Model
                        model1 = quantum.vit.VisionTransformer(
                            img_size=shape[-1], num_channels=shape[0], num_classes=exp_config['num_classes'],
                            patch_size=p1['1_patch_size'], hidden_size= shape[0]* p1['1_patch_size']**2, num_heads=p1['1_num_head'], Attention_N = p1['1_Attention_N'],
                            num_transformer_blocks=p1['1_num_transf'], attention_selection= p1['1_attention_selection'], selection_amount = p1['1_selection_amount'], special_cls = p1['1_special_cls'], 
                            mlp_hidden_size=p1['1_mlp_size'], quantum_mlp = False, dropout = make_dropout( p1['1_dropout']) , channels_last=exp_config['channels_last'], quantum_classification = False,
                            paralel = p1['1_paralel'], RD = p1['1_RD'], q_stride = p1['1_q_stride'], connectivity = 'chain'
                        )

                        # Train first model
                        test_auc_sel, test_acc_sel, val_auc_sel, val_acc_sel, train_auc_sel, _, params_sel = training.train_and_evaluate(
                            model1, train_dl, val_dl, test_dl, num_classes=exp_config['num_classes'],
                            learning_rate=p1['1_learning_rate'], num_epochs=exp_config['N1'], device=exp_config['device'], mapping=False,
                            res_folder=str(save_path), hidden_size=p1['1_hidden_size'], dropout= make_dropout( p1['1_dropout']),
                            num_heads=p1['1_num_head'], patch_size=p1['1_patch_size'], num_transf=p1['1_num_transf'],
                            mlp=p1['1_mlp_size'], wd=p1['1_weight_decay'], patience= p1['1_patience'], scheduler_factor=p1['1_scheduler_factor'], autoencoder=False,
                            augmentation_prob = p1['1_augmentation_prob'], val_train_pond=p1['1_val_train_pond']
                        )

                        print(f"\nSelector training completed succesfully.\nTest, Val, Train AUC (first step): {test_auc_sel:.5f}, {val_auc_sel:.5f}, {train_auc_sel:.5f}\n")

                    trained_once_none_config = False

                    for pack_data in itertools.product(*data_iter.values()):
                    
                        current_params = dict(zip(data_iter.keys(), pack_data))
                        print("Current params for data:", current_params)
                        custom_update_dict(p1, current_params)
                        custom_update_dict(p2, current_params)
                        custom_update_dict(exp_config, current_params)

                        if exp_config['B'] >= 8 and exp_config['special_batch_for_data']:
                            notrans_train_dl, train_dl, val_dl, test_dl, shape = data.get_medmnist_dataloaders(
                                pixel=28, data_flag='dermamnist', extra_tr_without_trans = True, batch_size=1, num_workers=4, pin_memory=True
                            )
                            
                        # Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
                        DataLoaders = [notrans_train_dl, val_dl, test_dl]
                        paddings = { 2 : { 'Up': 1, 'Down': 0, 'Left': 1, 'Right': 0 }, 3 : { 'Up': 1, 'Down': 1, 'Left': 1, 'Right': 1 } }

                        Kernels = { 'none' :        torch.nn.Identity(),
                                    'patchwise' :   quantum.quanvolution.QuantumConv2D(
                                                        kernel_size = p1['1_quanv_kernel_size'],
                                                        stride = 1,
                                                        padding = paddings[p1['1_quanv_kernel_size']],
                                                        channels_out = p1['1_channels_out'],
                                                        ancilla= p1['1_ancilla'],
                                                        graph = p1['1_connectivity'],
                                                        entangle_method = p1['1_entangle_method']
                                                    )
                                    }

                        Kernels = { k : v for k, v in Kernels.items() if ( (k == 'none') and NoneBool ) or ( (k == 'patchwise') and PatchBool ) }
                        print("Kernels to be used:", list(Kernels.keys()) )
                        
                        Latents = preprocess_and_save(
                            B = exp_config['B'],
                            DataLoaders = DataLoaders,
                            kernels = Kernels,
                            save_path = f"../QTransformer_Results_and_Datasets/selformer_results/quantum_datasets",
                            mode = 'by_selected_patches',
                            model1 = model1,
                            p1 = p1,
                            num_channels = num_channels,
                            flatten_extra_channels = p1['1_flatten_extra_channels'],
                            device = exp_config['device'],
                            flatten = not exp_config['augmenting'], 
                            concatenate_original = exp_config.get('concatenate_original', False)
                        )

                        original_measured = p1['1_channels_out']
                        original_channels_out = len(p1['1_channels_out'])
                        TrainedFlattenedOnce = False

                        channel_iterator =  range( ( original_channels_out if 'patchwise' in Kernels else 0 ) * (exp_config['rewind_channels'] != 0), 0 , - exp_config['rewind_channels']  ) if exp_config['rewind_channels'] > 0 else [ original_channels_out ]

                        print( "Channel iterator for classifier:" , list(channel_iterator) )

                        # Create second model for the second step) 

                        for config in list(Kernels.keys()):   

                            for i in channel_iterator:

                                if config == 'patchwise':

                                    if p1['1_flatten_extra_channels']:
                                    
                                        if TrainedFlattenedOnce:
                                            print("Skipping redundant configuration with flattened extra channels already trained.")
                                            continue
                                        else:
                                            TrainedFlattenedOnce = True

                                    else:

                                        p1.update( {'1_channels_out' : original_measured[:i] } )
                                        Aux_Latents = cut_extra_channels_from_latents( Latents, i, original_channels_out )

                                else:
                                    Aux_Latents = Latents

                                for pack_class in itertools.product(*m2_iter.values()):

                                    if trained_once_none_config and config == 'none':
                                        continue  
                        
                                    current_params = dict( zip(m2_iter.keys(), pack_class) )
                                    print("Current params for classification:", current_params)
                                    custom_update_dict(p1, current_params)
                                    custom_update_dict(p2, current_params)
                                    custom_update_dict( p2, {'selection_amount' : p1['1_selection_amount'] + (1 - p1['1_flatten_extra_channels']) * p2['len_channels_scaler'] * original_channels_out}  )  # Adjust selection amount for classifier})  
                                    
                                    if exp_config.get('square_arc', True):
                                        custom_update_dict( p1, {'1_num_transf' : p1['1_paralel'] } )
                                        custom_update_dict( p2, {'num_transf'   : p2['paralel']   } )

                                    config_dataset = Aux_Latents[config]
                                    shape2 = config_dataset[-1]  # Shape of one sample in the test set
                                    print("Shape2:", shape2, "1_channels_out:", p1['1_channels_out'])

                                    hidden_size = shape2[-1] if not exp_config['augmenting'] else shape2[-1] * shape2[-2] * shape2[-3]
                                    p2.update( {'hidden_size' : hidden_size } )
                                    print("Updated p2 hidden_size:", p2['hidden_size'])

                                    model2 = quantum.vit.VisionTransformer(
                                        img_size=shape[-1], num_channels= 3, num_classes=exp_config['num_classes'],
                                        patch_size=p2['patch_size'], hidden_size= hidden_size, num_heads=p2['num_head'], Attention_N = p2['Attention_N'],
                                        num_transformer_blocks=p2['num_transf'], attention_selection= p2['attention_selection'], special_cls = p2['special_cls'], 
                                        mlp_hidden_size=p2['mlp_size'], quantum_mlp = False, dropout = make_dropout(p2['dropout']), channels_last=exp_config['channels_last'], quantum_classification = False,
                                        paralel = p2['paralel'] , RD = p2['RD'], q_stride = p2['q_stride'], connectivity = 'chain', patch_embedding_required = 'flatten' if exp_config['augmenting'] else 'false'
                                    )

                                    print('\nTraining second model: classifier ViT on latent representations\n')
                                    print(f'QUANTUM SETTING IS: {config} and current lr is: {p2["learning_rate"]}')
                                    
                                    # Train second model
                                    test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params = training.train_and_evaluate(
                                        model2, config_dataset[0], config_dataset[1], config_dataset[2], num_classes=exp_config['num_classes'],
                                        learning_rate=p2['learning_rate'], num_epochs=exp_config['N2'], device=exp_config['device'], mapping=False,
                                        res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=make_dropout(p2['dropout']),
                                        num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
                                        mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor'], 
                                        autoencoder=False, augmentation_prob = p2['augmentation_prob'], val_train_pond=p2['val_train_pond']
                                    )

                                    # Save results
                                    row = {
                                        'idx': idx, 
                                            'q_config': config, 'channels_out' : len(p1['1_channels_out']) if config == 'patchwise' else 0, 'latent_shape' : [*shape2], '2_selection_amount' : p2['selection_amount'],
                                            'test_auc_sel': test_auc_sel, 'test_acc_sel': test_acc_sel, '#params_sel': params_sel, 
                                            'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc, 'train_auc': train_auc, 'train_acc' : train_acc,
                                            '#params_class': params,
                                            **p1,
                                            **p2
                                    }

                                    pd.DataFrame([row], columns=columns).to_csv(
                                        csv_path, mode='a', header=False, index=False
                                    )

                                if config == 'none':
                                    trained_once_none_config = True     

            if exp_config['send_telegram']:
                SendToTelegram(csv_file = csv_path, columns = graph_columns,
                            title = exp_config['experiment_name'] )

        except Exception as e:
            SendToTelegram(progress = progress, error_message=str(e))
            print(str(e))
