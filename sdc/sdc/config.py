import argparse


def build_parser():
    """Build parser."""
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    ###########################################################################
    # #### Wandb ##############################################################
    ###########################################################################
    parser.add_argument(
        '--wandb_project', type=str, default='sdc-debug',
        help='Wandb project name.')

    ###########################################################################
    # #### Directories ########################################################
    ###########################################################################
    parser.add_argument(
        '--dir_wandb', type=str, default='./',
        help='Directory to which wandb logs.')
    parser.add_argument(
        '--dir_tensorboard', type=str, default='tb',
        help='Directory to which TensorBoard logs.')
    # /home/master/10/cytseng/shifts_sdc_dataset/data
    # /tmp3/cytseng/shifts_sdc_dataset/data
    parser.add_argument(
        '--dir_data', type=str, default='/tmp3/cytseng/shifts_sdc_dataset/data',
        help='Directory where SDC data is stored. We also use this to cache '
             'torch hub models.')
    parser.add_argument(
        '--dir_metrics', type=str, default=None,
        help='Directory where intermediate metrics are stored. Default will '
             'be set to `{dir_data}/metrics`.')
    parser.add_argument(
        '--dir_metadata_cache', type=str, default=None,
        help='Directory where dataset stats are stored (cowntrolled by '
             '`debug_collect_dataset_stats` attribute).')
    parser.add_argument(
        '--dir_checkpoint', type=str, default='model_checkpoints',
        help='Directory to which model checkpoints are stored.')

    ###########################################################################
    # #### General ############################################################
    ###########################################################################
    parser.add_argument(
        '--verbose', type='bool', default=False,
        help='Extra logging, sanity checks.')
    parser.add_argument(
        '--tb_logging', type='bool', default=False,
        help='Use TensorBoard logging for losses and visualizations.')
    parser.add_argument(
        '--np_seed', type=int, default=42,
        help='NumPy seed (data processing, train/test splits, etc.).')
    parser.add_argument(
        '--torch_seed', type=int, default=42,
        help='Model seed.')

    ###########################################################################
    # #### Data ###############################################################
    ###########################################################################
    parser.add_argument(
        '--data_num_workers', type=int, default=4,
        help='Number of workers to use in PyTorch data loading.')
    parser.add_argument(
        '--data_prefetch_factor', type=int, default=2,
        help='Number of samples loaded in advance by each worker.')
    parser.add_argument(
        '--data_dtype',
        default='float32',
        type=str, help='Data type (supported for float32, float64) '
                         'used for data (e.g. inputs, ground truth).')
    parser.add_argument(
        '--data_use_prerendered',
        default=False,
        type='bool', help='Use pre-rendered data.')

    ###########################################################################
    # #### Experiment #########################################################
    ###########################################################################
    parser.add_argument(
        '--exp_sweep', type='bool', default=False,
        help=f'Run an experiment sweep. Will clear out previous wandb logs.')
    parser.add_argument(
        '--exp_name', type=str, default=None,
        help=f'Specify an explicit name for the experiment - otherwise we '
             f'generate a unique wandb ID.')
    parser.add_argument(
        '--exp_group', type=str, default=None,
        help=f'Specify an explicit name for the experiment group.')
    parser.add_argument(
        '--exp_use_cuda', type='bool', default=True,
        help=f'Attempt to use CUDA device for training.')
    parser.add_argument(
        '--exp_print_every_nth_forward', dest='exp_print_every_nth_forward',
        default=False, type=int,
        help='Print during mini-batch as well for large epochs.')
    parser.add_argument(
        '--exp_lr', type=float, default=1e-4)
    parser.add_argument(
        '--exp_num_lr_warmup_epochs', type=int, default=1)
    parser.add_argument(
        '--exp_batch_size', type=int, default=650)
    parser.add_argument(
        '--exp_num_epochs', type=int, default=100)
    parser.add_argument(
        '--exp_image_downsize_hw', type=int, default=None,
        help='Downsize image input to '
             '(exp_image_downsize, exp_image_downsize). '
             'Do not downsize if None.')
    parser.add_argument(
        '--exp_checkpoint_frequency', type=int, default=1,
        help='Checkpoint every `exp_checkpoint_frequency` times the '
             'validation loss improves.')
    parser.add_argument(
        '--exp_checkpoint_validation_loss', type=str,
        default='moscow__development__ade',
        help='Loss to use for model checkpointing (i.e., checkpoint if model '
             'improves this validation loss. Note that by default, this uses '
             'the in-distribution (Moscow, no precipitation) development set.')

    ###########################################################################
    # #### Model ##############################################################
    ###########################################################################

    parser.add_argument(
        '--model_name', type=str, default='bc',
        help="The backbone model. See "
             "sdc/oatomobile/torch/baselines/__init__.py.")
    parser.add_argument(
        '--model_prefix', type=str, default='Default',
        help="Used for logging/plotting purposes for RIP. Specify a prefix "
             "which will appear before the model (e.g., "
             "`Trained on Partial Data`).")
    parser.add_argument(
        '--model_checkpoint_key', type=str, default=None,
        help="Optionally specify the name of the subdirectory to "
             "which we checkpoint.")
    parser.add_argument(
        '--model_dim_hidden', type=int, default=512,
        help="Number of hidden dims, generally the size "
             "of the embedding passed from the vision model "
             "(e.g., MobileNetV2) to the autoregressive "
             "decoder.")
    parser.add_argument(
        '--model_in_channels', type=int, default=17,
        help="Number of feature map channels.")
    parser.add_argument(
        '--model_output_shape', type=tuple, default=(25, 2),
        help="Predict position for 25 timesteps.")
    parser.add_argument('--model_weight_decay', type=float, default=0.0,
                        help="The L2 penalty (regularization) coefficient.")
    parser.add_argument(
        '--model_clip_gradients', type='bool', default=True,
        help='Clips gradients to 1.0.')

    parser.add_argument(
        '--model_checkpoint_load_number', type=int, default=0,
        help="Load model checkpoints.")
    ###########################################################################
    # #### Method-Specific Hypers #############################################
    ###########################################################################

    parser.add_argument(
        '--bc_deterministic', type='bool', default=False,
        help='Train BC on the ADE objective, instead of using a valid '
             'likelihood.')
    parser.add_argument(
        '--bc_generation_mode', type=str, default='sampling',
        help='Specifies the mode used in generating trajectories from DIM/BC '
             'at train and evaluation time. Options:'
             'teacher-forcing: condition on ground-truth in forming the '
             'autoregressive likelihood.'
             'sampling: condition on samples.')
    parser.add_argument(
        '--dim_scale_eps', type=float, default=1e-7,
        help="Additive epsilon constant to avoid a 0 or negative scale.")
    parser.add_argument(
        '--rip_per_plan_algorithm', type=str, default=None,
        help="Use Robust Imitative Planning wrapper to select a robust "
             "log-likelihood estimate for each generated plan."
             "`None` will disable RIP."
             "`WCM`: worst-case model."
             "`MA`: Bayesian model averaging."
             "`BCM`: best-case model."
             "`UQ`: upper quartile."
             "`LQ`: lower quartile.")
    parser.add_argument(
        '--rip_per_scene_algorithm', type=str, default=None,
        help="Use Robust Imitative Planning wrapper to aggregate a per-scene "
             "confidence based on the aggregated per-plan log-likelihoods. "
             "Same settings as above.")
    parser.add_argument(
        '--rip_k', type=int, default=3,
        help="Number of models in the RIP ensemble.")
    parser.add_argument(
        '--rip_samples_per_model', type=int, default=10,
        help="Number of stochastic trajectory generations per RIP "
             "ensemble member.")
    parser.add_argument(
        '--rip_num_preds', type=int, default=5,
        help="Number of plan predictions that are actually made by the RIP "
             "model for a given prediction request input. "
             "We select these as the top samples from all stochastic "
             "generations from the ensemble members.")
    parser.add_argument(
        '--rip_eval_subgroup', type=str, default=None,
        help="If specified, RIP will only evaluate on either train/development "
             "datasets. Helpful to parallelize evaluation load across jobs.")
    parser.add_argument(
        '--rip_cache_all_preds', type='bool', default=False,
        help='Cache every predicted plan and their scores under each of the '
             'models. No aggregation is applied. '
             'Used for post-hoc analysis with whatever RIP algorithm '
             'is desired, or a different (a smaller) number of ensemble '
             'members.')

    ###########################################################################
    # #### Metrics ############################################################
    ###########################################################################

    parser.add_argument(
        '--fbeta_threshold', type=float,
        default=1.0,
        help='The error threshold below which we consider the prediction '
             'acceptable.')
    parser.add_argument(
        '--fbeta_beta', type=float,
        default=1.0, help='The beta value for the F_beta metric.')

    ###########################################################################
    # #### Debug ##############################################################
    ###########################################################################

    parser.add_argument(
        '--debug_overfit_eval', type='bool', default=False,
        help='Train on a very small subset, try to overfit.')
    parser.add_argument(
        '--debug_overfit_dev_data_only', type='bool', default=False,
        help='If True, only use the dev data.')
    parser.add_argument(
        '--debug_overfit_n_examples', type=int, default=10,
        help='Size of the subset of train on which we try to overfit.')
    parser.add_argument(
        '--debug_collect_dataset_stats', type='bool', default=False,
        help='When enabled, collect dataset statistics on train/val/test. '
             'Caches predictions, ground truth, uncertainty estimates, '
             'scene ID, and trajectory tags for every evaluated datapoint.')
    parser.add_argument(
        '--debug_eval_mode', type='bool', default=False,
        help='Only run evaluation. Can be manually triggered to evaluate '
             'a trained BC/DIM model.')
    return parser


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")
