from typing import Mapping
from typing import Optional
from functools import partial
import os
import torch
import numpy as np
import tqdm.notebook as tqdm
from sdc.metrics import SDCLoss
from sdc.config import build_parser
from sdc.oatomobile.torch.baselines import init_rip 
from sdc.oatomobile.torch.baselines.robust_imitative_planning import load_rip_checkpoints
from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.evaluation.utils import load_submission_proto, save_submission_proto
from ysdc_dataset_api.evaluation import Submission, object_prediction_from_model_output, save_submission_proto
from sdc.oatomobile.torch.baselines import batch_transform


# Softmax for normalize
def _softmax_normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.exp(weights - np.max(weights))
    return weights / weights.sum(axis=0)

class Model:
    def __init__(self, c):
        self.c = c
        # Initialize torch hub dir to cache MobileNetV2
        torch.hub.set_dir(f'{c.dir_checkpoint}/torch_hub')
        
    def load(self):
        model, self.full_model_name, _, _ = init_rip(c=self.c)
        checkpoint_dir = f'{c.dir_checkpoint}/{self.full_model_name}'
        self.model = load_rip_checkpoints(
            model=model, device=c.exp_device, k=c.rip_k,
            checkpoint_dir=checkpoint_dir)
        
    
    def predict(self, batch: Mapping[str, torch.Tensor], sdc_loss: Optional[SDCLoss]):
        """
        Args:
            batch: Mapping[str, torch.Tensor], with 'feature_maps' key/value

        Returns:
            Sequence of dicts. Each has the following structure:
                {
                    predictions_list: Sequence[np.ndarray],
                    plan_confidence_scores_list: Sequence[np.ndarray],
                    pred_request_uncertainty_measure: float,
                }
        """
        self.model.eval()
        with torch.no_grad():
            predictions, plan_confidence_scores, pred_request_confidence_scores = (
                self.model(**batch))
            
        predictions = predictions.detach().cpu().numpy()
#         print("predictions: ", predictions)

        plan_confidence_scores = plan_confidence_scores.detach().cpu().numpy()
        for i in range(predictions.shape[0]):
            plan_confidence_scores[i] = _softmax_normalize(plan_confidence_scores[i])
#         print("plan_confidence_scores: ", plan_confidence_scores)

        pred_request_confidence_scores = pred_request_confidence_scores.detach().cpu().numpy()
#         print("pred_request_confidence_scores: ", pred_request_confidence_scores)

        if sdc_loss is not None:
            ground_truth = batch['ground_truth_trajectory'].detach().cpu().numpy()
            sdc_loss.cache_batch_losses(
                predictions_list=predictions,
                ground_truth_batch=ground_truth,
                plan_confidence_scores_list=plan_confidence_scores,
                pred_request_confidence_scores=pred_request_confidence_scores)
        
        return [
            {
                'predictions_list': predictions[i],
                'plan_confidence_scores_list': plan_confidence_scores[i],
                # Negate, as we need to provide an uncertainty measure
                # for the submission pb, not a confidence score.
                'pred_request_uncertainty_measure':
                    -(pred_request_confidence_scores[i])
            } for i in range(predictions.shape[0])]



if __name__ == '__main__':
    # Setting args
    parser = build_parser()
    args = parser.parse_args('')
    ######################################################################################################
    # Please alter path 
    ######################################################################################################
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Dataset path 
    evaluation_dataset_path = \
    '/home/master/10/cytseng/shifts_sdc_dataset/canonical-eval-data/evaluation'
    prerendered_dataset_path = \
    '/home/master/10/cytseng/shifts_sdc_dataset/canonical-eval-data/evaluation_renders'
    # Submission_dir
    submission_dir = '/home/master/10/cytseng/shifts_eval/shifts/sdc/submissions_eval/'
    # checkpoint_dir
    args.dir_checkpoint = '/home/master/10/cytseng/shifts_eval/shifts/sdc/model_checkpoints'
    # Cache loss metrics here
    args.dir_metrics = '/home/master/10/cytseng/shifts_sdc_dataset/data/metrics' 
    ######################################################################################################
    
    ######################################################################################################
    # Ensumble Model Setting
    ######################################################################################################
    # The below configuration was our best performing in baseline experiments.
    # See paper for more details and the configurations considered.
    
    # Backbone model details
    # Behavioral Cloning: 
    # MobileNetv2 feature encoder, GRU decoder
    args.model_name = 'bc_nfnets_attention_loss'
    args.model_dim_hidden = 512
    args.exp_device = 'cuda:0'
    
    # Used in scoring generated trajectories and obtaining 
    # per-plan/per-scene confidence scores.
    # See 
    #   `sdc.oatomobile.torch.baselines.robust_imitative_planning.py` 
    # for details.
    args.rip_per_plan_algorithm = 'WCM'
    args.rip_per_scene_algorithm = 'WCM'
    
    # Number of ensemble members
    args.rip_k = 10
    args.rip_num_preds = 5
    args.rip_samples_per_model = 50

    # Data loading
    # https://pytorch.org/docs/stable/data.html
    args.exp_batch_size = 512
    args.data_num_workers = 4
    args.data_prefetch_factor = 2
    ######################################################################################################

    evaluation_dataset = MotionPredictionDataset(
        dataset_path=evaluation_dataset_path,
        prerendered_dataset_path=prerendered_dataset_path,
    )
    c = args
    # Initialize and load ensemble of k models from checkpoints
    # On first run, will fail and create a directory where checkpoints
    # should be placed.
    model = Model(c=c)
    model.load()
    
    # Init dataloader
    dataloader_kwargs = {
        'batch_size': c.exp_batch_size,
        'num_workers': c.data_num_workers,
        'prefetch_factor': c.data_prefetch_factor,
        'pin_memory': True
    }

    print(f'Building dataloaders with kwargs {dataloader_kwargs}.')
    evaluation_dataloader = torch.utils.data.DataLoader(evaluation_dataset, **dataloader_kwargs)

    submission = Submission()

    batch_cast = partial(
        batch_transform, device=c.exp_device, downsample_hw=None,
        data_use_prerendered=True)

    for batch_id, batch in enumerate(tqdm.tqdm(evaluation_dataloader)):
        batch = batch_cast(batch)
        batch_output = model.predict(batch, None)

        for i, data_item_output in enumerate(batch_output):
            proto = object_prediction_from_model_output(
                track_id=batch['track_id'][i],
                scene_id=batch['scene_id'][i],
                model_output=data_item_output,
                is_ood=False)  # Set fake value as we do not know is_ood for evaluation data.

            submission.predictions.append(proto)
    
    # First, let's write out a submission protobuf (as one should submit for the competition).
    save_submission_proto(submission_dir \
                          + '/'+ model.full_model_name + \
                          '_submission.pb', submission=submission)
    
    # Can check that things were written correctly as follows:
    new_sub = load_submission_proto(submission_dir +\
                                      '/'+ model.full_model_name + \
                                    '_submission.pb')
    
    print("Number of submission:", len(new_sub.predictions))
    
    
    
    
    
    
    