# Shift sdc

## Envionment

```
cd ./sdc
conda env create -f environment.yml
```
### Training
```
CUDA_VISIBLE_DEVICES=6 python run.py \
--model_name bc_nfnets_attention_loss \
--data_use_prerendered True \
--bc_generation_mode sampling \
--exp_batch_size=512 \
--data_num_workers=12 \
--data_prefetch_factor=2 \
--debug_overfit_eval=False \
--torch_seed=50 \
--model_checkpoint_key=bc_nfnets_attention_loss_dev\
--exp_lr=1e-5 \
--model_checkpoint_load_number=29 \
--debug_overfit_dev_data_only=True
```
or
```
./run.sh
```

### Evaluation
```
python submission_eval.py
```