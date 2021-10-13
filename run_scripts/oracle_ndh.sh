nvcc -V
nvidia-smi

python -m torch.distributed.launch --nproc_per_node=4 --master_port=2523  train_ndh_multiloss.py \
    --mlWeight 0.2 \
    --task NDH \
    --path_type oracle \
    --train listener \
    --name oracle_multiloss \
    --project orist \
    --maxAction 10 \
    --config config/train-ndh-multiloss.json \
    --drop_region_feat \
    --region_drop_p 0.5 \
    --use_lstm \
    --n_gpu 4 \
    --train_batch_size 2 \
    --val_batch_size 128 \
    --progress_loss \
    --angle_loss \
    --next_region_loss \
    --target_region_loss \
    --pretrained_model pretrained/uniter-base.pt

