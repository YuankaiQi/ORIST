nvcc -V
nvidia-smi

python -m torch.distributed.launch --nproc_per_node=4 --master_port=2563 train_reverie_multiloss.py \
    --max_bb 25 \
    --train listener \
    --name reverie_multiloss \
    --project orist \
    --maxAction 10 \
    --task REVERIE \
    --mlWeight 0.2 \
    --drop_region_feat \
    --region_drop_p 0.3 \
    --config config/train-reverie-multiloss.json \
    --n_gpu 4 \
    --train_batch_size 2 \
    --val_batch_size 128 \
    --pretrained_model pretrained/uniter-base.pt \
    --use_lstm \
    --progress_loss \
    --angle_loss \
    --next_region_loss \
    --target_region_loss

