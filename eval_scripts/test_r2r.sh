python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=2335 train_r2r_multiloss.py \
    --n_gpu 1 \
    --train validlistener \
    --name r2r-multiloss-aug \
    --task R2R \
    --project orist \
    --maxAction 10 \
    --train_batch_size 2 \
    --val_batch_size 128 \
    --max_bb 10 \
    --config config/train-r2r-multiloss-aug.json \
    --features resnet \
    --resume ckpt/r2r_best_val_unseen \
    --drop_region_feat \
    --region_drop_p 0.5 \
    --use_lstm \
    --mlWeight 0.2 \
    --progress_loss \
    --angle_loss \
    --next_region_loss \
    --target_region_loss \
    --eval_only \

                        