export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export NCCL_CROSS_NIC=1
export TOKENIZERS_PARALLELISM=false

# python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=1 train_textual_inversion.py \
#     --pn=512 \
#     --depth=30 \
#     --use_swiglu_ffn=true \
#     --data_path="data/backpack" \
#     --vae_ckpt="yresearch/VQVAE-Switti" \
#     --max_iters=10000 \
#     --bs=1 \
#     --eval_batch_size=16 \
#     --log_iters=10 \
#     --log_images_iters=50 \
#     --save_iters=300 \
#     --global_save_iters=10 \
#     --fp16=2 \
#     --alng=1e-3 \
#     --tblr=1e-3 \
#     --vfast=1 \
#     --wp=50 \
#     --twd=0.05 \
#     --placeholder_token="<sks>" \
#     --initializer_token="backpack" \
#     --learnable_property="object" \
#     --use_captions=true \


# tensorboard --logdir local_output/tb_logs --port 8200



# python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=1 train.py \
#     --pn=256 \
#     --depth=20 \
#     --use_swiglu_ffn=true \
#     --data_path="data/dreambench" \
#     --vae_ckpt="yresearch/VQVAE-Switti" \
#     --max_iters=300 \
#     --bs=16 \
#     --eval_batch_size=16 \
#     --log_iters=10 \
#     --log_images_iters=50 \
#     --save_iters=100 \
#     --global_save_iters=10 \
#     --fp16=2 \
#     --alng=1e-3 \
#     --tblr=1e-4 \
#     --vfast=1 \
#     --wp=50 \
#     --twd=0.05 

# set grad accum
# not use aug crop 
# -m debugpy --listen 0.0.0.0:8102 --wait-for-client
# python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=1 train_textual_inversion.py \
#     --use_fsdp=False \
#     --grad_accum=4 \
#     --dataset_repeats=100 \
#     --pn=512 \
#     --depth=30 \
#     --rope_size=128 \
#     --rope_theta=10000 \
#     --use_ar=False \
#     --use_crop_cond=True \
#     --use_swiglu_ffn=True \
#     --data_path="data/backpack" \
#     --vae_ckpt="yresearch/VQVAE-Switti" \
#     --max_iters=10000 \
#     --bs=1 \
#     --eval_batch_size=16 \
#     --log_iters=10 \
#     --log_images_iters=200 \
#     --save_iters=100 \
#     --global_save_iters=10 \
#     --fp16=2 \
#     --alng=1e-3 \
#     --tblr=1e-3 \
#     --vfast=1 \
#     --wp=50 \
#     --twd=0.05 \
#     --placeholder_token="<sks>" \
#     --initializer_token="backpack" \
#     --learnable_property="object" \
#     --use_captions=True \
#     --local_out_dir_path="local_output_backpack_use_captions"


python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=1 train_textual_inversion.py \
    --use_fsdp=False \
    --grad_accum=4 \
    --dataset_repeats=100 \
    --pn=512 \
    --depth=30 \
    --rope_size=128 \
    --rope_theta=10000 \
    --use_ar=False \
    --use_crop_cond=True \
    --use_swiglu_ffn=True \
    --data_path="data/backpack" \
    --vae_ckpt="yresearch/VQVAE-Switti" \
    --max_iters=3000 \
    --bs=1 \
    --eval_batch_size=16 \
    --log_iters=10 \
    --log_images_iters=200 \
    --save_iters=100 \
    --global_save_iters=10 \
    --fp16=2 \
    --alng=1e-3 \
    --tblr=1e-3 \
    --vfast=1 \
    --wp=50 \
    --twd=0.05 \
    --placeholder_token="<sks>" \
    --initializer_token="backpack" \
    --learnable_property="object" \
    --use_captions=True \
    --use_text_encoder=0 \
    --local_out_dir_path="exps/local_output_backpack_use_captions_use_T1"


python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=1 train_textual_inversion.py \
    --use_fsdp=False \
    --grad_accum=4 \
    --dataset_repeats=100 \
    --pn=512 \
    --depth=30 \
    --rope_size=128 \
    --rope_theta=10000 \
    --use_ar=False \
    --use_crop_cond=True \
    --use_swiglu_ffn=True \
    --data_path="data/backpack" \
    --vae_ckpt="yresearch/VQVAE-Switti" \
    --max_iters=3000 \
    --bs=1 \
    --eval_batch_size=16 \
    --log_iters=10 \
    --log_images_iters=200 \
    --save_iters=100 \
    --global_save_iters=10 \
    --fp16=2 \
    --alng=1e-3 \
    --tblr=1e-3 \
    --vfast=1 \
    --wp=50 \
    --twd=0.05 \
    --placeholder_token="<sks>" \
    --initializer_token="backpack" \
    --learnable_property="object" \
    --use_captions=True \
    --use_text_encoder=1 \
    --local_out_dir_path="exps/local_output_backpack_use_captions_use_T2"




# python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=1 train_textual_inversion.py \
#     --use_fsdp=False \
#     --grad_accum=4 \
#     --dataset_repeats=100 \
#     --pn=512 \
#     --depth=30 \
#     --rope_size=128 \
#     --rope_theta=10000 \
#     --use_ar=False \
#     --use_crop_cond=True \
#     --use_swiglu_ffn=True \
#     --data_path="data/dog" \
#     --vae_ckpt="yresearch/VQVAE-Switti" \
#     --max_iters=10000 \
#     --bs=1 \
#     --eval_batch_size=16 \
#     --log_iters=10 \
#     --log_images_iters=200 \
#     --save_iters=100 \
#     --global_save_iters=10 \
#     --fp16=2 \
#     --alng=1e-3 \
#     --tblr=1e-3 \
#     --vfast=1 \
#     --wp=50 \
#     --twd=0.05 \
#     --placeholder_token="<sks>" \
#     --initializer_token="dog" \
#     --learnable_property="object" \
#     --use_captions=True \
#     --local_out_dir_path="local_output_dog_use_captions"


# python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=1 train_textual_inversion.py \
#     --use_fsdp=False \
#     --grad_accum=4 \
#     --dataset_repeats=100 \
#     --pn=512 \
#     --depth=30 \
#     --rope_size=128 \
#     --rope_theta=10000 \
#     --use_ar=False \
#     --use_crop_cond=True \
#     --use_swiglu_ffn=True \
#     --data_path="data/can" \
#     --vae_ckpt="yresearch/VQVAE-Switti" \
#     --max_iters=10000 \
#     --bs=1 \
#     --eval_batch_size=16 \
#     --log_iters=10 \
#     --log_images_iters=200 \
#     --save_iters=100 \
#     --global_save_iters=10 \
#     --fp16=2 \
#     --alng=1e-3 \
#     --tblr=1e-3 \
#     --vfast=1 \
#     --wp=50 \
#     --twd=0.05 \
#     --placeholder_token="<sks>" \
#     --initializer_token="can" \
#     --learnable_property="object" \
#     --use_captions=True \
#     --local_out_dir_path="local_output_can_use_captions"