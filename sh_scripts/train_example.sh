export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export NCCL_CROSS_NIC=1
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1243 --nproc_per_node=2 train.py \
    --pn=256 \
    --depth=20 \
    --use_swiglu_ffn=true \
    --data_path="/path/to/datasets" \
    # --vae_ckpt="yresearch/VQVAE-Switti" \ optional, will work only in 512
    --max_iters=800000 \
    --bs=16 \
    --eval_batch_size=16 \
    --log_iters=10 \
    --log_images_iters=50 \
    --save_iters=100 \
    --global_save_iters=10 \
    --fp16=2 \
    --alng=1e-3 \
    --tblr=1e-4 \
    --vfast=1 \
    --wp=50 \
    --twd=0.05 