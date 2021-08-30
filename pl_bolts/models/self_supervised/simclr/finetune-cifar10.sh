python simclr_finetuner.py \
--gpus 4 \
--ckpt_path /root/share/pretrained_model/simclr-cifar10-sgd.ckpt \
--dataset cifar10 \
--batch_size 64 \
--num_workers 8 \
--learning_rate 0.3 \
--num_epochs 100