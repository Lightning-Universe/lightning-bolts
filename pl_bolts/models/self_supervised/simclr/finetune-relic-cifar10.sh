python simclr_finetuner.py --use_relic_loss=True \
--gpus 2 \
--ckpt_path /root/share/pretrained_model/simclr-cifar10-sgd.ckpt \
--dataset cifar10 \
--batch_size 512 \
--num_workers 8 \
--learning_rate 0.3 \
--num_epochs 200