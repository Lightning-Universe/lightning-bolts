python simclr_module.py \
--gpus 4 \
--dataset cifar10 \
--batch_size 512 \
--num_workers 16 \
--optimizer sgd \
--learning_rate 1.5 \
--exclude_bn_bias \
--max_epochs 800 \
--online_ft \
