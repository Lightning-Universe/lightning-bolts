python simclr_module.py --relic True \
--gpus 2 --dataset cifar10 \
--batch_size 256 --num_workers 8  \
--optimizer adam --learning_rate 0.3 \
--exclude_bn_bias --max_epochs 800 \
--online_ft
