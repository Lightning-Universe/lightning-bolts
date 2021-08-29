python simclr_module.py --relic True \
--gpus 4 --dataset cifar10 \
--batch_size 256 --num_workers 8  \
--optimizer sgd --learning_rate 1.5 \
--exclude_bn_bias --max_epochs 800 \
--online_ft --optimizer='lars'
