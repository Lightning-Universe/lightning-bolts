python simclr_module.py --relic True \
--gpus 2 --dataset cifar10 \
--batch_size 256 --num_workers 8  \
--optimizer adam --learning_rate 0.3 \
--exclude_bn_bias --max_epochs 800 \
<<<<<<< HEAD
--online_ft
=======
--online_ft --optimizer='lars'
>>>>>>> 3eb270b704af88b200abe8123fcd9012325a59dd
