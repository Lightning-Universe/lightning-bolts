import os
from argparse import ArgumentParser


def submit(master_address, master_port, world_size, node_rank, local_rank):
    os.environ['MASTER_ADDR'] = str(master_address)
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['NODE_RANK'] = str(node_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('master_address', default=str)
    parser.add_argument('master_port', default=str)
    parser.add_argument('node_rank', default=str)
    parser.add_argument('world_size', default=str)
    parser.add_argument('local_rank', default=str)


# grid train main.py --local --world_size 16 --local_gpus '0,1,2,3' --node_rank 0
