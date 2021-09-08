from argparse import ArgumentParser
from models import Trainer


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/train.yaml')
    args = parser.parse_args()
    Trainer(args.config).run()


if __name__ == '__main__':
    main()
