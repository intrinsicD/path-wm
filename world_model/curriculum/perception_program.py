"""Explicit training and evaluation entry points for overnight perception work."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['cache', 'train'])
    p.add_argument('--encoder', choices=['cnn', 'vit'], required=True)
    p.add_argument('--seed', type=int, default=9107)
    p.add_argument('--development', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.command == 'cache':
        from .perception_cache import prepare
        prepare(args.encoder, args.development, args.device)
    else:
        from .perception_training import train
        train(args.encoder, args.seed, args.development, args.device, args.resume)


if __name__ == '__main__':
    main()
