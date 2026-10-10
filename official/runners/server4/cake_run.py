"""Run the shared native CAKE implementation with server4 asset bindings."""
import argparse

from official.runners.server1.cake_run import run, validate
from official.runners.server1.common import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = validate(read(args.config))
    if config['server'] != 'server4':
        raise ValueError('CAKE_SERVER4_BINDING_REQUIRED')
    run(args.config, args.lock, args.resume)


if __name__ == '__main__':
    main()
