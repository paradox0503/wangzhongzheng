"""Train a model using the project's existing experiment pipeline.

Run from the project root, for example:
    python train.py -C conf/origin/AutoTimes.json
"""
import argparse

from utils.conf import Configuration
from utils.expe import Experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-C', '--conf', required=True, help='Path to the model configuration JSON')
    args = parser.parse_args(argv)

    experiment = Experiment(Configuration(args.conf))
    experiment.run()


if __name__ == '__main__':
    main()
