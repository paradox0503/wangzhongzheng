import sys
import argparse

# import torchvision
# torchvision.disable_beta_transforms_warning()
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

from utils.conf import Configuration
from utils.expe import Experiment


def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters')
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    args = parser.parse_args(argv[1: ])
    conf = Configuration(args.confpath)
    expe = Experiment(conf)
    expe.run()

if __name__ == '__main__':
    main(sys.argv)