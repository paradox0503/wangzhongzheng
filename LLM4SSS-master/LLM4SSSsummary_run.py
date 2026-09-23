"""Original launch command, now exporting the full dataset and query only.

Training is deliberately not imported or invoked, even when an older JSON
configuration still contains epoch_max=100. Default pretrained mode uses the
configured backbone and untrained task layers; explicit checkpoint mode loads
the complete task model weights.
"""
import sys

from export_embeddings import main as export_main


def main(argv):
    export_main(argv[1:])

if __name__ == '__main__':
    main(sys.argv)
