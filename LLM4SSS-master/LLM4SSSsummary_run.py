"""Original launch command, now exporting the full dataset and query only.

Training is deliberately not imported or invoked, even when an older JSON
configuration still contains epoch_max=100. The exporter requires an existing
checkpoint and does not fall back to training or random initialization.
"""
import sys

from export_embeddings import main as export_main


def main(argv):
    export_main(argv[1:])

if __name__ == '__main__':
    main(sys.argv)
