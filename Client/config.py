import argparse
import numpy as np

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--sf',
                        type=int,
                        default=7,
                        help='The spreading factor.')
    parser.add_argument('--bw',
                        type=int,
                        default=125000,
                        help='The bandwidth.')
    parser.add_argument('--fs',
                        type=int,
                        default=1000000,
                        help='The sampling rate.')


    return parser