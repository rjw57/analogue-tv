#!/usr/bin/env python3
"""
Encode a TV signal

Usage:
    tv-encode.py (-h | --help)
    tv-encode.py <inputimage> <outputnpz>
"""
import docopt
import numpy as np

from tv import *


def main():
    opts = docopt.docopt(__doc__)
    blocks = []
    for block in signal_blocks(encode_fields(opts["<inputimage>"])):
        blocks.append(block.astype(np.float16))
        if len(blocks) >= 6:
            break

    np.savez_compressed(
        opts["<outputnpz>"],
        signal=np.concatenate(blocks),
        sample_rate=SAMPLE_RATE,
        input_filename=opts["<inputimage>"],
    )


if __name__ == "__main__":
    main()
