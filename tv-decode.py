#!/usr/bin/env python3
"""
Decode a TV signal

Usage:
    tv-encode.py (-h | --help)
    tv-encode.py <inputnpz>
"""

import docopt
import PIL.Image

from tv import *


def main():
    opts = docopt.docopt(__doc__)
    signal = Signal(opts["<inputnpz>"])
    print(f"Input signal sampled at {signal.sample_rate:.2f}MHz")

    decode = Decoder(signal.sample_rate)

    received_signal_blocks = channel_filter(
        signal, sample_rate=signal.sample_rate, bandwidth=6, noise=0
    )
    frame_idx = 0
    for block in received_signal_blocks:
        frame_images = decode(block)

        for im in frame_images:
            im = PIL.Image.fromarray(np.clip(im * 255, 0, 255).astype(np.uint8))
            frame_filename = f"decoded-{1+frame_idx:04d}.png"
            im.save(frame_filename)
            print(f"saved {frame_filename}")
            frame_idx += 1


if __name__ == "__main__":
    main()
