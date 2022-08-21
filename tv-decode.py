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

    received_signal_blocks = channel_filter(
        signal, sample_rate=signal.sample_rate, bandwidth=6, noise=0
    )

    # Approximate number of samples per field
    expected_field_sample_count = (
        0.5 * FRAME_TOTAL_LINES * LINE_DURATION * signal.sample_rate
    )

    frame_image = 0.5 * np.ones((FRAME_ACTIVE_LINES, FRAME_H_PIXELS, 3))
    field_time = 0  # µs
    for field_idx, field in enumerate(
        decode_fields(received_signal_blocks, sample_rate=signal.sample_rate)
    ):
        print(
            f"Detected field at {field_time*1e-3:.3f}ms of duration {1e-3 * field.shape[0] / signal.sample_rate:.3f}ms"
        )

        # Reject field which is too short
        if field.shape[0] < 0.9 * expected_field_sample_count:
            print("Rejecting field which is too short")
            continue

        is_even, field_image = decode_field(field_time, field, signal.sample_rate)
        if is_even:
            frame_image[1::2, ...] = field_image
        else:
            frame_image[0::2, ...] = field_image

        if field_idx > 1:
            outpath = f"decoded-{field_idx-1:04d}.png"
            print(f"Writing frame to {outpath!r}")
            PIL.Image.fromarray((255 * frame_image).astype(np.uint8)).save(outpath)

        field_time += field.shape[0] / signal.sample_rate


if __name__ == "__main__":
    main()
