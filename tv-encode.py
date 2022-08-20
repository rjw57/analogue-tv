#!/usr/bin/env python3

import dataclasses
import enum
import math

import scipy.interpolate
import numpy as np
import numpy.typing as npt
import PIL.Image

# Signal levels in IRE
BLACK_LEVEL = 0.0
WHITE_LEVEL = 100.0
SYNC_LEVEL = -40.0

# For NTSC. Taken from https://www.avrfreaks.net/sites/default/files/ntsctime.pdf
LINE_DURATION = 63.6  # µs
BACK_PORCH_DURATION = 6.2  # µs
FRONT_PORCH_DURATION = 1.5  # µs
H_SYNC_DURATION = 4.7  # µs
ACTIVE_DURATION = (
    LINE_DURATION - BACK_PORCH_DURATION - FRONT_PORCH_DURATION - H_SYNC_DURATION
)

# Vertical sync signals
EQUALIZING_PULSE_DURATION = 2.3  # µs
EQUALIZING_PULSE_PERIOD = LINE_DURATION / 2
SERRATION_PULSE_DURATION = 4.5  # µs
SERRATION_PULSE_PERIOD = LINE_DURATION / 2

# Colour

# For NTSC "the horizontal scanning frequency shall be 2/455 times the color subcarrier
# frequency" (https://antiqueradio.org/art/NTSC%20Signal%20Specifications.pdf)
COLOUR_CARRIER_FREQUENCY = 455 / (2 * LINE_DURATION)  # MHz
COLOUR_CARRIER_AMPLITUDE = 0.5 * SYNC_LEVEL
COLOUR_BURST_START_TIME = H_SYNC_DURATION + 1.0  # µs
COLOUR_BURST_DURATION = 10 / COLOUR_CARRIER_FREQUENCY  # derived from no. of cycles

# Total and active scanlines in a single frame.
FRAME_TOTAL_LINES = 525
FRAME_ACTIVE_LINES = 480

# Nominal horizontal resolution of active area
FRAME_H_PIXELS = 640

# Internal sample rate is such that there are an integer number of samples per
# line. We choose it so we have about 10 samples per colour clock to make sure sampling
# noise does not contribute to signal degredation.
SAMPLES_PER_LINE = int(math.ceil(10 * LINE_DURATION * COLOUR_CARRIER_FREQUENCY))
SAMPLE_RATE = SAMPLES_PER_LINE / LINE_DURATION  # MHz


class Polarity(enum.Enum):
    EVEN = enum.auto()
    ODD = enum.auto()


@dataclasses.dataclass
class Field:
    polarity: Polarity
    image: npt.ArrayLike


def fields(image_file: str):
    """
    Generate fields from the video sequence.

    """
    # Load image
    im = (
        np.asarray(
            PIL.Image.open(image_file)
            .convert("RGB")
            .resize(
                (FRAME_H_PIXELS, FRAME_ACTIVE_LINES),
                resample=PIL.Image.Resampling.LANCZOS,
            )
        ).astype(np.float32)
        / 255.0
    )

    # Pre-calculate odd and even field images normalised
    # to (0, 1)
    odd, even = im[0::2, ...], im[1::2, ...]

    while True:
        yield Field(polarity=Polarity.EVEN, image=even)
        yield Field(polarity=Polarity.ODD, image=odd)


def signal_blocks(fields, *, start_t=0.0):
    """
    Generate blocks of sampled video signal.

    *fields* iterable of Fields
    *start_t* start time of signal in µs.

    """
    # Starting phase of colour carrier expressed as a time offset within the
    # signal. To keep precision we keep this within one period of the colour
    # carrier.
    colour_carrier_offset = np.fmod(start_t, 1 / COLOUR_CARRIER_FREQUENCY)

    # How long will a frame be
    frame_duration = FRAME_TOTAL_LINES * LINE_DURATION

    # Get even and odd fields
    field_iter = iter(fields)
    while True:
        even_field = next(field_iter)
        if even_field.polarity == Polarity.EVEN:
            odd_field = next(field_iter)
            frame_block = encode_frame(
                even_field, odd_field, colour_carrier_offset=colour_carrier_offset
            )
            colour_carrier_offset = np.fmod(
                colour_carrier_offset + frame_duration, 1 / COLOUR_CARRIER_FREQUENCY
            )
            yield frame_block


def encode_frame(even_field: Field, odd_field: Field, *, colour_carrier_offset=0.0):
    # Stack even and odd fields
    combined_frame = np.zeros((FRAME_ACTIVE_LINES, FRAME_H_PIXELS, 3))
    combined_frame[0::2, ...] = odd_field.image
    combined_frame[1::2, ...] = even_field.image

    # Convert to YIQ. Y is scaled between BLACK_LEVEL and WHITE_LEVEL. I and Q
    # are nominally (-0.5, 0.5).
    frame_y = (
        0.3 * combined_frame[..., 0]
        + 0.59 * combined_frame[..., 1]
        + 0.11 * combined_frame[..., 2]
    )
    frame_yiq = np.stack(
        (
            (WHITE_LEVEL - BLACK_LEVEL) * frame_y + BLACK_LEVEL,
            combined_frame[..., 0] - frame_y,  # R - Y
            combined_frame[..., 2] - frame_y,  # B - Y
        ),
        axis=2,
    )

    # Compute (fractional) sample indices for each horizontal pixel based on sample
    # rate, samples per pixel and the start sample for video data
    pixel_duration = ACTIVE_DURATION / frame_yiq.shape[1]
    samples_per_pixel = SAMPLE_RATE * pixel_duration
    pixel_sample_indices = SAMPLE_RATE * (
        H_SYNC_DURATION + BACK_PORCH_DURATION
    ) + samples_per_pixel * np.arange(frame_yiq.shape[1])

    # Resample the frame so that there are the correct number of samples per line. Outside
    # of the active area we return the black level.
    line_interpolator = scipy.interpolate.interp1d(
        x=pixel_sample_indices,
        y=frame_yiq,
        kind="quadratic",
        axis=1,
        fill_value=BLACK_LEVEL,
        bounds_error=False,
    )
    active_yiq = line_interpolator(np.arange(SAMPLES_PER_LINE))
    assert active_yiq.shape == (FRAME_ACTIVE_LINES, SAMPLES_PER_LINE, 3)

    # Construct a full frame
    full_frame_yiq = np.zeros((FRAME_TOTAL_LINES, SAMPLES_PER_LINE, 3))
    blank_lines = FRAME_TOTAL_LINES - FRAME_ACTIVE_LINES
    full_frame_yiq[
        blank_lines:,
    ] = active_yiq

    # Re-order to split even and odd lines
    full_frame_yiq = np.vstack((full_frame_yiq[1::2, ...], full_frame_yiq[0::2, ...]))

    # Add in the hsync pulse to the luminance channel.
    h_sync_n_samples = int(math.ceil(H_SYNC_DURATION * SAMPLE_RATE))
    full_frame_yiq[:, 0:h_sync_n_samples, 0] = SYNC_LEVEL

    # Set colour burst strip as being active in I channel
    cb_start = math.floor(COLOUR_BURST_START_TIME * SAMPLE_RATE)
    cb_end = cb_start + math.ceil(COLOUR_BURST_DURATION * SAMPLE_RATE)
    full_frame_yiq[:, cb_start:cb_end, 1] = 1

    # Compute colour carrier for frame
    n_samples = FRAME_TOTAL_LINES * SAMPLES_PER_LINE
    carrier_times = colour_carrier_offset + np.arange(n_samples) / SAMPLE_RATE
    colour_subcarrier = (
        COLOUR_CARRIER_AMPLITUDE
        * np.exp(1j * (COLOUR_CARRIER_FREQUENCY * 2 * np.pi * carrier_times))
    ).reshape(full_frame_yiq.shape[:2])

    # Start to build actual video signal by summing luminance and modulated chrominance
    video_signal = (
        full_frame_yiq[..., 0]
        + np.real(colour_subcarrier) * full_frame_yiq[..., 1]
        + np.imag(colour_subcarrier) * full_frame_yiq[..., 2]
    )

    # Time within each line
    line_times = np.arange(SAMPLES_PER_LINE) / SAMPLE_RATE

    # Create equalizing and seratation pulse lines.
    equalizing_line = np.where(
        np.fmod(line_times, EQUALIZING_PULSE_PERIOD) < EQUALIZING_PULSE_DURATION,
        SYNC_LEVEL,
        BLACK_LEVEL,
    )
    serration_line = np.where(
        np.fmod(line_times, SERRATION_PULSE_PERIOD)
        >= SERRATION_PULSE_PERIOD - SERRATION_PULSE_DURATION,
        BLACK_LEVEL,
        SYNC_LEVEL,
    )

    # Looking at https://www.avrfreaks.net/sites/default/files/ntsctime.pdf carefully,
    # we can see which lines are have which vertical sync pulses.

    # Even (first) field: lines 1-3 and 7-9 are equalizing and lines 4-6 are serration.
    # Note 1-index to 0-index correction.
    video_signal[[0, 1, 2, 6, 7, 8], ...] = np.repeat(
        equalizing_line.reshape((1, -1)), 6, axis=0
    )
    video_signal[[3, 4, 5], ...] = np.repeat(serration_line.reshape((1, -1)), 3, axis=0)

    # Odd (second) field: lines 263-265 and 270-271 are equalizing and lines 267 and 268 are serration.
    # Note 1-index to 0-index correction.
    video_signal[[262, 263, 264, 269, 270], ...] = np.repeat(
        equalizing_line.reshape((1, -1)), 5, axis=0
    )
    video_signal[[266, 267], ...] = np.repeat(
        serration_line.reshape((1, -1)), 2, axis=0
    )

    # Line 266 is half equalizing and half serration
    video_signal[265] = np.hstack(
        (
            equalizing_line[: equalizing_line.shape[0] >> 1],
            serration_line[equalizing_line.shape[0] >> 1 :],
        )
    )

    # Line 269 is the opposite
    video_signal[268] = np.hstack(
        (
            serration_line[: equalizing_line.shape[0] >> 1],
            equalizing_line[equalizing_line.shape[0] >> 1 :],
        )
    )

    return video_signal.reshape((-1,))


def main():
    image_file = "./img/testcard-f-hires.jpg"
    blocks = []
    for block in signal_blocks(fields(image_file)):
        blocks.append(block)
        if len(blocks) >= 3:
            break

    np.savez_compressed(
        "tv-signal.npz",
        signal=np.concatenate(blocks),
        sample_rate=SAMPLE_RATE,
        input_filename=image_file,
    )


if __name__ == "__main__":
    main()
