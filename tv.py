import math
import dataclasses
import enum
import math

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.signal
import PIL.Image

from ntsc import *

# Internal sample rate is such that there are an integer number of samples per
# line. We choose it so we have enough samples per colour clock to make sure sampling
# noise does not contribute to signal degredation.
SAMPLES_PER_LINE = int(math.ceil(8 * LINE_DURATION * COLOUR_CARRIER_FREQUENCY))
SAMPLE_RATE = SAMPLES_PER_LINE / LINE_DURATION  # MHz


class Polarity(enum.Enum):
    EVEN = enum.auto()
    ODD = enum.auto()


@dataclasses.dataclass
class Field:
    polarity: Polarity
    image: npt.ArrayLike


def encode_fields(image_file: str):
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

    # Set scale of I & Q relative to colourburst
    frame_yiq[..., 1:] *= 2

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


class Signal:
    def __init__(self, input_filename):
        with np.load(input_filename) as f:
            self.sample_rate = f["sample_rate"]
            self._signal = f["signal"]

    def __iter__(self):
        return iter(self._generate())

    def _generate(self):
        # Each block is ~100µs of samples
        block_ptr = 0
        block_len = int(math.ceil(100 * self.sample_rate))
        while block_ptr < self._signal.shape[0]:
            yield self._signal[block_ptr : block_ptr + block_len, ...]
            block_ptr += block_len


def noise_blocks(blocks, noise_level):
    for b in blocks:
        yield b + np.random.normal(size=b.shape, scale=noise_level)


def blocks_filtered_by(blocks, filter_, *, sample_rate):
    buffer_size = int(math.ceil(100 * sample_rate))  # ~ 100µs
    prior_block = np.zeros((buffer_size,))
    for block in blocks:
        # Prepend prior block to avoid discontinuities around blocks
        filtered_signal = filter_(np.concatenate((prior_block, block)))
        yield filtered_signal[prior_block.shape[0] :]
        prior_block = filtered_signal[-buffer_size:]


def channel_filter(blocks, sample_rate, *, bandwidth=6, noise=0):
    # Simulate a bandwidth limit with a Butterworth low-pass filter.
    sos = scipy.signal.butter(
        5, bandwidth, analog=False, btype="low", fs=sample_rate, output="sos"
    )
    for b in blocks_filtered_by(
        noise_blocks(blocks, noise),
        lambda s: scipy.signal.sosfilt(sos, s),
        sample_rate=sample_rate,
    ):
        yield b


def composite_sync(block):
    "Recover a composite sync signal for a block by thresholding"
    # Implement sync separator from https://www.ntsc-tv.com/images/tv/sync-sp.gif
    clip_level = -25
    return np.where(block >= clip_level, 1.0, 0.0)


def recover_vsync(cs, sample_rate):
    # We restore the vertical sync by low-pass filtering the composite sync.
    # We need a -3dB frequency which is above the line frequency (so we
    # retrive the vsync pulse) but which is well below the frequency implied by
    # the hsync pulse length.

    # The frequency implied by the h-sync pulse. (I.e. a period of two
    # times the pulse width.)
    h_sync_implied_frequency = 1 / (2 * H_SYNC_DURATION)

    # Choose a cutoff frequency well below the implied pulse frequence.
    cutoff_frequency = 0.1 * h_sync_implied_frequency

    # A simple first order Butterworth filter. Some TVs just use an passive RC filter for this
    # and so we don't need to be too clever.
    sos = scipy.signal.butter(
        1, cutoff_frequency, analog=False, btype="low", fs=sample_rate, output="sos"
    )
    lowpass_signal = scipy.signal.sosfilt(sos, cs)
    return lowpass_signal

    # Compute VSync via comparator.
    return np.where(lowpass_signal < 0.66, 0, 1)


def decode_fields(blocks, *, sample_rate):
    "Yield pairs giving snipped region from vsync to vsync and the recovered hsync for the region."
    # Build up a buffer of blocks. When we find a start end end vsync pulse in the
    # buffer we yield a section of the buffer as a field between the pulses.
    block_buffer = []
    block_buffer_len = 0

    # Minimum number of samples in the buffer to attempt field detection. A full
    # frame should contain two vsync pulses and so we buffer just over one frame.
    min_buffer_len = int(
        math.ceil(FRAME_TOTAL_LINES * LINE_DURATION * sample_rate * 1.2)
    )

    for block in blocks:
        block_buffer.append(block)
        block_buffer_len += block.shape[0]
        if block_buffer_len < min_buffer_len:
            continue

        # We have a buffer we expect to contain at least two vsyncs. Snip it out
        buffer_samples = np.concatenate(block_buffer)

        # Get composite sync
        cs = composite_sync(buffer_samples)

        # Low-pass filter and threshold to recover vsync
        vs = recover_vsync(cs, sample_rate)

        # Find vsync -ve going edges in the buffer.
        vsync_edges = (np.logical_and(vs[:-1] > 0.5, vs[1:] < 0.5)).nonzero()[0]

        # Ignore edges within 1 line delay to account for lag in lowpass filter.
        vsync_edges = vsync_edges[vsync_edges > LINE_DURATION * sample_rate]

        # There must be at least one edge
        if vsync_edges.shape[0] < 1:
            continue

        # Now find the next edge which should be at least one half field in the future
        vsync_start = vsync_edges[0]
        vsync_edges = vsync_edges[
            vsync_edges
            > vsync_start + 0.25 * FRAME_TOTAL_LINES * LINE_DURATION * sample_rate
        ]

        # We expect at least one other edge. Otherwise, snip at the one we have and continue
        if vsync_edges.shape[0] < 1:
            block_buffer = [buffer_samples[vsync_start:]]
            block_buffer_len = block_buffer[0].shape[0]
            continue

        vsync_end = vsync_edges[0]

        # Put region of buffer around a line and a half before second edge back
        # in block buffer
        block_buffer = [
            buffer_samples[
                max(0, vsync_end - int(math.ceil(LINE_DURATION * 1.5 * sample_rate))) :
            ]
        ]
        block_buffer_len = block_buffer[0].shape[0]

        # Yield snipped region
        yield buffer_samples[vsync_start:vsync_end]


def decode_field(field_time, field, sample_rate, *, colour=True):
    # The horizontal sync is recovered by triggering on low-going pulses from the
    # composite sync but not re-triggering for some hold time. (In the sync separator
    # circuit this is done by having the reset line for the one-shot trigger driven
    # by an RC-filter.) Implementing this efficiently in numpy is a little tricky.

    # Compute indices of low-going edges from composite sync.
    cs = composite_sync(field)
    cs_low_going_edge_indices = np.logical_and(cs[:-1] > 0.5, cs[1:] <= 0.5).nonzero()[
        0
    ]

    # Now many samples in the future must the next low-going edge be? We make this
    # most of a line.
    min_sample_separation = 0.8 * LINE_DURATION * sample_rate

    # Recovered hsync edges
    h_sync_edges = []

    # Walk edges adding in sync pulses.
    for edge_idx in cs_low_going_edge_indices:
        if (
            len(h_sync_edges) == 0
            or (edge_idx - h_sync_edges[-1]) >= min_sample_separation
        ):
            h_sync_edges.append(edge_idx)

    # Reject first 18 edges to allow hsync to have actually synchronised and stopped
    # being confused by vsync. This should line us up with the start of the active area.
    h_sync_edges = h_sync_edges[18:]

    # We must have enough for the actual field
    assert len(h_sync_edges) > FRAME_ACTIVE_LINES >> 1

    # What is the fractional difference between hsync and vsync location measured in
    # line delays?
    hv_drift = np.fmod(h_sync_edges[0] / (LINE_DURATION * sample_rate), 1.0)

    # Use this to detect even vs odd fields
    is_even = hv_drift > 0.5

    # Expected samples per line
    samples_per_line = int(math.ceil(LINE_DURATION * sample_rate))

    # Decode lines
    line_time = field_time
    decoded_lines = np.zeros((FRAME_ACTIVE_LINES >> 1, FRAME_H_PIXELS, 3))
    decode_line_cb = decode_line if colour else decode_line_bw
    for line_idx, (start_idx, end_idx) in enumerate(
        zip(h_sync_edges[: FRAME_ACTIVE_LINES >> 1], h_sync_edges[1:])
    ):
        line_samples = field[start_idx:end_idx]
        decoded_lines[line_idx, ...] = decode_line_cb(
            line_idx, field_time + start_idx / sample_rate, line_samples, sample_rate
        )

    return is_even, decoded_lines


def reconstruct_colourburst_phase(cb_freq, cb_times, cb_samples):
    complex_cb = COLOUR_CARRIER_AMPLITUDE * np.exp(1j * cb_freq * 2 * np.pi * cb_times)
    response = np.sum(cb_samples * complex_cb)
    cb_phase = -np.angle(response)
    return cb_phase


def decode_line_bw(line_idx, line_time, line, sample_rate):
    # Times within line of each sample.
    line_times = line_time + np.arange(line.shape[0]) / sample_rate
    line_luma = (line - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL)

    # Convert Y to RGB
    line_rgb = np.zeros((line_luma.shape[0], 3))
    for c in (0, 1, 2):
        line_rgb[..., c] = line_luma

    # Resample line into pixels
    line_interpolator = scipy.interpolate.interp1d(
        x=line_times, y=line_rgb, axis=0, kind="quadratic"
    )
    pixel_duration = ACTIVE_DURATION / FRAME_H_PIXELS
    pixel_times = (
        line_time
        + np.arange(FRAME_H_PIXELS) * pixel_duration
        + H_SYNC_DURATION
        + BACK_PORCH_DURATION
    )
    return np.clip(line_interpolator(pixel_times), 0, 1)


def decode_line(line_idx, line_time, line, sample_rate):
    # Times within line of each sample.
    line_times = line_time + np.arange(line.shape[0]) / sample_rate

    # Prepare buffer for line luma and I/Q components
    line_yiq = np.zeros((line.shape[0], 3))

    # Compute chroma bandpass filter based on the nominal colour burst frequency.
    colourbust_bp_ba = scipy.signal.iirpeak(
        COLOUR_CARRIER_FREQUENCY, Q=3, fs=sample_rate
    )

    # Filter line into luma and chroma components
    chroma = scipy.signal.lfilter(*colourbust_bp_ba, x=line)
    line_yiq[..., 0] = (line - chroma - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL)

    # Extract colour burst from chroma bandpass filtered signal
    cb_start_sample = int(math.floor(COLOUR_BURST_START_TIME * sample_rate))
    cb_sample_count = int(math.ceil(COLOUR_BURST_START_TIME * sample_rate))
    cb_samples = chroma[cb_start_sample : cb_start_sample + cb_sample_count]
    cb_times = line_times[cb_start_sample : cb_start_sample + cb_sample_count]

    # For the moment we do not attempt to track colourbust frequency.
    cb_phase = reconstruct_colourburst_phase(
        COLOUR_CARRIER_FREQUENCY, cb_times, cb_samples
    )

    # Demodulate I and Q components.
    cb_subcarrier = np.exp(
        1j * (COLOUR_CARRIER_FREQUENCY * 2 * np.pi * line_times + cb_phase)
    )
    line_yiq[..., 1] = np.real(cb_subcarrier) * chroma / COLOUR_CARRIER_AMPLITUDE
    line_yiq[..., 2] = np.imag(cb_subcarrier) * chroma / COLOUR_CARRIER_AMPLITUDE

    # Pass demodulation through lowpass filter using half nominal chroma burst frequency
    # as the -3dB point.
    colourburst_low_sos = scipy.signal.butter(
        3,
        0.5 * COLOUR_CARRIER_FREQUENCY,
        analog=False,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    )
    # Using the filtfilt variant has zero phase. In the real world we'd probably
    # use a delay line on the luminance.
    line_yiq[..., 1:] = scipy.signal.sosfiltfilt(
        colourburst_low_sos, line_yiq[..., 1:], axis=0
    )
    # line_yiq[..., 1:] -= scipy.signal.lfilter(*colourbust_bp_ba, x=line_yiq[..., 1:], axis=0)

    # Convert YIQ to RGB
    line_rgb = np.zeros_like(line_yiq)
    line_rgb[..., 0] = line_yiq[..., 1] + line_yiq[..., 0]
    line_rgb[..., 2] = line_yiq[..., 2] + line_yiq[..., 0]
    line_rgb[..., 1] = (
        line_yiq[..., 0] - 0.3 * line_rgb[..., 0] - 0.11 * line_rgb[..., 2]
    ) / 0.59

    # Resample line into pixels
    line_interpolator = scipy.interpolate.interp1d(
        x=line_times, y=line_rgb, axis=0, kind="quadratic"
    )
    pixel_duration = ACTIVE_DURATION / FRAME_H_PIXELS
    pixel_times = (
        line_time
        + np.arange(FRAME_H_PIXELS) * pixel_duration
        + H_SYNC_DURATION
        + BACK_PORCH_DURATION
    )
    return np.clip(line_interpolator(pixel_times), 0, 1)


class Decoder:
    """
    Initialise with a sample rate in MHz.

    """

    sample_rate: float  # MHz
    time: float  # µs
    colour_burst_freq: float  # MHz
    colour_burst_phase: float  # Radians

    h_sync_frequency: float  # MHz
    h_sync_offset: float  # µs

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.time = 0.0

        # Initial values for state
        self.colour_burst_freq = COLOUR_CARRIER_FREQUENCY
        self.colour_burst_phase = 0.0
        self.h_sync_frequency = 1.0 / LINE_DURATION
        self.h_sync_offset = 0.0

        # Compute chroma bandpass filter based on the nominal colour burst frequency.
        self._chroma_bp_ba = scipy.signal.iirpeak(
            COLOUR_CARRIER_FREQUENCY, Q=3, fs=self.sample_rate
        )

        # Compute demodulated chroma sigla lowpass filter based on the nominal colour burst frequency.
        self._chroma_low_sos = scipy.signal.butter(
            1,
            0.2 * COLOUR_CARRIER_FREQUENCY,
            analog=False,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )

        # We restore the vertical sync by low-pass filtering the composite sync.
        # Create a simple second order Butterworth filter for filtering out the vsync.
        self._vsync_lp_sos = scipy.signal.butter(
            2,
            0.8 * self.h_sync_frequency,
            analog=False,
            btype="low",
            fs=self.sample_rate,
            output="sos",
        )

        # We keep a small portion of the previous block around so that our filters
        # don't need to be padded backwards in time
        self.previous_block = np.zeros((0,))

        # Previous hsync time to help us filter out transient edges
        self.previous_h_sync_edge_time = 0.0

        # Previous vsync time to help generate ramp signal
        self.previous_v_sync_edge_time = 0.0

        # Previous hsync and vsync ramp signal end samples to help detect edges
        self.previous_h_sync_ramp_value = 0.0
        self.previous_v_sync_ramp_value = 0.0

        # Undecoded line portion from previous block(s)
        self.previous_line_v_sync_ramp = np.zeros((0,))
        self.previous_line_luma = np.zeros((0,))
        self.previous_line_chroma = np.zeros((0,))

        # 0-based index of line we're currently decoding
        self.line_index = 0

        # Current full frame
        self.full_frame = np.zeros((FRAME_ACTIVE_LINES, FRAME_H_PIXELS, 3))

    def _separarate_chroma(self, signal):
        """
        Return the luma and choma components of the passed signal.

        """
        chroma = scipy.signal.filtfilt(*self._chroma_bp_ba, x=signal)
        return signal - chroma, chroma

    def _compute_h_sync(self, times):
        """
        Given absolute times, return hsync ramp signal

        """
        return np.fmod((times - self.h_sync_offset) * self.h_sync_frequency, 1)

    def _reconstruct_colourburst_phase(self, cb_times, cb_samples):
        complex_cb = COLOUR_CARRIER_AMPLITUDE * np.exp(
            1j * self.colour_burst_freq * 2 * np.pi * cb_times
        )
        response = np.sum(cb_samples * complex_cb)
        cb_phase = -np.angle(response)
        return cb_phase

    def _decode_line(self, line_time, luma, chroma):
        # The line must be roughly the right length.
        if luma.shape[0] < 0.5 * LINE_DURATION / self.sample_rate:
            return np.ones((FRAME_H_PIXELS, 3)) * 0.5

        # Luma is fairly easy
        line_y = (luma - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL)

        # Times of line samples.
        line_times = line_time + np.arange(luma.shape[0]) / self.sample_rate

        # Extract colour burst from chroma bandpass filtered signal
        cb_start_sample = int(math.floor(COLOUR_BURST_START_TIME * self.sample_rate))
        cb_sample_count = int(math.ceil(COLOUR_BURST_START_TIME * self.sample_rate))
        cb_samples = chroma[cb_start_sample : cb_start_sample + cb_sample_count]
        cb_times = line_times[cb_start_sample : cb_start_sample + cb_sample_count]

        # For the moment we do not attempt to track colourbust frequency.
        cb_phase = self._reconstruct_colourburst_phase(cb_times, cb_samples)

        # Fold in phase gradually
        cb_mix = 0.1
        self.colour_burst_phase = (
            cb_mix * cb_phase + (1 - cb_mix) * self.colour_burst_phase
        )

        # Compute I and Q chroma via demodulation.
        cb_subcarrier = np.exp(
            1j
            * (
                self.colour_burst_freq * 2 * np.pi * line_times
                + self.colour_burst_phase
            )
        )
        line_i = np.real(cb_subcarrier) * chroma / COLOUR_CARRIER_AMPLITUDE
        line_q = np.imag(cb_subcarrier) * chroma / COLOUR_CARRIER_AMPLITUDE

        # Pass demodulation through lowpass filter using nominal chroma burst frequency.
        line_i = scipy.signal.sosfilt(self._chroma_low_sos, line_i)
        line_q = scipy.signal.sosfilt(self._chroma_low_sos, line_q)

        # Compute RGB for line.
        line_rgb = np.zeros((luma.shape[0], 3))
        line_rgb[..., 0] = line_i + line_y
        line_rgb[..., 2] = line_q + line_y
        line_rgb[..., 1] = (
            line_y - 0.3 * line_rgb[..., 0] - 0.11 * line_rgb[..., 2]
        ) / 0.59

        # Re-sample line into pixels/
        pixel_times = (
            H_SYNC_DURATION
            + BACK_PORCH_DURATION
            + np.arange(FRAME_H_PIXELS) * ACTIVE_DURATION / FRAME_H_PIXELS
        )
        rgb_interp = scipy.interpolate.interp1d(
            x=line_times - line_time,
            y=line_rgb,
            axis=0,
            kind="quadratic",
            bounds_error=False,
            fill_value=0,
        )
        return rgb_interp(pixel_times)

    def __call__(self, block):
        """
        Decode a block of samples. Returns a list of decoded fields. This
        list may be empty if more samples need to be consumed.

        """
        # Frames to return
        decoded_frames = []

        # Number of blank lines in a full frame
        n_blank_lines = FRAME_TOTAL_LINES - FRAME_ACTIVE_LINES

        # Number of samples within this block and its starting time.
        n_samples = block.shape[0]
        block_start_time = self.time

        # Time of samples within block
        block_times = np.arange(n_samples) / self.sample_rate + block_start_time

        # Pad block with previous block.
        padded_block = np.concatenate((self.previous_block, block))

        # Separate chroma and luma signal.
        luma, chroma = self._separarate_chroma(padded_block)

        # Compute composite sync by thresholding block.
        cs = np.where(padded_block < -25, 0, 1)

        # Compute vsync lowpass filter and edges where we detect vsync.
        v_sync_lowpass = scipy.signal.sosfilt(self._vsync_lp_sos, cs)
        v_sync_edge_indices = np.logical_and(
            v_sync_lowpass[:-1] > 0.66, v_sync_lowpass[1:] <= 0.66
        ).nonzero()[0]

        # Detect hsync edges as -ve going pulses in composite sync.
        h_sync_edge_indices = np.logical_and(cs[:-1] > 0.5, cs[1:] <= 0.5).nonzero()[0]

        # Prune edge indices and signals outside of the current block
        luma = luma[self.previous_block.shape[0] :]
        chroma = chroma[self.previous_block.shape[0] :]
        cs = cs[self.previous_block.shape[0] :]
        v_sync_lowpass = v_sync_lowpass[self.previous_block.shape[0] :]
        h_sync_edge_indices -= self.previous_block.shape[0]
        h_sync_edge_indices = h_sync_edge_indices[h_sync_edge_indices >= 0]
        v_sync_edge_indices -= self.previous_block.shape[0]
        v_sync_edge_indices = v_sync_edge_indices[v_sync_edge_indices >= 0]

        # Compute the vsync ramp signal
        v_sync_ramp = np.zeros_like(block)
        prior_edge_idx = 0
        for edge_idx in np.concatenate((v_sync_edge_indices, [n_samples])):
            # Times *within* frame for this region of the signal
            times = (
                block_times[prior_edge_idx:edge_idx] - self.previous_v_sync_edge_time
            )

            # Compute vsync ramp signal based on field duration.
            v_sync_ramp[prior_edge_idx:edge_idx] = np.fmod(
                times / (LINE_DURATION * FRAME_TOTAL_LINES * 0.5), 1
            )
            prior_edge_idx = edge_idx

            # That's if if we're not a "real" edge but just the last part of the block.
            if edge_idx == n_samples:
                continue

            # If this is a "real" edge, record the edge time.
            self.previous_v_sync_edge_time = block_times[edge_idx]

        # Compute the hsync ramp signal
        h_sync_ramp = np.zeros_like(block)
        prior_edge_idx = 0
        for edge_idx in np.concatenate((h_sync_edge_indices, [n_samples])):
            # Times for this region of the signal
            times = block_times[prior_edge_idx:edge_idx]

            # Compute hsync signal which ranges from 0 to 1
            h_sync_ramp[prior_edge_idx:edge_idx] = self._compute_h_sync(times)
            prior_edge_idx = edge_idx

            # That's if if we're not a "real" edge but just the last part of the block.
            if edge_idx == n_samples:
                continue

            # If this is a "real" edge, adjust the h_sync offset.

            # Ignore pulses which are not close to where we expect to find one. This is
            # to ignore transient hsync pulses and the vsync pulses.
            edge_time = block_times[edge_idx]
            if edge_time - self.previous_h_sync_edge_time < LINE_DURATION * 0.75:
                continue

            # Get a measure of the position within the hsync ramp this edge lies
            # ranging from -0.5 to 0.5. If we're perfectly in sync, this should be
            # zero. If it is +ve we need to increase offset to shift hsync to the
            # right and vice versa.
            position_within_h_sync = self._compute_h_sync(edge_time)
            if position_within_h_sync > 0.5:
                position_within_h_sync -= 1.0

            # Merge in phase change gradually. We don't yet update hsync frequency estimates.
            self.h_sync_offset += 0.01 * position_within_h_sync / self.h_sync_frequency

            # Record edge time.
            self.previous_h_sync_edge_time = edge_time

        # Re-construct "clean" hsync and vsync edges.
        augmented_h_sync_ramp = np.concatenate(
            ([self.previous_h_sync_ramp_value], h_sync_ramp)
        )
        clean_h_sync_edge_indices = np.logical_and(
            augmented_h_sync_ramp[:-1] > 0.5, augmented_h_sync_ramp[1:] <= 0.5
        ).nonzero()[0]

        # Decode each line
        prior_edge_idx = 0
        for edge_idx in clean_h_sync_edge_indices:
            line_v_sync_ramp = np.concatenate(
                (self.previous_line_v_sync_ramp, v_sync_ramp[prior_edge_idx:edge_idx])
            )
            line_luma = np.concatenate(
                (self.previous_line_luma, luma[prior_edge_idx:edge_idx])
            )
            line_chroma = np.concatenate(
                (self.previous_line_chroma, chroma[prior_edge_idx:edge_idx])
            )

            # Was there a vsync edge on this line?
            line_v_sync_edge_indices = np.logical_and(
                line_v_sync_ramp[:-1] > 0.5, line_v_sync_ramp[1:] <= 0.5
            ).nonzero()[0]
            if line_v_sync_edge_indices.shape[0] > 0:
                decoded_frames.append(self.full_frame.copy())

                # If we're before half way through, we're an even frame
                if line_v_sync_edge_indices[0] < line_luma.shape[0] * 0.5:
                    self.line_index = 7
                else:
                    self.line_index = 6

            # Decode line and record in full frame if appropriate.
            line_rgb = self._decode_line(
                block_times[edge_idx] - line_luma.shape[0] / self.sample_rate,
                line_luma,
                line_chroma,
            )
            if self.line_index >= n_blank_lines and self.line_index < FRAME_TOTAL_LINES:
                self.full_frame[self.line_index - n_blank_lines, ...] = line_rgb

            prior_edge_idx = edge_idx
            self.previous_line_luma = np.zeros((0,))
            self.previous_line_chroma = np.zeros((0,))
            self.previous_line_v_sync_ramp = np.zeros((0,))
            self.line_index += 2

        # Save undecoded line portion
        if clean_h_sync_edge_indices.shape[0] > 0:
            self.previous_line_luma = luma[clean_h_sync_edge_indices[-1] :]
            self.previous_line_chroma = chroma[clean_h_sync_edge_indices[-1] :]
            self.previous_line_v_sync_ramp = v_sync_ramp[
                clean_h_sync_edge_indices[-1] :
            ]
        else:
            self.previous_line_luma = np.concatenate((self.previous_line_luma, luma))
            self.previous_line_chroma = np.concatenate(
                (self.previous_line_chroma, chroma)
            )
            self.previous_line_v_sync_ramp = np.concatenate(
                (self.previous_line_v_sync_ramp, v_sync_ramp)
            )

        # Record last ~100µs of block as the next "previous" block.
        self.previous_block = block[-int(math.ceil(100 / self.sample_rate)) :]

        # Update rolling state.
        self.time += n_samples / self.sample_rate
        self.previous_h_sync_ramp_value = h_sync_ramp[-1]
        self.previous_v_sync_ramp_value = v_sync_ramp[-1]

        return decoded_frames
