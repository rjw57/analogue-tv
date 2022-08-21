import math

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
