# Pulsometer

Real-time heart rate estimation from a webcam using remote photoplethysmography (rPPG).

Uses the CHROM (Chrominance-based) algorithm to extract pulse signal from subtle skin color changes captured by an ordinary laptop camera.

## Setup

```
uv sync
```

## Usage

```
uv run python pulsometer.py
```

Sit still facing the webcam in good, steady lighting. BPM appears after ~4 seconds of calibration. Press `q` to quit.

## How it works

1. Haar cascade detects face, forehead ROI is extracted
2. Per-frame spatial mean of R, G, B channels is buffered (10s sliding window)
3. CHROM algorithm separates pulse signal from noise
4. Butterworth bandpass filter (45–200 BPM) + FFT finds dominant frequency
5. BPM is displayed with exponential smoothing

## Dependencies

- opencv-python
- numpy
- scipy
