# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time heart rate (BPM) estimation from a laptop webcam using remote photoplethysmography (rPPG) with the CHROM algorithm. Single-file Python app (`pulsometer.py`).

## Commands

```bash
uv sync              # install dependencies
uv run python pulsometer.py  # run the app (requires webcam access)
```

No tests, no linter configured yet.

## Architecture

Single-file app with this pipeline running per-frame in a loop:

1. **Capture & face detect** — OpenCV Haar cascade (`haarcascade_frontalface_default.xml`), picks largest face
2. **ROI extraction** — forehead region (top 25% of face bbox, horizontally narrowed)
3. **Signal buffering** — spatial mean of R, G, B channels stored in a 10s sliding window
4. **CHROM** — chrominance-based rPPG: builds two linear combinations of normalized RGB, uses adaptive alpha to cancel noise
5. **Bandpass + FFT** — Butterworth filter (45–180 BPM range), FFT to find dominant frequency
6. **Display** — OpenCV window with BPM overlay (exponentially smoothed) and live waveform plot

Key tuning constants are at the top of `pulsometer.py` (window size, BPM range, forehead fraction, plot dimensions).

## Code standards

- Follow SOLID, YAGNI, DRY
- Always specify return types and parameter types on all functions

## Tech

- Python >=3.13, managed with **uv** (never global python)
- opencv-python (full, not headless — needs GUI), numpy, scipy
