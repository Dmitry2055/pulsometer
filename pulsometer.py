"""
Real-time pulse measurement from webcam using rPPG (CHROM method).

Based on: De Haan & Jezemik, "Robust pulse rate from chrominance-based rPPG" (2013).

Usage: uv run pulsometer.py
"""

import time

import cv2
import numpy as np
from scipy.signal import butter, filtfilt


# --- Configuration ---
WINDOW_SEC = 10        # seconds of signal to keep for FFT
MIN_WINDOW_SEC = 4     # minimum seconds before attempting BPM estimation
BPM_LOW = 45           # low end of bandpass (BPM)
BPM_HIGH = 200         # high end of bandpass (BPM)
PLOT_HEIGHT = 120       # height of the signal plot overlay (px)
PLOT_WIDTH = 300        # width of the signal plot overlay (px)
FOREHEAD_FRAC = 0.4     # fraction of face bbox height to use as forehead ROI


def bandpass(signal: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    # Need at least padlen=3*max(len(a),len(b)) samples for filtfilt
    min_samples = 3 * max(len(a), len(b)) + 1
    if len(signal) < min_samples:
        return signal
    return filtfilt(b, a, signal)


def chrom(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    CHROM: extract pulse signal from RGB time-series.
    Returns a 1-D signal whose dominant frequency corresponds to heart rate.
    """
    # Normalize each channel by its mean to get unit-mean signals
    r_n = r / (np.mean(r) + 1e-8)
    g_n = g / (np.mean(g) + 1e-8)
    b_n = b / (np.mean(b) + 1e-8)

    # CHROM linear combinations
    xs = 3.0 * r_n - 2.0 * g_n
    ys = 1.5 * r_n + g_n - 1.5 * b_n

    # Adaptive alpha
    alpha = np.std(xs) / (np.std(ys) + 1e-8)
    return xs - alpha * ys


def estimate_bpm(signal: np.ndarray, fs: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Run FFT on signal, return (dominant_bpm, freqs_bpm, magnitude).
    Only considers frequencies within [BPM_LOW, BPM_HIGH].
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs) * 60.0  # convert Hz -> BPM
    fft_mag = np.abs(np.fft.rfft(signal))

    # Mask to valid BPM range
    mask = (freqs >= BPM_LOW) & (freqs <= BPM_HIGH)
    if not np.any(mask):
        return 0.0, freqs, fft_mag

    fft_masked = fft_mag.copy()
    fft_masked[~mask] = 0.0

    peak_idx = np.argmax(fft_masked)
    return freqs[peak_idx], freqs, fft_mag


def draw_plot(frame: np.ndarray, signal: np.ndarray, x0: int, y0: int) -> None:
    """Draw a small waveform plot onto the frame."""
    if len(signal) < 2:
        return
    # Subsample / take last PLOT_WIDTH points
    sig = signal[-PLOT_WIDTH:]
    n = len(sig)
    # Normalize to [0, 1]
    lo, hi = sig.min(), sig.max()
    if hi - lo < 1e-8:
        normed = np.full_like(sig, 0.5)
    else:
        normed = (sig - lo) / (hi - lo)

    # Background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + PLOT_WIDTH, y0 + PLOT_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw line
    for i in range(1, n):
        px = x0 + int(i * PLOT_WIDTH / n)
        py = y0 + PLOT_HEIGHT - int(normed[i] * (PLOT_HEIGHT - 10)) - 5
        px_prev = x0 + int((i - 1) * PLOT_WIDTH / n)
        py_prev = y0 + PLOT_HEIGHT - int(normed[i - 1] * (PLOT_HEIGHT - 10)) - 5
        cv2.line(frame, (px_prev, py_prev), (px, py), (0, 255, 0), 1, cv2.LINE_AA)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Ring buffers for RGB channel means + timestamps
    timestamps: list[float] = []
    r_vals: list[float] = []
    g_vals: list[float] = []
    b_vals: list[float] = []

    bpm_display = 0.0
    bpm_smooth = 0.0
    last_face_rect = None

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]
        now = time.monotonic()

        # --- Face detection (every frame — Haar is fast enough) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        if len(faces) > 0:
            # Pick largest face
            areas = [fw * fh for (_, _, fw, fh) in faces]
            idx = int(np.argmax(areas))
            last_face_rect = faces[idx]

        if last_face_rect is not None:
            fx, fy, fw, fh = last_face_rect

            # Forehead ROI: top portion of face box, narrowed a bit horizontally
            roi_x = fx + fw // 6
            roi_y = fy
            roi_w = fw - fw // 3
            roi_h = int(fh * FOREHEAD_FRAC)

            # Clamp
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(roi_w, w - roi_x)
            roi_h = min(roi_h, h - roi_y)

            if roi_w > 5 and roi_h > 5:
                roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                means = roi.mean(axis=(0, 1))  # BGR

                b_vals.append(float(means[0]))
                g_vals.append(float(means[1]))
                r_vals.append(float(means[2]))
                timestamps.append(now)

                # Draw ROI rectangle
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

            # Draw face rectangle
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), 1)

        # --- Trim to WINDOW_SEC ---
        cutoff = now - WINDOW_SEC
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)
            r_vals.pop(0)
            g_vals.pop(0)
            b_vals.pop(0)

        # --- Estimate BPM ---
        elapsed = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        if elapsed >= MIN_WINDOW_SEC and len(timestamps) >= 30:
            fs = len(timestamps) / elapsed  # empirical frame rate

            pulse = chrom(np.array(r_vals), np.array(g_vals), np.array(b_vals))
            pulse = bandpass(pulse, fs, BPM_LOW / 60.0, BPM_HIGH / 60.0)

            bpm, _, _ = estimate_bpm(pulse, fs)

            if bpm > 0:
                # Exponential smoothing
                if bpm_smooth == 0:
                    bpm_smooth = bpm
                else:
                    bpm_smooth = 0.7 * bpm_smooth + 0.3 * bpm
                bpm_display = bpm_smooth

            # Draw signal plot
            draw_plot(frame, pulse, w - PLOT_WIDTH - 10, 10)

        # --- HUD ---
        if bpm_display > 0:
            txt = f"BPM: {bpm_display:.0f}"
            color = (0, 255, 0)
        else:
            secs_left = max(0, MIN_WINDOW_SEC - elapsed)
            txt = f"Calibrating... {secs_left:.0f}s"
            color = (0, 200, 255)

        cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        if last_face_rect is None:
            cv2.putText(frame, "No face detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Pulsometer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
