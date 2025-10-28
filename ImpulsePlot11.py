"""
Multi-channel real-time oscilloscope with selectable FFT per-channel
and on-plot FFT status overlay.

(Full control summary same as before)
"""

import serial, threading, queue, csv, logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import time

# === CONFIG ===
PORT = 'COM4'
BAUDRATE = 115200
SAMPLE_RATE = 500
WINDOW_SECONDS = 4
MAX_CHANNELS = 15
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
FFT_WINDOW_SAMPLES = 1000
TRIGGER_THRESHOLD = 2.5
y_range = 2.0

# === STATE ===
running = True
log_enabled = True
trigger_enabled = True
fft_enabled = False
fft_window_type = 'hann'
fft_span = SAMPLE_RATE / 2
start_time = datetime.now()

# === FILES & STRUCTURES ===
LOG_FILENAME = "accel_log.csv"
TRIGGER_FILENAME = "trigger_log.csv"
data_queue = queue.Queue()
log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

num_channels = 0
channel_names = []
time_buffers = []
data_buffers = []
arrival_times = deque(maxlen=BUFFER_SIZE)
time_buffer = deque(maxlen=BUFFER_SIZE)
fft_selected = set()

# === SERIAL READER THREAD ===
def serial_reader():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)
        ser.reset_input_buffer()
        logging.info(f"Serial connected to {PORT} @ {BAUDRATE}")
        while True:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            parts = line.split('\t')
            vals = []
            for p in parts[:MAX_CHANNELS]:
                try:
                    vals.append(float(p))
                except Exception:
                    break
            if vals:
                try:
                    data_queue.put(tuple(vals), block=False)
                except queue.Full:
                    pass
    except Exception as e:
        logging.error(f"Serial thread error: {e}")

threading.Thread(target=serial_reader, daemon=True).start()

# === CSV WRITER THREAD ===
def csv_writer():
    while True:
        batch = []
        while not log_queue.empty():
            batch.append(log_queue.get())
        if batch:
            with open(LOG_FILENAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(batch)
        time.sleep(1.0)

threading.Thread(target=csv_writer, daemon=True).start()

# === PLOTS ===
plt.ion()
fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle("Multi-channel Oscilloscope")

time_lines = []
fft_lines = {}
time_legend = None

ax_time.set_xlabel('Time (s)')
ax_time.set_ylabel('Acceleration (g)')
ax_fft.set_xlabel('Frequency (Hz)')
ax_fft.set_ylabel('Amplitude (g)')
ax_fft.set_visible(False)
fig.tight_layout()

# 🆕 Overlay text handle (initially empty)
fft_overlay = ax_fft.text(
    0.01, 0.98, "", transform=ax_fft.transAxes,
    fontsize=9, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.6)
)

help_text = (
    "Controls:\n"
    "  SPACE Run/Pause | ↑/↓ Zoom Y | ←/→ Zoom time | L Toggle logging\n"
    "  T Toggle trigger | F Toggle FFT | W Cycle window | I/D FFT size +/-\n"
    "  Z/X FFT zoom in/out | 1..9,0 toggle CH1..CH10 for FFT | ESC Quit\n"
    "Legend: click a label to show/hide that channel"
)
ax_time.text(0.01, 0.98, help_text, transform=ax_time.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

# Colors
cmap = plt.get_cmap('tab20')
channel_colors = [cmap(i) for i in range(20)]

# === Functions ===
def init_channels(n):
    global num_channels, channel_names, time_buffers, data_buffers, time_lines, time_legend
    num_channels = n
    channel_names = [f"CH{i+1}" for i in range(num_channels)]
    time_buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]
    data_buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]
    ax_time.clear()
    time_lines.clear()
    for ch in range(num_channels):
        ln, = ax_time.plot([], [], label=channel_names[ch], color=channel_colors[ch % len(channel_colors)])
        time_lines.append(ln)
    time_legend = ax_time.legend(loc='upper right', fancybox=True, shadow=True)
    for legline, origline in zip(time_legend.get_lines(), time_lines):
        legline.set_picker(True)
        legline._orig_line = origline
    fig.canvas.draw_idle()

def get_window(kind, N):
    kind = (kind or 'hann').lower()
    if kind == 'hann':
        return np.hanning(N)
    elif kind == 'hamming':
        return np.hamming(N)
    elif kind == 'blackman':
        return np.blackman(N)
    else:
        return np.ones(N)

def clear_fft_plot():
    global fft_lines
    for ln in fft_lines.values():
        try:
            ln.remove()
        except Exception:
            pass
    fft_lines = {}
    ax_fft.cla()
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Amplitude (g)')
    # 🆕 reset overlay
    fft_overlay.set_text("")
    fig.canvas.draw_idle()

def append_sample(vals):
    global num_channels
    if num_channels == 0:
        init_channels(min(len(vals), MAX_CHANNELS))
    for ch in range(num_channels):
        v = vals[ch] if ch < len(vals) else 0.0
        data_buffers[ch].append(v)
    if not time_buffer:
        t = 0.0
    else:
        t = time_buffer[-1] + 1.0 / SAMPLE_RATE
    time_buffer.append(t)
    arrival_times.append(datetime.now().timestamp())
    if log_enabled:
        row = [f"{t:.6f}"] + [f"{data_buffers[ch][-1]:.6f}" for ch in range(num_channels)]
        log_queue.put(row)

# 🆕 Overlay updater
def update_fft_overlay():
    """Update the small FFT overlay box with selection info."""
    if not fft_enabled:
        fft_overlay.set_text("")
        return
    sel = ", ".join([f"CH{ch+1}" for ch in sorted(fft_selected)]) or "None"
    text = f"FFT Channels: {sel}\nWindow: {fft_window_type}\nN = {FFT_WINDOW_SAMPLES}\nSpan: 0–{fft_span:.0f} Hz"
    fft_overlay.set_text(text)

def compute_and_plot_fft():
    if not fft_selected:
        clear_fft_plot()
        update_fft_overlay()
        return
    N = FFT_WINDOW_SAMPLES
    if any(len(data_buffers[ch]) < N for ch in fft_selected):
        clear_fft_plot()
        update_fft_overlay()
        return
    clear_fft_plot()
    win = get_window(fft_window_type, N)
    freqs = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)
    scale = 2.0 / N
    for ch in sorted(fft_selected):
        if ch >= num_channels:
            continue
        arr = np.array(list(data_buffers[ch])[-N:]) * win
        fft_vals = np.abs(np.fft.rfft(arr)) * scale
        ln, = ax_fft.plot(freqs, fft_vals, label=f"CH{ch+1}", color=channel_colors[ch % len(channel_colors)])
        fft_lines[ch] = ln
    ax_fft.set_xlim(0, min(fft_span, SAMPLE_RATE / 2))
    ax_fft.legend(loc='upper right')
    update_fft_overlay()
    fig.canvas.draw_idle()

def on_pick(event):
    legline = event.artist
    if not hasattr(legline, "_orig_line"):
        return
    orig = legline._orig_line
    vis = not orig.get_visible()
    orig.set_visible(vis)
    legline.set_alpha(1.0 if vis else 0.25)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('pick_event', on_pick)

# === Key handler (same as before, but updates overlay on FFT changes) ===
def on_key(event):
    global running, y_range, WINDOW_SECONDS, BUFFER_SIZE
    global log_enabled, trigger_enabled, fft_enabled, fft_window_type, FFT_WINDOW_SAMPLES, fft_span

    key = event.key.lower() if event.key else ""
    if key == ' ':
        running = not running
        print("▶️ Resumed" if running else "⏸ Paused")
    elif key == 'f':
        fft_enabled = not fft_enabled
        ax_fft.set_visible(fft_enabled)
        if fft_enabled:
            compute_and_plot_fft()
        else:
            clear_fft_plot()
        fig.tight_layout()
    elif key == 'w':
        types = ['hann', 'hamming', 'blackman', 'rect']
        idx = types.index(fft_window_type) if fft_window_type in types else 0
        fft_window_type = types[(idx + 1) % len(types)]
        print("FFT window:", fft_window_type)
        if fft_enabled:
            compute_and_plot_fft()
    elif key == 'i':
        FFT_WINDOW_SAMPLES = min(int(FFT_WINDOW_SAMPLES * 1.25), BUFFER_SIZE)
        print("FFT samples:", FFT_WINDOW_SAMPLES)
        if fft_enabled:
            compute_and_plot_fft()
    elif key == 'd':
        FFT_WINDOW_SAMPLES = max(int(FFT_WINDOW_SAMPLES / 1.25), 100)
        print("FFT samples:", FFT_WINDOW_SAMPLES)
        if fft_enabled:
            compute_and_plot_fft()
    elif key == 'z':
        fft_span = max(5.0, fft_span / 2.0)
        if fft_enabled:
            ax_fft.set_xlim(0, fft_span)
            update_fft_overlay()
            fig.canvas.draw_idle()
    elif key == 'x':
        fft_span = min(fft_span * 2.0, SAMPLE_RATE / 2.0)
        if fft_enabled:
            ax_fft.set_xlim(0, fft_span)
            update_fft_overlay()
            fig.canvas.draw_idle()
    elif key in [str(n) for n in range(10)]:
        idx = 10 if key == '0' else int(key)
        ch_index = idx - 1
        if ch_index in fft_selected:
            fft_selected.remove(ch_index)
            print(f"FFT: removed CH{ch_index+1}")
        else:
            fft_selected.add(ch_index)
            print(f"FFT: added CH{ch_index+1}")
        clear_fft_plot()
        if fft_enabled:
            compute_and_plot_fft()
        update_fft_overlay()
    elif key == 'escape':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)

# === MAIN LOOP ===
try:
    while plt.get_fignums():
        if running:
            while not data_queue.empty():
                append_sample(data_queue.get())
            if num_channels:
                t_arr = np.array(time_buffer)
                for ch in range(num_channels):
                    y_arr = np.array(data_buffers[ch])
                    time_lines[ch].set_data(t_arr[-len(y_arr):], y_arr)
                if len(t_arr):
                    last_t = t_arr[-1]
                    ax_time.set_xlim(max(0, last_t - WINDOW_SECONDS), last_t)
                    ax_time.set_ylim(-y_range, y_range)
            if fft_enabled:
                compute_and_plot_fft()
            fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            plt.pause(0.05)
except KeyboardInterrupt:
    print("Interrupted.")
finally:
    print("🛑 Exiting.")
