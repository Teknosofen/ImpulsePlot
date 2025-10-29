"""
Multi-channel real-time oscilloscope with selectable FFT per-channel,
clickable legend, per-channel trigger, FFT overlay, amplitude dB toggle,
measured data-rate title, and helpful inline comments.

Designed for desktop Python (VS Code) with matplotlib interactive backend.
Requires: pyserial, numpy, matplotlib
"""

import serial
import serial.tools.list_ports
import threading
import queue
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import time

# ----------------------------
# === User-configurable top ===
# ----------------------------
PORT = None                  # will be selected from available ports
BAUDRATE = 115200
SAMPLE_RATE = 2000           # Hz
WINDOW_SECONDS = 4          # visible time window in seconds
MAX_CHANNELS = 15           # accept up to 15 values per line
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)  # circular buffer size
FFT_WINDOW_SAMPLES = 1000   # initial FFT window length (samples)
TRIGGER_THRESHOLD = 2.5     # g, per-channel trigger threshold
y_range = 2.0               # Y axis +/- range for time plot
DISPLAY_DECIMATION = 4      # show every Nth point (1=all, 2=every 2nd, 4=every 4th, etc.)

# ----------------------------
# === Internal run-time state
# ----------------------------
running = True                     # pause/resume state
log_enabled = False                # start logging disabled as requested
trigger_enabled = True
fft_enabled = False
fft_window_type = 'hann'           # 'hann'|'hamming'|'blackman'|'rect'
fft_span = SAMPLE_RATE / 2        # displayed frequency span (Hz)
start_time = datetime.now()
serial_connected = False           # tracks if serial connection is active
data_flowing = False               # tracks if valid data is being received
last_data_time = None              # timestamp of last received data

# amplitude display mode: 'linear' or 'dB'; default linear
amplitude_mode = 'linear'

# files and queues
LOG_FILENAME = "accel_log.csv"
TRIGGER_FILENAME = "trigger_log.csv"
data_queue = queue.Queue()
log_queue = queue.Queue()

# logging config
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# dynamic buffers and channel metadata (initialized on first valid line)
num_channels = 0
channel_names = []
data_buffers = []               # list of deques (one per channel)
time_buffer = deque(maxlen=BUFFER_SIZE)
arrival_times = deque(maxlen=BUFFER_SIZE)
first_sample_time = None        # absolute timestamp of first sample (never changes)

# FFT selection state: channels included in FFT plotting (0-based)
# Keys 1..9 and 0 map to CH1..CH10
fft_selected = set()

# UI state
show_help_overlay = True

# color map for channels
cmap = plt.get_cmap('tab20')
channel_colors = [cmap(i) for i in range(20)]

# ----------------------------
# === Serial port selection ===
# ----------------------------
def select_serial_port():
    """
    Display available serial ports and let user select one.
    Returns the selected port name or None if cancelled.
    """
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports found!")
        return None
    
    print("\n" + "="*60)
    print("Available Serial Ports:")
    print("="*60)
    for i, port in enumerate(ports, 1):
        print(f"{i}. {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Manufacturer: {port.manufacturer or 'N/A'}")
        print("-"*60)
    
    while True:
        try:
            choice = input(f"\nSelect port (1-{len(ports)}) or 'q' to quit: ").strip().lower()
            if choice == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(ports):
                selected = ports[idx].device
                print(f"✓ Selected: {selected}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(ports)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            return None

# Select serial port before starting
PORT = select_serial_port()
if PORT is None:
    print("No port selected. Exiting.")
    exit(0)

# ----------------------------
# === Serial reader thread ===
# ----------------------------
def serial_reader():
    """
    Background thread that reads lines from serial port, parses up to MAX_CHANNELS
    floats (tab-separated), and puts tuples into data_queue.
    """
    global serial_connected, data_flowing, last_data_time
    
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)
        ser.reset_input_buffer()
        serial_connected = True
        logging.info(f"✓ Serial connected to {PORT} @ {BAUDRATE}")
        print(f"✓ Serial connected to {PORT} @ {BAUDRATE}")
        
        while True:
            raw = ser.readline().decode('ascii', errors='ignore').strip()
            if not raw:
                continue
            parts = raw.split('\t')
            vals = []
            for p in parts[:MAX_CHANNELS]:
                try:
                    vals.append(float(p))
                except Exception:
                    # ignore unparsable tokens
                    break
            if vals:
                try:
                    data_queue.put(tuple(vals), block=False)
                    data_flowing = True
                    last_data_time = time.time()
                except queue.Full:
                    # if queue is full we drop the sample
                    pass
    except Exception as e:
        serial_connected = False
        logging.error(f"❌ Serial thread error: {e}")
        print(f"❌ Serial connection failed: {e}")

# start serial thread as daemon
threading.Thread(target=serial_reader, daemon=True).start()

# ----------------------------
# === CSV writer thread ===
# ----------------------------
def csv_writer():
    """
    Background thread that writes queued log rows to disk once per second.
    """
    while True:
        batch = []
        while not log_queue.empty():
            batch.append(log_queue.get())
        if batch:
            try:
                with open(LOG_FILENAME, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(batch)
            except Exception as e:
                logging.error(f"CSV writer error: {e}")
        time.sleep(1.0)

threading.Thread(target=csv_writer, daemon=True).start()

# ----------------------------
# === Plot setup (single figure, 2 subplots) ===
# ----------------------------
plt.ion()  # interactive mode
# Configure matplotlib for better rendering of dense data
plt.rcParams['path.simplify'] = False
plt.rcParams['path.simplify_threshold'] = 0.0
plt.rcParams['agg.path.chunksize'] = 0
# Use fast rendering that still shows all points
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['lines.antialiased'] = True

fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle("Multi-channel Oscilloscope")

# Try using blitting for faster updates (may help with rendering)
ax_time.set_animated(False)

# time plot initial config
ax_time.set_xlabel('Time (s)')
ax_time.set_ylabel('Acceleration (g)')
ax_time.set_ylim(-y_range, y_range)
ax_time.set_xlim(0, WINDOW_SECONDS)

# fft plot initial config (hidden until toggled)
ax_fft.set_xlabel('Frequency (Hz)')
ax_fft.set_ylabel('Amplitude (g)')
ax_fft.set_visible(False)  # hidden by default
fig.tight_layout()

# lines & legend placeholders (populated on first data line)
time_lines = []    # list of Line2D objects (time-domain)
time_legend = None
fft_lines = {}     # map channel index -> Line2D for FFT plot

# overlay objects
fft_overlay = ax_fft.text(0.01, 0.98, "", transform=ax_fft.transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(facecolor='white', alpha=0.75))
help_overlay = ax_time.text(0.01, 0.98, "", transform=ax_time.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.75))

# initial help text (visible by default)
help_text = (
    "Controls:\n"
    " SPACE: Pause/Resume   ↑/↓: Zoom Y-axis   ←/→: Zoom time window\n"
    " L: Toggle logging (starts OFF)   T: Toggle trigger\n"
    " F: Toggle FFT display   W: Cycle FFT window   I/D: FFT size +/-\n"
    " Z/X: FFT freq zoom in/out   A: Toggle amplitude (Linear/dB)\n"
    " 1..9,0: Toggle CH1..CH10 for FFT   U: Toggle this help\n"
    " Click legend entries to show/hide channels. ESC: Quit"
)

# ----------------------------
# === Utility functions ===
# ----------------------------
def init_channels(n):
    """
    Initialize data buffers, time-plot lines and legend for 'n' channels.
    Called once upon first received sample or during startup.
    """
    global num_channels, channel_names, data_buffers, time_lines, time_legend, help_overlay
    
    num_channels = n
    channel_names[:] = [f"CH{i+1}" for i in range(num_channels)]
    data_buffers[:] = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]
    # clear and re-create time axes and lines
    ax_time.clear()
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Acceleration (g)')
    ax_time.set_ylim(-y_range, y_range)
    ax_time.set_xlim(0, WINDOW_SECONDS)
    time_lines.clear()
    for ch in range(num_channels):
        ln, = ax_time.plot([], [], label=channel_names[ch],
                           color=channel_colors[ch % len(channel_colors)],
                           linewidth=1.0)
        time_lines.append(ln)
    # clickable legend
    time_legend = ax_time.legend(loc='upper right', fancybox=True, shadow=True)
    for legline, origline in zip(time_legend.get_lines(), time_lines):
        legline.set_picker(True)  # enable picking on the legend entries
        legline._orig_line = origline
    
    # Recreate help overlay after clearing axes
    help_overlay = ax_time.text(0.01, 0.98, "", transform=ax_time.transAxes,
                                fontsize=9, verticalalignment='top',
                                bbox=dict(facecolor='white', alpha=0.75))
    update_help_overlay()
    
    fig.canvas.draw_idle()

def get_window(kind, N):
    """Return window array of length N for the chosen kind."""
    k = (kind or 'hann').lower()
    if k == 'hann':
        return np.hanning(N)
    if k == 'hamming':
        return np.hamming(N)
    if k == 'blackman':
        return np.blackman(N)
    return np.ones(N)

def clear_fft_plot():
    """Clear FFT axes and overlay, removing existing lines."""
    global fft_lines
    for ln in fft_lines.values():
        try:
            ln.remove()
        except Exception:
            pass
    fft_lines = {}
    ax_fft.cla()
    ax_fft.set_xlabel('Frequency (Hz)')
    # y-label depends on amplitude mode (set below)
    if amplitude_mode == 'linear':
        ax_fft.set_ylabel('Amplitude (g)')
    else:
        ax_fft.set_ylabel('Amplitude (dB re 1 g)')
    fft_overlay.set_text("")
    fig.canvas.draw_idle()

def update_help_overlay():
    """Show or hide the help overlay depending on show_help_overlay flag."""
    if show_help_overlay:
        help_overlay.set_text(help_text)
    else:
        help_overlay.set_text("Press U for help")
    fig.canvas.draw_idle()

def update_fft_overlay_text():
    """
    Update small overlay on FFT axis showing:
      - selected channels
      - window type
      - N samples
      - span
      - amplitude mode
      - trigger state
    """
    if not fft_enabled:
        fft_overlay.set_text("")
        return
    sel = ", ".join([f"CH{c+1}" for c in sorted(fft_selected)]) or "None"
    amp = "Linear" if amplitude_mode == 'linear' else "dB re 1 g"
    trig = "ON" if trigger_enabled else "OFF"
    txt = (f"FFT Channels: {sel}\nWindow: {fft_window_type}\n"
           f"N = {FFT_WINDOW_SAMPLES}\nSpan: 0–{fft_span:.0f} Hz\n"
           f"Amplitude: {amp}\nTrigger: {trig} (th={TRIGGER_THRESHOLD} g)")
    fft_overlay.set_text(txt)
    fig.canvas.draw_idle()

def append_sample(vals):
    """
    Append a new incoming sample (tuple of floats) to buffers.
    Initialize channels on first call.
    Also queue a CSV row if logging enabled.
    """
    global num_channels, first_sample_time
    if num_channels == 0:
        # determine channel count from first valid line
        init_channels(min(len(vals), MAX_CHANNELS))
    # append each channel value (pad with 0.0 if incoming had fewer columns)
    for ch in range(num_channels):
        v = vals[ch] if ch < len(vals) else 0.0
        data_buffers[ch].append(v)
    
    # append monotonic time based on SAMPLE_RATE (original working method)
    if not time_buffer:
        t = 0.0
    else:
        t = time_buffer[-1] + 1.0 / SAMPLE_RATE
    time_buffer.append(t)
    
    # Store actual arrival time for rate calculation
    arrival_times.append(datetime.now().timestamp())
    
    # enqueue CSV logging row (only timestamp + all channels) if logging is enabled
    if log_enabled:
        try:
            row = [f"{t:.6f}"] + [f"{(data_buffers[ch][-1] if data_buffers[ch] else 0.0):.6f}" for ch in range(num_channels)]
            log_queue.put(row)
        except Exception:
            pass

# ----------------------------
# === Trigger & FFT routines
# ----------------------------
def compute_and_plot_fft():
    """
    Compute normalized FFT for each selected channel and plot them.
    Normalization: one-sided amplitude = 2 * |rfft| / N
    dB conversion: 20*log10(A/1g + eps), where 1 g = 1.0 reference.
    """
    if not fft_selected:
        clear_fft_plot()
        update_fft_overlay_text()
        return
    N = FFT_WINDOW_SAMPLES
    # ensure enough samples for each selected channel
    if any(len(data_buffers[ch]) < N for ch in fft_selected):
        clear_fft_plot()
        update_fft_overlay_text()
        return
    # clear previous lines and draw fresh
    clear_fft_plot()
    win = get_window(fft_window_type, N)
    freqs = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)
    scale = 2.0 / N  # one-sided amplitude scale
    eps = 1e-12      # avoid log(0)
    for ch in sorted(fft_selected):
        if ch >= num_channels:
            continue
        arr = np.array(list(data_buffers[ch])[-N:]) * win
        fft_vals = np.abs(np.fft.rfft(arr)) * scale  # amplitude in g
        if amplitude_mode == 'dB':
            fft_plot_vals = 20.0 * np.log10(fft_vals / 1.0 + eps)  # dB re 1 g
            ax_fft.set_ylabel('Amplitude (dB re 1 g)')
        else:
            fft_plot_vals = fft_vals
            ax_fft.set_ylabel('Amplitude (g)')
        # plot and store handle
        ln, = ax_fft.plot(freqs, fft_plot_vals, label=f"CH{ch+1}", color=channel_colors[ch % len(channel_colors)])
        fft_lines[ch] = ln
    # set x-limits to current span (cap to Nyquist)
    ax_fft.set_xlim(0, min(fft_span, SAMPLE_RATE / 2))
    ax_fft.relim(); ax_fft.autoscale_view()
    ax_fft.legend(loc='upper right')
    update_fft_overlay_text()
    fig.canvas.draw_idle()

def check_triggers_for_latest_sample():
    """
    Check per-channel trigger for each FFT-selected channel using the latest sample.
    If triggered, print and append to trigger CSV file.
    """
    if not trigger_enabled or not fft_selected or not time_buffer:
        return
    t_now = time_buffer[-1]
    for ch in sorted(fft_selected):
        if ch < num_channels and data_buffers[ch]:
            val = data_buffers[ch][-1]
            if abs(val) > TRIGGER_THRESHOLD:
                # print to console
                print(f"🚨 Trigger CH{ch+1} @ {t_now:.3f}s: {val:.3f} g")
                # append to trigger csv
                try:
                    with open(TRIGGER_FILENAME, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{t_now:.6f}", f"CH{ch+1}", f"{val:.6f}"])
                except Exception as e:
                    logging.error(f"Error writing trigger log: {e}")

# ----------------------------
# === Interactive handlers ===
# ----------------------------
def on_pick(event):
    """Legend pick handler toggles visibility of the corresponding time-domain line."""
    legline = event.artist
    if not hasattr(legline, "_orig_line"):
        return
    orig = legline._orig_line
    visible = not orig.get_visible()
    orig.set_visible(visible)
    # fade legend line to indicate hidden state
    legline.set_alpha(1.0 if visible else 0.25)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('pick_event', on_pick)

def _resize_buffers(new_maxlen):
    """Resize all data deques to a new maxlen while keeping recent samples."""
    global BUFFER_SIZE, time_buffer, data_buffers, arrival_times
    BUFFER_SIZE = int(new_maxlen)
    # resize time buffer
    tvals = list(time_buffer)
    time_buffer.clear(); time_buffer.extend(tvals[-BUFFER_SIZE:])
    # resize data buffers
    for i in range(len(data_buffers)):
        vals = list(data_buffers[i])
        data_buffers[i] = deque(vals[-BUFFER_SIZE:], maxlen=BUFFER_SIZE)
    at = list(arrival_times)
    arrival_times.clear(); arrival_times.extend(at[-BUFFER_SIZE:])

def on_key(event):
    """
    Keyboard handler. Key mappings:
      SPACE: pause/resume
      arrow keys: zoom/time window
      L: toggle logging
      T: toggle trigger
      F: toggle FFT panel
      W: cycle FFT window type
      I/D: FFT window length +/- (1.25x)
      Z/X: FFT frequency zoom in/out
      1..9,0: toggle CH1..CH10 inclusion in FFT
      A: toggle amplitude linear/dB
      U: toggle help overlay
      ESC: quit
    """
    global running, y_range, WINDOW_SECONDS, BUFFER_SIZE
    global log_enabled, trigger_enabled, fft_enabled, fft_window_type, FFT_WINDOW_SAMPLES, fft_span
    global amplitude_mode, show_help_overlay

    key = event.key
    if not key:
        return
    k = key.lower()

    if k == ' ':
        running = not running
        print("▶️ Resumed" if running else "⏸ Paused")
    elif k == 'up':
        y_range = max(0.01, y_range * 0.8)
        ax_time.set_ylim(-y_range, y_range)
        print(f"Y-range ±{y_range:.3f}")
    elif k == 'down':
        y_range *= 1.25
        ax_time.set_ylim(-y_range, y_range)
        print(f"Y-range ±{y_range:.3f}")
    elif k == 'left':
        WINDOW_SECONDS *= 1.25
        BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
        _resize_buffers(BUFFER_SIZE)
        print(f"Time window {WINDOW_SECONDS:.2f}s (buffer={BUFFER_SIZE})")
    elif k == 'right':
        WINDOW_SECONDS = max(0.5, WINDOW_SECONDS * 0.8)
        BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
        _resize_buffers(BUFFER_SIZE)
        print(f"Time window {WINDOW_SECONDS:.2f}s (buffer={BUFFER_SIZE})")
    elif k == 'l':
        log_enabled = not log_enabled
        print("Logging", "enabled" if log_enabled else "disabled")
    elif k == 't':
        trigger_enabled = not trigger_enabled
        print("Trigger", "enabled" if trigger_enabled else "disabled")
        update_fft_overlay_text()
    elif k == 'f':
        fft_enabled = not fft_enabled
        ax_fft.set_visible(fft_enabled)
        if not fft_enabled:
            clear_fft_plot()
        else:
            compute_and_plot_fft()
        fig.tight_layout()
        update_fft_overlay_text()
        print("FFT", "enabled" if fft_enabled else "disabled")
    elif k == 'w':
        types = ['hann', 'hamming', 'blackman', 'rect']
        idx = types.index(fft_window_type) if fft_window_type in types else 0
        fft_window_type = types[(idx + 1) % len(types)]
        print("FFT window:", fft_window_type)
        if fft_enabled:
            compute_and_plot_fft()
        update_fft_overlay_text()
    elif k == 'i':
        FFT_WINDOW_SAMPLES = min(int(FFT_WINDOW_SAMPLES * 1.25), BUFFER_SIZE if BUFFER_SIZE > 0 else FFT_WINDOW_SAMPLES)
        print("FFT samples:", FFT_WINDOW_SAMPLES)
        if fft_enabled:
            compute_and_plot_fft()
        update_fft_overlay_text()
    elif k == 'd':
        FFT_WINDOW_SAMPLES = max(int(FFT_WINDOW_SAMPLES / 1.25), 100)
        print("FFT samples:", FFT_WINDOW_SAMPLES)
        if fft_enabled:
            compute_and_plot_fft()
        update_fft_overlay_text()
    elif k == 'z':
        fft_span = max(5.0, fft_span / 2.0)
        print(f"FFT span 0–{fft_span:.1f} Hz")
        if fft_enabled:
            ax_fft.set_xlim(0, fft_span)
            fig.canvas.draw_idle()
        update_fft_overlay_text()
    elif k == 'x':
        fft_span = min(fft_span * 2.0, SAMPLE_RATE / 2.0)
        print(f"FFT span 0–{fft_span:.1f} Hz")
        if fft_enabled:
            ax_fft.set_xlim(0, fft_span)
            fig.canvas.draw_idle()
        update_fft_overlay_text()
    elif k == 'a':
        amplitude_mode = 'dB' if amplitude_mode == 'linear' else 'linear'
        print("Amplitude mode:", amplitude_mode)
        if fft_enabled:
            compute_and_plot_fft()
        update_fft_overlay_text()
    elif k == 'u':
        show_help_overlay = not show_help_overlay
        update_help_overlay()
    elif k in [str(n) for n in range(10)]:
        # toggle inclusion for CH1..CH10 (0 maps to CH10)
        idx = 10 if k == '0' else int(k)
        ch_index = idx - 1
        if ch_index < 0 or ch_index >= min(10, MAX_CHANNELS):
            print(f"Channel {idx} out of FFT selection range (allowed CH1..CH10)")
            return
        if ch_index in fft_selected:
            fft_selected.remove(ch_index)
            print(f"FFT: removed CH{ch_index+1}")
        else:
            fft_selected.add(ch_index)
            print(f"FFT: added CH{ch_index+1}")
        # per request: clear FFT when selection changes
        clear_fft_plot()
        if fft_enabled:
            compute_and_plot_fft()
        update_fft_overlay_text()
    elif k == 'escape':
        print("Exiting...")
        plt.close(fig)
    else:
        # ignore other keys
        pass

fig.canvas.mpl_connect('key_press_event', on_key)

# ----------------------------
# === Initialize UI ===
# ----------------------------
# Create initial empty legend so it's visible from the start
time_legend = ax_time.legend([], loc='upper right', fancybox=True, shadow=True)

# Set help overlay text directly
update_help_overlay()

# ----------------------------
# === Main update loop ===
# ----------------------------
try:
    while plt.get_fignums():
        if running:
            # pull all pending samples off the queue
            new_samples = 0
            while not data_queue.empty():
                vals = data_queue.get()
                append_sample(vals)
                new_samples += 1

            if new_samples == 0:
                # yield to GUI event loop
                plt.pause(0.001)
                continue

            # Update time-domain lines if channels are present
            if num_channels > 0:
                t_arr = np.array(time_buffer)
                # update each channel's line with decimation for display
                for ch in range(num_channels):
                    if len(data_buffers[ch]) > 0:
                        y_arr = np.array(data_buffers[ch])
                        # Use the full time array, not a subset
                        if len(t_arr) == len(y_arr):
                            # Apply decimation for smoother rendering
                            # Take every Nth point for display (but keep all data in buffers)
                            if DISPLAY_DECIMATION > 1 and len(t_arr) > DISPLAY_DECIMATION:
                                display_indices = slice(0, len(t_arr), DISPLAY_DECIMATION)
                                time_lines[ch].set_data(t_arr[display_indices], y_arr[display_indices])
                            else:
                                time_lines[ch].set_data(t_arr, y_arr)
                        else:
                            # If arrays don't match, align them properly
                            min_len = min(len(t_arr), len(y_arr))
                            if DISPLAY_DECIMATION > 1 and min_len > DISPLAY_DECIMATION:
                                display_indices = slice(0, min_len, DISPLAY_DECIMATION)
                                time_lines[ch].set_data(t_arr[-min_len:][display_indices], 
                                                       y_arr[-min_len:][display_indices])
                            else:
                                time_lines[ch].set_data(t_arr[-min_len:], y_arr[-min_len:])
                # update sliding x-limits
                if len(time_buffer):
                    last_t = time_buffer[-1]
                    ax_time.set_xlim(max(0.0, last_t - WINDOW_SECONDS), last_t)
                    ax_time.set_ylim(-y_range, y_range)
                # adjust legend alpha to indicate hidden lines
                if ax_time.get_legend():
                    for legline, origline in zip(ax_time.get_legend().get_lines(), time_lines):
                        legline.set_alpha(1.0 if origline.get_visible() else 0.25)

            # compute measured sampling rate (from arrival_times) and update time plot title
            if len(arrival_times) > 1:
                intervals = np.diff(np.array(arrival_times))
                avg_dt = np.mean(intervals)
                measured_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
                
                # Add connection status and buffer info to title
                status = ""
                if serial_connected:
                    if data_flowing:
                        decim_info = f" | Decim: 1/{DISPLAY_DECIMATION}" if DISPLAY_DECIMATION > 1 else ""
                        status = f" | ✓ Data flowing | Buf: {len(time_buffer)}/{BUFFER_SIZE}{decim_info}"
                    else:
                        status = " | ⚠ Connected, waiting for data"
                else:
                    status = " | ❌ Not connected"
                
                ax_time.set_title(f"Oscilloscope — Δt={avg_dt*1000:.1f} ms ({measured_hz:.1f} Hz){status}")
            else:
                status = ""
                if serial_connected:
                    status = " | ⚠ Connected, waiting for data..."
                else:
                    status = " | ❌ Not connected"
                ax_time.set_title(f"Oscilloscope — {PORT}{status}")

            # check triggers on latest sample (per-channel)
            check_triggers_for_latest_sample()

            # if FFT is enabled, compute and plot it
            if fft_enabled:
                compute_and_plot_fft()

            # draw and pause briefly (keeps GUI responsive)
            fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            # paused: still keep GUI responsive
            plt.pause(0.05)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    print("Exiting cleanly.")