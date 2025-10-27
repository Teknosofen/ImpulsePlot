import serial, threading, queue, csv, os, logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import time

# === CONFIG ===
PORT = 'COM4'          # your serial port
BAUDRATE = 115200
SAMPLE_RATE = 500       # Hz (update from 100 to 500)
WINDOW_SECONDS = 4
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS
FFT_WINDOW_SAMPLES = 1000
PLOT_EVERY = 1
TRIGGER_THRESHOLD = 2.5
y_range = 2.0

# === STATE ===
running = True
log_enabled = True
trigger_active = True
fft_enabled = False
start_time = datetime.now()
fft_window_type = 'hann'   # default

# === FILES ===
LOG_FILENAME = "accel_log.csv"
TRIGGER_FILENAME = "trigger_log.csv"

# === BUFFERS ===
time_buffer = deque(maxlen=BUFFER_SIZE)
x_buffer = deque(maxlen=BUFFER_SIZE)
y_buffer = deque(maxlen=BUFFER_SIZE)
z_buffer = deque(maxlen=BUFFER_SIZE)
arrival_times = deque(maxlen=BUFFER_SIZE)

# === QUEUES ===
data_queue = queue.Queue()
log_queue = queue.Queue()

# === SERIAL THREAD ===
def serial_reader():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)
        ser.reset_input_buffer()
        logging.info(f"Serial connected to {PORT}")
        while True:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 3:
                try:
                    data_queue.put(tuple(map(float, parts)), block=False)
                except queue.Full:
                    pass
    except Exception as e:
        logging.error(f"Serial thread error: {e}")

threading.Thread(target=serial_reader, daemon=True).start()

# === CSV THREAD ===
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

# === PLOT SETUP (combined) ===
plt.ion()
fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
line_x, = ax_time.plot([], [], label='X')
line_y, = ax_time.plot([], [], label='Y')
line_z, = ax_time.plot([], [], label='Z')
ax_time.legend(loc='upper right')
ax_time.set_xlabel('Time (s)')
ax_time.set_ylabel('Acceleration (g)')
ax_time.set_ylim(-y_range, y_range)
ax_time.set_xlim(0, WINDOW_SECONDS)

fft_line_x, = ax_fft.plot([], [], label='X (FFT)')
fft_line_y, = ax_fft.plot([], [], label='Y (FFT)')
fft_line_z, = ax_fft.plot([], [], label='Z (FFT)')
ax_fft.legend(loc='upper right')
ax_fft.set_xlabel('Frequency (Hz)')
ax_fft.set_ylabel('Amplitude')
ax_fft.set_visible(False)
fig.tight_layout()
fig.canvas.draw_idle()

help_text = (
    "Controls:\n"
    "  SPACE → Run/Pause\n"
    "  ↑ / ↓ → Zoom Y-axis\n"
    "  ← / → → Zoom time axis\n"
    "  L     → Toggle logging\n"
    "  T     → Toggle trigger\n"
    "  F     → Toggle FFT display\n"
    "  W     → Cycle FFT window type\n"
    "  I / D → Increase / Decrease FFT window length\n"
    "  ESC   → Quit"
)
ax_time.text(0.01, 0.95, help_text, transform=ax_time.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))

# === TRIGGER ===
def trigger_action(t, x, y, z):
    magnitude = (x**2 + y**2 + z**2)**0.5
    if magnitude > TRIGGER_THRESHOLD:
        print(f"🚨 {t:.2f}s |a|={magnitude:.2f}")
        if trigger_active:
            with open(TRIGGER_FILENAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{magnitude:.3f}"])

# === FFT WINDOW FUNCTION ===
def get_fft_window(n, kind='hann'):
    if kind == 'hann':
        return np.hanning(n)
    elif kind == 'hamming':
        return np.hamming(n)
    elif kind == 'blackman':
        return np.blackman(n)
    else:  # rectangular
        return np.ones(n)

# === KEYBOARD HANDLER ===
def on_key(event):
    global running, y_range, WINDOW_SECONDS, BUFFER_SIZE
    global time_buffer, x_buffer, y_buffer, z_buffer, arrival_times
    global log_enabled, trigger_active, fft_enabled, start_time
    global fft_window_type, FFT_WINDOW_SAMPLES

    if event.key == ' ':
        running = not running
        if running:
            print("▶️  Resumed")
            time_buffer.clear()
            x_buffer.clear()
            y_buffer.clear()
            z_buffer.clear()
            arrival_times.clear()
            start_time = datetime.now()
        else:
            print("⏸️  Paused")

    elif event.key == 'up':
        y_range = max(0.1, y_range * 0.8)
        print(f"🔍 Zoomed in Y: ±{y_range:.2f}")

    elif event.key == 'down':
        y_range *= 1.25
        print(f"🔎 Zoomed out Y: ±{y_range:.2f}")

    elif event.key == 'left':
        WINDOW_SECONDS *= 1.25
        BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
        time_buffer = deque(time_buffer, maxlen=BUFFER_SIZE)
        x_buffer = deque(x_buffer, maxlen=BUFFER_SIZE)
        y_buffer = deque(y_buffer, maxlen=BUFFER_SIZE)
        z_buffer = deque(z_buffer, maxlen=BUFFER_SIZE)
        arrival_times = deque(arrival_times, maxlen=BUFFER_SIZE)
        print(f"⏪ Zoomed out time: {WINDOW_SECONDS:.2f}s")

    elif event.key == 'right':
        WINDOW_SECONDS = max(0.5, WINDOW_SECONDS * 0.8)
        BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
        time_buffer = deque(time_buffer, maxlen=BUFFER_SIZE)
        x_buffer = deque(x_buffer, maxlen=BUFFER_SIZE)
        y_buffer = deque(y_buffer, maxlen=BUFFER_SIZE)
        z_buffer = deque(z_buffer, maxlen=BUFFER_SIZE)
        arrival_times = deque(arrival_times, maxlen=BUFFER_SIZE)
        print(f"⏩ Zoomed in time: {WINDOW_SECONDS:.2f}s")

    elif event.key == 'l':
        log_enabled = not log_enabled
        print(f"📝 Logging {'enabled' if log_enabled else 'disabled'}")

    elif event.key == 't':
        trigger_active = not trigger_active
        print(f"🎯 Trigger {'enabled' if trigger_active else 'disabled'}")

    elif event.key == 'f':
        fft_enabled = not fft_enabled
        ax_fft.set_visible(fft_enabled)
        fig.tight_layout()
        fig.canvas.draw_idle()
        print(f"📊 FFT display {'enabled' if fft_enabled else 'disabled'}")

    elif event.key == 'w':
        windows = ['hann', 'hamming', 'blackman', 'rect']
        idx = windows.index(fft_window_type)
        fft_window_type = windows[(idx + 1) % len(windows)]
        print(f"🎚️ FFT window set to: {fft_window_type}")

    elif event.key == 'i':
        FFT_WINDOW_SAMPLES = min(BUFFER_SIZE, int(FFT_WINDOW_SAMPLES * 1.25))
        print(f"📈 FFT window length increased: {FFT_WINDOW_SAMPLES} samples")

    elif event.key == 'd':
        FFT_WINDOW_SAMPLES = max(100, int(FFT_WINDOW_SAMPLES * 0.8))
        print(f"📉 FFT window length decreased: {FFT_WINDOW_SAMPLES} samples")

    elif event.key == 'escape':
        print("🛑 Exiting...")
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)

# === MAIN LOOP ===
try:
    while plt.get_fignums():
        if running:
            try:
                new_samples = 0
                while not data_queue.empty():
                    x_val, y_val, z_val = data_queue.get()
                    if len(time_buffer) == 0:
                        t = 0.0
                    else:
                        t = time_buffer[-1] + 1.0 / SAMPLE_RATE

                    arrival = datetime.now().timestamp()
                    time_buffer.append(t)
                    x_buffer.append(x_val)
                    y_buffer.append(y_val)
                    z_buffer.append(z_val)
                    arrival_times.append(arrival)
                    new_samples += 1

                    if log_enabled:
                        log_queue.put([f"{t:.3f}", f"{x_val:.3f}", f"{y_val:.3f}", f"{z_val:.3f}"])

                    if trigger_active:
                        trigger_action(t, x_val, y_val, z_val)

                if new_samples == 0:
                    plt.pause(0.001)
                    continue

                # Time plot
                t_arr = np.array(time_buffer)
                x_arr = np.array(x_buffer)
                y_arr = np.array(y_buffer)
                z_arr = np.array(z_buffer)
                line_x.set_data(t_arr, x_arr)
                line_y.set_data(t_arr, y_arr)
                line_z.set_data(t_arr, z_arr)
                ax_time.set_xlim(max(t_arr[-1] - WINDOW_SECONDS, 0), t_arr[-1])
                ax_time.set_ylim(-y_range, y_range)

                # Timing diagnostics
                intervals = np.diff(np.array(arrival_times))
                if len(intervals) > 0:
                    avg_dt = np.mean(intervals)
                    fps = 1 / avg_dt if avg_dt > 0 else 0
                    ax_time.set_title(f"Oscilloscope — Δt={avg_dt*1000:.1f} ms ({fps:.1f} Hz)")

                # === FFT update ===
                if fft_enabled and len(x_buffer) >= FFT_WINDOW_SAMPLES:
                    x_win = np.array(list(x_buffer)[-FFT_WINDOW_SAMPLES:])
                    y_win = np.array(list(y_buffer)[-FFT_WINDOW_SAMPLES:])
                    z_win = np.array(list(z_buffer)[-FFT_WINDOW_SAMPLES:])

                    window = get_fft_window(len(x_win), fft_window_type)
                    freqs = np.fft.rfftfreq(len(x_win), d=1/SAMPLE_RATE)
                    fft_x = np.abs(np.fft.rfft(x_win * window))
                    fft_y = np.abs(np.fft.rfft(y_win * window))
                    fft_z = np.abs(np.fft.rfft(z_win * window))

                    fft_line_x.set_data(freqs, fft_x)
                    fft_line_y.set_data(freqs, fft_y)
                    fft_line_z.set_data(freqs, fft_z)
                    ax_fft.set_xlim(0, SAMPLE_RATE / 2)
                    ax_fft.relim()
                    ax_fft.autoscale_view()

                    # Display current FFT info
                    ax_fft.set_title(
                        f"FFT ({FFT_WINDOW_SAMPLES} samples, window={fft_window_type})"
                    )

                plt.pause(0.001)

            except Exception as e:
                logging.warning(f"⚠️ Runtime error: {e}")
                plt.pause(0.05)
        else:
            plt.pause(0.05)

except KeyboardInterrupt:
    print("Interrupted.")
finally:
    print("🛑 Exiting.")
