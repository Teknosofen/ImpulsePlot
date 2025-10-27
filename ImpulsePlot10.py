import serial, threading, queue, csv, os, logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import time

# === CONFIG ===
PORT = 'COM4'            # your COM port
BAUDRATE = 115200
SAMPLE_RATE = 500        # Hz
WINDOW_SECONDS = 4
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS
PLOT_EVERY = 1
FFT_WINDOW_SAMPLES = 1000
TRIGGER_THRESHOLD = 2.5
y_range = 2.0

# === STATE ===
running = True
log_enabled = True
trigger_active = True
fft_enabled = False
fft_window_type = "hann"
fft_span = SAMPLE_RATE / 2
start_time = datetime.now()

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

# === PLOT SETUP ===
plt.ion()
fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(10, 8),
                                      gridspec_kw={'height_ratios': [2, 1]})

# Time-domain lines
line_x, = ax_time.plot([], [], label='X')
line_y, = ax_time.plot([], [], label='Y')
line_z, = ax_time.plot([], [], label='Z')
ax_time.legend(loc='upper right')
ax_time.set_xlabel('Time (s)')
ax_time.set_ylabel('Acceleration (g)')
ax_time.set_ylim(-y_range, y_range)
ax_time.set_xlim(0, WINDOW_SECONDS)

# FFT lines
fft_line_x, = ax_fft.plot([], [], label='X (FFT)')
fft_line_y, = ax_fft.plot([], [], label='Y (FFT)')
fft_line_z, = ax_fft.plot([], [], label='Z (FFT)')
ax_fft.set_xlabel('Frequency (Hz)')
ax_fft.set_ylabel('Amplitude (g)')
ax_fft.legend(loc='upper right')
ax_fft.set_visible(False)
fig.tight_layout()

help_text = (
    "Controls:\n"
    "  SPACE → Run/Pause\n"
    "  ↑/↓ → Zoom Y-axis\n"
    "  ←/→ → Zoom time axis\n"
    "  L → Toggle logging\n"
    "  T → Toggle trigger\n"
    "  F → Toggle FFT display\n"
    "  W → Cycle FFT window type\n"
    "  I/D → Increase/Decrease FFT window samples\n"
    "  Z/X → Zoom in/out FFT freq span\n"
    "  ESC → Quit"
)
ax_time.text(0.01, 0.95, help_text, transform=ax_time.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))

# === UTILS ===
def get_window(win_type, N):
    win_type = win_type.lower()
    if win_type == "hann":
        return np.hanning(N)
    elif win_type == "hamming":
        return np.hamming(N)
    elif win_type == "blackman":
        return np.blackman(N)
    elif win_type == "rect":
        return np.ones(N)
    return np.hanning(N)

def trigger_action(t, x, y, z):
    magnitude = (x**2 + y**2 + z**2)**0.5
    if magnitude > TRIGGER_THRESHOLD:
        print(f"🚨 {t:.2f}s |a|={magnitude:.2f}")
        if trigger_active:
            with open(TRIGGER_FILENAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{magnitude:.3f}"])

# === KEYBOARD HANDLER ===
def on_key(event):
    global running, y_range, WINDOW_SECONDS, BUFFER_SIZE
    global time_buffer, x_buffer, y_buffer, z_buffer, arrival_times
    global log_enabled, trigger_active, fft_enabled, fft_window_type
    global FFT_WINDOW_SAMPLES, fft_span, start_time

    if event.key == ' ':
        running = not running
        if running:
            print("▶️ Resumed")
            time_buffer.clear(); x_buffer.clear(); y_buffer.clear(); z_buffer.clear()
            arrival_times.clear(); start_time = datetime.now()
        else:
            print("⏸️ Paused")

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
        print(f"⏪ Time window: {WINDOW_SECONDS:.2f}s")

    elif event.key == 'right':
        WINDOW_SECONDS = max(0.5, WINDOW_SECONDS * 0.8)
        BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
        time_buffer = deque(time_buffer, maxlen=BUFFER_SIZE)
        x_buffer = deque(x_buffer, maxlen=BUFFER_SIZE)
        y_buffer = deque(y_buffer, maxlen=BUFFER_SIZE)
        z_buffer = deque(z_buffer, maxlen=BUFFER_SIZE)
        arrival_times = deque(arrival_times, maxlen=BUFFER_SIZE)
        print(f"⏩ Time window: {WINDOW_SECONDS:.2f}s")

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
        print(f"📊 FFT {'enabled' if fft_enabled else 'disabled'}")

    elif event.key == 'w':
        types = ["hann", "hamming", "blackman", "rect"]
        idx = (types.index(fft_window_type) + 1) % len(types)
        fft_window_type = types[idx]
        print(f"🔁 FFT window → {fft_window_type}")

    elif event.key == 'i':
        FFT_WINDOW_SAMPLES = min(int(FFT_WINDOW_SAMPLES * 1.25), BUFFER_SIZE)
        print(f"➕ FFT samples = {FFT_WINDOW_SAMPLES}")

    elif event.key == 'd':
        FFT_WINDOW_SAMPLES = max(int(FFT_WINDOW_SAMPLES / 1.25), 100)
        print(f"➖ FFT samples = {FFT_WINDOW_SAMPLES}")

    elif event.key == 'z':
        fft_span = max(fft_span / 2, 5)
        print(f"🔍 FFT zoom in — span 0–{fft_span:.1f} Hz")

    elif event.key == 'x':
        fft_span = min(fft_span * 2, SAMPLE_RATE / 2)
        print(f"🔎 FFT zoom out — span 0–{fft_span:.1f} Hz")

    elif event.key == 'escape':
        print("🛑 Exiting...")
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)

# === MAIN LOOP ===
try:
    while plt.get_fignums():
        if running:
            new_samples = 0
            while not data_queue.empty():
                x, y, z = data_queue.get()
                t = 0 if not time_buffer else time_buffer[-1] + 1.0 / SAMPLE_RATE
                time_buffer.append(t)
                x_buffer.append(x); y_buffer.append(y); z_buffer.append(z)
                arrival_times.append(datetime.now().timestamp())
                new_samples += 1

                if log_enabled:
                    log_queue.put([f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
                if trigger_active:
                    trigger_action(t, x, y, z)

            if new_samples == 0:
                plt.pause(0.001)
                continue

            # Update time plot
            t = np.array(time_buffer)
            x = np.array(x_buffer)
            y = np.array(y_buffer)
            z = np.array(z_buffer)
            line_x.set_data(t, x)
            line_y.set_data(t, y)
            line_z.set_data(t, z)
            ax_time.set_xlim(max(t[-1] - WINDOW_SECONDS, 0), t[-1])
            ax_time.set_ylim(-y_range, y_range)

            # FFT update
            if fft_enabled and len(x_buffer) >= FFT_WINDOW_SAMPLES:
                N = FFT_WINDOW_SAMPLES
                win = get_window(fft_window_type, N)
                x_win = np.array(list(x_buffer)[-N:]) * win
                y_win = np.array(list(y_buffer)[-N:]) * win
                z_win = np.array(list(z_buffer)[-N:]) * win

                freqs = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)
                scale = 2.0 / N
                fft_x = np.abs(np.fft.rfft(x_win)) * scale
                fft_y = np.abs(np.fft.rfft(y_win)) * scale
                fft_z = np.abs(np.fft.rfft(z_win)) * scale

                fft_line_x.set_data(freqs, fft_x)
                fft_line_y.set_data(freqs, fft_y)
                fft_line_z.set_data(freqs, fft_z)
                ax_fft.set_xlim(0, fft_span)
                ax_fft.relim()
                ax_fft.autoscale_view()
                ax_fft.set_title(f"FFT — {fft_window_type} | {N} samples @ {SAMPLE_RATE} Hz | span: 0–{fft_span:.0f} Hz")

            plt.pause(0.001)
        else:
            plt.pause(0.05)
except KeyboardInterrupt:
    print("Interrupted.")
finally:
    print("🛑 Exiting.")
