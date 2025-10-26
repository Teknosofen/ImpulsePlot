import serial
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from datetime import datetime
import logging
import csv
import os

# === LOGGING SETUP ===
logging.basicConfig(level=logging.DEBUG)

# === CONFIGURATION ===
PORT = 'COM7'
BAUDRATE = 115200
SAMPLE_RATE = 100  # Hz
WINDOW_SECONDS = 4
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS
y_range = 2.0


plot_counter = 0
PLOT_EVERY = 5  # update every 5 samples

# === STATE ===
running = True
log_enabled = True
trigger_active = True
start_time = datetime.now()

# === TRIGGER CONFIG ===
TRIGGER_THRESHOLD = 2.5  # g

# === CSV LOGGING ===
LOG_FILENAME = "accel_log.csv"
TRIGGER_FILENAME = "trigger_log.csv"

if log_enabled and not os.path.exists(LOG_FILENAME):
    with open(LOG_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "X", "Y", "Z"])

if trigger_active and not os.path.exists(TRIGGER_FILENAME):
    with open(TRIGGER_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "X", "Y", "Z", "Magnitude"])

# === BUFFERS ===
time_buffer = deque(maxlen=BUFFER_SIZE)
x_buffer = deque(maxlen=BUFFER_SIZE)
y_buffer = deque(maxlen=BUFFER_SIZE)
z_buffer = deque(maxlen=BUFFER_SIZE)
arrival_times = deque(maxlen=BUFFER_SIZE)

# === SERIAL SETUP ===
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)
    ser.reset_input_buffer()  # ✅ Clear buffer immediately
    logging.debug(f"Opened and flushed serial port {PORT}")
except Exception as e:
    print(f"❌ Could not open serial port {PORT}: {e}")
    exit(1)

# === PLOT SETUP ===
plt.ion()
fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(10, 8))
line_x, = ax_time.plot([], [], label='X')
line_y, = ax_time.plot([], [], label='Y')
line_z, = ax_time.plot([], [], label='Z')
ax_time.set_xlabel('Time (s)')
ax_time.set_ylabel('Acceleration')
ax_time.legend()

ax_fft.set_title('FFT of Acceleration')
ax_fft.set_xlabel('Frequency (Hz)')
ax_fft.set_ylabel('Amplitude')

# === HELP TEXT ===
help_text = (
    "Controls:\n"
    "  SPACE → Run/Pause\n"
    "  ↑ / ↓ → Zoom Y-axis\n"
    "  ← / → → Zoom time axis\n"
    "  L     → Toggle logging\n"
    "  T     → Toggle trigger\n"
    "  ESC   → Quit"
)
ax_time.text(0.01, 0.95, help_text, transform=ax_time.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))

# === TRIGGER ACTION ===
def trigger_action(t, x, y, z):
    magnitude = (x**2 + y**2 + z**2)**0.5
    print(f"🚨 Triggered at {t:.2f}s: x={x:.2f}, y={y:.2f}, z={z:.2f}, |a|={magnitude:.2f}")
    if trigger_active:
        with open(TRIGGER_FILENAME, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{magnitude:.3f}"])

# === EVENT HANDLER ===
def on_key(event):
    global running, y_range, WINDOW_SECONDS, BUFFER_SIZE, start_time
    global log_enabled, trigger_active
    if event.key == ' ':
        running = not running
        if running:
            print("▶️  Resumed")
            ser.reset_input_buffer()
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
    elif event.key == 'escape':
        print("🛑 Exiting...")
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)

# === PLOT UPDATE FUNCTION ===
def update_plot():
    if len(time_buffer) < 2:
        return

    t = np.array(time_buffer)
    x = np.array(x_buffer)
    y = np.array(y_buffer)
    z = np.array(z_buffer)

    line_x.set_data(t, x)
    line_y.set_data(t, y)
    line_z.set_data(t, z)

    if t[-1] > 0:
        ax_time.set_xlim(max(t[-1] - WINDOW_SECONDS, 0), t[-1])
    else:
        ax_time.set_xlim(0, WINDOW_SECONDS)

    ax_time.set_ylim(-y_range, y_range)
    ax_time.relim()
    ax_time.autoscale_view(True, True, False)

    # === Interval diagnostics ===
    intervals = np.diff(np.array(arrival_times))
    if len(intervals) > 0:
        avg_interval = np.mean(intervals)
        fps = 1 / avg_interval if avg_interval > 0 else 0
        ax_time.set_title(f'Oscilloscope View — Δt={avg_interval*1000:.1f} ms ({fps:.1f} Hz)')

    # === FFT plot ===
    if len(x) >= 32:
        freqs = np.fft.rfftfreq(len(x), d=1/SAMPLE_RATE)
        fft_x = np.abs(np.fft.rfft(x))
        fft_y = np.abs(np.fft.rfft(y))
        fft_z = np.abs(np.fft.rfft(z))

        ax_fft.clear()
        ax_fft.plot(freqs, fft_x, label='X')
        ax_fft.plot(freqs, fft_y, label='Y')
        ax_fft.plot(freqs, fft_z, label='Z')
        ax_fft.set_title('FFT of Acceleration')
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('Amplitude')
        ax_fft.legend()

    plt.pause(0.001)

# === MAIN LOOP ===
try:
    #plt.pause(0.001)
    while plt.get_fignums():
        if running:
            try:
                line = ser.readline().decode('ascii', errors='ignore').strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 3:
                        x, y, z = map(float, parts)
                        t = (datetime.now() - start_time).total_seconds()
                        arrival = datetime.now().timestamp()

                        # Append to buffers
                        time_buffer.append(t)
                        x_buffer.append(x)
                        y_buffer.append(y)
                        z_buffer.append(z)
                        arrival_times.append(arrival)

                        # CSV logging
                        if log_enabled:
                            with open(LOG_FILENAME, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])

                        # Trigger detection
                        if trigger_active and (
                            abs(x) > TRIGGER_THRESHOLD or
                            abs(y) > TRIGGER_THRESHOLD or
                            abs(z) > TRIGGER_THRESHOLD
                        ):
                            trigger_action(t, x, y, z)

                        # Update plots

                        plot_counter += 1
                        if plot_counter >= PLOT_EVERY:
                            update_plot()
                            plot_counter = 0

            except Exception as e:
                logging.warning(f"⚠️ Runtime error: {e}")
                plt.pause(0.1)
        else:
            plt.pause(0.1)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    ser.close()