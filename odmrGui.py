#!/usr/bin/env python3
"""
odmr_light_gui.py

Tkinter GUI for Pico-based ODMR and TSL2591 light-stream measurements.

This is a GUI version of the two-mode serial script:

1) ODMR mode
   - Sends Pico timing/config commands
   - Sends MEAS <freq_hz>
   - Expects RESULT,freq_hz,off1,on,off2,off_ref,delta,contrast,drift
   - Saves raw CSV
   - Plots mean delta vs frequency with SEM error bars

2) Light mode
   - Sends Pico timing/config commands
   - Optionally sends MODE STREAM / STREAM OFF
   - Expects DATA,<ms_since_boot>,<ch0>,<ch1>,<lux>
   - Live plots CH0, CH1, and lux vs time
   - Optionally logs CSV

Requires:
    pip install pyserial numpy matplotlib

Run:
    python odmr_light_gui.py
"""

from __future__ import annotations

import csv
import queue
import re
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Deque, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import serial
import serial.tools.list_ports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# -----------------------------------------------------------------------------
# Serial line formats expected from Pico
# -----------------------------------------------------------------------------

RE_RESULT = re.compile(
    r"^\s*RESULT\s*,\s*"
    r"(\d+)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)

RE_DATA = re.compile(
    r"^\s*DATA\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([+-]?\d+(?:\.\d+)?))?\s*$",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class PicoTimingConfig:
    it_ms: int = 400
    extra_wait_ms: int = 20
    freq_settle_ms: int = 100
    rf_on_settle_ms: int = 100
    rf_off_settle_ms: int = 100
    discard_samples: int = 1
    avg_samples_per_state: int = 4


@dataclass
class SerialConfig:
    port: str = "COM5"
    baud: int = 115200
    timeout_s: float = 0.2
    assert_dtr: bool = True
    assert_rts: bool = False


@dataclass
class ODMRConfig:
    f_start_hz: float = 2.81e9
    f_stop_hz: float = 2.93e9
    f_step_hz: float = 0.0001e9
    repeats_per_freq: int = 4
    meas_timeout_s: float = 25.0
    csv_path: str = "noField.csv"
    print_each_measurement: bool = True
    print_freq_summary: bool = True


@dataclass
class LightConfig:
    rolling_seconds: float = 60.0
    assumed_sample_hz: float = 10.0
    max_plot_hz: float = 20.0
    csv_log: bool = True
    csv_path: str = "tsl2591_log.csv"
    send_stream_commands: bool = True
    print_nondata_lines: bool = True
    print_raw_data_lines: bool = False
    heartbeat_period_s: float = 1.0


@dataclass
class ODMRPoint:
    freq_hz: float
    repeat_idx: int
    off1: float
    on: float
    off2: float
    off_ref: float
    delta: float
    contrast: float
    drift: float


@dataclass
class LightSample:
    t_pico_s: float
    ch0: int
    ch1: int
    lux: Optional[float]
    raw: str


# -----------------------------------------------------------------------------
# Parsing and utility functions
# -----------------------------------------------------------------------------

def build_frequency_list(start_hz: float, stop_hz: float, step_hz: float) -> np.ndarray:
    """Build an inclusive frequency list using the same convention as the CLI script."""
    if step_hz <= 0:
        raise ValueError("Frequency step must be positive.")
    if stop_hz < start_hz:
        raise ValueError("Stop frequency must be greater than or equal to start frequency.")

    n = int(round((stop_hz - start_hz) / step_hz)) + 1
    return start_hz + step_hz * np.arange(n, dtype=np.float64)


def parse_result_line(line: str) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
    """Parse RESULT,freq_hz,off1,on,off2,off_ref,delta,contrast,drift."""
    match = RE_RESULT.match(line)
    if not match:
        return None

    freq_hz = float(match.group(1))
    off1 = float(match.group(2))
    on = float(match.group(3))
    off2 = float(match.group(4))
    off_ref = float(match.group(5))
    delta = float(match.group(6))
    contrast = float(match.group(7))
    drift = float(match.group(8))

    return freq_hz, off1, on, off2, off_ref, delta, contrast, drift


def parse_data_line(line: str) -> Optional[Tuple[float, int, int, Optional[float]]]:
    """Parse DATA,<ms_since_boot>,<ch0>,<ch1>,<lux>. Lux may be absent."""
    match = RE_DATA.match(line)
    if not match:
        return None

    ms = int(match.group(1))
    ch0 = int(match.group(2))
    ch1 = int(match.group(3))
    lux = float(match.group(4)) if match.group(4) is not None else None
    return ms / 1000.0, ch0, ch1, lux


def available_serial_ports() -> List[Tuple[str, str]]:
    """Return available serial ports as (device, description)."""
    return [(p.device, p.description) for p in serial.tools.list_ports.comports()]


def safe_float(value: str, name: str) -> float:
    """Convert a string to float with a useful error message."""
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc


def safe_int(value: str, name: str) -> int:
    """Convert a string to int with a useful error message."""
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


# -----------------------------------------------------------------------------
# Serial manager
# -----------------------------------------------------------------------------

class SerialManager:
    """Small thread-safe wrapper around pyserial."""

    def __init__(self) -> None:
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @property
    def port_name(self) -> str:
        if self._ser is None:
            return ""
        return str(self._ser.port)

    def open(self, config: SerialConfig) -> None:
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                raise RuntimeError("Serial port is already open.")

            ser = serial.Serial(
                config.port,
                config.baud,
                timeout=config.timeout_s,
                write_timeout=config.timeout_s,
                rtscts=False,
                dsrdtr=False,
                xonxoff=False,
            )

            try:
                ser.setDTR(config.assert_dtr)
                ser.setRTS(config.assert_rts)
            except Exception:
                pass

            # Give Pico time to reset/enumerate if opening the port resets it.
            time.sleep(2.0)

            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass

            self._ser = ser

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None

    def send_line(self, line: str) -> None:
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise RuntimeError("Serial port is not open.")
            self._ser.write((line.rstrip() + "\n").encode("utf-8"))
            self._ser.flush()

    def readline(self) -> bytes:
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise RuntimeError("Serial port is not open.")
            return self._ser.readline()

    def drain(self, seconds: float, stop_event: Optional[threading.Event] = None) -> List[str]:
        lines: List[str] = []
        t0 = time.time()
        while (time.time() - t0) < seconds:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                raw = self.readline()
            except RuntimeError:
                break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
        return lines


# -----------------------------------------------------------------------------
# Worker functions run in background threads
# -----------------------------------------------------------------------------

def configure_pico_timings(
    serial_manager: SerialManager,
    timing: PicoTimingConfig,
    out_queue: "queue.Queue[Tuple[str, Any]]",
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Send Pico timing configuration commands and echo any responses."""
    commands = [
        f"CFG IT_MS {int(timing.it_ms)}",
        f"CFG EXTRA_WAIT_MS {int(timing.extra_wait_ms)}",
        f"CFG FREQ_SETTLE_MS {int(timing.freq_settle_ms)}",
        f"CFG RF_ON_SETTLE_MS {int(timing.rf_on_settle_ms)}",
        f"CFG RF_OFF_SETTLE_MS {int(timing.rf_off_settle_ms)}",
        f"CFG DISCARD {int(timing.discard_samples)}",
        f"CFG AVG {int(timing.avg_samples_per_state)}",
        "GETCFG",
    ]

    out_queue.put(("log", "Configuring Pico timings..."))
    for command in commands:
        if stop_event is not None and stop_event.is_set():
            return
        serial_manager.send_line(command)
        out_queue.put(("log", f"> {command}"))
        time.sleep(0.15)
        for line in serial_manager.drain(0.15, stop_event=stop_event):
            out_queue.put(("log", f"[PICO] {line}"))


def wait_for_result(
    serial_manager: SerialManager,
    expected_freq_hz: float,
    timeout_s: float,
    out_queue: "queue.Queue[Tuple[str, Any]]",
    stop_event: threading.Event,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Wait for a RESULT line from the Pico, ignoring unrelated lines."""
    t0 = time.time()
    expected = int(round(expected_freq_hz))

    while (time.time() - t0) < timeout_s:
        if stop_event.is_set():
            raise RuntimeError("Measurement stopped by user.")

        raw = serial_manager.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        parsed = parse_result_line(line)
        if parsed is None:
            out_queue.put(("log", f"[PICO] {line}"))
            continue

        freq_hz, off1, on, off2, off_ref, delta, contrast, drift = parsed
        if int(round(freq_hz)) != expected:
            out_queue.put(("log", f"[PICO] Warning: got result for unexpected frequency {freq_hz:.0f} Hz"))
            continue

        return freq_hz, off1, on, off2, off_ref, delta, contrast, drift

    raise TimeoutError(f"Timed out waiting for RESULT for {expected} Hz")


def run_odmr_worker(
    serial_manager: SerialManager,
    timing: PicoTimingConfig,
    config: ODMRConfig,
    out_queue: "queue.Queue[Tuple[str, Any]]",
    stop_event: threading.Event,
) -> None:
    """Run an ODMR sweep in a background thread."""
    try:
        configure_pico_timings(serial_manager, timing, out_queue, stop_event=stop_event)
        if stop_event.is_set():
            out_queue.put(("odmr_done", "stopped"))
            return

        freqs_hz = build_frequency_list(config.f_start_hz, config.f_stop_hz, config.f_step_hz)
        n_freqs = len(freqs_hz)
        total_meas = n_freqs * config.repeats_per_freq
        out_queue.put(("odmr_start", {"n_freqs": n_freqs, "total_meas": total_meas}))
        out_queue.put((
            "log",
            f"Planned ODMR run: {n_freqs} frequencies, {config.repeats_per_freq} repeats/frequency, {total_meas} total measurements",
        ))

        csv_path = Path(config.csv_path).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        freq_mean_delta = np.full(n_freqs, np.nan, dtype=np.float64)
        freq_sem_delta = np.full(n_freqs, np.nan, dtype=np.float64)

        completed_meas = 0
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "freq_hz",
                "repeat_idx",
                "off1",
                "on",
                "off2",
                "off_ref",
                "delta_on_minus_offref",
                "contrast_on_minus_offref_over_offref",
                "drift_off2_minus_off1",
            ])

            for fi, freq_hz in enumerate(freqs_hz):
                if stop_event.is_set():
                    out_queue.put(("log", "ODMR run stopped."))
                    break

                out_queue.put(("log", f"Requesting measurements at {freq_hz / 1e9:.6f} GHz ..."))
                deltas_this_freq: List[float] = []

                for repeat_idx in range(config.repeats_per_freq):
                    if stop_event.is_set():
                        break

                    serial_manager.send_line(f"MEAS {int(round(freq_hz))}")
                    result = wait_for_result(
                        serial_manager=serial_manager,
                        expected_freq_hz=freq_hz,
                        timeout_s=config.meas_timeout_s,
                        out_queue=out_queue,
                        stop_event=stop_event,
                    )

                    result_freq_hz, off1, on, off2, off_ref, delta, contrast, drift = result
                    if np.isfinite(delta):
                        deltas_this_freq.append(delta)

                    point = ODMRPoint(
                        freq_hz=result_freq_hz,
                        repeat_idx=repeat_idx,
                        off1=off1,
                        on=on,
                        off2=off2,
                        off_ref=off_ref,
                        delta=delta,
                        contrast=contrast,
                        drift=drift,
                    )

                    writer.writerow([
                        f"{result_freq_hz:.0f}",
                        repeat_idx,
                        off1,
                        on,
                        off2,
                        off_ref,
                        delta,
                        contrast,
                        drift,
                    ])
                    file.flush()

                    completed_meas += 1
                    out_queue.put(("odmr_point", point))
                    out_queue.put(("progress", {"completed": completed_meas, "total": total_meas}))

                    if config.print_each_measurement:
                        out_queue.put((
                            "log",
                            f"  rep {repeat_idx + 1:2d}/{config.repeats_per_freq} | "
                            f"OFF1={off1:.6g} ON={on:.6g} OFF2={off2:.6g} "
                            f"OFF_REF={off_ref:.6g} Δ={delta:+.3e} "
                            f"contrast={contrast:+.3e} drift={drift:+.3e}",
                        ))

                d = np.asarray(deltas_this_freq, dtype=np.float64)
                if d.size > 0:
                    freq_mean_delta[fi] = float(np.mean(d))
                    freq_sem_delta[fi] = float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0

                if d.size > 0:
                    out_queue.put((
                        "odmr_summary_point",
                        {
                            "freq_hz": float(freq_hz),
                            "mean_delta": float(freq_mean_delta[fi]),
                            "sem_delta": float(freq_sem_delta[fi]),
                            "n": int(d.size),
                        },
                    ))

                if config.print_freq_summary and d.size > 0:
                    out_queue.put((
                        "log",
                        f"Freq {freq_hz / 1e9:.6f} GHz summary: "
                        f"mean Δ = {freq_mean_delta[fi]:+.6e}, "
                        f"SEM = {freq_sem_delta[fi]:.6e}, n = {d.size}",
                    ))

        if stop_event.is_set():
            out_queue.put(("odmr_done", "stopped"))
        else:
            out_queue.put(("log", f"Saved {csv_path}"))
            out_queue.put(("odmr_done", "completed"))

    except Exception as exc:
        out_queue.put(("error", f"ODMR error: {exc}"))
        out_queue.put(("odmr_done", "error"))


def run_light_worker(
    serial_manager: SerialManager,
    timing: PicoTimingConfig,
    config: LightConfig,
    out_queue: "queue.Queue[Tuple[str, Any]]",
    stop_event: threading.Event,
) -> None:
    """Run TSL2591 serial streaming in a background thread."""
    csv_file = None
    csv_writer = None

    try:
        configure_pico_timings(serial_manager, timing, out_queue, stop_event=stop_event)
        if stop_event.is_set():
            out_queue.put(("light_done", "stopped"))
            return

        if config.csv_log:
            csv_path = Path(config.csv_path).expanduser()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["t_pico_s", "ch0", "ch1", "lux", "raw"])
            out_queue.put(("log", f"Light CSV logging to {csv_path}"))

        if config.send_stream_commands:
            out_queue.put(("log", "Sending Pico to stream mode..."))
            serial_manager.send_line("MODE STREAM")
            time.sleep(0.2)
            for line in serial_manager.drain(1.0, stop_event=stop_event):
                out_queue.put(("log", f"[PICO] {line}"))

        out_queue.put(("light_start", None))

        total_bytes = 0
        total_lines = 0
        total_samples = 0
        last_hb = time.time()

        while not stop_event.is_set():
            raw = serial_manager.readline()
            if not raw:
                continue

            total_bytes += len(raw)
            total_lines += 1

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            parsed = parse_data_line(line)
            if parsed is None:
                if config.print_nondata_lines:
                    out_queue.put(("log", f"[MSG] {line}"))
                continue

            if config.print_raw_data_lines:
                out_queue.put(("log", f"[DATA] {line}"))

            t_s, ch0, ch1, lux = parsed
            sample = LightSample(t_pico_s=t_s, ch0=ch0, ch1=ch1, lux=lux, raw=line)
            total_samples += 1

            if csv_writer is not None and csv_file is not None:
                csv_writer.writerow([
                    f"{t_s:.3f}",
                    ch0,
                    ch1,
                    "" if lux is None else f"{lux:.4f}",
                    line,
                ])
                csv_file.flush()

            out_queue.put(("light_sample", sample))

            now = time.time()
            if (now - last_hb) >= config.heartbeat_period_s:
                last_hb = now
                out_queue.put((
                    "light_status",
                    {
                        "bytes": total_bytes,
                        "lines": total_lines,
                        "samples": total_samples,
                    },
                ))

        if config.send_stream_commands and serial_manager.is_open:
            try:
                serial_manager.send_line("STREAM OFF")
                time.sleep(0.1)
            except Exception:
                pass

        out_queue.put(("light_done", "stopped"))

    except Exception as exc:
        out_queue.put(("error", f"Light-stream error: {exc}"))
        out_queue.put(("light_done", "error"))

    finally:
        try:
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# GUI application
# -----------------------------------------------------------------------------

class ODMRLightGUI(tk.Tk):
    """Main Tkinter application."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Pico ODMR / Light Measurement GUI")
        self.geometry("1250x820")
        self.minsize(1000, 680)

        self.serial_manager = SerialManager()
        self.message_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self.odmr_freqs_ghz: List[float] = []
        self.odmr_mean_delta: List[float] = []
        self.odmr_sem_delta: List[float] = []

        self.light_t: Deque[float] = deque(maxlen=600)
        self.light_ch0: Deque[float] = deque(maxlen=600)
        self.light_ch1: Deque[float] = deque(maxlen=600)
        self.light_lux: Deque[float] = deque(maxlen=600)

        self._make_variables()
        self._make_widgets()
        self.refresh_ports()
        self._set_running_state(False)

        self.after(50, self._poll_queue)
        self.after(200, self._refresh_light_plot)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _make_variables(self) -> None:
        self.port_var = tk.StringVar(value="COM5")
        self.baud_var = tk.StringVar(value="115200")
        self.timeout_var = tk.StringVar(value="0.2")
        self.dtr_var = tk.BooleanVar(value=True)
        self.rts_var = tk.BooleanVar(value=False)

        self.it_ms_var = tk.StringVar(value="400")
        self.extra_wait_ms_var = tk.StringVar(value="20")
        self.freq_settle_ms_var = tk.StringVar(value="100")
        self.rf_on_settle_ms_var = tk.StringVar(value="100")
        self.rf_off_settle_ms_var = tk.StringVar(value="100")
        self.discard_var = tk.StringVar(value="1")
        self.avg_var = tk.StringVar(value="4")

        self.f_start_ghz_var = tk.StringVar(value="2.81")
        self.f_stop_ghz_var = tk.StringVar(value="2.93")
        self.f_step_ghz_var = tk.StringVar(value="0.0001")
        self.repeats_var = tk.StringVar(value="4")
        self.meas_timeout_var = tk.StringVar(value="25.0")
        self.odmr_csv_var = tk.StringVar(value="noField.csv")
        self.print_each_var = tk.BooleanVar(value=True)
        self.print_summary_var = tk.BooleanVar(value=True)

        self.rolling_seconds_var = tk.StringVar(value="60")
        self.assumed_sample_hz_var = tk.StringVar(value="10")
        self.max_plot_hz_var = tk.StringVar(value="20")
        self.light_csv_log_var = tk.BooleanVar(value=True)
        self.light_csv_var = tk.StringVar(value="tsl2591_log.csv")
        self.send_stream_var = tk.BooleanVar(value=True)
        self.print_nondata_var = tk.BooleanVar(value=True)
        self.print_raw_data_var = tk.BooleanVar(value=False)
        self.heartbeat_var = tk.StringVar(value="1.0")

        self.status_var = tk.StringVar(value="Disconnected")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0 / 0")

    def _make_widgets(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        self._make_serial_frame(left)
        self._make_timing_frame(left)
        self._make_odmr_frame(left)
        self._make_light_frame(left)
        self._make_action_frame(left)

        self._make_plot_tabs(right)
        self._make_log_frame(right)

    def _make_serial_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Serial", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(frame, textvariable=self.port_var, width=18)
        self.port_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(frame, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=(6, 0))

        ttk.Label(frame, text="Baud").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(frame, textvariable=self.baud_var, width=12).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))

        ttk.Label(frame, text="Timeout s").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(frame, textvariable=self.timeout_var, width=12).grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))

        ttk.Checkbutton(frame, text="Assert DTR", variable=self.dtr_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(frame, text="Assert RTS", variable=self.rts_var).grid(row=4, column=0, columnspan=2, sticky="w")

        self.connect_button = ttk.Button(frame, text="Connect", command=self.connect_serial)
        self.connect_button.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.disconnect_button = ttk.Button(frame, text="Disconnect", command=self.disconnect_serial)
        self.disconnect_button.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=(8, 0))

    def _make_timing_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Pico timing config", padding=8)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        rows = [
            ("IT ms", self.it_ms_var),
            ("Extra wait ms", self.extra_wait_ms_var),
            ("Freq settle ms", self.freq_settle_ms_var),
            ("RF on settle ms", self.rf_on_settle_ms_var),
            ("RF off settle ms", self.rf_off_settle_ms_var),
            ("Discard samples", self.discard_var),
            ("Avg/state", self.avg_var),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 4, 0))
            ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=(0 if row == 0 else 4, 0))

        self.configure_button = ttk.Button(frame, text="Send config only", command=self.send_config_only)
        self.configure_button.grid(row=len(rows), column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _make_odmr_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="ODMR sweep", padding=8)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        rows = [
            ("Start GHz", self.f_start_ghz_var),
            ("Stop GHz", self.f_stop_ghz_var),
            ("Step GHz", self.f_step_ghz_var),
            ("Repeats/freq", self.repeats_var),
            ("Meas timeout s", self.meas_timeout_var),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 4, 0))
            ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=(0 if row == 0 else 4, 0))

        csv_row = len(rows)
        ttk.Label(frame, text="CSV path").grid(row=csv_row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(frame, textvariable=self.odmr_csv_var, width=18).grid(row=csv_row, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
        ttk.Button(frame, text="...", width=3, command=self.browse_odmr_csv).grid(row=csv_row, column=2, padx=(6, 0), pady=(4, 0))

        ttk.Checkbutton(frame, text="Log each measurement", variable=self.print_each_var).grid(row=csv_row + 1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(frame, text="Log frequency summaries", variable=self.print_summary_var).grid(row=csv_row + 2, column=0, columnspan=3, sticky="w")

    def _make_light_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Light stream", padding=8)
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        rows = [
            ("Rolling seconds", self.rolling_seconds_var),
            ("Assumed sample Hz", self.assumed_sample_hz_var),
            ("Max plot Hz", self.max_plot_hz_var),
            ("Heartbeat s", self.heartbeat_var),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 4, 0))
            ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=(0 if row == 0 else 4, 0))

        csv_row = len(rows)
        ttk.Checkbutton(frame, text="CSV log", variable=self.light_csv_log_var).grid(row=csv_row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(frame, textvariable=self.light_csv_var, width=18).grid(row=csv_row, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
        ttk.Button(frame, text="...", width=3, command=self.browse_light_csv).grid(row=csv_row, column=2, padx=(6, 0), pady=(4, 0))

        ttk.Checkbutton(frame, text="Send MODE STREAM / STREAM OFF", variable=self.send_stream_var).grid(row=csv_row + 1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(frame, text="Print non-DATA lines", variable=self.print_nondata_var).grid(row=csv_row + 2, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(frame, text="Print raw DATA lines", variable=self.print_raw_data_var).grid(row=csv_row + 3, column=0, columnspan=3, sticky="w")

    def _make_action_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Run", padding=8)
        frame.grid(row=4, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.start_odmr_button = ttk.Button(frame, text="Start ODMR", command=self.start_odmr)
        self.start_odmr_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.start_light_button = ttk.Button(frame, text="Start Light", command=self.start_light)
        self.start_light_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.stop_button = ttk.Button(frame, text="Stop", command=self.stop_worker)
        self.stop_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        ttk.Label(frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.progress = ttk.Progressbar(frame, variable=self.progress_var, maximum=100.0)
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(frame, textvariable=self.progress_text_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _make_plot_tabs(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.odmr_tab = ttk.Frame(self.notebook)
        self.light_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.odmr_tab, text="ODMR plot")
        self.notebook.add(self.light_tab, text="Light plot")

        self.odmr_tab.columnconfigure(0, weight=1)
        self.odmr_tab.rowconfigure(0, weight=1)
        self.light_tab.columnconfigure(0, weight=1)
        self.light_tab.rowconfigure(0, weight=1)

        self.odmr_fig = Figure(figsize=(7, 5), dpi=100)
        self.odmr_ax = self.odmr_fig.add_subplot(111)
        self.odmr_ax.set_title("ODMR sweep mean Δ vs frequency")
        self.odmr_ax.set_xlabel("Frequency (GHz)")
        self.odmr_ax.set_ylabel("Mean Δ = ON - OFF_REF")
        self.odmr_ax.grid(True)
        self.odmr_canvas = FigureCanvasTkAgg(self.odmr_fig, master=self.odmr_tab)
        self.odmr_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.odmr_toolbar = NavigationToolbar2Tk(self.odmr_canvas, self.odmr_tab, pack_toolbar=False)
        self.odmr_toolbar.grid(row=1, column=0, sticky="ew")

        self.light_fig = Figure(figsize=(7, 5), dpi=100)
        self.light_ax = self.light_fig.add_subplot(111)
        self.light_ax.set_title("TSL2591 live")
        self.light_ax.set_xlabel("Pico time (s)")
        self.light_ax.set_ylabel("Counts / Lux")
        self.light_ax.grid(True)
        (self.line_ch0,) = self.light_ax.plot([], [], label="CH0 (vis+IR)")
        (self.line_ch1,) = self.light_ax.plot([], [], label="CH1 (IR)")
        (self.line_lux,) = self.light_ax.plot([], [], label="Lux (approx)")
        self.light_ax.legend(loc="upper right")
        self.light_canvas = FigureCanvasTkAgg(self.light_fig, master=self.light_tab)
        self.light_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.light_toolbar = NavigationToolbar2Tk(self.light_canvas, self.light_tab, pack_toolbar=False)
        self.light_toolbar.grid(row=1, column=0, sticky="ew")

    def _make_log_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Log", padding=6)
        frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(frame, height=10, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        ttk.Button(frame, text="Clear log", command=self.clear_log).grid(row=1, column=0, sticky="e", pady=(4, 0))

    # ------------------------------------------------------------------
    # Config extraction
    # ------------------------------------------------------------------

    def get_serial_config(self) -> SerialConfig:
        return SerialConfig(
            port=self.port_var.get().strip(),
            baud=safe_int(self.baud_var.get(), "Baud"),
            timeout_s=safe_float(self.timeout_var.get(), "Timeout"),
            assert_dtr=bool(self.dtr_var.get()),
            assert_rts=bool(self.rts_var.get()),
        )

    def get_timing_config(self) -> PicoTimingConfig:
        return PicoTimingConfig(
            it_ms=safe_int(self.it_ms_var.get(), "IT ms"),
            extra_wait_ms=safe_int(self.extra_wait_ms_var.get(), "Extra wait ms"),
            freq_settle_ms=safe_int(self.freq_settle_ms_var.get(), "Freq settle ms"),
            rf_on_settle_ms=safe_int(self.rf_on_settle_ms_var.get(), "RF on settle ms"),
            rf_off_settle_ms=safe_int(self.rf_off_settle_ms_var.get(), "RF off settle ms"),
            discard_samples=safe_int(self.discard_var.get(), "Discard samples"),
            avg_samples_per_state=safe_int(self.avg_var.get(), "Avg/state"),
        )

    def get_odmr_config(self) -> ODMRConfig:
        return ODMRConfig(
            f_start_hz=safe_float(self.f_start_ghz_var.get(), "Start GHz") * 1e9,
            f_stop_hz=safe_float(self.f_stop_ghz_var.get(), "Stop GHz") * 1e9,
            f_step_hz=safe_float(self.f_step_ghz_var.get(), "Step GHz") * 1e9,
            repeats_per_freq=safe_int(self.repeats_var.get(), "Repeats/freq"),
            meas_timeout_s=safe_float(self.meas_timeout_var.get(), "Meas timeout"),
            csv_path=self.odmr_csv_var.get().strip() or "noField.csv",
            print_each_measurement=bool(self.print_each_var.get()),
            print_freq_summary=bool(self.print_summary_var.get()),
        )

    def get_light_config(self) -> LightConfig:
        rolling_seconds = safe_float(self.rolling_seconds_var.get(), "Rolling seconds")
        assumed_sample_hz = safe_float(self.assumed_sample_hz_var.get(), "Assumed sample Hz")
        maxlen = max(20, int(rolling_seconds * assumed_sample_hz))

        self.light_t = deque(self.light_t, maxlen=maxlen)
        self.light_ch0 = deque(self.light_ch0, maxlen=maxlen)
        self.light_ch1 = deque(self.light_ch1, maxlen=maxlen)
        self.light_lux = deque(self.light_lux, maxlen=maxlen)

        return LightConfig(
            rolling_seconds=rolling_seconds,
            assumed_sample_hz=assumed_sample_hz,
            max_plot_hz=safe_float(self.max_plot_hz_var.get(), "Max plot Hz"),
            csv_log=bool(self.light_csv_log_var.get()),
            csv_path=self.light_csv_var.get().strip() or "tsl2591_log.csv",
            send_stream_commands=bool(self.send_stream_var.get()),
            print_nondata_lines=bool(self.print_nondata_var.get()),
            print_raw_data_lines=bool(self.print_raw_data_var.get()),
            heartbeat_period_s=safe_float(self.heartbeat_var.get(), "Heartbeat"),
        )

    # ------------------------------------------------------------------
    # GUI actions
    # ------------------------------------------------------------------

    def refresh_ports(self) -> None:
        ports = available_serial_ports()
        values = [device for device, _desc in ports]
        self.port_combo["values"] = values
        if values and self.port_var.get() not in values:
            self.port_var.set(values[0])
        self.log("Available ports:")
        if ports:
            for device, desc in ports:
                self.log(f"  {device:>8}  {desc}")
        else:
            self.log("  (none)")

    def connect_serial(self) -> None:
        try:
            config = self.get_serial_config()
            if not config.port:
                raise ValueError("Select a serial port first.")
            self.status_var.set(f"Opening {config.port} @ {config.baud}...")
            self.update_idletasks()
            self.serial_manager.open(config)
            self.status_var.set(f"Connected to {config.port}")
            self.log(f"Connected. DTR={config.assert_dtr}, RTS={config.assert_rts}")
            for line in self.serial_manager.drain(1.0):
                self.log(f"[PICO] {line}")
            self._set_connection_state(True)
        except Exception as exc:
            self.status_var.set("Disconnected")
            messagebox.showerror("Serial connection error", str(exc))
            self.log(f"ERROR opening serial port: {exc}")
            self._set_connection_state(False)

    def disconnect_serial(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "Stop the running measurement before disconnecting.")
            return
        self.serial_manager.close()
        self.status_var.set("Disconnected")
        self.log("Serial port closed.")
        self._set_connection_state(False)

    def send_config_only(self) -> None:
        if not self._check_connected():
            return
        if self._worker_running():
            messagebox.showwarning("Busy", "A measurement is already running.")
            return

        try:
            timing = self.get_timing_config()
        except Exception as exc:
            messagebox.showerror("Invalid timing config", str(exc))
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._config_only_worker,
            args=(timing,),
            daemon=True,
        )
        self.worker_thread.start()
        self._set_running_state(True)

    def _config_only_worker(self, timing: PicoTimingConfig) -> None:
        try:
            configure_pico_timings(self.serial_manager, timing, self.message_queue, stop_event=self.stop_event)
        except Exception as exc:
            self.message_queue.put(("error", f"Config error: {exc}"))
        finally:
            self.message_queue.put(("config_done", None))

    def start_odmr(self) -> None:
        if not self._check_connected():
            return
        if self._worker_running():
            messagebox.showwarning("Busy", "A measurement is already running.")
            return

        try:
            timing = self.get_timing_config()
            config = self.get_odmr_config()
            _ = build_frequency_list(config.f_start_hz, config.f_stop_hz, config.f_step_hz)
            if config.repeats_per_freq <= 0:
                raise ValueError("Repeats/freq must be positive.")
        except Exception as exc:
            messagebox.showerror("Invalid ODMR config", str(exc))
            return

        self.clear_odmr_plot()
        self.stop_event.clear()
        self.progress_var.set(0.0)
        self.progress_text_var.set("0 / 0")
        self.status_var.set("Running ODMR...")
        self.notebook.select(self.odmr_tab)

        self.worker_thread = threading.Thread(
            target=run_odmr_worker,
            args=(self.serial_manager, timing, config, self.message_queue, self.stop_event),
            daemon=True,
        )
        self.worker_thread.start()
        self._set_running_state(True)

    def start_light(self) -> None:
        if not self._check_connected():
            return
        if self._worker_running():
            messagebox.showwarning("Busy", "A measurement is already running.")
            return

        try:
            timing = self.get_timing_config()
            config = self.get_light_config()
        except Exception as exc:
            messagebox.showerror("Invalid light config", str(exc))
            return

        self.clear_light_plot()
        self.stop_event.clear()
        self.progress_var.set(0.0)
        self.progress_text_var.set("Streaming")
        self.status_var.set("Running light stream...")
        self.notebook.select(self.light_tab)

        self.worker_thread = threading.Thread(
            target=run_light_worker,
            args=(self.serial_manager, timing, config, self.message_queue, self.stop_event),
            daemon=True,
        )
        self.worker_thread.start()
        self._set_running_state(True)

    def stop_worker(self) -> None:
        if self._worker_running():
            self.log("Stop requested...")
            self.stop_event.set()
            self.status_var.set("Stopping...")
        else:
            self.log("No worker is running.")

    def browse_odmr_csv(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save ODMR CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=self.odmr_csv_var.get() or "noField.csv",
        )
        if filename:
            self.odmr_csv_var.set(filename)

    def browse_light_csv(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save light-stream CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=self.light_csv_var.get() or "tsl2591_log.csv",
        )
        if filename:
            self.light_csv_var.set(filename)

    # ------------------------------------------------------------------
    # Queue handling and plots
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                self._handle_message(kind, payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _handle_message(self, kind: str, payload: Any) -> None:
        if kind == "log":
            self.log(str(payload))
        elif kind == "error":
            self.log(str(payload))
            messagebox.showerror("Error", str(payload))
        elif kind == "progress":
            completed = int(payload["completed"])
            total = int(payload["total"])
            self.progress_var.set(100.0 * completed / max(1, total))
            self.progress_text_var.set(f"{completed} / {total}")
        elif kind == "odmr_start":
            self.log(f"ODMR started: {payload['n_freqs']} frequency points, {payload['total_meas']} total measurements")
        elif kind == "odmr_point":
            # Raw measurement points are logged by the worker if enabled.
            pass
        elif kind == "odmr_summary_point":
            self.odmr_freqs_ghz.append(float(payload["freq_hz"]) / 1e9)
            self.odmr_mean_delta.append(float(payload["mean_delta"]))
            self.odmr_sem_delta.append(float(payload["sem_delta"]))
            self.update_odmr_plot()
        elif kind == "odmr_done":
            self.status_var.set(f"ODMR {payload}")
            self._set_running_state(False)
        elif kind == "light_start":
            self.log("Light stream started.")
        elif kind == "light_sample":
            sample: LightSample = payload
            self.light_t.append(sample.t_pico_s)
            self.light_ch0.append(sample.ch0)
            self.light_ch1.append(sample.ch1)
            self.light_lux.append(float("nan") if sample.lux is None else sample.lux)
        elif kind == "light_status":
            self.log(
                f"[HB] bytes={payload['bytes']} lines={payload['lines']} samples={payload['samples']} window_n={len(self.light_t)}"
            )
            self.light_ax.set_title(f"TSL2591 live | samples={payload['samples']}")
        elif kind == "light_done":
            self.status_var.set(f"Light stream {payload}")
            self._set_running_state(False)
        elif kind == "config_done":
            self.status_var.set("Config sent")
            self._set_running_state(False)
        else:
            self.log(f"Unknown message: {kind} {payload}")

    def clear_odmr_plot(self) -> None:
        self.odmr_freqs_ghz.clear()
        self.odmr_mean_delta.clear()
        self.odmr_sem_delta.clear()
        self.update_odmr_plot()

    def update_odmr_plot(self) -> None:
        self.odmr_ax.clear()
        self.odmr_ax.set_title("ODMR sweep mean Δ vs frequency")
        self.odmr_ax.set_xlabel("Frequency (GHz)")
        self.odmr_ax.set_ylabel("Mean Δ = ON - OFF_REF")
        self.odmr_ax.grid(True)

        if self.odmr_freqs_ghz:
            self.odmr_ax.errorbar(
                self.odmr_freqs_ghz,
                self.odmr_mean_delta,
                yerr=self.odmr_sem_delta,
                fmt="o-",
                capsize=4,
            )
            self.odmr_fig.tight_layout()

        self.odmr_canvas.draw_idle()

    def clear_light_plot(self) -> None:
        self.light_t.clear()
        self.light_ch0.clear()
        self.light_ch1.clear()
        self.light_lux.clear()
        self._draw_light_plot(force=True)

    def _refresh_light_plot(self) -> None:
        self._draw_light_plot(force=False)

        try:
            max_plot_hz = safe_float(self.max_plot_hz_var.get(), "Max plot Hz")
        except Exception:
            max_plot_hz = 20.0
        interval_ms = int(1000 / max(1.0, max_plot_hz))
        self.after(max(50, interval_ms), self._refresh_light_plot)

    def _draw_light_plot(self, force: bool) -> None:
        if len(self.light_t) < 2 and not force:
            return

        t = list(self.light_t)
        self.line_ch0.set_data(t, list(self.light_ch0))
        self.line_ch1.set_data(t, list(self.light_ch1))
        self.line_lux.set_data(t, list(self.light_lux))

        self.light_ax.relim()
        self.light_ax.autoscale_view()
        self.light_ax.grid(True)
        self.light_canvas.draw_idle()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _worker_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _check_connected(self) -> bool:
        if not self.serial_manager.is_open:
            messagebox.showwarning("Not connected", "Connect to the Pico serial port first.")
            return False
        return True

    def _set_connection_state(self, connected: bool) -> None:
        self.connect_button.configure(state="disabled" if connected else "normal")
        self.disconnect_button.configure(state="normal" if connected else "disabled")
        self.configure_button.configure(state="normal" if connected else "disabled")
        self.start_odmr_button.configure(state="normal" if connected else "disabled")
        self.start_light_button.configure(state="normal" if connected else "disabled")

    def _set_running_state(self, running: bool) -> None:
        connected = self.serial_manager.is_open
        self.stop_button.configure(state="normal" if running else "disabled")
        self.configure_button.configure(state="disabled" if running or not connected else "normal")
        self.start_odmr_button.configure(state="disabled" if running or not connected else "normal")
        self.start_light_button.configure(state="disabled" if running or not connected else "normal")
        self.disconnect_button.configure(state="disabled" if running or not connected else "normal")
        self.connect_button.configure(state="disabled" if running or connected else "normal")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _on_close(self) -> None:
        if self._worker_running():
            if not messagebox.askyesno("Measurement running", "Stop the running measurement and exit?"):
                return
            self.stop_event.set()
            # Do not block indefinitely on close; the serial timeout should let worker exit.
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=2.0)

        self.serial_manager.close()
        self.destroy()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    app = ODMRLightGUI()
    app.mainloop()


if __name__ == "__main__":
    main()


