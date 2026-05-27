"""
PID Controller Tuner GUI
========================
Simulates a motor speed controller with PID + Feed-forward (F).

Motor model : three cascaded first-order lags  G(s) = 1/(τs+1)³
              input  u  ∈ [0, 1]
              output ω  ∈ [0, 6000] RPM

Controller  : output = clamp(P + I + D + F·setpoint_norm, 0, 1)
              F is a direct feed-forward on the normalised setpoint
              (bypasses PID dynamics)

Ziegler-Nichols (P-only, Ki=Kd=F=0):
              Ultimate gain    Ku = 8
              Ultimate period  Tu = 2*pi*tau/sqrt(3)  ~  3.63*tau
"""

"""
TODO:
 * change graphs to show only k most recent readouts (like dashboard)
 * add kS
 * add noise checkboxes to GUI
 * 
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import collections

# ── Constants ────────────────────────────────────────────────────────────────
MAX_RPM       = 6000          # full-scale motor speed
HISTORY_SEC   = 20            # seconds of plot history
GUI_UPDATE_MS = 50            # GUI refresh interval (~20 FPS)
MAX_HIST_PTS  = HISTORY_SEC * 1000 + 200   # buffer for 1 ms resolution


# ── PID Controller ────────────────────────────────────────────────────────────
class PIDController:
    """Discrete PID with integral anti-windup (integral clamping)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint_norm, measurement_norm, dt, Kp, Ki, Kd, F):
        """
        All signals are normalised to [0, 1].

        Returns
        -------
        output  : clamped command to motor [0, 1]
        P, I, D : individual PID contributions
        ff      : feed-forward contribution  (F x setpoint_norm)
        error   : normalised error
        """
        error = setpoint_norm - measurement_norm

        # Feed-forward (F x setpoint, bypasses PID)
        ff = F * setpoint_norm

        # Proportional
        P = Kp * error

        # Integral with anti-windup
        self.integral += error * dt
        if Ki != 0.0:
            lim = 1.0 / abs(Ki)
            self.integral = float(np.clip(self.integral, -lim, lim))
        I = Ki * self.integral

        # Derivative (on error)
        D = Kd * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # Total output
        output = float(np.clip(P + I + D + ff, 0.0, 1.0))
        return output, P, I, D, ff, error


# ── Motor Model ───────────────────────────────────────────────────────────────
class MotorModel:
    """
    Three cascaded first-order lags:  G(s) = 1 / (tau*s+1)^3

    This 3rd-order plant exhibits classic Ziegler-Nichols behaviour
    when driven by a pure P controller (Ki = Kd = F = 0):

        Kp < Ku  ->  error converges  (stable)
        Kp = Ku  ->  sustained constant-amplitude oscillation
        Kp > Ku  ->  error diverges   (unstable)

    For equal stage time constants tau:
        Ultimate gain    Ku = 8
        Ultimate period  Tu = 2*pi*tau / sqrt(3)  ~  3.63*tau
    """

    def __init__(self):
        self.x1 = 0.0   # output of stage 1
        self.x2 = 0.0   # output of stage 2
        self.x3 = 0.0   # output of stage 3  (= normalised motor speed)

    @property
    def speed_norm(self) -> float:
        return self.x3
    
    # simulate sensor noise (hopefully)
    def get_noisy_speed_readout(self) -> float:
        return self.x3 + np.random.normal(0, 0.01)

    def reset(self):
        self.x1 = self.x2 = self.x3 = 0.0

    def update(self, command: float, dt: float, tau: float) -> float:
        """Euler integration through three cascaded stages; returns speed in RPM."""
        if tau > 0.0:
            self.x1 += (command - self.x1) / tau * dt
            self.x2 += (self.x1  - self.x2) / tau * dt
            self.x3 += (self.x2  - self.x3) / tau * dt
        else:
            self.x1 = self.x2 = self.x3 = command
        self.x3 = float(np.clip(self.x3, 0.0, 1.0))
        return self.x3 * MAX_RPM


# ── Main Application ──────────────────────────────────────────────────────────
class PIDTunerApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PID Controller Tuner  -  Motor Speed (0-6000 RPM)")
        self.root.minsize(1100, 700)

        self.pid   = PIDController()
        self.motor = MotorModel()

        self.running  = False
        self.sim_time = 0.0

        # History ring-buffers
        def _buf():
            return collections.deque(maxlen=MAX_HIST_PTS)

        self.t_buf     = _buf()
        self.speed_buf = _buf()
        self.sp_buf    = _buf()
        self.err_buf   = _buf()
        self.out_buf   = _buf()
        self.P_buf     = _buf()
        self.I_buf     = _buf()
        self.D_buf     = _buf()
        self.F_buf     = _buf()

        # param_var dict: holds the *committed* DoubleVar the simulation reads
        self.params: dict[str, tk.DoubleVar] = {}

        self._build_controls()
        self._build_plots()

    # ── GUI: left control panel ───────────────────────────────────────────────
    def _build_controls(self):
        left = ttk.Frame(self.root, width=320, padding=(8, 8))
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)
        left.pack_propagate(False)

        ttk.Label(left, text="PID Controller Tuner",
                  font=("Segoe UI", 13, "bold")).pack(pady=(0, 6))

        # Setpoint (RPM)
        ttk.Label(left, text="Setpoint  (RPM,  0 - 6000)",
                  font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=4)
        self._param_row(left, "setpoint", 0.0, 6000.0, 3000.0, 1.0, "%.1f")

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=5)
        ttk.Label(left, text="Controller Parameters",
                  font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=4)

        for label, key, lo, hi, default, step, fmt in [
            ("Kp  - Proportional gain",  "kp",  0.0,  20.0,  1.0,   0.01,  "%.3f"),
            ("Ki  - Integral gain",       "ki",  0.0,  20.0,  0.1,   0.01,  "%.3f"),
            ("Kd  - Derivative gain",     "kd",  0.0,  10.0,  0.05,  0.001, "%.4f"),
            ("F   - Feed-forward x SP",   "f",   0.0,   2.0,  0.0,   0.01,  "%.3f"),
        ]:
            ttk.Label(left, text=label, font=("Segoe UI", 8)).pack(
                anchor=tk.W, padx=6, pady=(3, 0))
            self._param_row(left, key, lo, hi, default, step, fmt)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=5)
        ttk.Label(left, text="Simulation Parameters",
                  font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=4)

        for label, key, lo, hi, default, step, fmt in [
            ("Loop time  dt  (ms)",         "dt_ms", 1,    200,  20,    1,    "%.0f"),
            ("Motor time constant  tau (s)", "tau",   0.01,  5.0,  0.5,   0.01, "%.2f"),
        ]:
            ttk.Label(left, text=label, font=("Segoe UI", 8)).pack(
                anchor=tk.W, padx=6, pady=(3, 0))
            self._param_row(left, key, lo, hi, default, step, fmt)

        # Ziegler-Nichols reference note (updates live with tau)
        zn_frame = ttk.Frame(left)
        zn_frame.pack(fill=tk.X, padx=6, pady=(2, 0))
        self.zn_var = tk.StringVar()
        self._update_zn_label()
        ttk.Label(zn_frame, textvariable=self.zn_var,
                  font=("Courier New", 8), foreground="#555555").pack(anchor=tk.W)
        self.params["tau"].trace_add("write", lambda *_: self._update_zn_label())

        # Buttons
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=5)
        btn = ttk.Frame(left)
        btn.pack(fill=tk.X, padx=4)

        self.start_btn = ttk.Button(btn, text="Start",  command=self.start_sim)
        self.stop_btn  = ttk.Button(btn, text="Stop",   command=self.stop_sim,
                                    state=tk.DISABLED)
        self.reset_btn = ttk.Button(btn, text="Reset",  command=self.reset_sim)

        for b in (self.start_btn, self.stop_btn, self.reset_btn):
            b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Component visibility
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=5)
        ttk.Label(left, text="Component Visibility  (bottom plot)",
                  font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=4)

        cb_row = ttk.Frame(left)
        cb_row.pack(fill=tk.X, padx=8, pady=2)

        self.show_P = tk.BooleanVar(value=True)
        self.show_I = tk.BooleanVar(value=True)
        self.show_D = tk.BooleanVar(value=True)
        self.show_F = tk.BooleanVar(value=True)

        for col, (text, var) in enumerate(
                [("P", self.show_P), ("I", self.show_I),
                 ("D", self.show_D), ("F", self.show_F)]):
            ttk.Checkbutton(cb_row, text=text, variable=var).grid(
                row=0, column=col, padx=8)

        # Live readouts
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(left, textvariable=self.status_var,
                  font=("Segoe UI", 9)).pack(anchor=tk.W, padx=6)

        readout = ttk.Frame(left)
        readout.pack(fill=tk.X, padx=6, pady=2)

        self.speed_var = tk.StringVar(value="Speed    :      0.0 RPM")
        self.sp_var    = tk.StringVar(value="Setpoint :      0.0 RPM")
        self.err_var   = tk.StringVar(value="Error    :      0.0 RPM")
        self.out_var   = tk.StringVar(value="Output   :   0.0000")
        self.p_var     = tk.StringVar(value="P        :   0.0000")
        self.i_var     = tk.StringVar(value="I        :   0.0000")
        self.d_var     = tk.StringVar(value="D        :   0.0000")
        self.f_var     = tk.StringVar(value="F (ff)   :   0.0000")

        for sv in (self.speed_var, self.sp_var, self.err_var, self.out_var,
                   self.p_var, self.i_var, self.d_var, self.f_var):
            ttk.Label(readout, textvariable=sv,
                      font=("Courier New", 9)).pack(anchor=tk.W)

    # ─────────────────────────────────────────────────────────────────────────
    def _update_zn_label(self):
        """Refresh the Ziegler-Nichols reference values shown below tau."""
        try:
            tau = self.params["tau"].get()
        except Exception:
            tau = 0.5
        ku = 8.0
        tu = 2.0 * np.pi * tau / np.sqrt(3.0)
        self.zn_var.set(
            f"ZN (P-only): Ku={ku:.1f}  Tu={tu:.2f}s"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _param_row(self, parent, key, lo, hi, default, step, fmt):
        """
        Slider + Spinbox row.

        Slider  -> updates the committed param_var in real time.
        Spinbox -> displays the current value but only commits to param_var
                   when the user presses Enter, clicks away (FocusOut), or
                   uses the up/down arrow buttons.
        """
        param_var = tk.DoubleVar(value=default)
        self.params[key] = param_var

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=6, pady=1)

        slider = ttk.Scale(row, from_=lo, to=hi, variable=param_var,
                           orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        # Spinbox uses its own StringVar so typing doesn't affect the sim
        spin_var = tk.StringVar(value=fmt % default)
        spin = ttk.Spinbox(row, from_=lo, to=hi, textvariable=spin_var,
                           width=9, increment=step, format=fmt)
        spin.pack(side=tk.RIGHT)

        # Track whether the spinbox currently has keyboard focus
        spin_focused = [False]

        # When slider moves -> refresh spinbox display (only if not typing)
        def _on_param_change(*_):
            if not spin_focused[0]:
                try:
                    spin_var.set(fmt % param_var.get())
                except Exception:
                    pass

        param_var.trace_add("write", _on_param_change)

        # Commit spinbox text -> param_var
        def _commit(*_):
            try:
                val = float(spin_var.get())
                val = float(np.clip(val, lo, hi))
                param_var.set(val)
                spin_var.set(fmt % val)
            except ValueError:
                spin_var.set(fmt % param_var.get())

        def _on_focus_in(*_):
            spin_focused[0] = True

        def _on_focus_out(*_):
            spin_focused[0] = False
            _commit()

        spin.bind("<FocusIn>",  _on_focus_in)
        spin.bind("<FocusOut>", _on_focus_out)
        spin.bind("<Return>",   _commit)

        # Arrow buttons update spin_var first, then fire the virtual event;
        # schedule commit for the next event-loop tick so spin_var is ready.
        spin.bind("<<Increment>>", lambda _e: self.root.after(1, _commit))
        spin.bind("<<Decrement>>", lambda _e: self.root.after(1, _commit))

    # ── GUI: right plot area ──────────────────────────────────────────────────
    def _build_plots(self):
        right = ttk.Frame(self.root, padding=4)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 9), dpi=95)
        self.fig.patch.set_facecolor("#f5f5f5")

        gs = self.fig.add_gridspec(
            4, 1, hspace=0.55,
            left=0.07, right=0.97, top=0.97, bottom=0.05
        )

        self.ax_speed  = self.fig.add_subplot(gs[0])
        self.ax_error  = self.fig.add_subplot(gs[1])
        self.ax_output = self.fig.add_subplot(gs[2])
        self.ax_comp   = self.fig.add_subplot(gs[3])

        for ax in (self.ax_speed, self.ax_error, self.ax_output, self.ax_comp):
            ax.set_facecolor("#fefefe")
            ax.grid(True, alpha=0.35, linewidth=0.6)
            ax.tick_params(labelsize=8)

        self.ax_speed.set_title("Motor Speed", fontsize=9, fontweight="bold", pad=3)
        self.ax_speed.set_ylabel("RPM", fontsize=8)
        self.ax_speed.set_ylim(-150, 6450)

        self.ax_error.set_title("Error  (Setpoint - Actual)", fontsize=9,
                                fontweight="bold", pad=3)
        self.ax_error.set_ylabel("RPM", fontsize=8)

        self.ax_output.set_title("Correction Command to Motor", fontsize=9,
                                 fontweight="bold", pad=3)
        self.ax_output.set_ylabel("Command  (0-1)", fontsize=8)
        self.ax_output.set_ylim(-0.05, 1.05)

        self.ax_comp.set_title("PID + F  Components", fontsize=9,
                               fontweight="bold", pad=3)
        self.ax_comp.set_ylabel("Contribution", fontsize=8)
        self.ax_comp.set_xlabel("Time  (s)", fontsize=8)

        # Pre-create line objects - updated via set_data() for efficiency
        self.ln_sp,    = self.ax_speed.plot([], [], "r--", lw=1.6,
                                            label="Setpoint", alpha=0.85)
        self.ln_speed, = self.ax_speed.plot([], [], color="#1a6faf", lw=1.8,
                                            label="Actual speed")
        self.ax_speed.legend(loc="upper right", fontsize=7, framealpha=0.75)

        self.ln_err,   = self.ax_error.plot([], [], color="#e67e22", lw=1.5)
        self.ax_error.axhline(0, color="#555", lw=0.7, ls="--", alpha=0.6)

        self.ln_out,   = self.ax_output.plot([], [], color="#8e44ad", lw=1.5)
        self.ax_output.axhline(0, color="#555", lw=0.5, alpha=0.4)
        self.ax_output.axhline(1, color="#555", lw=0.5, alpha=0.4)

        self.ln_P,     = self.ax_comp.plot([], [], color="#2980b9", lw=1.3,
                                           label="P")
        self.ln_I,     = self.ax_comp.plot([], [], color="#27ae60", lw=1.3,
                                           label="I")
        self.ln_D,     = self.ax_comp.plot([], [], color="#c0392b", lw=1.3,
                                           label="D")
        self.ln_F,     = self.ax_comp.plot([], [], color="#e67e22", lw=1.3,
                                           label="F (ff)", ls="--")
        self.ax_comp.axhline(0, color="#555", lw=0.7, ls="--", alpha=0.6)
        self.ax_comp.legend(loc="upper right", fontsize=7, framealpha=0.75)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Simulation control ────────────────────────────────────────────────────
    def start_sim(self):
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Running")
            self._sim_loop()

    def stop_sim(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def reset_sim(self):
        was_running = self.running
        self.stop_sim()

        self.pid.reset()
        self.motor.reset()
        self.sim_time = 0.0

        for buf in (self.t_buf, self.speed_buf, self.sp_buf, self.err_buf,
                    self.out_buf, self.P_buf, self.I_buf, self.D_buf, self.F_buf):
            buf.clear()

        for ln in (self.ln_sp, self.ln_speed, self.ln_err, self.ln_out,
                   self.ln_P, self.ln_I, self.ln_D, self.ln_F):
            ln.set_data([], [])

        self.canvas.draw_idle()
        self._update_readouts(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        if was_running:
            self.start_sim()

    # ── Simulation loop ───────────────────────────────────────────────────────
    def _sim_loop(self):
        if not self.running:
            return

        dt_ms = max(1, int(round(self.params["dt_ms"].get())))
        dt_noise = np.random.normal(loc=0,scale=dt_ms/5)
        dt_ms = max(0,dt_ms + dt_noise) # clamp s.t. no values are negative
        dt    = dt_ms / 1000.0

        # Run enough steps to fill one GUI frame
        steps = int(max(1, np.ceil(GUI_UPDATE_MS // dt_ms)))

        # Setpoint is in RPM -> normalise to [0, 1] for the controller
        sp_rpm = float(np.clip(self.params["setpoint"].get(), 0.0, MAX_RPM))
        sp     = sp_rpm / MAX_RPM

        Kp  = self.params["kp"].get()
        Ki  = self.params["ki"].get()
        Kd  = self.params["kd"].get()
        F   = self.params["f"].get()
        tau = max(1e-6, self.params["tau"].get())

        last = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for _ in range(steps):
            meas = self.motor.get_noisy_speed_readout()
            output, P, I, D, ff, error = self.pid.compute(
                sp, meas, dt, Kp, Ki, Kd, F)
            self.motor.update(output, dt, tau)
            speed_rpm = self.motor.get_noisy_speed_readout()*MAX_RPM

            self.sim_time += dt
            self.t_buf.append(self.sim_time)
            self.speed_buf.append(speed_rpm)
            self.sp_buf.append(sp_rpm)
            self.err_buf.append(error * MAX_RPM)
            self.out_buf.append(output)
            self.P_buf.append(P)
            self.I_buf.append(I)
            self.D_buf.append(D)
            self.F_buf.append(ff)

            last = (speed_rpm, sp_rpm, error * MAX_RPM,
                    output, P, I, D, ff)

        self._update_readouts(*last)
        self._update_plots()
        self.root.after(GUI_UPDATE_MS, self._sim_loop)

    # ── Live text readouts ────────────────────────────────────────────────────
    def _update_readouts(self, speed, sp, err, out, P, I, D, ff):
        self.speed_var.set(f"Speed    : {speed:9.1f} RPM")
        self.sp_var   .set(f"Setpoint : {sp:9.1f} RPM")
        self.err_var  .set(f"Error    : {err:9.1f} RPM")
        self.out_var  .set(f"Output   : {out:9.4f}")
        self.p_var    .set(f"P        : {P:9.4f}")
        self.i_var    .set(f"I        : {I:9.4f}")
        self.d_var    .set(f"D        : {D:9.4f}")
        self.f_var    .set(f"F (ff)   : {ff:9.4f}")

    # ── Plot update ───────────────────────────────────────────────────────────
    def _update_plots(self):
        if not self.t_buf:
            return

        t     = np.asarray(self.t_buf)
        t_max = t[-1]
        t_min = max(0.0, t_max - HISTORY_SEC)

        mask = t >= t_min
        tw   = t[mask]

        def _w(buf):
            return np.asarray(buf)[mask]

        speed_w = _w(self.speed_buf)
        sp_w    = _w(self.sp_buf)
        err_w   = _w(self.err_buf)
        out_w   = _w(self.out_buf)
        P_w     = _w(self.P_buf)
        I_w     = _w(self.I_buf)
        D_w     = _w(self.D_buf)
        F_w     = _w(self.F_buf)

        x_lim = (t_min, t_max + max(0.5, (t_max - t_min) * 0.02))

        # Speed
        self.ln_sp   .set_data(tw, sp_w)
        self.ln_speed.set_data(tw, speed_w)
        self.ax_speed.set_xlim(*x_lim)

        # Error
        self.ln_err.set_data(tw, err_w)
        self.ax_error.set_xlim(*x_lim)
        if len(err_w) > 1:
            rng = np.ptp(err_w)
            pad = max(50.0, rng * 0.15)
            self.ax_error.set_ylim(err_w.min() - pad, err_w.max() + pad)

        # Output
        self.ln_out.set_data(tw, out_w)
        self.ax_output.set_xlim(*x_lim)

        # Components
        def _set(ln, show, xd, yd):
            ln.set_data(xd if show else [], yd if show else [])

        _set(self.ln_P, self.show_P.get(), tw, P_w)
        _set(self.ln_I, self.show_I.get(), tw, I_w)
        _set(self.ln_D, self.show_D.get(), tw, D_w)
        _set(self.ln_F, self.show_F.get(), tw, F_w)
        self.ax_comp.set_xlim(*x_lim)

        visible = []
        if self.show_P.get(): visible.append(P_w)
        if self.show_I.get(): visible.append(I_w)
        if self.show_D.get(): visible.append(D_w)
        if self.show_F.get(): visible.append(F_w)

        if visible:
            all_v = np.concatenate(visible)
            if len(all_v) > 1:
                rng = np.ptp(all_v)
                pad = max(0.02, rng * 0.15)
                self.ax_comp.set_ylim(all_v.min() - pad, all_v.max() + pad)

        self.canvas.draw_idle()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.state("zoomed")          # maximise on Windows
    except Exception:
        root.geometry("1400x900")

    style = ttk.Style(root)
    available = style.theme_names()
    for preferred in ("vista", "winnative", "clam", "alt", "default"):
        if preferred in available:
            style.theme_use(preferred)
            break

    app = PIDTunerApp(root)
    root.mainloop()