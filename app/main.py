import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from control.timeresp import step_info
from control import tf, step_response, bode, margin, poles, zeros, nyquist, nichols, rlocus
import control as ctrl
import io
import base64
import warnings
from functools import wraps
import threading
from flask import Flask, render_template, request, url_for



warnings.filterwarnings('ignore')

# Thread-safe kontrol mekanizması
warnings.filterwarnings('ignore')
control_lock = threading.Lock()


def thread_safe_control(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with control_lock:
            return func(*args, **kwargs)

    return wrapper


@thread_safe_control
def get_controller_response(controller_type, Gp, H, Kp, Ki, Kd, Ka=0):
    try:
        # Normalizasyon ekleyin (büyük sayılar için)
        max_den = max(abs(np.array(Gp.den[0][0])))
        if max_den > 1e8:
            scale = 10 ** np.floor(np.log10(max_den))
            Gp = ctrl.tf(np.array(Gp.num[0][0]) / scale, np.array(Gp.den[0][0]) / scale)

        if controller_type == "PID":
            C = ctrl.tf([Kd, Kp, Ki], [1, 0])
        else:  # PIDA
            # PIDA için doğru transfer fonksiyonu: (Ka*s^3 + Kd*s^2 + Kp*s + Ki) / s
            C = ctrl.tf([Ka, Kd, Kp, Ki], [1, 0])  # Payda [1, 0] olmalı!

        sys_closed = ctrl.feedback(C * Gp, H)

        # PIDA için zaman ölçeğini dinamik ayarla
        poles_closed = ctrl.poles(sys_closed)
        if len(poles_closed) > 0:
            max_pole = max(abs(np.real(poles_closed)))
            t_max = min(0.1, 10 / max_pole) if max_pole > 0 else 0.1
        else:
            t_max = 0.1

        t, y = step_response(sys_closed, T=np.linspace(0, t_max, 10000))
        return t, y

    except Exception as e:
        print(f"Error in controller response: {str(e)}")
        return np.linspace(0, 10, 100), np.zeros(100)


# Thread-safe grafik fonksiyonları
@thread_safe_control
def generate_pid_step_response(sys_closed):
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        t, y = step_response(sys_closed, T=np.linspace(0, 10, 1000))
        ax.plot(t, y, linewidth=2.5, color='#1f77b4')
        ax.set_title('PID - Step Response')
        ax.grid(True)

        # Performans metrikleri
        y_final = y[-1]
        if y_final != 0:
            rise_time = t[np.argmax(y >= 0.9 * y_final)] - t[np.argmax(y >= 0.1 * y_final)]
            settling_idx = np.argmax((np.abs(y - y_final) <= 0.02 * y_final)[::-1])
            settling_time = t[-settling_idx] if settling_idx != 0 else t[-1]
            overshoot = (np.max(y) - y_final) / y_final * 100
            steady_error = 1 - y[-1] / y_final

            metrics_text = (f"Rise Time: {rise_time:.4f} s\n"
                            f"Settling Time: {settling_time:.4f} s\n"
                            f"Overshoot: {overshoot:.2f}%\n"
                            f"Steady Error: {steady_error:.4f}")

            ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='bottom', horizontalalignment='right')

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pid_bode_diagram(sys_open):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    try:
        mag, phase, omega = bode(sys_open, dB=True, plot=False)

        ax1.semilogx(omega, 20 * np.log10(mag), linewidth=2, color='blue')
        ax1.set_ylabel('Magnitude [dB]')
        ax1.set_title('PID - Bode Diagram')
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)

        ax2.semilogx(omega, phase * 180 / np.pi, linewidth=2, color='green')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.set_ylabel('Phase [deg]')
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)

        gm, pm, wpc, wgc = margin(sys_open)
        gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf

        params_text = (f"Gain Margin: {gm_db:.2f} dB @ {wpc:.2f} rad/s\n"
                       f"Phase Margin: {pm:.2f}° @ {wgc:.2f} rad/s")

        ax2.text(0.98, 0.02, params_text, transform=ax2.transAxes,
                 fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        ax1.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')
        ax2.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')

@thread_safe_control
def generate_pida_step_response(sys_closed):
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        # Makaleye uygun sabit zaman ekseni (50 ms gibi)
        t = np.linspace(0, 0.05, 10000)  # Gerekirse 0.1 saniyeye çıkarılabilir
        t, y = step_response(sys_closed, T=t)

        # Step grafiği çizimi
        ax.plot(t, y, linewidth=2.5, color='#ff7f0e')
        ax.set_title('PIDA - Step Response')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

        # Performans metriklerini hesapla (daha doğru)
        info = step_info(sys_closed, T=t)
        rise_time = info.get("RiseTime", 0.0)
        settling_time = info.get("SettlingTime", 0.0)
        overshoot = info.get("Overshoot", 0.0)
        steady_state = info.get("SteadyStateValue", 1.0)
        steady_error = abs(1 - y[-1] / steady_state) if steady_state != 0 else float('nan')

        # Metin kutusu
        metrics_text = (f"Rise Time: {rise_time:.6f} s\n"
                        f"Settling Time: {settling_time:.6f} s\n"
                        f"Overshoot: {overshoot:.2f}%\n"
                        f"Steady Error: {steady_error:.6f}")

        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='bottom', horizontalalignment='right')

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')

    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pida_bode_diagram(sys_open):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    try:
        omega = np.logspace(-1, 6, 1000)
        mag, phase, omega = bode(sys_open, omega=omega, dB=True, plot=False)

        ax1.semilogx(omega, 20 * np.log10(mag), linewidth=2, color='#2ca02c')
        ax1.set_ylabel('Magnitude [dB]')
        ax1.set_title('PIDA - Bode Diagram')
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)

        ax2.semilogx(omega, phase * 180 / np.pi, linewidth=2, color='#d62728')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.set_ylabel('Phase [deg]')
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)

        # Performans metrikleri
        gm, pm, wpc, wgc = margin(sys_open)
        gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf
        params_text = (f"Gain Margin: {gm_db:.2f} dB @ {wpc:.2f} rad/s\n"
                       f"Phase Margin: {pm:.2f}° @ {wgc:.2f} rad/s")

        ax2.text(0.98, 0.02, params_text, transform=ax2.transAxes,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.85),
                 verticalalignment='bottom', horizontalalignment='right')

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        ax1.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')
        ax2.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')

    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)  # DPI artırıldı
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pid_pole_zero_map(sys_closed):
    """Thread-safe pole-zero map generation for PID"""
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        p = poles(sys_closed)
        z = zeros(sys_closed)

        ax.scatter(np.real(p), np.imag(p), marker='x', color='red', s=100, label='Poles')
        if len(z) > 0:
            ax.scatter(np.real(z), np.imag(z), marker='o', color='blue', s=100,
                       label='Zeros', facecolors='none', edgecolors='blue', linewidths=2)

        # Otomatik eksen ayarı (real/imag + padding)
        all_points = np.concatenate([p, z]) if len(z) > 0 else p
        real_parts = np.real(all_points)
        imag_parts = np.imag(all_points)

        rmin, rmax = np.min(real_parts), np.max(real_parts)
        imin, imax = np.min(imag_parts), np.max(imag_parts)

        x_padding = 0.2 * (rmax - rmin) if rmax != rmin else 1
        y_padding = 0.2 * (imax - imin) if imax != imin else 1

        ax.set_xlim(rmin - x_padding, rmax + x_padding)
        ax.set_ylim(imin - y_padding, imax + y_padding)

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title('PID - Pole-Zero Map')
        ax.set_xlabel('Real Axis (seconds⁻¹)')
        ax.set_ylabel('Imaginary Axis (seconds⁻¹)')
        ax.legend(loc='best')
        ax.grid(True)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pida_pole_zero_map(sys_closed):
    """Thread-safe pole-zero map generation for PIDA"""
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        # Kutuplar ve sıfırlar
        p = poles(sys_closed)
        z = zeros(sys_closed)

        # Noktaları çiz
        ax.scatter(np.real(p), np.imag(p), marker='x', color='red', s=100, label='Poles')
        if len(z) > 0:
            ax.scatter(np.real(z), np.imag(z), marker='o', color='blue', s=100,
                       label='Zeros', facecolors='none', edgecolors='blue', linewidths=2)

        # Otomatik eksen ayarı (real/imaginary)
        all_points = np.concatenate([p, z]) if len(z) > 0 else p
        real_parts = np.real(all_points)
        imag_parts = np.imag(all_points)

        rmin, rmax = np.min(real_parts), np.max(real_parts)
        imin, imax = np.min(imag_parts), np.max(imag_parts)

        x_padding = 0.2 * (rmax - rmin) if rmax != rmin else 1
        y_padding = 0.2 * (imax - imin) if imax != imin else 1

        ax.set_xlim(rmin - x_padding, rmax + x_padding)
        ax.set_ylim(imin - y_padding, imax + y_padding)

        # Eksen çizgileri ve başlık
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title('PIDA - Pole-Zero Map')
        ax.set_xlabel('Real Axis (seconds⁻¹)')
        ax.set_ylabel('Imaginary Axis (seconds⁻¹)')
        ax.legend(loc='best')
        ax.grid(True)

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')

    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pid_root_locus(sys_open):
    """Thread-safe root locus plot for PID"""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        roots, gains = rlocus(sys_open, plot=False)

        # Plot root locus branches
        for i in range(roots.shape[1]):
            ax.plot(np.real(roots[:, i]), np.imag(roots[:, i]), 'b-', linewidth=1, alpha=0.7,
                    label='Root Locus' if i == 0 else "")

        # Kutupları ve sıfırları çizdirir.
        p = poles(sys_open)
        z = zeros(sys_open)

        ax.scatter(np.real(p), np.imag(p), marker='x', color='red', s=100, label='Open-loop Poles')
        if len(z) > 0:
            ax.scatter(np.real(z), np.imag(z), marker='o', color='blue', s=100,
                       label='Open-loop Zeros', facecolors='none', linewidths=2)

        all_points = np.concatenate([p, z]) if len(z) > 0 else p
        real_parts = np.real(all_points)
        imag_parts = np.imag(all_points)

        rmin, rmax = np.min(real_parts), np.max(real_parts)
        imin, imax = np.min(imag_parts), np.max(imag_parts)

        x_padding = 0.2 * (rmax - rmin) if rmax != rmin else 1
        y_padding = 0.2 * (imax - imin) if imax != imin else 1

        ax.set_xlim(rmin - x_padding, rmax + x_padding)
        ax.set_ylim(imin - y_padding, imax + y_padding)

        # Genel grafik ayarları
        ax.set_title('PID - Root Locus')
        ax.set_xlabel('Real Axis (seconds⁻¹)')
        ax.set_ylabel('Imaginary Axis (seconds⁻¹)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='best')
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pida_root_locus(Gp, Kp, Ki, Kd, Ka):
    """PIDA için Root Locus: Sistem ve kontrolör parametreleriyle root locus çizer"""

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # 1. PIDA kontrolörü tanımla
        # (Ka*s^3 + Kd*s^2 + Kp*s + Ki) / s = tf([Ka, Kd, Kp, Ki], [1, 0])
        C = tf([Ka, Kd, Kp, Ki], [1, 0])

        # 2. Açık çevrim sistemi oluştur (H(s) = 1 olduğu varsayımıyla)
        sys_open = C * Gp

        # 3. Root Locus verilerini al
        roots, gains = rlocus(sys_open, kvect=np.linspace(0, 1000, 1000), plot=False)

        # 4. Root locus dallarını çiz
        for i in range(roots.shape[1]):
            ax.plot(np.real(roots[:, i]), np.imag(roots[:, i]), 'b-', linewidth=1, alpha=0.7,
                    label='Root Locus' if i == 0 else "")

        # 5. Kutupları ve sıfırları göster
        p = poles(sys_open)
        z = zeros(sys_open)
        ax.scatter(np.real(p), np.imag(p), marker='x', color='red', s=100, label='Open-loop Poles')
        if len(z) > 0:
            ax.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='blue',
                       s=100, linewidths=2, label='Open-loop Zeros')

        # 6. Otomatik eksen ölçekleme
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('PIDA - Root Locus')
        ax.set_xlabel('Real Axis (1/s)')
        ax.set_ylabel('Imaginary Axis (rad/s)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='best')
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.axvline(0, color='black', linewidth=0.8, linestyle=':')

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pid_nyquist_diagram(sys_open):
    """Thread-safe Nyquist diagram for PID"""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        omega = np.logspace(-2, 4, 5000)  # Geniş frekans aralığı
        response = np.array([sys_open(1j * w) for w in omega])
        real = np.real(response)
        imag = np.imag(response)

        # Nyquist eğrileri
        ax.plot(real, imag, 'b-', linewidth=1.5, label='Nyquist Curve')
        ax.plot(real, -imag, 'r--', linewidth=1, alpha=0.7, label='Mirror Image')
        ax.plot(-1, 0, 'ro', markersize=8, label='Critical Point (-1,0)')

        #Otomatik ve simetrik eksen ayarı
        x_min, x_max = np.min(real), np.max(real)
        y_all = np.concatenate([imag, -imag])  # Tüm imag verisi
        y_min, y_max = np.min(y_all), np.max(y_all)

        x_pad = 0.1 * (x_max - x_min) if x_max != x_min else 1
        y_pad = 0.1 * (y_max - y_min) if y_max != y_min else 1

        #Imaginer eksen simetrik olacak şekilde ayarlanır
        y_abs = max(abs(y_min), abs(y_max))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(-y_abs - y_pad, y_abs + y_pad)

        ax.set_title('PID - Nyquist Diagram')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()

        # Kararlılık çemberi (isteğe bağlı)
        stability_circle = plt.Circle((-1, 0), 0.5, color='gray',
                                      fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(stability_circle)

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pida_nyquist_diagram(sys_open):
    """Thread-safe Nyquist diagram for PIDA """
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Geniş ama kontrollü frekans aralığı
        omega = np.logspace(-1, 4, 10000)  # 0.1 rad/s ile 10,000 rad/s arası
        response = np.array([sys_open(1j * w) for w in omega])
        real = np.real(response)
        imag = np.imag(response)

        # Ana Nyquist eğrisi ve simetriği
        ax.plot(real, imag, 'b-', linewidth=1.5, label='Nyquist Curve')
        ax.plot(real, -imag, 'r--', linewidth=1, alpha=0.7, label='Mirror Image')
        ax.plot(-1, 0, 'ro', markersize=8, label='Critical Point (-1,0)')

        # Otomatik eksen ayarı (padding’li)
        x_min, x_max = np.min(real), np.max(real)
        y_min, y_max = np.min(imag), np.max(imag)
        x_pad = 0.1 * (x_max - x_min) if x_max != x_min else 1
        y_pad = 0.1 * (y_max - y_min) if y_max != y_min else 1
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_title('PIDA - Nyquist Diagram')
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()

        # Kararlılık çemberi
        stability_circle = plt.Circle((-1, 0), 0.5, color='gray',
                                      fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(stability_circle)

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pid_nichols_chart(sys_open):
    """Thread-safe Nichols chart for PID"""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Frekans cevabını hesaplar
        freq = np.logspace(-3, 3, 5000)
        mag = []
        phase = []

        for w in freq:
            if w == 0:
                w = 1e-10  # Sıfıra bölünmeyi engeller
            s = 1j * w
            response = sys_open(s)

            if isinstance(response, (float, complex)):
                mag_val = 20 * np.log10(np.abs(response))
                phase_val = np.angle(response) * 180 / np.pi
            else:
                mag_val = 20 * np.log10(np.abs(response[0][0]))
                phase_val = np.angle(response[0][0]) * 180 / np.pi

            mag.append(mag_val)
            phase.append(phase_val)

        mag = np.array(mag)
        phase = np.array(phase)

        # Nichols chart için Gridleri ekler
        def add_nichols_grid(ax):
            dB_levels = np.array([-40, -30, -20, -12, -6, -3, -1, 0, 1, 3, 6, 12, 20])
            grid_style = {'color': '0.7', 'linestyle': '--', 'linewidth': 0.6, 'alpha': 0.7}

            for dB in dB_levels:
                M = 10 ** (dB / 20.0)
                phase_rad = np.linspace(-2 * np.pi, 0, 1000)
                mag_plot = 20 * np.log10(M / np.abs(1 - M * np.exp(-1j * phase_rad)))
                phase_plot = np.degrees(phase_rad + np.angle(1 - M * np.exp(-1j * phase_rad)))
                phase_plot = np.where(phase_plot > 0, phase_plot - 360, phase_plot)

                # Handle discontinuities
                breaks = np.where(np.abs(np.diff(phase_plot)) > 180)[0] + 1
                phase_segs = np.split(phase_plot, breaks)
                mag_segs = np.split(mag_plot, breaks)

                for p, m in zip(phase_segs, mag_segs):
                    if len(p) > 2:
                        ax.plot(p, m, **grid_style)

        add_nichols_grid(ax)

        # Sistem cevabını çizdirir.
        ax.plot(phase, mag, 'b-', linewidth=1.5, label='System Response')
        ax.set_xlim(-360, 0)
        ax.set_ylim(-40, 40)
        ax.set_title('PID - Nichols Chart')
        ax.set_xlabel('Open Loop Phase [deg]')
        ax.set_ylabel('Open Loop Gain [dB]')
        ax.legend()
        ax.grid(True)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


@thread_safe_control
def generate_pida_nichols_chart(sys_open):
    """Thread-safe Nichols chart for PIDA"""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # PIDA için farklı frekans aralığı
        freq = np.logspace(-1, 5, 10000)  # Daha geniş frekans aralığı
        mag = []
        phase = []

        for w in freq:
            if w == 0:
                w = 1e-10  # Sıfıra bölünmeyi engeller
            s = 1j * w
            response = sys_open(s)

            # Farklı cevapları ele alır
            if isinstance(response, (float, complex)):
                mag_val = 20 * np.log10(np.abs(response))
                phase_val = np.angle(response) * 180 / np.pi
            else:
                mag_val = 20 * np.log10(np.abs(response[0][0]))
                phase_val = np.angle(response[0][0]) * 180 / np.pi

            mag.append(mag_val)
            phase.append(phase_val)

        mag = np.array(mag)
        phase = np.array(phase)

        # Nichols chart için özel gridleri ekler
        def add_nichols_grid(ax):
            dB_levels = np.array([-40, -30, -20, -12, -6, -3, -1, 0, 1, 3, 6, 12, 20])
            grid_style = {'color': '0.7', 'linestyle': '--', 'linewidth': 0.6, 'alpha': 0.7}

            for dB in dB_levels:
                M = 10 ** (dB / 20.0)
                phase_rad = np.linspace(-2 * np.pi, 0, 1000)
                mag_plot = 20 * np.log10(M / np.abs(1 - M * np.exp(-1j * phase_rad)))
                phase_plot = np.degrees(phase_rad + np.angle(1 - M * np.exp(-1j * phase_rad)))
                phase_plot = np.where(phase_plot > 0, phase_plot - 360, phase_plot)

                breaks = np.where(np.abs(np.diff(phase_plot)) > 180)[0] + 1
                phase_segs = np.split(phase_plot, breaks)
                mag_segs = np.split(mag_plot, breaks)

                for p, m in zip(phase_segs, mag_segs):
                    if len(p) > 2:
                        ax.plot(p, m, **grid_style)

        add_nichols_grid(ax)

        # Plot system response
        ax.plot(phase, mag, 'b-', linewidth=1.5, label='System Response')
        ax.set_xlim(-360, 0)
        ax.set_ylim(-80, 80)
        ax.set_title('PIDA - Nichols Chart')
        ax.set_xlabel('Open Loop Phase [deg]')
        ax.set_ylabel('Open Loop Gain [dB]')
        ax.legend()
        ax.grid(True)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
    finally:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')


def parse_scientific_notation(s):
    """Convert European format (comma decimal, e exponent) to float"""
    try:
        s = s.replace(',', '.')

        s = s.replace(' ', '')

        return float(s)
    except ValueError:
        print(f"Error parsing number: {s}")
        return 0.0


def parse_polynomial(poly_str):
    """Parse polynomial string with European number format"""
    if not poly_str.strip():
        return []

    terms = poly_str.split()

    coefficients = []
    for term in terms:
        try:
            coeff = parse_scientific_notation(term)
            coefficients.append(coeff)
        except Exception as e:
            print(f"Error parsing term: {term}, Error: {str(e)}")
            coefficients.append(0.0)

    return coefficients


def normalize_system(Gp):
    """Büyük sayılar içeren sistemleri normalize eder"""
    num = np.array(Gp.num[0][0])
    den = np.array(Gp.den[0][0])

    # Pay ve paydadaki maksimum mutlak değeri bul
    max_num = max(abs(num))
    max_den = max(abs(den))
    scale = max(max_num, max_den)

    if scale > 1e6:  # Sadece çok büyük sayılar için ölçekle
        return ctrl.tf(num / scale, den / scale)
    return Gp


def get_time_scale(sys_closed):
    """Sistem dinamiklerine göre uygun zaman ölçeğini belirler"""
    poles_closed = ctrl.poles(sys_closed)
    if len(poles_closed) > 0:
        max_pole = max(abs(np.real(poles_closed)))
        t_max = min(0.1, 10 / max_pole) if max_pole > 0 else 0.1
    else:
        t_max = 0.1
    return np.linspace(0, t_max, 10000)


def create_controller(controller_type, Kp, Ki, Kd, Ka=0):
    """Doğru kontrolör transfer fonksiyonunu oluşturur"""
    if controller_type == "PID":
        return ctrl.tf([Kd, Kp, Ki], [1, 0])
    else:  # PIDA
        # Doğru form: (Ka*s^3 + Kd*s^2 + Kp*s + Ki) / s
        return ctrl.tf([Ka, Kd, Kp, Ki], [1, 0])


app = Flask(__name__)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')


def compare_controllers(Gp, H, controllers, plot_type):
    if plot_type == "step":
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    performance_data = []

    if plot_type == "step":
        performance_data.append(['Controller', 'Rise Time (s)', 'Settling Time (s)', 'Overshoot (%)', 'Steady State Error'])
    else:
        performance_data.append(['Controller', 'Gain Margin (dB)', 'Phase Margin (°)', 'Gain Crossover (rad/s)',
                                 'Phase Crossover (rad/s)'])

    for i, (c_type, Kp, Ki, Kd, Ka) in enumerate(controllers):
        if c_type == "OFF":
            continue  # "OFF" durumundaki denetleyicileri atla

        try:
            if c_type == "PID":
                C = ctrl.tf([Kd, Kp, Ki], [1, 0])
                label = f"PID (Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f})"
            elif c_type == "PIDA":
                C = ctrl.tf([Ka, Kd, Kp, Ki], [1, 0, 0])
                label = f"PIDA (Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}, Ka={Ka:.2f})"
            else:
                continue

            sys_open = C * Gp
            sys_closed = ctrl.feedback(sys_open, H)

            t, y = step_response(sys_closed)
            y_final = np.mean(y[-100:])

            rise_time = t[np.argmax(y >= 0.9 * y_final)] - t[np.argmax(y >= 0.1 * y_final)]
            settling_idx = np.argmax((np.abs(y - y_final) <= 0.02 * y_final)[::-1])
            settling_time = t[-settling_idx] if settling_idx != 0 else t[-1]
            overshoot = (np.max(y) - y_final) / y_final * 100 if y_final != 0 else 0
            steady_error = 1 - y[-1] / y_final if y_final != 0 else np.nan

            if plot_type == "step":
                ax.plot(t, y, color=colors[i], linewidth=2, label=label)
                performance_data.append([
                    label,
                    f"{rise_time:.3f}",
                    f"{settling_time:.3f}",
                    f"{overshoot:.2f}",
                    f"{steady_error:.4f}"
                ])
            else:
                mag, phase, omega = bode(sys_open, dB=True, plot=False)
                gm, pm, wpc, wgc = margin(sys_open)
                gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf

                ax1.semilogx(omega, 20 * np.log10(mag), color=colors[i], linewidth=2, label=label)
                ax2.semilogx(omega, phase * 180 / np.pi, color=colors[i], linewidth=2)

                performance_data.append([
                    label,
                    f"{gm_db:.2f}" if gm_db != np.inf else "∞",
                    f"{pm:.2f}",
                    f"{wgc:.2f}",
                    f"{wpc:.2f}" if wpc != np.inf else "∞",
                ])
        except Exception as e:
            print(f"Error processing controller {i+1}: {str(e)}")
            continue

    if plot_type == "step":
        ax.set_title('Step Response Comparison', pad=15)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
    else:
        ax1.set_title('Bode Magnitude Comparison', pad=15)
        ax1.set_ylabel('Magnitude [dB]')
        ax2.set_title('Bode Phase Comparison', pad=15)
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.set_ylabel('Phase [deg]')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

    plt.tight_layout()
    return plot_to_base64(fig), performance_data


# Default değerleri global olarak tanımla (fonksiyon dışında)
DEFAULT_VALUES = {
    'num': '1',
    'den': '1 1',
    'h_num': '1',
    'h_den': '1',
    'kp': '1.0',
    'ki': '0.1',
    'kd': '0.01',
    'ka': '0.0',
    'controller_type': 'PID',
    'compare_type_1': 'PID',
    'compare_kp_1': '1.0',
    'compare_ki_1': '0.1',
    'compare_kd_1': '0.01',
    'compare_ka_1': '0.0',
    'compare_type_2': 'PID',
    'compare_kp_2': '1.0',
    'compare_ki_2': '0.1',
    'compare_kd_2': '0.01',
    'compare_ka_2': '0.0',
    'compare_type_3': 'PID',
    'compare_kp_3': '1.0',
    'compare_ki_3': '0.1',
    'compare_kd_3': '0.01',
    'compare_ka_3': '0.0'
}


@app.route('/', methods=['GET', 'POST'])
def index():
    values = DEFAULT_VALUES.copy()
    plots = {}
    performance_data = []
    comparison_plot = None
    error = None

    if request.method == 'POST':
        try:
            action = request.form.get("action", "")
            controller_type = request.form.get("controller_type", "PID")
            values['controller_type'] = controller_type

            # Sayısal parametreleri al
            Kp = parse_scientific_notation(request.form.get("kp", "1"))
            Ki = parse_scientific_notation(request.form.get("ki", "0.1"))
            Kd = parse_scientific_notation(request.form.get("kd", "0.01"))
            Ka = parse_scientific_notation(request.form.get("ka", "0"))

            # Önceki compare değerlerini koru (yeni eklenen kısım)
            prev_values = {
                'compare_type_1': values.get('compare_type_1', 'PID'),
                'compare_kp_1': values.get('compare_kp_1', str(Kp)),
                'compare_ki_1': values.get('compare_ki_1', str(Ki)),
                'compare_kd_1': values.get('compare_kd_1', str(Kd)),
                'compare_ka_1': values.get('compare_ka_1', "0"),
                'compare_type_2': values.get('compare_type_2', 'PID'),
                'compare_kp_2': values.get('compare_kp_2', str(Kp)),
                'compare_ki_2': values.get('compare_ki_2', str(Ki)),
                'compare_kd_2': values.get('compare_kd_2', str(Kd)),
                'compare_ka_2': values.get('compare_ka_2', "0"),
                'compare_type_3': values.get('compare_type_3', 'PID'),
                'compare_kp_3': values.get('compare_kp_3', str(Kp)),
                'compare_ki_3': values.get('compare_ki_3', str(Ki)),
                'compare_kd_3': values.get('compare_kd_3', str(Kd)),
                'compare_ka_3': values.get('compare_ka_3', "0")
            }

            # Formdan gelen string değerleri al ve kaydet
            values.update({
                'num': request.form.get("num", ""),
                'den': request.form.get("den", ""),
                'h_num': request.form.get("h_num", ""),
                'h_den': request.form.get("h_den", ""),
                'kp': Kp,
                'ki': Ki,
                'kd': Kd,
                'ka': Ka,
                'compare_num': request.form.get('compare_num', values.get('compare_num', DEFAULT_VALUES['num'])),
                'compare_den': request.form.get('compare_den', values.get('compare_den', DEFAULT_VALUES['den'])),
                'compare_h_num': request.form.get('compare_h_num',
                                                  values.get('compare_h_num', DEFAULT_VALUES['h_num'])),
                'compare_h_den': request.form.get('compare_h_den',
                                                  values.get('compare_h_den', DEFAULT_VALUES['h_den'])),

                # Controller 1 - Önceki değerleri koru, yeni değer gelmemişse
                'compare_type_1': request.form.get("compare_type_1", prev_values['compare_type_1']),
                'compare_kp_1': request.form.get("compare_kp_1", prev_values['compare_kp_1']),
                'compare_ki_1': request.form.get("compare_ki_1", prev_values['compare_ki_1']),
                'compare_kd_1': request.form.get("compare_kd_1", prev_values['compare_kd_1']),
                'compare_ka_1': request.form.get("compare_ka_1", prev_values['compare_ka_1']),

                # Controller 2 ve 3 için aynı mantık
                'compare_type_2': request.form.get("compare_type_2", values.get('compare_type_2', 'PID')),
                'compare_kp_2': request.form.get("compare_kp_2", values.get('compare_kp_2', str(Kp))),
                'compare_ki_2': request.form.get("compare_ki_2", values.get('compare_ki_2', str(Ki))),
                'compare_kd_2': request.form.get("compare_kd_2", values.get('compare_kd_2', str(Kd))),
                'compare_ka_2': request.form.get("compare_ka_2", values.get('compare_ka_2', "0")),

                'compare_type_3': request.form.get("compare_type_3", values.get('compare_type_3', 'PID')),
                'compare_kp_3': request.form.get("compare_kp_3", values.get('compare_kp_3', str(Kp))),
                'compare_ki_3': request.form.get("compare_ki_3", values.get('compare_ki_3', str(Ki))),
                'compare_kd_3': request.form.get("compare_kd_3", values.get('compare_kd_3', str(Kd))),
                'compare_ka_3': request.form.get("compare_ka_3", values.get('compare_ka_3', "0"))
            })

            # Karşılaştırma analizi
            if action in ("compare_step", "compare_bode"):
                compare_num = parse_polynomial(values['compare_num'])
                compare_den = parse_polynomial(values['compare_den'])
                compare_h_num = parse_polynomial(values['compare_h_num'])
                compare_h_den = parse_polynomial(values['compare_h_den'])

                if not any(compare_den) or all(x == 0 for x in compare_den):
                    error = "Gp(s) denominator cannot be zero or empty."
                elif not any(compare_h_den) or all(x == 0 for x in compare_h_den):
                    error = "H(s) denominator cannot be zero or empty."
                else:
                    Gp = ctrl.tf(compare_num, compare_den)
                    H = ctrl.tf(compare_h_num, compare_h_den)

                    controllers = []
                    for i in range(1, 4):
                        c_type = values[f'compare_type_{i}']
                        if c_type == "OFF":
                            continue

                        kp = parse_scientific_notation(values[f'compare_kp_{i}'])
                        ki = parse_scientific_notation(values[f'compare_ki_{i}'])
                        kd = parse_scientific_notation(values[f'compare_kd_{i}'])
                        ka = parse_scientific_notation(values[f'compare_ka_{i}'])
                        controllers.append((c_type, kp, ki, kd, ka))

                    plot_type = 'step' if action == 'compare_step' else 'bode'
                    comparison_plot, performance_data = compare_controllers(Gp, H, controllers, plot_type)

            #Tekli analiz
            elif action == "analyze":
                num = parse_polynomial(values['num'])
                den = parse_polynomial(values['den'])
                h_num = parse_polynomial(values['h_num'])
                h_den = parse_polynomial(values['h_den'])

                if not any(num) or not any(den):
                    error = "Gp(s) numerator and denominator cannot be empty."
                elif not any(h_num) or not any(h_den):
                    error = "H(s) numerator and denominator cannot be empty."
                else:
                    Gp = ctrl.tf(num, den)
                    H = ctrl.tf(h_num, h_den)

                    C = create_controller(controller_type, Kp, Ki, Kd, Ka)
                    sys_open = C * Gp
                    sys_closed = ctrl.feedback(sys_open, H)

                    if controller_type == "PID":
                        plots["step"] = generate_pid_step_response(sys_closed)
                        plots["bode"] = generate_pid_bode_diagram(sys_open)
                        plots["pole_zero"] = generate_pid_pole_zero_map(sys_closed)
                        plots["root_locus"] = generate_pid_root_locus(sys_open)
                        plots["nyquist"] = generate_pid_nyquist_diagram(sys_open)
                        plots["nichols"] = generate_pid_nichols_chart(sys_open)

                    else:
                        plots["step"] = generate_pida_step_response(sys_closed)
                        plots["bode"] = generate_pida_bode_diagram(sys_open)
                        plots["pole_zero"] = generate_pida_pole_zero_map(sys_closed)
                        plots["root_locus"] = generate_pida_root_locus(Gp, Kp, Ki, Kd, Ka)
                        plots["nyquist"] = generate_pida_nyquist_diagram(sys_open)
                        plots["nichols"] = generate_pida_nichols_chart(sys_open)

        except Exception as e:
            error = f"Bir hata oluştu: {str(e)}"

    return render_template('index.html',
                            values=values,
                            plots=plots,
                            comparison_plot=comparison_plot,
                            performance_data=performance_data,
                            error=error,
                            url_for=url_for)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)