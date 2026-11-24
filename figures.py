"""
figures.py  (الإصدار ٢) — يرسم كل أشكال البحث من ملفات النتائج.
لوحة ألوان آمنة لعمى الألوان (تم التحقق منها بأداة فحص).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import physics as ph
import simulate as S

C_RAD, C_THERM, C_FUSE = "#0072B2", "#D55E00", "#009E73"
C_INK, C_MUTED, C_GRID = "#1a1a1a", "#5c5c5c", "#d9d9d6"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": C_MUTED,
    "axes.linewidth": 0.7, "axes.labelcolor": C_INK, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "text.color": C_INK, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "legend.frameon": False,
})
CH_STYLE = {"radiation": ("Radiation channel only", C_RAD, "o", "-"),
            "thermal":   ("Thermal channel only",   C_THERM, "s", "--"),
            "fused":     ("Fused (both channels)",  C_FUSE, "^", "-")}


def fig1_signals(path="fig1_signals.png"):
    """الحالات الثلاث: سليمة / احتباس وريدي / تسرّب."""
    scen = [("Normal injection", 0.0, 0.003, 45.0),
            ("Venous retention", 0.0, 0.08, 90.0),
            ("Extravasation (10% of dose)", 0.10, 0.003, 45.0)]
    fig, ax = plt.subplots(2, 3, figsize=(9.4, 4.8), sharex=True, sharey="row")
    for j, (name, f, vr, tau) in enumerate(scen):
        rng = np.random.default_rng(4)
        pt = dict(depth=0.5, depth_ref=0.5, sys_asym=1.0, volume=6.0,
                  room_drift=0.0, perfusion=1.0, clear_h=6.0)
        r_i, r_r, T_i, T_r = S.simulate_curves("Lu-177", f, vr, rng,
                                               venous_tau_s=tau, pt=pt)
        a = ax[0, j]
        a.plot(S.T_GRID, r_i / 1e3, color=C_RAD, label="Injection arm")
        a.plot(S.T_GRID, r_r / 1e3, color=C_MUTED, ls="--", lw=1.4,
               label="Reference arm")
        a.set_yscale("log"); a.set_title(name, pad=6); a.grid(True, axis="y")
        b = ax[1, j]
        b.plot(S.T_GRID, T_i, color=C_THERM, label="Injection site")
        b.plot(S.T_GRID, T_r, color=C_MUTED, ls="--", lw=1.4,
               label="Reference site")
        b.grid(True, axis="y"); b.set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Count rate\n(kcps, log)")
    ax[1, 0].set_ylabel("Skin temperature\n(°C)")
    ax[0, 0].legend(loc="upper right", fontsize=7.5)
    ax[1, 0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Simulated dual-channel signals for the three injection classes",
                 y=1.02, fontsize=10.5)
    fig.savefig(path); plt.close(fig)
    return path


def fig2_latency(res, path="fig2_latency.png"):
    """أصغر تسرّب قابل للكشف مقابل زمن الإنذار — لكل قناة (Lu-177)."""
    wins = res["windows"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for ch, (lbl, col, mk, ls) in CH_STYLE.items():
        y = [res["mdf"]["Lu-177"][str(w)][ch] for w in wins]
        y = [np.nan if v is None else 100 * v for v in y]
        ax.plot(wins, y, marker=mk, ms=6, color=col, ls=ls, label=lbl)
    ax.set_xscale("log"); ax.set_xticks(wins); ax.set_xticklabels(wins)
    ax.set_xlabel("Decision window = alarm latency (s)")
    ax.set_ylabel("Smallest detectable\nextravasated fraction (%)")
    ax.set_title("Detectability vs. alarm latency, Lu-177\n"
                 "(90% sensitivity at 95% specificity; missing point = not detectable)",
                 pad=8)
    ax.grid(True); ax.legend(fontsize=8)
    fig.savefig(path); plt.close(fig)
    return path


def fig3_false_alarm(res, path="fig3_false_alarm.png"):
    """الإنذار الكاذب على حقنات سليمة مقابل الاحتباس الوريدي (نافذة 30 ث)."""
    fa = res["false_alarm"]
    vrs = sorted(float(k) for k in fa["radiation"])
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for ch, (lbl, col, mk, ls) in CH_STYLE.items():
        ax.plot([100 * v for v in vrs], [100 * fa[ch][str(v)] for v in vrs],
                marker=mk, ms=6, color=col, ls=ls, label=lbl)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("Venous retention near puncture site (% of dose)")
    ax.set_ylabel("False-alarm rate on\nnon-extravasated injections (%)")
    ax.set_title("False alarms at a 30-s decision window\n"
                 "(sensitivity fixed at 90% for a 10% leak)", pad=8)
    ax.grid(True); ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path); plt.close(fig)
    return path


def fig4_confusion(ai, path="fig4_ai_confusion.png"):
    """مصفوفة الالتباس للانحدار اللوجستي: الاختبار العادي والصعب."""
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for a, split, title in [(ax[0], "test", "Standard test set"),
                            (ax[1], "hard", "Hard test set (unseen conditions)")]:
        cm = np.array(ai["Logistic regression"][split]["confusion"], float)
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        a.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{pct[i, j]:.0f}%", ha="center", va="center",
                       color="white" if pct[i, j] > 55 else C_INK, fontsize=9)
        a.set_xticks(range(3)); a.set_yticks(range(3))
        a.set_xticklabels(["Normal", "Venous\nretention", "Extrav."])
        a.set_yticklabels(["Normal", "Venous\nretention", "Extrav."])
        acc = 100 * ai["Logistic regression"][split]["accuracy"]
        a.set_xlabel("Predicted"); a.set_title(f"{title}\naccuracy {acc:.1f}%", pad=6)
        for s in a.spines.values():
            s.set_visible(False)
    ax[0].set_ylabel("True class")
    fig.suptitle("AI classifier (logistic regression): confusion matrices",
                 y=1.03, fontsize=10.5)
    fig.savefig(path); plt.close(fig)
    return path


def fig5_comparison(ai, path="fig5_ai_vs_baseline.png"):
    """هل تفوّق الذكاء الاصطناعي على القواعد البسيطة؟ (الاختبار الصعب)"""
    names = ["Radiation only", "Hand rule (baseline)", "Random forest",
             "Logistic regression"]
    vals = [100 * ai[n]["hard"]["sens_at_5far"] for n in names]
    cols = [C_MUTED, C_MUTED, C_RAD, C_FUSE]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.barh(names, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Extravasation sensitivity at a 5% false-alarm rate (%)")
    ax.set_title("Hard test set: small leaks, deep veins, strong venous retention",
                 pad=8)
    ax.grid(True, axis="x")
    fig.savefig(path); plt.close(fig)
    return path


def fig6_weights(ai, path="fig6_ai_weights.png"):
    """ماذا تعلّم النموذج؟ أوزان سمات فئة 'تسرّب'."""
    names = ai["feature_names"]
    w = np.array(ai["lr_weights_extrav"])
    order = np.argsort(np.abs(w))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh([names[i] for i in order], w[order],
            color=[C_RAD if names[i].startswith("rad") else C_THERM for i in order],
            height=0.6)
    ax.axvline(0, color=C_MUTED, lw=0.8)
    ax.set_xlabel("Weight toward 'extravasation' (standardized features)")
    ax.set_title("What the model learned (blue = radiation, orange = thermal)",
                 pad=8)
    ax.grid(True, axis="x")
    fig.savefig(path); plt.close(fig)
    return path
