"""
physics.py  
=========================================================
النموذج الفيزيائي الأمامي للصقة الذكية لرصد التسرّب الإشعاعي
لا يحتوي هذا الملف أي تصنيف ولا تعلّم آلي:
يأخذ "السبب" (كم تسرّب؟) ويُرجع "النتيجة" (ماذا تقرأ المستشعرات؟)
"""
import numpy as np

# ------------------------------------------------------------------
# 1. خصائص النظائر العلاجية
# ------------------------------------------------------------------
# A0_MBq      : النشاط العلاجي للجرعة الواحدة (ميغابكريل)
# gamma_yield : فوتونات قابلة للكشف خارج الجلد لكل تحلل
# mu_tissue   : معامل التوهين الخطي للنسيج الرخو (1/سم)
ISOTOPES = {
    # Lutathera: 7.4 GBq لكل دورة علاجية
    "Lu-177": dict(A0_MBq=7400.0, gamma_yield=0.166, mu_tissue=0.150,
                   E_keV=208, half_life_h=159.6),
    # [١] Zevalin الوريدي: 14.8 MBq/kg، بحد أقصى 1184 MBq (نشرة الدواء)
    "Y-90":   dict(A0_MBq=1184.0, gamma_yield=0.014, mu_tissue=0.145,
                   E_keV=200, half_life_h=64.1),
    # Xofigo: 55 kBq/kg × 70 kg ≈ 4 MBq
    "Ra-223": dict(A0_MBq=4.0,    gamma_yield=0.250, mu_tissue=0.135,
                   E_keV=269, half_life_h=273.6),
}

# ------------------------------------------------------------------
# 2. هندسة الكاشف وكفاءته
# ------------------------------------------------------------------
DET = dict(
    area_cm2      = 0.09,   # بلورة 3×3 مم — نفس حجم بلورة نظام Lara
    eff_intrinsic = 0.30,   # ⚠ تقدير: نسبة الفوتونات الواصلة التي تُحسب
    hydrogel_cm   = 0.20,   # سماكة الهيدروجيل
    mu_hydrogel   = 0.150,  # الهيدروجيل ماء أساساً → توهين قريب من النسيج
    dead_time_us  = 1.0,    # الزمن الميت للكاشف
    g_systemic    = 2.0e-5, # ⚠ تقدير: ما يراه الكاشف من نشاط باقي الجسم
)

def geometric_efficiency(depth_cm, area_cm2):
    """قانون التربيع العكسي: نسبة الفوتونات المتجهة نحو الكاشف."""
    return area_cm2 / (4.0 * np.pi * np.maximum(depth_cm, 0.05) ** 2)

def observed_rate(true_rate, dead_time_us):
    """الزمن الميت (الكاشيرة المزحومة): المعدل المرصود لا يتجاوز 1/τ."""
    tau = dead_time_us * 1e-6
    return true_rate / (1.0 + true_rate * tau)

# ------------------------------------------------------------------
# 3. القناة الإشعاعية: معدل العد عند الكاشفين لحظة بلحظة
# ------------------------------------------------------------------
def radiation_tac(t_s, iso, frac_extrav, depth_cm, bolus_t0=10.0,
                  bolus_width=4.0, local_clear_h=6.0,
                  depth_ref_cm=None, sys_asym=1.0,
                  venous_residual=0.0, venous_tau_s=45.0):
    """
    يُرجع (معدل العد عند ذراع الحقن، معدل العد عند الذراع المرجعية).

    frac_extrav     : الكسر المتسرّب من الجرعة (0 = لا تسرّب)
    venous_residual : كسر الجرعة المحتبس داخل الوريد قرب موقع الوخز
    venous_tau_s    : [٤] زمن غسل الاحتباس الوريدي (ثانية)
    """
    p  = ISOTOPES[iso]
    A0 = p["A0_MBq"] * 1e6
    if depth_ref_cm is None:
        depth_ref_cm = depth_cm

    def k_loc(d):
        att = np.exp(-p["mu_tissue"] * d - DET["mu_hydrogel"] * DET["hydrogel_cm"])
        return (p["gamma_yield"] * DET["eff_intrinsic"] * att
                * geometric_efficiency(d, DET["area_cm2"]))

    k_inj = k_loc(depth_cm)
    k_ref = k_loc(depth_ref_cm)
    k_sys = p["gamma_yield"] * DET["eff_intrinsic"] * DET["g_systemic"]

    bolus = np.exp(-0.5 * ((t_s - bolus_t0) / bolus_width) ** 2)
    step  = (t_s >= bolus_t0).astype(float)
    dt    = np.maximum(t_s - bolus_t0, 0.0)

    # ① التسرّب: يبقى (نصفه يزول في ساعات)
    lam_local = np.log(2) / (local_clear_h * 3600.0)
    retained  = frac_extrav * A0 * np.exp(-lam_local * dt) * step
    # ② الاحتباس الوريدي: يُغسل خلال دقيقة أو دقيقتين
    venous    = venous_residual * A0 * np.exp(-dt / venous_tau_s) * step
    # ③ النشاط الدوراني في باقي الجسم: يصعد ثم يستقر
    A_circ    = (1.0 - frac_extrav) * A0 * (1 - np.exp(-dt / 60.0)) * step

    r_inj = (k_inj * (retained + venous + 0.02 * (1 - frac_extrav) * A0 * bolus)
             + k_sys * A_circ * sys_asym)
    r_ref = (k_sys * A_circ / sys_asym
             + k_ref * 0.004 * (1 - frac_extrav) * A0
               * np.exp(-0.5 * ((t_s - bolus_t0 - 12.0) / 8.0) ** 2))

    r_inj += 0.3   # الخلفية الإشعاعية الطبيعية
    r_ref += 0.3
    return (observed_rate(r_inj, DET["dead_time_us"]),
            observed_rate(r_ref, DET["dead_time_us"]))

# ------------------------------------------------------------------
# 4. القناة الحرارية: حرارة الجلد لحظة بلحظة
# ------------------------------------------------------------------
THERM = dict(
    T_skin_base  = 33.6,  # حرارة جلد الساعد قبل الحقن — Kageyama 2026
    T_fluid      = 22.0,  # حرارة المحلول (حرارة الغرفة)
    c_water      = 4.18,  # الحرارة النوعية للماء J/(g·K)
    c_tissue     = 3.60,  # الحرارة النوعية للنسيج J/(g·K) — IT'IS
    m_eff_g      = 12.0,  # ⚠ تقدير: كتلة النسيج التي "يحسّها" المستشعر
    tau_pool_s   = 300.0, # ⚠ تقدير: دفء السائل المتسرّب (~5 دقائق)
    tau_vein_s   = 18.0,  # ⚠ تقدير: دفء الجلد بعد مرور السائل في الوريد
    vein_fraction= 0.50,  # [٢] كان 0.12 — رُفع ليطابق التبريد الوريدي القوي
                          #     المرصود فوق الوريد أثناء حقنة سليمة (Kageyama 2026)
    ntc_sigma    = 0.06,  # ضجيج مستشعر NTC (°م)
    coupling     = 0.45,  # ⚠ تقدير: نسبة التبريد الواصلة لسطح الجلد
)

def thermal_curves(t_s, volume_ml, frac_extrav, bolus_t0=10.0,
                   room_drift_C=0.0, perfusion_scale=1.0, rng=None):
    """
    يُرجع (حرارة موقع الحقن، حرارة الموقع المرجعي).
    السائل أبرد من الجسم → يبرّد الجلد (انخفاض لا ارتفاع):
      • داخل الوريد : تبريد قوي لكنه يزول خلال ثوانٍ
      • متسرّب      : تبريد يبقى دقائق
    """
    if rng is None:
        rng = np.random.default_rng()
    T = THERM
    V_ex = volume_ml * frac_extrav
    V_iv = volume_ml * (1.0 - frac_extrav)

    # معادلة الاتزان الحراري: كم تبرد كتلة النسيج لكل 1 مل سائل
    dT_per_ml = ((T["T_skin_base"] - T["T_fluid"]) * T["c_water"]
                 / (T["m_eff_g"] * T["c_tissue"]))

    step = (t_s >= bolus_t0).astype(float)
    dt   = np.maximum(t_s - bolus_t0, 0.0)
    tau_p = T["tau_pool_s"] / perfusion_scale
    tau_v = T["tau_vein_s"] / perfusion_scale

    drop_ex = T["coupling"] * dT_per_ml * V_ex * np.exp(-dt / tau_p) * step
    drop_iv = (T["coupling"] * dT_per_ml * V_iv * T["vein_fraction"]
               * np.exp(-dt / tau_v) * step)

    drift = room_drift_C * (t_s / max(t_s[-1], 1.0))
    T_inj = T["T_skin_base"] - drop_ex - drop_iv + drift
    T_ref = T["T_skin_base"] + drift
    T_inj = T_inj + rng.normal(0, T["ntc_sigma"], t_s.size)
    T_ref = T_ref + rng.normal(0, T["ntc_sigma"], t_s.size)
    return T_inj, T_ref

# ------------------------------------------------------------------
# 5. السمتان للقرار البسيط (تحليل قابلية الكشف)
# ------------------------------------------------------------------
def features(t_s, r_inj, r_ref, T_inj, T_ref, rng, bolus_t0=10.0,
             win_start=2.0, win_len=30.0):
    """
    rad_ratio : نسبة العد (ذراع الحقن ÷ المرجعية) داخل نافذة القرار
    therm_dip : [٣] متوسط الهبوط الحراري التفاضلي في آخر 10 ثوانٍ من النافذة
                — يقيس ما "بقي" وقت القرار، لا أعمق نقطة.
                النافذة القصيرة تلتقط التبريد الوريدي العابر (صعب)،
                والطويلة تلتقط بقاء تبريد التسرّب (سهل).
    """
    t0, t1 = bolus_t0 + win_start, bolus_t0 + win_start + win_len
    w    = (t_s >= t0) & (t_s <= t1)
    tail = (t_s >= t1 - 10.0) & (t_s <= t1)

    n_inj = rng.poisson(np.maximum(r_inj[w], 0)).sum()
    n_ref = rng.poisson(np.maximum(r_ref[w], 0)).sum()
    rad_ratio = (n_inj + 1.0) / (n_ref + 1.0)

    therm_dip = -float(np.mean((T_inj - T_ref)[tail]))
    return rad_ratio, therm_dip
