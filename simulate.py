"""
simulate.py  (الإصدار ٢)
=========================================================
محاكاة مونت كارلو وتحليل قابلية الكشف — بلا أي تعلّم آلي.
الحقيقة الأرضية من الفيزياء (نحن نُدخل كسر التسرّب)،
والعتبة من الحقنات السليمة وحدها → التحليل غير دائري.
"""
import numpy as np
import physics as ph

T_GRID = np.arange(0.0, 180.0, 1.0)   # 3 دقائق، قراءة كل ثانية


def random_patient(rng):
    """مريض افتراضي بصفات عشوائية — عوامل الإرباك الواقعية."""
    depth = rng.uniform(0.35, 0.80)
    return dict(
        depth      = depth,                                  # عمق الوريد (سم)
        depth_ref  = max(depth + rng.normal(0, 0.10), 0.2),  # لصق المرجعي ليس مطابقاً
        sys_asym   = np.exp(rng.normal(0, 0.12)),            # اختلاف الذراعين
        volume     = rng.uniform(3.0, 10.0),                 # حجم الحقنة (مل)
        room_drift = rng.normal(0, 0.25),                    # انسياق حرارة الغرفة
        perfusion  = np.exp(rng.normal(0, 0.25)),            # سرعة الدورة الدموية
        clear_h    = rng.uniform(3.0, 12.0),                 # إزالة المتسرّب (ساعات)
    )


def simulate_curves(iso, frac, vr, rng, venous_tau_s=45.0, pt=None):
    """يحاكي حقنة واحدة ويُرجع المنحنيات الأربعة."""
    if pt is None:
        pt = random_patient(rng)
    r_inj, r_ref = ph.radiation_tac(
        T_GRID, iso, frac, pt["depth"], depth_ref_cm=pt["depth_ref"],
        sys_asym=pt["sys_asym"], venous_residual=vr,
        venous_tau_s=venous_tau_s, local_clear_h=pt["clear_h"])
    T_inj, T_ref = ph.thermal_curves(
        T_GRID, pt["volume"], frac, room_drift_C=pt["room_drift"],
        perfusion_scale=pt["perfusion"], rng=rng)
    return r_inj, r_ref, T_inj, T_ref


def population(iso, frac, n, seed, win_len, vr_range=(0.0, 0.02)):
    """n حقنة بنفس كسر التسرّب، كل واحدة لمريض عشوائي مختلف."""
    rng = np.random.default_rng(seed)
    rad, th = np.empty(n), np.empty(n)
    for i in range(n):
        vr = rng.uniform(*vr_range)
        c = simulate_curves(iso, frac, vr, rng)
        rad[i], th[i] = ph.features(T_GRID, *c, rng=rng, win_len=win_len)
    return dict(rad=rad, therm=th)


# ثلاث طرق قرار بسيطة (بلا ذكاء اصطناعي)
def score(channel, d):
    lr = np.log(np.maximum(d["rad"], 1e-3))
    th = np.maximum(d["therm"], 0.0)
    if channel == "radiation":
        return lr
    if channel == "thermal":
        return th
    return lr + 2.0 * np.log1p(th)   # "fused": وزن 2.0 مُختار يدوياً


CHANNELS = ["radiation", "thermal", "fused"]
FRACS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50]


def min_detectable_fraction(iso, win_len, n=1000, seed=3,
                            spec=0.95, sens=0.90):
    """
    أصغر كسر تسرّب يُكشف بنوعية 95% وحساسية 90%، لكل قناة.
      1) نحاكي حقنات سليمة → العتبة = المئين 95 لدرجاتها
      2) لكل كسر تسرّب نحاكي حقنات ونحسب نسبة تجاوز العتبة
      3) أول كسر تصل عنده النسبة 90% = الحد الأدنى للكشف
    """
    normal = population(iso, 0.0, n, seed, win_len)
    thr = {ch: np.quantile(score(ch, normal), spec) for ch in CHANNELS}
    out = {ch: None for ch in CHANNELS}
    for j, f in enumerate(FRACS):
        d = population(iso, f, n, seed + 100 + j, win_len)
        for ch in CHANNELS:
            if out[ch] is None and (score(ch, d) > thr[ch]).mean() >= sens:
                out[ch] = f
        if all(v is not None for v in out.values()):
            break
    return out


def false_alarm_vs_venous(iso="Lu-177", vr_levels=(0.0, 0.01, 0.02, 0.04,
                          0.06, 0.10, 0.15), win_len=30.0, target_f=0.10,
                          n=1000, seed=7):
    """
    معدل الإنذار الكاذب على حقنات سليمة حسب مقدار الاحتباس الوريدي،
    بعد ضبط العتبة لكشف 90% من تسرّبات حجمها target_f.
    """
    pos = population(iso, target_f, n, seed, win_len)
    thr = {ch: np.quantile(score(ch, pos), 0.10) for ch in CHANNELS}
    out = {ch: {} for ch in CHANNELS}
    for k, vr in enumerate(vr_levels):
        d = population(iso, 0.0, n, seed + 50 + k, win_len, vr_range=(vr, vr))
        for ch in CHANNELS:
            out[ch][vr] = float((score(ch, d) > thr[ch]).mean())
    return out
