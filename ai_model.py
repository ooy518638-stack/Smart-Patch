"""
ai_model.py
=========================================================
نموذج الذكاء الاصطناعي: تصنيف كل حقنة إلى ثلاث حالات
    0 = سليمة   1 = احتباس وريدي   2 = تسرّب
(التقسيم الثلاثي من الأدبيات: Iori 2023؛ Sanchez & Currie 2020)

قواعد التقييم الصادق:
  ① التسمية من الفيزياء: نحدد الحالة قبل المحاكاة، لا عتبة تسمّي البيانات
  ②بيانات التدريب والاختبار من مرضى مختلفين تماماً (بذور عشوائية مختلفة)
  """
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import simulate as S

CLASSES = ["Normal", "Venous retention", "Extravasation"]
CLASSES_AR = ["سليمة", "احتباس وريدي", "تسرّب"]
BOLUS_T0 = 10.0

FEATURE_NAMES = [
    "rad_early",     # 1 نسبة الإشعاع مبكراً     (5–15 ث بعد الحقن)
    "rad_mid",       # 2 نسبة الإشعاع متوسطاً    (30–60 ث)
    "rad_late",      # 3 نسبة الإشعاع متأخراً    (90–150 ث)
    "rad_washout",   # 4 سرعة الغسل = مبكر − متأخر (لوغاريتمياً)
    "therm_early",   # 5 الهبوط الحراري مبكراً   (2–15 ث)
    "therm_late",    # 6 الهبوط الحراري متأخراً  (90–150 ث)
    "therm_persist", # 7 بقاء التبريد = متأخر ÷ مبكر
    "therm_slope",   # 8 ميل الحرارة المتأخر (هل تعود لطبيعتها؟)
]


def _window(a, b):
    t = S.T_GRID
    return (t >= BOLUS_T0 + a) & (t <= BOLUS_T0 + b)


W_EARLY, W_MID, W_LATE = _window(5, 15), _window(30, 60), _window(90, 150)
W_T_EARLY = _window(2, 15)


def extract_features(r_inj, r_ref, T_inj, T_ref, rng):
    """يحوّل المنحنيات الأربعة إلى 8 سمات لكل منها معنى فيزيائي."""
    n_inj = rng.poisson(np.maximum(r_inj, 0))   # عدّ حقيقي بضجيج بواسون
    n_ref = rng.poisson(np.maximum(r_ref, 0))

    def log_ratio(w):
        return np.log((n_inj[w].sum() + 1.0) / (n_ref[w].sum() + 1.0))

    e, m, l = log_ratio(W_EARLY), log_ratio(W_MID), log_ratio(W_LATE)
    dip = -(T_inj - T_ref)                        # موجب = تبريد
    te, tl = dip[W_T_EARLY].mean(), dip[W_LATE].mean()
    slope = np.polyfit(S.T_GRID[W_LATE], dip[W_LATE], 1)[0] * 60.0  # °م/دقيقة
    return [e, m, l, e - l, te, tl, tl / (abs(te) + 0.2), slope]


def make_dataset(n_per_class, seed, hard=False):
    """
    يولّد حقنات مسمّاة من الفيزياء. hard=True يولّد ظروفاً أصعب
    لم يرها النموذج في التدريب: أوردة أعمق، تسرّبات أصغر، احتباس أقوى وأبطأ.
    """
    rng = np.random.default_rng(seed)
    X, y, frac_out = [], [], []
    for cls in (0, 1, 2):
        for _ in range(n_per_class):
            pt = S.random_patient(rng)
            iso = rng.choice(list(S.ph.ISOTOPES))
            if hard:
                pt["depth"] = rng.uniform(0.8, 1.2)
                pt["depth_ref"] = max(pt["depth"] + rng.normal(0, 0.10), 0.2)
            if cls == 0:      # سليمة: بقية وريدية صغيرة تُغسل بسرعة
                frac, vr, tau = 0.0, rng.uniform(0, 0.01), rng.uniform(30, 60)
            elif cls == 1:    # احتباس وريدي: بقية كبيرة تُغسل ببطء
                frac = 0.0
                vr  = rng.uniform(0.10, 0.25) if hard else rng.uniform(0.03, 0.15)
                tau = rng.uniform(90, 150)    if hard else rng.uniform(40, 120)
            else:             # تسرّب حقيقي
                frac = rng.uniform(0.01, 0.05) if hard else rng.uniform(0.02, 0.30)
                vr, tau = rng.uniform(0, 0.01), rng.uniform(30, 60)
            c = S.simulate_curves(iso, frac, vr, rng, venous_tau_s=tau, pt=pt)
            X.append(extract_features(*c, rng=rng))
            y.append(cls)
            frac_out.append(frac)
    return np.array(X), np.array(y), np.array(frac_out)


def baseline_score(X):
    """
    خط الأساس: القاعدة اليدوية المدمجة (بوزن 2.0 المُختار يدوياً)،
    على نفس النافذة الزمنية الكاملة، حتى تكون المقارنة عادلة.
    """
    lr_all = (X[:, 0] + X[:, 1] + X[:, 2]) / 3.0
    return lr_all + 2.0 * np.log1p(np.maximum(X[:, 5], 0.0))


def sens_at_far(score, is_extrav, far=0.05):
    """حساسية كشف التسرّب عند معدل إنذار كاذب ثابت (5%) على غير المتسرّبة."""
    thr = np.quantile(score[~is_extrav], 1.0 - far)
    return float((score[is_extrav] > thr).mean())


def run(n_train=2000, n_test=1000, seed=2026):
    X_tr, y_tr, _ = make_dataset(n_train, seed)
    X_te, y_te, _ = make_dataset(n_test, seed + 1)
    X_hd, y_hd, _ = make_dataset(n_test, seed + 2, hard=True)

    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=0),
    }
    res = {"feature_names": FEATURE_NAMES, "classes": CLASSES}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        res[name] = {}
        for split, X, y in [("test", X_te, y_te), ("hard", X_hd, y_hd)]:
            pred = m.predict(X)
            p_ex = m.predict_proba(X)[:, 2]
            res[name][split] = dict(
                accuracy=float((pred == y).mean()),
                confusion=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
                sens_at_5far=sens_at_far(p_ex, y == 2),
            )
    for split, X, y in [("test", X_te, y_te), ("hard", X_hd, y_hd)]:
        res.setdefault("Hand rule (baseline)", {})[split] = dict(
            sens_at_5far=sens_at_far(baseline_score(X), y == 2))
        res.setdefault("Radiation only", {})[split] = dict(
            sens_at_5far=sens_at_far(X[:, 2], y == 2))

    # أوزان الانحدار اللوجستي لفئة "تسرّب": ماذا تعلّم النموذج؟
    lr = models["Logistic regression"][-1]
    res["lr_weights_extrav"] = lr.coef_[2].tolist()
    return res


if __name__ == "__main__":
    import json
    r = run()
    json.dump(r, open("results_ai.json", "w"), indent=1)
    for k in ["Logistic regression", "Random forest",
              "Hand rule (baseline)", "Radiation only"]:
        print(k, {s: {m: (round(v, 3) if isinstance(v, float) else v)
                      for m, v in r[k][s].items()} for s in r[k]})
    print("weights:", dict(zip(FEATURE_NAMES, np.round(r["lr_weights_extrav"], 2))))
