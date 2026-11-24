"""
demo.py — عرض حي لقرار الذكاء الاصطناعي
=========================================================
التشغيل:   python demo.py
يدرّب النموذج (ثوانٍ)، ثم يحاكي حقنات جديدة لم يرها من قبل،
ويطبع قراره لكل واحدة مقابل الحالة الحقيقية.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import simulate as S
import ai_model as AI

ALARM = 0.50   # يُطلق الإنذار إذا تجاوز احتمال التسرّب 50%

# الحالات المعروضة: (الوصف، النظير، كسر التسرّب، الاحتباس الوريدي، زمن غسله، عمق الوريد)
# عمق None = عمق عشوائي عادي (0.35–0.8 سم)
CASES = [
    ("حقنة سليمة",                    "Lu-177", 0.00, 0.005, 45,  None),
    ("احتباس وريدي قوي (بلا تسرّب)",  "Lu-177", 0.00, 0.12,  100, None),
    ("تسرّب صغير 3%",                 "Lu-177", 0.03, 0.005, 45,  None),
    ("تسرّب كبير 15%",                "Lu-177", 0.15, 0.005, 45,  None),
    ("تسرّب 5% — إيتريوم-90",          "Y-90",   0.05, 0.005, 45,  None),
    ("تسرّب 5% — راديوم-223",          "Ra-223", 0.05, 0.005, 45,  None),
    # حالة صعبة عمداً: خارج ظروف التدريب — قد يخطئ فيها النموذج، وهذا متوقع
    ("حالة صعبة: تسرّب 1% في وريد عميق", "Lu-177", 0.01, 0.005, 45,  1.1),
]


def bar(p, width=20):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def main(seed=7):
    print("جارٍ تدريب نموذج الذكاء الاصطناعي...")
    X, y, _ = AI.make_dataset(2000, seed=2026)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X, y)
    print("تم التدريب.\n")

    rng = np.random.default_rng(seed)   # مرضى جدد لم يرهم النموذج
    for name, iso, frac, vr, tau, depth in CASES:
        pt = S.random_patient(rng)
        if depth is not None:
            pt["depth"] = depth
            pt["depth_ref"] = depth
        curves = S.simulate_curves(iso, frac, vr, rng, venous_tau_s=tau, pt=pt)
        x = np.array([AI.extract_features(*curves, rng=rng)])
        p = model.predict_proba(x)[0]
        print("=" * 52)
        print(f"الحالة الحقيقية : {name}")
        print("قرار الذكاء الاصطناعي:")
        for k, cls in enumerate(AI.CLASSES_AR):
            print(f"   {cls:14s} {bar(p[k])} {100 * p[k]:5.1f}%")
        if p[2] >= ALARM:
            print("   ⚠️  إنذار: اشتباه تسرّب — أوقفي الحقن وافحصي الموقع")
        elif p[1] >= ALARM:
            print("   ℹ️  احتباس وريدي — لا تسرّب، تابعي المراقبة")
        else:
            print("   ✓  الحقنة سليمة")
    print("=" * 52)


if __name__ == "__main__":
    main()
