"""
run_all.py — شغّلي هذا الملف وحده:  python run_all.py
يحسب كل النتائج ويرسم الأشكال الستة. المدة المتوقعة: 2–5 دقائق.
"""
import json, time
import simulate as S
import ai_model as AI
import figures as F

t0 = time.time()
WINDOWS = [10, 30, 60, 120]

print("١/٤  تحليل قابلية الكشف (النماذج البسيطة)...")
res = {"windows": WINDOWS, "mdf": {}}
for iso in ["Lu-177", "Y-90", "Ra-223"]:
    res["mdf"][iso] = {}
    for w in WINDOWS:
        res["mdf"][iso][str(w)] = S.min_detectable_fraction(iso, float(w))
        print(f"     {iso:7s} نافذة {w:4d} ث → {res['mdf'][iso][str(w)]}")

print("٢/٤  الإنذار الكاذب مقابل الاحتباس الوريدي...")
fa = S.false_alarm_vs_venous(win_len=30.0)
res["false_alarm"] = {ch: {str(k): v for k, v in d.items()} for ch, d in fa.items()}
json.dump(res, open("results.json", "w"), indent=1)

print("٣/٤  تدريب نموذج الذكاء الاصطناعي واختباره...")
ai = AI.run()
json.dump(ai, open("results_ai.json", "w"), indent=1)
for k in ["Logistic regression", "Random forest",
          "Hand rule (baseline)", "Radiation only"]:
    print(f"     {k:22s} حساسية (اختبار صعب، إنذار كاذب 5%) = "
          f"{100 * ai[k]['hard']['sens_at_5far']:.1f}%")

print("٤/٤  رسم الأشكال...")
for p in [F.fig1_signals(), F.fig2_latency(res), F.fig3_false_alarm(res),
          F.fig4_confusion(ai), F.fig5_comparison(ai), F.fig6_weights(ai)]:
    print("     ✓", p)
print(f"تم في {time.time() - t0:.0f} ثانية.")
