import sys, os, time, math, subprocess, csv
sys.path.insert(0, r"C:\Users\adora\DeepfakeLab\models\pellav2")
import torch
from sklearn.metrics import roc_auc_score
import pellav2_infer as P

torch.set_num_threads(8)
src = sys.argv[1]
cache = os.path.join(os.path.dirname(os.path.abspath(src)), "audio_cache")
os.makedirs(cache, exist_ok=True)

model = P.Detector()
model.load_state_dict(torch.load(r"C:\Users\adora\DeepfakeLab\models\pellav2\pellav2_detector.pt", map_location="cpu"))
model.eval()

exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".mp4"}
rows = []
for f in sorted(os.listdir(src)):
    base, ext = os.path.splitext(f)
    low = base.lower()
    if ext.lower() not in exts or not (low.startswith("real") or low.startswith("fake")):
        continue
    truth = 1 if low.startswith("fake") else 0
    mp3, wav = os.path.join(cache, base + ".mp3"), os.path.join(cache, base + ".wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(src, f), "-b:a", "64k", mp3], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav], check=True)
    except FileNotFoundError:
        sys.exit("ffmpeg not found in this window. Open a new terminal, or tell Claude.")
    except subprocess.CalledProcessError:
        print("could not convert", f); continue
    x = torch.from_numpy(P.load_clip(wav))[None]
    t = time.time()
    with torch.no_grad():
        logit = model(x).item()
    p = 1 / (1 + math.exp(-logit))
    pred = int(p >= 0.5)
    rows.append((f, truth, p, pred))
    print(f"{f:28} {'fake' if truth else 'real':5} logit={logit:.1f} p_fake={p:.3f} -> {'fake' if pred else 'real':5} {'' if truth == pred else '<-- WRONG'} ({time.time()-t:.1f}s)")

tp = sum(1 for r in rows if r[1] and r[3]); fn = sum(1 for r in rows if r[1] and not r[3])
fp = sum(1 for r in rows if not r[1] and r[3]); tn = sum(1 for r in rows if not r[1] and not r[3])
n = len(rows)
print(f"\nClips: {n} (real={tn+fp}, fake={tp+fn})")
print(f"Confusion: real->real {tn} | real->fake {fp} | fake->real {fn} | fake->fake {tp}")
print(f"Accuracy: {(tp+tn)/n:.1%}")
if tn + fp: print(f"False-positive rate (real called fake): {fp/(tn+fp):.1%}")
if tp + fn: print(f"Fake detection rate: {tp/(tp+fn):.1%}")
if len({r[1] for r in rows}) == 2: print(f"ROC-AUC: {roc_auc_score([r[1] for r in rows], [r[2] for r in rows]):.3f}")
with open(os.path.join(os.path.dirname(os.path.abspath(src)), "eval_audio_results.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "truth", "p_fake", "pred"]); w.writerows(rows)
