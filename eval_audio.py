import sys, time, glob, os, math
sys.path.insert(0, r"C:\Users\adora\DeepfakeLab\models\pellav2")
import torch, psutil
import pellav2_infer as P

torch.set_num_threads(8)
proc = psutil.Process(os.getpid())

t = time.time()
model = P.Detector()
model.load_state_dict(torch.load(r"C:\Users\adora\DeepfakeLab\models\pellav2\pellav2_detector.pt", map_location="cpu"))
model.eval()
print(f"Model load: {time.time()-t:.1f}s   RAM now: {proc.memory_info().rss/1e9:.2f} GB")

rows = []
for f in sorted(glob.glob(r"C:\Users\adora\DeepfakeLab\test_audio_wav\*.wav")):
    name = os.path.basename(f)
    x = torch.from_numpy(P.load_clip(f))[None]
    t = time.time()
    with torch.no_grad():
        logit = model(x).item()
    dt = time.time() - t
    p = 1 / (1 + math.exp(-logit))
    truth = "fake" if name.startswith("fake") else "real"
    pred = "fake" if p >= 0.5 else "real"
    rows.append((name, truth, pred))
    print(f"{name:14} truth={truth:5} model score p_fake={p:.3f} -> {pred:5} {'' if truth == pred else '<-- WRONG'}  ({dt:.1f}s)")

ok = sum(1 for r in rows if r[1] == r[2])
print(f"\nCorrect: {ok}/{len(rows)}   RAM now: {proc.memory_info().rss/1e9:.2f} GB")
