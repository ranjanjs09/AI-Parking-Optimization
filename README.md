# Module 2: Backend + Live Dashboard

Ye Module 1 (CV model) ke JSON output ko lekar ek live-updating web dashboard
me dikhata hai. Maine ye sandbox me poora test kiya hai — server start hua,
API ne GET/POST dono sahi handle kiya, aur counts (available/occupied/total)
sahi calculate hue.

## Folder structure (isi tarah rakhna)

```
parking_dashboard/
    app.py                 <- Flask backend + dashboard route
    push_live_status.py    <- Module 1 ke CV model ko dashboard se jodta hai
    requirements.txt
    static/
        index.html          <- Dashboard webpage (auto-refresh)
```

**IMPORTANT:** `index.html` ko `static` naam ke folder ke ANDAR hi rakhna,
Flask isi folder se serve karta hai.

## 0. Setup

```bash
cd parking_dashboard
pip install -r requirements.txt
```

## 1. Server start karo

```bash
python app.py
```

Terminal me ye dikhna chahiye:
```
Starting Parking Dashboard server...
Dashboard: http://localhost:5000
* Running on http://127.0.0.1:5000
```

Ab browser me kholo: **http://localhost:5000**

Abhi dashboard khaali dikhega ("Koi data nahi mila abhi tak") — kyunki
koi status push nahi hui hai. Ye terminal isi tarah chalta rehna do
(band mat karna), ek **naya terminal tab/window** kholo agle step ke liye.

## 2. Quick test (bina CV model ke) - dashboard dikh raha hai ya nahi confirm karo

Naye terminal me:
```bash
curl -X POST http://localhost:5000/api/status \
    -H "Content-Type: application/json" \
    -d '{"spot_1": "empty", "spot_2": "occupied", "spot_3": "empty", "spot_4": "occupied"}'
```

Browser me dashboard turant update ho jaana chahiye (3 second ke andar) —
4 boxes dikhenge, 2 green (empty) aur 2 red (occupied), upar counts bhi
update honge.

## 3. Ab Module 1 (CV model) ko jodo — real live feed

`push_live_status.py` ko Module 1 ke files (`model.pth`, `spots.json`) chahiye
— unhe is `parking_dashboard` folder me copy kar lo, ya path bata do.

**Static image ko baar-baar process karke "live feed" simulate karna** (demo
ke liye sabse aasaan):
```bash
python push_live_status.py --mode image --source test.jpg \
    --spots spots.json --model model.pth --interval 5
```
Ye har 5 second me `test.jpg` ko phir se CV model se predict karke dashboard
ko push karega. (Static image hai to result har baar same hi aayega — asli
demo effect ke liye kisi image ko manually replace kar dena beech me, ya
Step 4 wala webcam mode try karo.)

**Webcam se live feed** (agar laptop ka camera use karna hai):
```bash
python push_live_status.py --mode video --source 0 \
    --spots spots.json --model model.pth --interval 5
```

Ab dashboard **real-time me** update hota rahega, jaise real production
system — CCTV/camera → CV model → backend → live dashboard.

## Architecture (report/PPT ke liye)

```
[Camera/Video] --> [detect_occupancy.py ya push_live_status.py]
                            |  (CV model: ResNet18)
                            v
                    JSON: {spot_1: "empty", ...}
                            |
                            v POST /api/status
                    [Flask Backend - app.py]
                            |
                            v GET /api/status (every 3s poll)
                    [Dashboard - static/index.html]
                            |
                            v
                    Browser me live green/red grid
```

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard webpage |
| `/api/status` | GET | Current status JSON return karta hai |
| `/api/status` | POST | Naya status set karta hai (body: `{"spot_1": "empty", ...}`) |

## Why this design (report me likhne ke liye)

- **Polling (3s) instead of WebSockets:** Simple HTTP polling use kiya hai
  taaki koi extra dependency (Socket.IO, Firebase account) na chahiye ho —
  college project ke liye setup-free rehna zyada important hai. Production
  system me WebSockets/Firebase Realtime DB better hoga.
- **File-backed state:** `current_status.json` me bhi status save hota hai,
  isliye server restart hone pe bhi last known status yaad rehta hai.
- **Separation of concerns:** CV model (`push_live_status.py`) aur backend
  (`app.py`) alag services hain, jaise real microservice architecture —
  isse Module 3 (prediction) aur Module 4 (pricing) baad me easily isi
  backend me add ho sakte hain.

## Next Steps (Module 3 & 4)

- **Module 3 (Prediction):** `current_status.json` ko timestamp ke saath
  history me log karna shuru karo (`app.py` me ek chhota addition), phir
  us history pe Prophet/time-series model train karke "15 min baad
  availability" predict karna aur dashboard pe dikhana.
- **Module 4 (Dynamic Pricing):** occupancy % ke hisab se ek simple
  rule-based price calculate karke dashboard pe har spot ke saath dikhana.

Chalao aur bata dena kaisa gaya — screenshot bhi bhej sakte ho agar kuch
galat dikhe.
# AI-Parking-Optimization
