import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template_string, Response, request, jsonify, url_for
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import json
import uuid
from datetime import datetime
import html as html_mod  # for escaping when needed
import base64  # for client image uploads

# ========== OpenAI (GPT) ==========
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
OPENAI_MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# ========== Pose ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========== Session & Logs ==========
SESSION_ID = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
FRAME_LOG_PATH = os.path.join(LOG_DIR, f"frames-{SESSION_ID}.jsonl")
REP_LOG_PATH   = os.path.join(LOG_DIR, f"reps-{SESSION_ID}.jsonl")

# ========== Shared State ==========
state_lock = threading.Lock()
app_state = {
    "mode": None,                # None | "curl" | "squat"
    "curl_count": 0,
    "curl_correct": 0,
    "curl_stage": None,
    "show_debug": False,
    "ema_state": {},

    # UI fields
    "posture_ok": False,
    "last_reasons": [],
    "last_feedback": "",
    "exercise_name": "—",
    "pose_detected": False,
    "analysis_text_html": "",

    # squat-specific helper for robust counting
    "squat_min_knee": None,
}

# ========== Client Frame Buffer (browser -> server) ==========
latest_client_frame = None        # numpy BGR image
latest_frame_lock = threading.Lock()

# ========== Helpers ==========
def ema(value, name, alpha=0.35):
    if value is None: return None
    d = app_state["ema_state"]
    if name not in d:
        d[name] = float(value)
    else:
        d[name] = alpha * float(value) + (1.0 - alpha) * d[name]
    return d[name]

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y], dtype=float)
    b = np.array([b.x, b.y], dtype=float)
    c = np.array([c.x, c.y], dtype=float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 180.0
    cosv = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

def vec(p, q):
    return np.array([q.x - p.x, q.y - p.y, q.z - p.z], dtype=float)

def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosv = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

def visible(lm, *ids, thresh=0.3):
    for i in ids:
        l = lm[i]
        if getattr(l, "visibility", 0.0) < thresh:
            return False
    return True

# ---- 3D helpers for camera-agnostic squat checks ----
def asnp(l):  # landmark -> np.array([x,y,z])
    return np.array([l.x, l.y, l.z], dtype=float)

def angle_3d(a, b, c):  # true 3D joint angle at b
    va, vc = asnp(a) - asnp(b), asnp(c) - asnp(b)
    return angle_between(va, vc)

def estimate_yaw_deg_world(LSHw, RSHw):
    # larger z-separation between shoulders => more side-on
    return float(min(45.0, abs(LSHw.z - RSHw.z) * 260.0))

def lateral_offset_from_foot_axis(knee_w, ankle_w, heel_w, toe_w):
    """
    Knee valgus in 3D: distance of knee from the foot forward axis (through ankle),
    normalized later by hip-ankle length.
    """
    fwd = asnp(toe_w) - asnp(heel_w)
    nf = np.linalg.norm(fwd)
    if nf < 1e-6:
        fwd = asnp(toe_w) - asnp(ankle_w)
        nf = np.linalg.norm(fwd)
    if nf < 1e-6:
        return None
    fwd /= nf
    rel = asnp(knee_w) - asnp(ankle_w)
    off = np.linalg.norm(np.cross(rel, fwd))  # magnitude of perpendicular component
    return float(off)

# ---------- JSON Sanitizer ----------
def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(x) for x in obj]
    if isinstance(obj, (np.bool_,)):      return bool(obj)
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, (np.floating,)):   return float(obj)
    if isinstance(obj, np.ndarray):       return obj.tolist()
    try:
        json.dumps(obj); return obj
    except Exception:
        return str(obj)

# ========== Logging ==========
def _ts():
    return datetime.utcnow().isoformat() + "Z"

def log_jsonl(path, obj):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitize(obj), ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOG WARN] Could not write log to {path}: {e}")

def log_frame(data): log_jsonl(FRAME_LOG_PATH, data)
def log_rep(data):   log_jsonl(REP_LOG_PATH, data)

# ========== Posture Check (curl) ==========
def posture_check_curl(lm_img, lm_world, debug=None):
    reasons = []
    per_arm_reasons = {"L": [], "R": []}

    LSH = lm_img[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    RSH = lm_img[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    LHP = lm_img[mp_pose.PoseLandmark.LEFT_HIP.value]
    RHP = lm_img[mp_pose.PoseLandmark.RIGHT_HIP.value]
    LAK = lm_img[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    RAK = lm_img[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    l_torso = calculate_angle(LSH, LHP, LAK)
    r_torso = calculate_angle(RSH, RHP, RAK)
    torso_ok = (l_torso > 160) and (r_torso > 160)
    if not torso_ok:
        reasons.append("Stand tall. Don’t lean forward/back.")

    if lm_world is None:
        return torso_ok, reasons, {"L": [], "R": []}, {"L":{}, "R":{}}

    LSHw = lm_world[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    RSHw = lm_world[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    LHPw = lm_world[mp_pose.PoseLandmark.LEFT_HIP.value]
    RHPw = lm_world[mp_pose.PoseLandmark.RIGHT_HIP.value]
    LELw = lm_world[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    RELw = lm_world[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    shoulder_depth_diff = abs(LSHw.z - RSHw.z)
    yaw_deg = min(25.0, shoulder_depth_diff * 180.0)

    ADDUCT_MAX = 30.0 + 0.6 * yaw_deg
    ADDUCT_MAX_RELAX = ADDUCT_MAX + 6.0
    LAT_FRAC = 0.55 + 0.35 * (yaw_deg / 25.0)
    shoulder_width = max(1e-6, np.linalg.norm(vec(LSHw, RSHw)))

    out_metrics = {"L": {}, "R": {}}

    def eval_arm(side):
        if side == "L":
            SHw, ELw, HPw = LSHw, LELw, LHPw
            vis_ok = visible(lm_img,
                             mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                             mp_pose.PoseLandmark.LEFT_ELBOW.value,
                             mp_pose.PoseLandmark.LEFT_HIP.value)
            pref = "L"
        else:
            SHw, ELw, HPw = RSHw, RELw, RHPw
            vis_ok = visible(lm_img,
                             mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                             mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                             mp_pose.PoseLandmark.RIGHT_HIP.value)
            pref = "R"

        if not vis_ok:
            per_arm_reasons[side].append("Make sure shoulder & elbow are visible in camera.")
            out_metrics[side]["adduct_deg"] = None
            out_metrics[side]["lat"] = None
            return False

        adduct = angle_between(vec(SHw, ELw), vec(SHw, HPw))
        adduct_sm = ema(adduct, f"{pref}_adduct")
        lat = abs(ELw.x - SHw.x)
        lat_sm = ema(lat, f"{pref}_lat")

        out_metrics[side]["adduct_deg"] = adduct_sm if adduct_sm is not None else adduct
        out_metrics[side]["lat"] = lat_sm if lat_sm is not None else lat

        adduct_ok = (adduct_sm if adduct_sm is not None else adduct) < ADDUCT_MAX_RELAX
        lat_ok = (lat_sm if lat_sm is not None else lat) < (LAT_FRAC * shoulder_width)

        if not adduct_ok:
            per_arm_reasons[side].append("Keep elbows close to ribs (don’t flare).")
        if not lat_ok:
            per_arm_reasons[side].append("Elbows drifting out — tuck them in.")
        return adduct_ok and lat_ok

    L_ok = eval_arm("L")
    R_ok = eval_arm("R")
    ok = torso_ok and (L_ok or R_ok)

    if not ok:
        if torso_ok and (not L_ok or not R_ok):
            if not L_ok and per_arm_reasons["L"]:
                reasons.append("Left arm: " + "; ".join(per_arm_reasons["L"]))
            if not R_ok and per_arm_reasons["R"]:
                reasons.append("Right arm: " + "; ".join(per_arm_reasons["R"]))
        elif not torso_ok:
            if not L_ok and per_arm_reasons["L"]:
                reasons.append("Left arm: " + "; ".join(per_arm_reasons["L"]))
            if not R_ok and per_arm_reasons["R"]:
                reasons.append("Right arm: " + "; ".join(per_arm_reasons["R"]))

    if debug is not None:
        debug["yaw_deg"] = yaw_deg
        debug["L_adduct"] = out_metrics["L"].get("adduct_deg")
        debug["R_adduct"] = out_metrics["R"].get("adduct_deg")
        debug["L_lat"] = out_metrics["L"].get("lat")
        debug["R_lat"] = out_metrics["R"].get("lat")
        debug["shoulder_width"] = shoulder_width
        debug["adduct_max_relax"] = ADDUCT_MAX_RELAX
        debug["lat_allowed"] = LAT_FRAC * shoulder_width

    return ok, reasons, per_arm_reasons, out_metrics

# ========== Posture Check (squat) — camera-agnostic ==========
def posture_check_squat(lm_img, lm_world, debug=None):
    """
    Camera-agnostic squat checks using world (3D) landmarks where possible.
    Returns: ok, reasons(list), per_side_reasons, metrics
    metrics includes per-side: knee_deg, valgus_norm, trunk_angle_deg, hip_below_knee (fallback)
    """
    reasons = []
    per_side_reasons = {"L": [], "R": []}
    out = {"L": {}, "R": {}}

    P = mp_pose.PoseLandmark
    need_ids = [
        P.LEFT_HIP, P.RIGHT_HIP, P.LEFT_KNEE, P.RIGHT_KNEE,
        P.LEFT_ANKLE, P.RIGHT_ANKLE, P.LEFT_HEEL, P.RIGHT_HEEL,
        P.LEFT_FOOT_INDEX, P.RIGHT_FOOT_INDEX, P.LEFT_SHOULDER, P.RIGHT_SHOULDER
    ]

    if not all(visible(lm_img, p.value) for p in need_ids):
        reasons.append("Step back: keep hips/knees/ankles/feet in frame.")
        return False, reasons, per_side_reasons, out

    # 2D refs (fallbacks)
    LHIP_i, RHIP_i = lm_img[P.LEFT_HIP.value], lm_img[P.RIGHT_HIP.value]
    LKNE_i, RKNE_i = lm_img[P.LEFT_KNEE.value], lm_img[P.RIGHT_KNEE.value]
    LANK_i, RANK_i = lm_img[P.LEFT_ANKLE.value], lm_img[P.RIGHT_ANKLE.value]
    LTOE_i, RTOE_i = lm_img[P.LEFT_FOOT_INDEX.value], lm_img[P.RIGHT_FOOT_INDEX.value]
    LSH_i, RSH_i   = lm_img[P.LEFT_SHOULDER.value], lm_img[P.RIGHT_SHOULDER.value]

    if lm_world is None:
        # Fallback: 2D angles and simple image-space constraints
        l_knee = calculate_angle(LHIP_i, LKNE_i, LANK_i)
        r_knee = calculate_angle(RHIP_i, RKNE_i, RANK_i)
        l_knee_sm = ema(l_knee, "L_knee") or l_knee
        r_knee_sm = ema(r_knee, "R_knee") or r_knee
        out["L"]["knee_deg"], out["R"]["knee_deg"] = l_knee_sm, r_knee_sm

        l_hip_below_knee = LHIP_i.y > LKNE_i.y
        r_hip_below_knee = RHIP_i.y > RKNE_i.y
        out["L"]["hip_below_knee"] = l_hip_below_knee
        out["R"]["hip_below_knee"] = r_hip_below_knee

        L_dx = abs(LKNE_i.x - LTOE_i.x)
        R_dx = abs(RKNE_i.x - RTOE_i.x)
        out["L"]["valgus_norm"] = ema(L_dx, "L_valgus") or L_dx
        out["R"]["valgus_norm"] = ema(R_dx, "R_valgus") or R_dx

        L_torso_dx = abs(LSH_i.x - LTOE_i.x); R_torso_dx = abs(RSH_i.x - RTOE_i.x)
        out["L"]["torso_toe_dx"] = ema(L_torso_dx, "L_torso_dx") or L_torso_dx
        out["R"]["torso_toe_dx"] = ema(R_torso_dx, "R_torso_dx") or R_torso_dx

        posture_ok = (out["L"]["valgus_norm"] < 0.08 and out["R"]["valgus_norm"] < 0.08 and
                      out["L"]["torso_toe_dx"] < 0.20 and out["R"]["torso_toe_dx"] < 0.20)

        if out["L"]["valgus_norm"] >= 0.08: per_side_reasons["L"].append("Knee over toes: track straight.")
        if out["R"]["valgus_norm"] >= 0.08: per_side_reasons["R"].append("Knee over toes: track straight.")
        if out["L"]["torso_toe_dx"] >= 0.20 or out["R"]["torso_toe_dx"] >= 0.20:
            reasons.append("Chest proud: don’t fold forward.")

        if per_side_reasons["L"]: reasons.append("Left: " + "; ".join(per_side_reasons["L"]))
        if per_side_reasons["R"]: reasons.append("Right: " + "; ".join(per_side_reasons["R"]))

        if debug is not None:
            debug.update({k: out["L"].get(k) for k in out["L"]})
            debug.update({("R_"+k): out["R"].get(k) for k in out["R"]})
        return posture_ok, reasons, per_side_reasons, out

    # World (3D) — robust and camera-agnostic
    LHIP, RHIP = lm_world[P.LEFT_HIP.value], lm_world[P.RIGHT_HIP.value]
    LKNE, RKNE = lm_world[P.LEFT_KNEE.value], lm_world[P.RIGHT_KNEE.value]
    LANK, RANK = lm_world[P.LEFT_ANKLE.value], lm_world[P.RIGHT_ANKLE.value]
    LHEL, RHEL = lm_world[P.LEFT_HEEL.value], lm_world[P.RIGHT_HEEL.value]
    LTOE, RTOE = lm_world[P.LEFT_FOOT_INDEX.value], lm_world[P.RIGHT_FOOT_INDEX.value]
    LSH,  RSH  = lm_world[P.LEFT_SHOULDER.value], lm_world[P.RIGHT_SHOULDER.value]

    # 3D knee flexion
    l_knee = angle_3d(LHIP, LKNE, LANK); r_knee = angle_3d(RHIP, RKNE, RANK)
    l_knee_sm = ema(l_knee, "L_knee3D") or l_knee
    r_knee_sm = ema(r_knee, "R_knee3D") or r_knee
    out["L"]["knee_deg"], out["R"]["knee_deg"] = l_knee_sm, r_knee_sm

    # Yaw estimate from shoulders
    yaw_deg = estimate_yaw_deg_world(LSH, RSH)

    # 3D valgus: knee lateral offset from foot axis, normalized
    L_off = lateral_offset_from_foot_axis(LKNE, LANK, LHEL, LTOE)
    R_off = lateral_offset_from_foot_axis(RKNE, RANK, RHEL, RTOE)
    L_norm = L_off / (np.linalg.norm(asnp(LHIP) - asnp(LANK)) + 1e-6) if L_off is not None else None
    R_norm = R_off / (np.linalg.norm(asnp(RHIP) - asnp(RANK)) + 1e-6) if R_off is not None else None

    base_valgus = 0.14
    valgus_tol = base_valgus + 0.04 * (yaw_deg / 45.0)

    out["L"]["valgus_norm"] = ema(L_norm, "L_valgus3D") if L_norm is not None else None
    out["R"]["valgus_norm"] = ema(R_norm, "R_valgus3D") if R_norm is not None else None

    L_valgus_ok = (out["L"]["valgus_norm"] is None) or (out["L"]["valgus_norm"] < valgus_tol)
    R_valgus_ok = (out["R"]["valgus_norm"] is None) or (out["R"]["valgus_norm"] < valgus_tol)
    if not L_valgus_ok: per_side_reasons["L"].append("Knee caves/out—track over foot.")
    if not R_valgus_ok: per_side_reasons["R"].append("Knee caves/out—track over foot.")

    # Torso lean vs vertical
    trunk = (asnp(LSH) + asnp(RSH)) / 2 - (asnp(LHIP) + asnp(RHIP)) / 2
    trunk_angle = angle_between(trunk, np.array([0.0, 1.0, 0.0]))
    out["L"]["trunk_angle_deg"] = trunk_angle
    out["R"]["trunk_angle_deg"] = trunk_angle

    trunk_tol = 35.0 + 12.0 * (yaw_deg / 45.0)
    trunk_ok = trunk_angle <= trunk_tol
    if not trunk_ok:
        reasons.append("Keep chest proud—don’t fold forward.")

    # Depth hint (3D): hip below knee => hip.y < knee.y (world y up)
    out["L"]["hip_below_knee"] = (LHIP.y < LKNE.y)
    out["R"]["hip_below_knee"] = (RHIP.y < RKNE.y)

    posture_ok = L_valgus_ok and R_valgus_ok and trunk_ok

    if per_side_reasons["L"]: reasons.append("Left: " + "; ".join(per_side_reasons["L"]))
    if per_side_reasons["R"]: reasons.append("Right: " + "; ".join(per_side_reasons["R"]))

    if debug is not None:
        debug.update({
            "yaw_deg": yaw_deg,
            "L_knee_deg": out["L"]["knee_deg"], "R_knee_deg": out["R"]["knee_deg"],
            "valgus_tol": valgus_tol, "trunk_angle": trunk_angle, "trunk_tol": trunk_tol,
            "L_valgus_norm": out["L"]["valgus_norm"], "R_valgus_norm": out["R"]["valgus_norm"],
            "L_hip_below_knee": out["L"]["hip_below_knee"], "R_hip_below_knee": out["R"]["hip_below_knee"],
        })
    return posture_ok, reasons, per_side_reasons, out

# ========== Video Drawing ==========
def draw_colored_pose(image, landmarks, color=(0, 255, 0), thickness=3):
    h, w = image.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for (i, j) in mp_pose.POSE_CONNECTIONS:
        cv2.line(image, pts[i], pts[j], color, thickness)
    for (x, y) in pts:
        cv2.circle(image, (x, y), 3, color, -1)

# ========== Flask ==========
app = Flask(__name__)

# ====== Browser uploads raw frames to this endpoint ======
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """
    Accepts either:
      - multipart 'frame' (image/jpeg),
      - or JSON { "image_b64": "data:image/jpeg;base64,..." }
    Stores BGR image into latest_client_frame.
    """
    global latest_client_frame
    try:
        img = None
        if request.files.get('frame'):
            file = request.files['frame'].read()
            arr = np.frombuffer(file, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get('image_b64', '')
            if b64.startswith('data:'):
                b64 = b64.split(',', 1)[1]
            if b64:
                arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"ok": False, "error": "bad image"}), 400

        # mirror to match the old flip(1) selfie view
        img = cv2.flip(img, 1)

        with latest_frame_lock:
            latest_client_frame = img
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ====== Generator that consumes uploaded frames and streams processed video ======
def video_frames():
    frame_idx = 0
    while True:
        # wait for a client frame
        with latest_frame_lock:
            frame = None if latest_client_frame is None else latest_client_frame.copy()
        if frame is None:
            time.sleep(0.03)
            continue

        frame_idx += 1
        frame_time = time.time()
        frame_utc = _ts()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        with state_lock:
            mode = app_state["mode"]
            curl_count = app_state["curl_count"]
            curl_correct = app_state["curl_correct"]
            curl_stage = app_state["curl_stage"]

        current_feedback = ""
        reasons = []
        form_ok = False
        exercise_name = "—"
        debug = {}

        l_ang = r_ang = avg_ang = None
        rep_happened = False
        rep_entry = None
        pose_detected = False

        if res.pose_landmarks:
            pose_detected = True
            lm_img = res.pose_landmarks.landmark
            lm_world = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

            if mode == "curl":
                exercise_name = "Bicep Curl"
                LSH_i = lm_img[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                LEL_i = lm_img[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                LWR_i = lm_img[mp_pose.PoseLandmark.LEFT_WRIST.value]
                RSH_i = lm_img[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                REL_i = lm_img[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                RWR_i = lm_img[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                l_ang = calculate_angle(LSH_i, LEL_i, LWR_i)
                r_ang = calculate_angle(RSH_i, REL_i, RWR_i)
                avg_ang = (l_ang + r_ang) / 2.0

                form_ok, reasons, per_arm, metrics = posture_check_curl(lm_img, lm_world, debug=debug)

                MIN_CURL_TOP = 37
                MAX_CURL_BOTTOM = 163
                if avg_ang is not None and avg_ang >= MAX_CURL_BOTTOM:
                    curl_stage = "down"
                    current_feedback = "Ready: curl up."
                elif avg_ang is not None and avg_ang <= MIN_CURL_TOP and curl_stage == "down":
                    curl_count += 1
                    if form_ok:
                        curl_correct += 1
                    curl_stage = "up"
                    current_feedback = "Great rep! Lower slowly."
                    rep_happened = True
                    rep_entry = {
                        "ts": frame_utc,
                        "epoch": frame_time,
                        "session_id": SESSION_ID,
                        "mode": "curl",
                        "rep_index": curl_count,
                        "rep_correct_total": curl_correct,
                        "angles": {"left_elbow": l_ang, "right_elbow": r_ang, "avg_elbow": avg_ang},
                        "posture_ok": form_ok,
                        "reasons": reasons,
                        "debug": debug
                    }

                draw_colored_pose(frame, lm_img, color=(0, 255, 0) if form_ok else (0, 0, 255))

            elif mode == "squat":
                exercise_name = "Squat"

                # 3D-aware posture check (front/side robust)
                form_ok, reasons, per_side, metrics = posture_check_squat(lm_img, lm_world, debug=debug)

                # Use knee flexion from metrics if present (3D), else fallback to 2D
                l_ang = metrics["L"].get("knee_deg")
                r_ang = metrics["R"].get("knee_deg")
                if l_ang is None or r_ang is None:
                    LKNE_i = lm_img[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    LHIP_i = lm_img[mp_pose.PoseLandmark.LEFT_HIP.value]
                    LANK_i = lm_img[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    RKNE_i = lm_img[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    RHIP_i = lm_img[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    RANK_i = lm_img[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    l_ang = calculate_angle(LHIP_i, LKNE_i, LANK_i)
                    r_ang = calculate_angle(RHIP_i, RKNE_i, RANK_i)

                avg_ang = (float(l_ang) + float(r_ang)) / 2.0 if (l_ang is not None and r_ang is not None) else None

                # Yaw-aware thresholds
                yaw_deg = debug.get("yaw_deg", 0.0)
                GOING_DOWN_TH = 152.0 - 2.0 * (yaw_deg / 45.0)   # ~152 → 150
                STAND_UP_TH   = 168.0 - 2.0 * (yaw_deg / 45.0)   # ~168 → 166
                DEPTH_ANGLE   = 112.0 + 3.0 * (yaw_deg / 45.0)   # ~112 → 115

                # Stage + min-angle tracking across the entire down phase
                if avg_ang is not None and avg_ang < GOING_DOWN_TH:
                    if curl_stage != "down":
                        app_state["squat_min_knee"] = 999.0
                    curl_stage = "down"
                    app_state["squat_min_knee"] = min(app_state["squat_min_knee"], avg_ang)
                    current_feedback = "Sit back and down. Heels heavy."

                # Count when standing back up past threshold AND depth was achieved during down phase
                if avg_ang is not None and avg_ang > STAND_UP_TH and curl_stage == "down":
                    min_in_down = app_state.get("squat_min_knee") or 999.0
                    depth_reached = (
                        (bool(metrics["L"].get("hip_below_knee")) and bool(metrics["R"].get("hip_below_knee")))
                        or (min_in_down <= DEPTH_ANGLE)
                    )
                    if depth_reached:
                        curl_count += 1
                        if form_ok:
                            curl_correct += 1
                        current_feedback = "Nice rep! Stand tall and reset."
                        rep_happened = True
                        rep_entry = {
                            "ts": frame_utc,
                            "epoch": frame_time,
                            "session_id": SESSION_ID,
                            "mode": "squat",
                            "rep_index": curl_count,
                            "rep_correct_total": curl_correct,
                            "angles": {
                                "L_knee": l_ang, "R_knee": r_ang, "avg_knee": avg_ang,
                                "min_in_down": min_in_down
                            },
                            "posture_ok": form_ok,
                            "reasons": reasons,
                            "debug": debug
                        }
                    curl_stage = "up"

                draw_colored_pose(frame, lm_img, color=(0, 255, 0) if form_ok else (0, 0, 255))

        # ---- LOG: frame ----
        frame_log = {
            "ts": frame_utc,
            "epoch": frame_time,
            "session_id": SESSION_ID,
            "frame_index": frame_idx,
            "mode": mode,
            "pose_detected": pose_detected,
            "posture_ok": form_ok,
            "reasons": reasons,
            "angles": {"left_elbow": l_ang, "right_elbow": r_ang, "avg_elbow": avg_ang},  # kept keys; squat logs knees in rep log
            "curl_stage": curl_stage,
            "counters": {"curl_count": curl_count, "curl_correct": curl_correct},
            "debug": debug
        }
        log_frame(frame_log)

        # ---- LOG: rep ----
        if rep_happened and rep_entry:
            log_rep(rep_entry)

        with state_lock:
            app_state["curl_count"] = int(curl_count)
            app_state["curl_correct"] = int(curl_correct)
            app_state["curl_stage"] = curl_stage if isinstance(curl_stage, str) or curl_stage is None else str(curl_stage)
            app_state["posture_ok"] = bool(form_ok)
            app_state["last_reasons"] = list(reasons)
            app_state["last_feedback"] = str(current_feedback)
            app_state["exercise_name"] = exercise_name if mode in ("curl","squat") else "—"
            app_state["pose_detected"] = bool(pose_detected)

        ok2, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>My Gym Workout Buddy</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg0:#0a0c15; --bg1:#0e1222; --bg2:#151b34;
      --panel:rgba(18,22,42,.55); --panel-strong:rgba(18,22,42,.75);
      --stroke:#1f2745; --text:#e8ecf8; --muted:#a4aec5;
      --accent:#d6b16a; --accent-2:#8ca9ff; --good:#18c77a; --bad:#ff5d5d; --amber:#ffb020;
      --shadow:0 24px 70px rgba(0,0,0,.45);
      --radius:18px;
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{
      margin:0; font-family:'Inter',ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; color:var(--text);
      background:
        radial-gradient(1400px 800px at 10% -10%, #1b2a5b44, transparent 60%),
        radial-gradient(1000px 700px at 110% 0, #5b1b5544, transparent 55%),
        linear-gradient(180deg, var(--bg0), var(--bg1) 40%, var(--bg2));
      min-height:100vh; overflow-y:overlay; animation: pageFade .5s ease-out;
    }
    @keyframes pageFade { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform:none;} }
    body:before{
      content:""; position:fixed; inset:-20%; pointer-events:none;
      background: conic-gradient(from 0deg, #ffffff0a, #d6b16a22, #ffffff08, #8ca9ff1a, #ffffff0a);
      filter: blur(80px); animation: sheen 22s linear infinite; opacity:.45; mix-blend-mode:screen;
    }
    @keyframes sheen { to { transform: rotate(360deg); } }
    .wrap{max-width:1200px; margin:26px auto; padding:0 16px}

    /* ===== Top Bar ===== */
    .appbar{
      display:grid; grid-template-columns:auto 1fr auto; gap:14px; align-items:center;
      padding:16px; border:1px solid var(--stroke); border-radius:var(--radius);
      background:linear-gradient(135deg, var(--panel-strong), rgba(12,16,34,.55));
      backdrop-filter: blur(12px); box-shadow:var(--shadow);
      animation: pop .4s ease-out; position:sticky; top:16px; z-index:10;
    }
    @keyframes pop { from { transform: scale(.99); opacity:.85 } to { transform: scale(1); opacity:1 } }
    .brand{display:flex; align-items:center; gap:12px}
    .brand img{width:72px;height:72px;border-radius:16px;border:1px solid var(--stroke)}
    .title{
      font-weight:800; letter-spacing:.3px; font-size:22px;
      background: linear-gradient(90deg, #fff, #e8ecf8 45%, var(--accent) 92%);
      -webkit-background-clip:text; background-clip:text; color:transparent;
      filter: drop-shadow(0 2px 10px rgba(214,177,106,.1));
    }
    .pill{
      padding:6px 12px; border-radius:999px; background:#0f142a; color:var(--muted);
      font-size:12px; border:1px solid var(--stroke); margin-left:4px
    }
    .led{width:10px;height:10px;border-radius:50%;margin-left:10px;box-shadow:0 0 12px currentColor}
    .dot-ok{color:var(--good)} .dot-warn{color:var(--amber)}
    .hints{display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-start; opacity:.88}
    .hint{border:1px dashed #2b355e; border-radius:999px; padding:6px 10px; background:#0e1326; font-size:12px; color:var(--muted)}

    /* ===== Primary Controls ===== */
    .controls{display:flex; gap:10px; flex-wrap:wrap; align-items:center; justify-content:flex-end}
    .btn{
      position:relative; overflow:hidden;
      border:1px solid var(--stroke); background:#12182e; color:#e8ecf8;
      padding:12px 18px; border-radius:14px; cursor:pointer; user-select:none;
      transition: transform .08s ease, background .25s ease, border-color .25s ease, box-shadow .25s ease, opacity .25s ease;
      box-shadow: 0 12px 28px rgba(0,0,0,.28); font-weight:800; letter-spacing:.2px
    }
    .btn:hover{background:#172041; border-color:#2c3966; transform: translateY(-1px)}
    .btn:active{transform: translateY(0)}
    .btn:before{
      content:""; position:absolute; inset:0; background:radial-gradient(200px 60px at var(--mx,50%) -20%, #ffffff2a, transparent 60%);
      opacity:0; transition:opacity .25s ease;
    }
    .btn:hover:before{opacity:.55}
    .btn-accent{
      background:linear-gradient(135deg, #c9a963, #d6b16a);
      color:#0b0f19; border:none; box-shadow:0 14px 34px rgba(214,177,106,.26)
    }
    .btn-danger{
      background:linear-gradient(135deg,#ff6b6b,#ff4a4a); color:#fff; border:none;
      box-shadow:0 14px 34px rgba(255,90,90,.26)
    }

    /* ===== Secondary Nav Tabs ===== */
    .secnav{
      margin-top:14px; border:1px solid var(--stroke);
      background:linear-gradient(135deg, rgba(16,20,40,.60), rgba(12,16,34,.54));
      backdrop-filter: blur(10px); border-radius:var(--radius); box-shadow:var(--shadow);
      display:flex; align-items:center; gap:8px; padding:8px; position:sticky; top:86px; z-index:9;
    }
    .tab{
      position:relative; padding:10px 14px; border-radius:12px; color:var(--muted); font-weight:700; cursor:pointer;
      transition: color .25s ease, transform .08s ease;
    }
    .tab:hover{ color:#dfe6ff; transform:translateY(-1px) }
    .tab.active{ color:#fff }
    .tab.active:after{
      content:""; position:absolute; left:10px; right:10px; bottom:4px; height:2px;
      background:linear-gradient(90deg, transparent, var(--accent), transparent);
      box-shadow:0 0 20px rgba(214,177,106,.35);
      border-radius:2px; animation: glowline 1.6s ease-in-out infinite alternate;
    }

    .tabpanes{ position:relative; overflow:hidden; height:auto; margin-top:10px }
    .pane{
      border:1px solid var(--stroke);
      background:linear-gradient(135deg, rgba(16,20,40,.60), rgba(12,16,34,.54));
      backdrop-filter: blur(10px); border-radius:var(--radius); box-shadow:var(--shadow); padding:14px; position:relative;
      animation: pop .4s ease-out;
    }
    .pane h3{ margin:0 0 8px; font-size:16px; opacity:.95 }
    .pane p{ margin:6px 0; color:#cbd6ff; font-size:14px; line-height:1.6 }

    /* ===== Main Layout ===== */
    .grid{display:grid; grid-template-columns:1.4fr .8fr; gap:16px; margin-top:16px}
    @media (max-width: 980px){ .grid{grid-template-columns:1fr; } }
    .card{
      border:1px solid var(--stroke);
      background:linear-gradient(135deg, rgba(16,20,40,.60), rgba(12,16,34,.54));
      backdrop-filter: blur(10px);
      border-radius:var(--radius); box-shadow:var(--shadow); padding:14px; position:relative;
      animation: pop .4s ease-out;
    }
    .video{
      border-radius:16px; overflow:hidden; border:1px solid #212a44; width:100%; height:auto; display:block; background:#0f1426;
      transition: box-shadow .3s ease, border-color .3s ease;
    }
    .video.okglow{ box-shadow: 0 0 0 1px rgba(24,199,122,.25), 0 0 40px rgba(24,199,122,.10); border-color:#1a3c2a; }

    .stats{display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px}
    .stat{ display:flex; align-items:center; justify-content:space-between; padding:14px; background:#0f1426; border:1px solid var(--stroke); border-radius:14px }
    .stat .v{font-weight:800; font-size:22px}

    .badge{padding:8px 12px; border-radius:999px; font-weight:800; font-size:12px; letter-spacing:.25px}
    .ok{background:#0b2a19; color:var(--good); border:1px solid #155b36}
    .notok{background:#2a1212; color:var(--bad); border:1px solid #5a1f1f}

    .muted{color:var(--muted); font-size:13px}
    ul.reasons{margin:8px 0 0 18px; padding:0}
    ul.reasons li{margin:6px 0; color:#cbd6ff}
  </style>
</head>
<body>
  <div id="app-root" class="wrap" role="application" aria-label="My Gym Buddy">
    <!-- ===== App Bar ===== -->
    <div class="appbar" aria-label="Top bar">
      <div class="brand">
        <img src="{{ url_for('static', filename='My_Gym_Logo.png') }}" alt="Brand logo" />
        <div class="title">My Gym Workout Assistant</div>
        <span id="mode-pill" class="pill" aria-live="polite">Bicep Curl</span>
        <span id="pose-led" class="led dot-warn" title="Pose detection status" aria-label="Pose detection status"></span>
      </div>
      <div class="hints">
        <span class="hint">Shortcuts — 1 Curl · 2 Squat · S Start · X Stop · R Reset · A Analyze · D Debug</span>
      </div>
      <div class="controls" aria-label="Controls">
        <!-- Mode selector -->
        <label for="mode-select" class="muted" style="margin-right:6px;">Mode</label>
        <select id="mode-select" class="pill" style="margin-right:8px">
          <option value="curl" selected>Bicep Curl</option>
          <option value="squat">Squat</option>
        </select>

        <form id="start-form" method="POST" action="{{ url_for('set_mode') }}" aria-label="Start session">
          <input id="mode-input" type="hidden" name="mode" value="curl" />
          <button class="btn btn-accent" type="submit" aria-keyshortcuts="S">Start</button>
        </form>
        <form id="stop-form" method="POST" action="{{ url_for('stop') }}" aria-label="Stop session">
          <button class="btn btn-danger" type="submit" aria-keyshortcuts="X">Stop</button>
        </form>
        <form id="reset-form" method="POST" action="{{ url_for('reset') }}" aria-label="Reset counters">
          <button class="btn" type="submit" aria-keyshortcuts="R">Reset</button>
        </form>
        <form id="debug-form" method="POST" action="{{ url_for('toggle_debug') }}" aria-label="Toggle debug">
          <button class="btn" type="submit" aria-keyshortcuts="D">Debug</button>
        </form>
        <form id="analyze-form" method="POST" action="{{ url_for('analyze') }}" aria-label="Analyze session">
          <input type="hidden" name="n_frames" value="800">
          <button id="analyze-btn" class="btn btn-analyze" type="submit" aria-keyshortcuts="A">Analyze</button>
        </form>
      </div>
    </div>

    <!-- ===== Main Workout Area ===== -->
    <div class="grid">
      <div class="card" aria-label="Camera feed">
        <img class="video" id="video-el" src="{{ url_for('video_feed') }}" alt="Live workout video">
      </div>

      <div class="card" aria-label="Session stats and guidance">
        <div class="stats" role="group" aria-label="Rep counters">
          <div class="stat"><div>Reps</div><div class="v" id="reps">0</div></div>
          <div class="stat"><div>Correct</div><div class="v" id="correct">0</div></div>
        </div>
        <div class="stat" style="margin-bottom:12px;">
          <div>Posture</div><div id="posture-badge" class="badge notok" aria-live="polite">Not OK</div>
        </div>
        <div class="stat" style="margin-bottom:12px;">
          <div>Mode</div><div class="v" id="mode">—</div>
        </div>
        <div>
          <div class="muted">Why it’s wrong</div>
          <ul class="reasons" id="reasons"></ul>
        </div>
        <div style="margin-top: 12px;">
          <div class="muted">Tip</div>
          <div id="feedback" style="margin-top:6px;">—</div>
        </div>
      </div>
    </div>

    <!-- Inline analysis fallback -->
    <div class="card" style="margin-top:16px;" aria-label="AI analysis">
      <div class="muted">AI Analysis</div>
      <div id="analysis" style="margin-top: 6px;">Click <b>Analyze</b> for a refined recap of your set.</div>
    </div>
  </div>

  <script>
    // gradient sheen follows cursor on buttons
    document.querySelectorAll('.btn').forEach(btn=>{
      btn.addEventListener('mousemove',e=>{
        const r = btn.getBoundingClientRect();
        btn.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      });
    });

    const toast = document.createElement('div');

    async function postForm(form, onJSON) {
      const fd = new FormData(form);
      const res = await fetch(form.action, { method: 'POST', body: fd });
      const type = res.headers.get('content-type') || '';
      if (type.includes('application/json')) {
        const data = await res.json();
        if (onJSON) onJSON(data);
      }
    }

    // ===== Forms =====
    const modeSelect = document.getElementById('mode-select');
    const modeInput  = document.getElementById('mode-input');
    const modePill   = document.getElementById('mode-pill');

    modeSelect.addEventListener('change', () => { modeInput.value = modeSelect.value; });

    document.getElementById('start-form').addEventListener('submit', e => { e.preventDefault(); postForm(e.target); });
    document.getElementById('stop-form').addEventListener('submit', e => {
      e.preventDefault();
      postForm(e.target, () => {
        document.getElementById('analysis').innerHTML = 'Stopped. Click <b>Analyze</b> for a recap or <b>Start</b> to resume.';
      });
    });
    document.getElementById('reset-form').addEventListener('submit', e => {
      e.preventDefault();
      postForm(e.target, () => {
        document.getElementById('analysis').innerHTML = 'Click <b>Analyze</b> for a refined recap of your set.';
      });
    });
    document.getElementById('debug-form').addEventListener('submit', e => { e.preventDefault(); postForm(e.target); });

    document.getElementById('analyze-form').addEventListener('submit', e => {
      e.preventDefault();
      const panel = document.getElementById('analysis');
      panel.innerHTML = 'Analyzing…';
      postForm(e.target, (data) => {
        panel.innerHTML = data.html || 'No analysis.';
      });
    });

    // ===== Live stats poll =====
    const elReps = document.getElementById('reps');
    const elCorrect = document.getElementById('correct');
    const elBadge = document.getElementById('posture-badge');
    const elMode = document.getElementById('mode');
    const elReasons = document.getElementById('reasons');
    const elFeedback = document.getElementById('feedback');
    const led = document.getElementById('pose-led');
    const videoEl = document.getElementById('video-el');

    function setBadge(ok) { elBadge.textContent = ok ? 'OK' : 'Not OK'; elBadge.className = 'badge ' + (ok ? 'ok' : 'notok'); }
    function setPoseLed(detected) { led.className = 'led ' + (detected ? 'dot-ok' : 'dot-warn'); }

    async function poll() {
      try {
        const r = await fetch('{{ url_for("stats") }}');
        const s = await r.json();
        elReps.textContent = s.curl_count ?? 0;
        elCorrect.textContent = s.curl_correct ?? 0;
        setBadge(!!s.posture_ok);
        setPoseLed(!!s.pose_detected);
        elMode.textContent = s.exercise_name || '—';
        elReasons.innerHTML = '';
        (s.last_reasons || []).slice(0,5).forEach(reason => {
          const li = document.createElement('li'); li.textContent = reason; elReasons.appendChild(li);
        });
        elFeedback.textContent = s.last_feedback || '—';

        if (s.exercise_name && s.exercise_name !== '—') {
          modePill.textContent = s.exercise_name;
          const normalized = s.exercise_name.toLowerCase().includes('squat') ? 'squat' : 'curl';
          if (modeSelect.value !== normalized) {
            modeSelect.value = normalized;
            modeInput.value = normalized;
          }
        }
      } catch (_) {}
      setTimeout(poll, 800);
    }
    poll();

    // ===== Keyboard shortcuts =====
    async function setMode(m){
      modeInput.value = m;
      await fetch('{{ url_for("set_mode") }}', {
        method:'POST',
        body: new URLSearchParams({mode:m})
      });
    }
    document.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (['1','2','s','x','r','d','a'].includes(k)) e.preventDefault();
      if (k === '1') setMode('curl');
      if (k === '2') setMode('squat');
      if (k === 's') document.querySelector('#start-form button').click();
      if (k === 'x') document.querySelector('#stop-form button').click();
      if (k === 'r') document.querySelector('#reset-form button').click();
      if (k === 'd') document.querySelector('#debug-form button').click();
      if (k === 'a') document.querySelector('#analyze-form button').click();
    });

    // ===== Client-side camera capture to server =====
    const camVideo = document.createElement('video');
    camVideo.autoplay = true; camVideo.playsInline = true; camVideo.muted = true;
    const camCanvas = document.createElement('canvas');
    let sending = false;

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        camVideo.srcObject = stream;

        const loop = async () => {
          if (!camVideo.videoWidth) { requestAnimationFrame(loop); return; }
          camCanvas.width = camVideo.videoWidth;
          camCanvas.height = camVideo.videoHeight;
          const ctx = camCanvas.getContext('2d', { willReadFrequently: true });
          ctx.drawImage(camVideo, 0, 0, camCanvas.width, camCanvas.height);

          if (!sending) {
            sending = true;
            camCanvas.toBlob(async (blob) => {
              const fd = new FormData();
              fd.append('frame', blob, 'frame.jpg');
              try { await fetch('{{ url_for("upload_frame") }}', { method: 'POST', body: fd }); }
              catch(e) {}
              finally { sending = false; }
            }, 'image/jpeg', 0.7);
          }
          setTimeout(()=>requestAnimationFrame(loop), 100); // ~10 fps
        };
        requestAnimationFrame(loop);
      } catch (err) {
        console.error('Camera error', err);
        alert('Please allow camera access to use Workout Buddy.');
      }
    }

    // Try MJPEG first; fall back to polling /peek.jpg if it doesn’t render quickly.
    let fallbackTimer;
    function tryMjpegThenFallback() {
      videoEl.src = '{{ url_for("video_feed") }}';
      // If no pixels appear within ~3s, switch to /peek.jpg loop
      fallbackTimer = setTimeout(startPeekPolling, 3000);
      videoEl.onload = () => { clearTimeout(fallbackTimer); };
    }

    function startPeekPolling() {
      function tick() {
        videoEl.src = '/peek.jpg?ts=' + Date.now();
        setTimeout(tick, 100); // ~10 fps
      }
      tick();
    }

    startCamera();
    tryMjpegThenFallback();
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        INDEX_HTML,
        session_id=SESSION_ID,
        frame_log=os.path.relpath(FRAME_LOG_PATH, os.getcwd()),
        rep_log=os.path.relpath(REP_LOG_PATH, os.getcwd())
    )

# --- STREAM: unbuffered MJPEG so Render/NGINX don’t stall it ---
@app.route('/video_feed')
def video_feed():
    return Response(
        video_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive',
        }
    )

# --- SNAPSHOT: latest processed frame (fallback/debug) ---
@app.route('/peek.jpg')
def peek_jpg():
    with latest_frame_lock:
        frame = None if latest_client_frame is None else latest_client_frame.copy()
    if frame is None:
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        ok, jpeg = cv2.imencode('.jpg', tiny, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(jpeg.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})
    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpeg.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    mode = (request.form.get("mode") or "curl").strip().lower()
    if mode not in {"curl", "squat"}:
        mode = "curl"
    with state_lock:
        app_state["mode"] = mode
        app_state["curl_stage"] = None
        app_state["ema_state"].clear()
        app_state["exercise_name"] = "Bicep Curl" if mode == "curl" else "Squat"
        app_state["posture_ok"] = False
        app_state["last_reasons"] = []
        app_state["last_feedback"] = ""
        app_state["squat_min_knee"] = None
    return jsonify({"ok": True, "mode": app_state["mode"]})

@app.route('/stop', methods=['POST'])
def stop():
    with state_lock:
        app_state["mode"] = None
        app_state["curl_stage"] = None
        app_state["ema_state"].clear()
        app_state["posture_ok"] = False
        app_state["last_reasons"] = []
        app_state["last_feedback"] = "Stopped."
        app_state["exercise_name"] = "—"
    return jsonify({"ok": True, "mode": app_state["mode"]})

@app.route('/reset', methods=['POST'])
def reset():
    with state_lock:
        app_state["curl_count"] = 0
        app_state["curl_correct"] = 0
        app_state["curl_stage"] = None
        app_state["ema_state"].clear()
        app_state["last_reasons"] = []
        app_state["posture_ok"] = False
        app_state["last_feedback"] = ""
        app_state["analysis_text_html"] = ""
        app_state["squat_min_knee"] = None
    return jsonify({"ok": True})

@app.route('/toggle_debug', methods=['POST'])
def toggle_debug():
    with state_lock:
        app_state["show_debug"] = not app_state["show_debug"]
    return jsonify({"ok": True, "show_debug": app_state["show_debug"]})

@app.route('/stats')
def stats():
    with state_lock:
        payload = {
            "curl_count": app_state["curl_count"],
            "curl_correct": app_state["curl_correct"],
            "posture_ok": app_state["posture_ok"],
            "last_reasons": app_state["last_reasons"],
            "last_feedback": app_state["last_feedback"],
            "exercise_name": app_state["exercise_name"],
            "pose_detected": app_state["pose_detected"],
        }
    return jsonify(sanitize(payload))

# ========== GPT Analysis ==========
def read_jsonl_tail(path, n=800):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    out = []
    for line in lines:
        try: out.append(json.loads(line))
        except: pass
    return out

def build_analysis_prompt(frames, reps):
    # concise, layman-readable HTML
    return (
        "You are a friendly strength coach. Using the logs below, generate CLEAR HTML (no markdown). "
        "Audience is a beginner. Use short sentences and simple words. Structure as:\n"
        "<h3>Summary</h3><div class='callout'><span class='dot warn'></span><div>One or two lines on overall form.</div></div>\n"
        "<h3>What you did well</h3><ul>3-5 bullet points.</ul>\n"
        "<h3>What to fix (in order)</h3><ol>Top 3 issues. Each 1 sentence.</ol>\n"
        "<h3>How to fix it</h3><ul>4-6 specific drills/cues. Example: 'Keep elbows pinned: squeeze a towel between elbow and ribs.'</ul>\n"
        "<div class='targets'><div class='target'><b>Next goal</b><br>e.g., Full extension each rep</div>"
        "<div class='target'><b>Tempo</b><br>2s up, 2s down</div>"
        "<div class='target'><b>Set target</b><br>2 sets × 10–12 reps</div></div>\n"
        "Avoid jargon and numbers unless meaningful. Do NOT wrap the whole thing in <html> or <body>.\n\n"
        f"Frames (recent {len(frames)}):\n" +
        "\n".join([
            f"- t={f.get('epoch'):.3f} ok={f.get('posture_ok')} avg_elbow={round((f.get('angles') or {}).get('avg_elbow') or 0,2)} "
            f"stage={f.get('curl_stage')} reasons={'; '.join(f.get('reasons') or [])}"
            for f in frames[-100:]
        ]) + "\n\n"
        f"Reps ({len(reps)} total):\n" +
        "\n".join([
            f"- rep#{r.get('rep_index')} ok={r.get('posture_ok')} "
            f"avg_elbow={round((r.get('angles') or {}).get('avg_elbow') or 0,2)} reasons={'; '.join(r.get('reasons') or [])}"
            for r in reps[-50:]
        ])
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    n_frames = int(request.form.get("n_frames", 800))
    frames = read_jsonl_tail(FRAME_LOG_PATH, n_frames)
    reps = read_jsonl_tail(REP_LOG_PATH, 200)
    prompt = build_analysis_prompt(frames, reps)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        max_output_tokens=700
    )
    text = resp.output_text.strip()

    # If the model somehow returns non-HTML, wrap safely
    if "<" not in text or ">" not in text:
        text = "<div class='callout'><span class='dot warn'></span><div>" + \
               html_mod.escape(text) + "</div></div>"

    with state_lock:
        app_state["analysis_text_html"] = text
    return jsonify(sanitize({"ok": True, "html": text}))

if __name__ == '__main__':
    print(f"[INFO] Session: {SESSION_ID}")
    print(f"[INFO] Frame logs -> {FRAME_LOG_PATH}")
    print(f"[INFO] Rep logs   -> {REP_LOG_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
