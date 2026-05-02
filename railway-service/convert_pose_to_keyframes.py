"""Convert RTMPose JSON output to HeroRally KEYFRAME_ANGLES format.

Reads /tmp/alcaraz_pose.json and emits TypeScript-ready keyframes
matching the polar-angle convention used in components/HeroRally.tsx.

v2 — joint-position-driven smoothing:
  1. Extract 13 joint positions per frame from RTMPose.
  2. Normalize each frame so hip-midpoint -> (0.5, 0.55) and the
     hip-shoulder distance matches the figure's trunk length.
  3. Apply a per-axis 5-frame moving average to every joint to remove
     pose-extraction jitter at the position level (smoother than
     smoothing derived angles, since position noise is well-behaved
     while angles can flip 180 deg at near-parallel limbs).
  4. Compute angles from the smoothed positions in HeroRally's polar
     convention (0 deg=+x, 90=+y/down, -90=-y/up).
  5. Unwrap each angle channel into a continuous time series and
     apply a second 3-frame moving average.
  6. Mirror angles around the y-axis (Alcaraz swings left->right; the
     rally figure hits right->left).
"""
import json
import math


def polar_angle_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def vec(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


def lm_xy(lm):
    return (lm["x"], lm["y"])


# BONE_LENGTHS.trunk in the figure (hip-center to shoulder-midpoint)
TRUNK_FIGURE_LEN = 0.25


def normalize_to_figure_space(pts, ref_hip_mid, ref_scale):
    """Translate + scale image-space points using a GLOBAL reference
    (the average hip-midpoint and torso scale across the window) so
    per-frame hip motion is preserved as a real shift in figure-space.
    Per-frame normalization would otherwise lock hip-center to (0.5,
    0.55) and erase the weight-transfer signal."""
    out = {}
    for name, (x, y) in pts.items():
        nx = 0.5 + (x - ref_hip_mid[0]) * ref_scale
        ny = 0.55 + (y - ref_hip_mid[1]) * ref_scale
        out[name] = (nx, ny)
    return out


def moving_average_2d(series, window=5):
    n = len(series)
    out = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sx = sum(p[0] for p in series[lo:hi]) / (hi - lo)
        sy = sum(p[1] for p in series[lo:hi]) / (hi - lo)
        out.append((sx, sy))
    return out


def unwrap_angle_series(angles):
    """Add +/-360 multiples so consecutive samples differ by < 180."""
    if not angles:
        return []
    out = [angles[0]]
    for a in angles[1:]:
        prev = out[-1]
        delta = a - prev
        while delta > 180:
            delta -= 360
        while delta <= -180:
            delta += 360
        out.append(prev + delta)
    return out


def moving_average_1d(series, window=3):
    n = len(series)
    out = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(series[lo:hi]) / (hi - lo))
    return out


def main():
    data = json.load(open("/tmp/alcaraz_pose.json"))
    frames = data["frames"]
    fps = data["fps_sampled"]

    # 1. Find swing peak (right-wrist speed maximum).
    speeds = []
    for i, f in enumerate(frames):
        if i == 0:
            speeds.append(0.0)
            continue
        prev = frames[i - 1]
        try:
            rw_now = lm_xy(f["landmarks"][16])
            rw_prev = lm_xy(prev["landmarks"][16])
            dx = rw_now[0] - rw_prev[0]
            dy = rw_now[1] - rw_prev[1]
            speeds.append(math.hypot(dx, dy))
        except Exception:
            speeds.append(0.0)
    peak_idx = max(range(len(speeds)), key=lambda i: speeds[i])
    print(f"# total frames: {len(frames)}, peak (contact): frame {peak_idx} "
          f"t={frames[peak_idx]['timestamp_ms']}ms")

    # 2. Window: ~1.0s pre-contact (full prep) + ~0.55s post (full
    # follow-through THROUGH return to ready) so the figure's pose at
    # the end of the cycle is close to its pose at the start, keeping
    # the loop seam smooth. Earlier window cropped at 0.2s post and
    # left the arm wrapped over the off-shoulder, forcing Hermite to
    # whip 130° back to idle in the recovery phase.
    pre_frames = int(1.0 * fps)
    post_frames = int(0.55 * fps)
    start = max(0, peak_idx - pre_frames)
    end = min(len(frames), peak_idx + post_frames + 1)
    window = frames[start:end]
    n_window = len(window)
    contact_local_idx = peak_idx - start
    print(f"# window: {n_window} frames, contact at local idx {contact_local_idx}")

    # 3. Extract & normalize per-frame joints + racket head.
    JOINT_INDEX = {
        "nose": 0,
        "ls": 11, "rs": 12, "le": 13, "re": 14, "lw": 15, "rw": 16,
        "lh": 23, "rh": 24, "lk": 25, "rk": 26, "la": 27, "ra": 28,
    }
    JOINT_NAMES = list(JOINT_INDEX.keys())

    # Compute GLOBAL reference: average hip-midpoint and average torso
    # scale across the entire window. Used for all per-frame
    # normalization so hip motion is preserved as a real shift.
    raw_per_frame = []
    for f in window:
        lms = f["landmarks"]
        raw = {name: lm_xy(lms[JOINT_INDEX[name]]) for name in JOINT_NAMES}
        raw_per_frame.append(raw)

    avg_hip = (
        sum((r["lh"][0] + r["rh"][0]) / 2 for r in raw_per_frame) / n_window,
        sum((r["lh"][1] + r["rh"][1]) / 2 for r in raw_per_frame) / n_window,
    )
    avg_sh = (
        sum((r["ls"][0] + r["rs"][0]) / 2 for r in raw_per_frame) / n_window,
        sum((r["ls"][1] + r["rs"][1]) / 2 for r in raw_per_frame) / n_window,
    )
    avg_torso_len = math.hypot(avg_sh[0] - avg_hip[0], avg_sh[1] - avg_hip[1])
    ref_scale = TRUNK_FIGURE_LEN / max(avg_torso_len, 1e-6)

    # Racket-head is derived from the SMOOTHED forearm direction (not
    # the per-frame racket_head detection). The detector wobbles ~50°
    # between adjacent frames during fast follow-through; using
    # forearm extension produces a clean, racket-tracks-wrist arc with
    # no jitter in the post-contact phase.
    per_frame_pts = []
    for raw in raw_per_frame:
        pts = normalize_to_figure_space(raw, avg_hip, ref_scale)
        per_frame_pts.append(pts)

    # 4. Smooth joint trajectories (5-frame centered moving average).
    smoothed = {}
    for name in JOINT_NAMES:
        series = [pts[name] for pts in per_frame_pts]
        smoothed[name] = moving_average_2d(series, window=5)

    # 5. Compute angles from smoothed positions.
    raw_angles = {key: [] for key in [
        "trunk", "neck", "uArmL", "fArmL", "uArmR", "fArmR",
        "thighL", "shinL", "thighR", "shinR", "racket"
    ]}
    hip_centers = []
    for i in range(n_window):
        nose = smoothed["nose"][i]
        ls = smoothed["ls"][i]; rs = smoothed["rs"][i]
        le = smoothed["le"][i]; re = smoothed["re"][i]
        lw = smoothed["lw"][i]; rw = smoothed["rw"][i]
        lh = smoothed["lh"][i]; rh = smoothed["rh"][i]
        lk = smoothed["lk"][i]; rk = smoothed["rk"][i]
        la = smoothed["la"][i]; ra = smoothed["ra"][i]

        hip_c = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        sh_c = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_centers.append(hip_c)

        raw_angles["trunk"].append(polar_angle_deg(*vec(hip_c, sh_c)))
        raw_angles["neck"].append(polar_angle_deg(*vec(sh_c, nose)))
        raw_angles["uArmR"].append(polar_angle_deg(*vec(rs, re)))
        raw_angles["fArmR"].append(polar_angle_deg(*vec(re, rw)))
        raw_angles["uArmL"].append(polar_angle_deg(*vec(ls, le)))
        raw_angles["fArmL"].append(polar_angle_deg(*vec(le, lw)))
        raw_angles["thighL"].append(polar_angle_deg(*vec(lh, lk)))
        raw_angles["shinL"].append(polar_angle_deg(*vec(lk, la)))
        raw_angles["thighR"].append(polar_angle_deg(*vec(rh, rk)))
        raw_angles["shinR"].append(polar_angle_deg(*vec(rk, ra)))

        # Racket extends along the smoothed forearm direction —
        # eliminates the per-frame detector noise that was causing
        # 50-degree jumps in the post-contact phase.
        raw_angles["racket"].append(polar_angle_deg(*vec(re, rw)))

    # Fixed racketLen for every frame — was bouncing 0.08 to 0.16
    # because of detection-distance noise.
    FIXED_RACKET_LEN = 0.10
    racket_lens = [FIXED_RACKET_LEN] * n_window

    # 6. Unwrap each angle channel + re-smooth.
    smoothed_angles = {}
    for key, series in raw_angles.items():
        unwrapped = unwrap_angle_series(series)
        averaged = moving_average_1d(unwrapped, window=3)
        rewrapped = []
        for a in averaged:
            while a > 180:
                a -= 360
            while a <= -180:
                a += 360
            rewrapped.append(a)
        smoothed_angles[key] = rewrapped

    # 7. Mirror around y-axis to flip swing direction.
    def mirror(a):
        m = 180.0 - a
        while m > 180:
            m -= 360
        while m <= -180:
            m += 360
        return m

    for key in smoothed_angles:
        smoothed_angles[key] = [mirror(a) for a in smoothed_angles[key]]

    # 7.5. Cap arm vertical reach so the racket tip during follow-
    # through stays at shoulder height instead of sweeping above the
    # head. Pulls each angle toward the nearest horizontal axis (0 or
    # +/-180) by 50% when the angle has a strong vertical component.
    # Preserves left/right sweep direction (key for the cross-body
    # follow-through arc that gives the swing its real-tennis feel).
    def cap_vertical(angle, gain=0.5):
        # angle in (-180, 180]; positive = pointing down (positive y)
        # Only cap negative angles (pointing up).
        if angle >= 0:
            return angle
        # Negative => upper half. Pull toward nearest horizontal:
        # cos<0 (angle in (-180, -90)): pull toward -180.
        # cos>=0 (angle in (-90, 0)):    pull toward 0.
        if angle < -90:
            target = -180.0
        else:
            target = 0.0
        return angle + (target - angle) * gain

    # fArmR uses the up-axis cap above (forearm rotates through
    # straight-up during follow-through).
    smoothed_angles["fArmR"] = [cap_vertical(a, 0.5) for a in smoothed_angles["fArmR"]]

    # uArmR is different — during post-contact the upper arm reaches
    # past horizontal-LEFT (uArmR raw ~170, or ~-170 wrapping into
    # ~190 unwrapped). The "up" angles for uArmR live in the
    # unwrapped (140, 260) range. Cap excess above 145 by 70%.
    def cap_uArmR_horizontal(raw):
        unwrapped = raw if raw >= 0 else raw + 360
        if unwrapped <= 145:
            return raw
        capped = 145 + (unwrapped - 145) * 0.30
        return capped if capped <= 180 else capped - 360
    smoothed_angles["uArmR"] = [cap_uArmR_horizontal(a) for a in smoothed_angles["uArmR"]]
    # racket follows fArmR (we set racket = fArmR earlier) — re-sync
    # after the cap so they stay aligned.
    smoothed_angles["racket"] = list(smoothed_angles["fArmR"])

    # 8. hipCenter: dampen motion (so the figure doesn't walk across
    # the screen) + mirror around 0.5 to flip swing direction.
    avg_hip_x = sum(c[0] for c in hip_centers) / n_window
    avg_hip_y = sum(c[1] for c in hip_centers) / n_window
    HIP_SHIFT_GAIN = 0.4
    final_hips = []
    for c in hip_centers:
        # Dampen relative to the per-window average.
        sx_dampened = avg_hip_x + (c[0] - avg_hip_x) * HIP_SHIFT_GAIN
        sy_dampened = avg_hip_y + (c[1] - avg_hip_y) * HIP_SHIFT_GAIN
        # Mirror around 0.5.
        final_hips.append((round(1.0 - sx_dampened, 4), round(sy_dampened, 4)))

    # 9. t values: contact at 0.65, final at 0.85.
    t_contact = 0.65
    t_final = 0.85
    ts = []
    for i in range(n_window):
        if i <= contact_local_idx:
            ts.append(round((i / contact_local_idx) * t_contact, 4) if contact_local_idx > 0 else 0.0)
        else:
            local = i - contact_local_idx
            post = n_window - 1 - contact_local_idx
            ts.append(round(t_contact + (local / post) * (t_final - t_contact), 4))

    # 10. Forward-evaluate the bone chain at the contact frame so the
    # emitted RACKET_CONTACT_NORM_X/Y match what HeroRally actually
    # renders. The raw RTMPose-normalized racket position would be
    # wrong: HeroRally renders with a 0.4-dampened hipCenter and the
    # figure's fixed BONE_LENGTHS, so the wrist (and racket extending
    # from it) lands at a different point than the raw pose.
    BONE_HIP_HALF = 0.04
    BONE_SH_HALF = 0.05
    BONE_TRUNK = 0.25
    BONE_UPPER_ARM = 0.139
    BONE_FOREARM = 0.133

    def polar_pt(L, deg):
        r = math.radians(deg)
        return (L * math.cos(r), L * math.sin(r))

    hcx, hcy = final_hips[contact_local_idx]
    a_trunk = smoothed_angles["trunk"][contact_local_idx]
    a_uArmR = smoothed_angles["uArmR"][contact_local_idx]
    a_fArmR = smoothed_angles["fArmR"][contact_local_idx]
    a_racket = smoothed_angles["racket"][contact_local_idx]
    racketLen_at_contact = FIXED_RACKET_LEN

    tx, ty = polar_pt(BONE_TRUNK, a_trunk)
    sh_mid_render = (hcx + tx, hcy + ty)
    rs_render = (sh_mid_render[0] + BONE_SH_HALF, sh_mid_render[1])
    ueRx, ueRy = polar_pt(BONE_UPPER_ARM, a_uArmR)
    re_render = (rs_render[0] + ueRx, rs_render[1] + ueRy)
    feRx, feRy = polar_pt(BONE_FOREARM, a_fArmR)
    rw_render = (re_render[0] + feRx, re_render[1] + feRy)
    rkx, rky = polar_pt(racketLen_at_contact, a_racket)
    rh_render = (rw_render[0] + rkx, rw_render[1] + rky)

    print(f"// CONTACT racket head (bone-chain rendered, figure-space):")
    print(f"//   x={round(rh_render[0], 4)}, y={round(rh_render[1], 4)}")
    print(f"// Update HeroRally.tsx:")
    print(f"//   const RACKET_CONTACT_NORM_X = {round(rh_render[0], 4)}")
    print(f"//   const RACKET_CONTACT_NORM_Y = {round(rh_render[1], 4)}")

    # 11. Emit keyframes.
    print(f"\n// === KEYFRAMES (Alcaraz, smoothed: 5-frame pos + 3-frame angle) ===")
    print(f"// {n_window} frames, contact at local idx {contact_local_idx} (t={t_contact})")
    for i in range(n_window):
        hcx, hcy = final_hips[i]
        a = {key: round(smoothed_angles[key][i], 1) for key in smoothed_angles}
        rl = round(racket_lens[i], 3)
        print(
            f'  {{ t: {ts[i]}, hipCenter: [{hcx}, {hcy}], '
            f'trunk: {a["trunk"]}, neck: {a["neck"]},\n'
            f'    uArmL: {a["uArmL"]}, fArmL: {a["fArmL"]}, '
            f'uArmR: {a["uArmR"]}, fArmR: {a["fArmR"]},\n'
            f'    thighL: {a["thighL"]}, shinL: {a["shinL"]}, '
            f'thighR: {a["thighR"]}, shinR: {a["shinR"]},\n'
            f'    racket: {a["racket"]}, racketLen: {rl} }},'
        )


if __name__ == "__main__":
    main()
