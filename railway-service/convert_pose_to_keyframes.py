"""Convert RTMPose JSON output to HeroRally KEYFRAME_ANGLES format.

Reads /tmp/alcaraz_pose.json and emits TypeScript-ready keyframes
matching the polar-angle convention used in components/HeroRally.tsx.

Polar convention (matches HeroRally):
  0° = +x (right), 90° = +y (down), 180° = -x (left), -90° = -y (up).

Each per-frame keyframe captures:
  hipCenter, trunk, neck, uArmL/R, fArmL/R, thighL/R, shinL/R,
  racket, racketLen (constant 0.13).
"""
import json
import math


def polar_angle_deg(dx: float, dy: float) -> float:
    """atan2(dy, dx) in degrees, using HeroRally's screen-space convention."""
    return math.degrees(math.atan2(dy, dx))


def vec(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


def lm_xy(lm):
    return (lm["x"], lm["y"])


def main():
    data = json.load(open("/tmp/alcaraz_pose.json"))
    frames = data["frames"]
    fps = data["fps_sampled"]

    # 1. Find the swing peak: frame with maximum racket-hand speed.
    speeds = []
    for i, f in enumerate(frames):
        if i == 0:
            speeds.append(0.0)
            continue
        prev = frames[i - 1]
        # Use right_wrist (id=16). Fall back to racket_head if available.
        try:
            rw_now = lm_xy(f["landmarks"][16])
            rw_prev = lm_xy(prev["landmarks"][16])
            dx = rw_now[0] - rw_prev[0]
            dy = rw_now[1] - rw_prev[1]
            speeds.append(math.hypot(dx, dy))
        except Exception:
            speeds.append(0.0)

    peak_idx = max(range(len(speeds)), key=lambda i: speeds[i])
    print(f"# total frames: {len(frames)}, peak (contact): frame {peak_idx} t={frames[peak_idx]['timestamp_ms']}ms")

    # 2. Window the swing: prep through mid follow-through. We cap the
    # post-contact frames at ~0.2s so the loop seam lands during early
    # follow-through (close to the idle/prep pose) rather than at the
    # high-wrap finish, which would force a 180° spin to recover.
    pre_frames = int(1.1 * fps)   # ~33 frames before contact (full prep)
    post_frames = int(0.2 * fps)  # ~6 frames after contact (mid follow-through)
    start = max(0, peak_idx - pre_frames)
    end = min(len(frames), peak_idx + post_frames + 1)
    window = frames[start:end]
    print(f"# window: frames {start}..{end-1} ({len(window)} frames)")
    print(f"# contact within window: index {peak_idx - start}")

    # 3. For each frame in window, compute the figure's polar angles.
    keyframes = []
    n_window = len(window)
    contact_local_idx = peak_idx - start
    for local_idx, f in enumerate(window):
        lms = f["landmarks"]

        nose = lm_xy(lms[0])
        ls = lm_xy(lms[11])
        rs = lm_xy(lms[12])
        le = lm_xy(lms[13])
        re = lm_xy(lms[14])
        lw = lm_xy(lms[15])
        rw = lm_xy(lms[16])
        lh = lm_xy(lms[23])
        rh = lm_xy(lms[24])
        lk = lm_xy(lms[25])
        rk = lm_xy(lms[26])
        la = lm_xy(lms[27])
        ra = lm_xy(lms[28])

        hip_cx = (lh[0] + rh[0]) / 2
        hip_cy = (lh[1] + rh[1]) / 2
        sh_cx = (ls[0] + rs[0]) / 2
        sh_cy = (ls[1] + rs[1]) / 2

        # Trunk angle: hipCenter -> shoulderCenter
        trunk_dx, trunk_dy = vec((hip_cx, hip_cy), (sh_cx, sh_cy))
        trunk = polar_angle_deg(trunk_dx, trunk_dy)

        # Neck: shoulderCenter -> nose
        neck_dx, neck_dy = vec((sh_cx, sh_cy), nose)
        neck = polar_angle_deg(neck_dx, neck_dy)

        # Right arm
        uArmR_dx, uArmR_dy = vec(rs, re)
        fArmR_dx, fArmR_dy = vec(re, rw)
        uArmR = polar_angle_deg(uArmR_dx, uArmR_dy)
        fArmR = polar_angle_deg(fArmR_dx, fArmR_dy)

        # Left arm
        uArmL_dx, uArmL_dy = vec(ls, le)
        fArmL_dx, fArmL_dy = vec(le, lw)
        uArmL = polar_angle_deg(uArmL_dx, uArmL_dy)
        fArmL = polar_angle_deg(fArmL_dx, fArmL_dy)

        # Legs
        thighL_dx, thighL_dy = vec(lh, lk)
        shinL_dx, shinL_dy = vec(lk, la)
        thighR_dx, thighR_dy = vec(rh, rk)
        shinR_dx, shinR_dy = vec(rk, ra)
        thighL = polar_angle_deg(thighL_dx, thighL_dy)
        shinL = polar_angle_deg(shinL_dx, shinL_dy)
        thighR = polar_angle_deg(thighR_dx, thighR_dy)
        shinR = polar_angle_deg(shinR_dx, shinR_dy)

        # Racket: wrist -> racket_head (forearm extension if no racket detected)
        rh_pt = f.get("racket_head")
        if rh_pt and rh_pt.get("confidence", 0) > 0.2:
            racket_target = (rh_pt["x"], rh_pt["y"])
        else:
            # Fallback: extend forearm direction by 0.10
            ext_len = 0.10
            mag = math.hypot(fArmR_dx, fArmR_dy) or 1.0
            racket_target = (rw[0] + fArmR_dx / mag * ext_len,
                             rw[1] + fArmR_dy / mag * ext_len)
        racket_dx, racket_dy = vec(rw, racket_target)
        racket = polar_angle_deg(racket_dx, racket_dy)
        racketLen = math.hypot(racket_dx, racket_dy)
        # Clamp racketLen to a reasonable figure-space length (0.10 - 0.16)
        racketLen = max(0.08, min(0.16, racketLen))

        # 4. Normalize hipCenter to figure-space.
        # The figure renders centered horizontally; use the player's hip
        # x relative to a moving average so we don't track the player
        # walking across the court. We'll do a final pass to recenter.
        keyframes.append({
            "raw_hip": (hip_cx, hip_cy),
            "trunk": trunk,
            "neck": neck,
            "uArmL": uArmL,
            "fArmL": fArmL,
            "uArmR": uArmR,
            "fArmR": fArmR,
            "thighL": thighL,
            "shinL": shinL,
            "thighR": thighR,
            "shinR": shinR,
            "racket": racket,
            "racketLen": round(racketLen, 3),
        })

    # 5. Recenter hipCenter to (0.5, 0.55), animate small shifts based on
    # the player's actual hip motion.
    avg_hip_x = sum(k["raw_hip"][0] for k in keyframes) / n_window
    avg_hip_y = sum(k["raw_hip"][1] for k in keyframes) / n_window
    # Scale hip-shift by 0.4 so we don't make the figure walk across the screen
    HIP_SHIFT_GAIN = 0.4
    for k in keyframes:
        rx, ry = k["raw_hip"]
        k["hipCenter"] = (
            round(0.5 + (rx - avg_hip_x) * HIP_SHIFT_GAIN, 4),
            round(0.55 + (ry - avg_hip_y) * HIP_SHIFT_GAIN, 4),
        )
        del k["raw_hip"]

    # 6. Assign t-values evenly across [0.0, 0.8], leaving a recovery
    # buffer to t=1.0 so the loop seams smoothly back to the start.
    # Contact should land near t=0.65 (matches SWING_CONTACT_PHASE).
    target_contact_t = 0.65
    final_t = 0.85
    for i, k in enumerate(keyframes):
        if i <= contact_local_idx:
            # Spread pre-contact frames over [0.0, target_contact_t]
            k["t"] = round((i / contact_local_idx) * target_contact_t, 4) if contact_local_idx > 0 else 0.0
        else:
            # Spread post-contact frames over (target_contact_t, final_t]
            post_count = n_window - 1 - contact_local_idx
            local = i - contact_local_idx
            k["t"] = round(target_contact_t + (local / post_count) * (final_t - target_contact_t), 4)

    # 6.5. Mirror around the Y axis. Alcaraz in this clip is facing
    # RIGHT (right arm sweeps left→right in the video). The HeroRally
    # right figure stands on screen-right and hits toward the left
    # (its opponent) — so we need the swing to sweep right→left. Flip
    # each polar angle θ → 180° - θ, which mirrors the geometry around
    # the vertical axis. Also flip hipCenter x.
    def mirror_angle(a):
        m = 180.0 - a
        # Normalize to (-180, 180]
        while m > 180:
            m -= 360
        while m <= -180:
            m += 360
        return m

    for k in keyframes:
        for key in ("trunk", "neck", "uArmL", "fArmL", "uArmR", "fArmR",
                    "thighL", "shinL", "thighR", "shinR", "racket"):
            k[key] = mirror_angle(k[key])
        hcx, hcy = k["hipCenter"]
        k["hipCenter"] = (round(1.0 - hcx, 4), hcy)
    # Note: we do NOT swap left/right joint identities. Mirroring the
    # angles is enough — the figure's right arm now does what Alcaraz's
    # right arm did, but on the opposite screen side.

    # Round angles to 1 decimal
    for k in keyframes:
        for key in ("trunk", "neck", "uArmL", "fArmL", "uArmR", "fArmR",
                    "thighL", "shinL", "thighR", "shinR", "racket"):
            k[key] = round(k[key], 1)

    # 7. Emit TypeScript.
    print("\n// === KEYFRAMES (Alcaraz forehand from RTMPose) ===")
    print(f"// {n_window} frames extracted, contact at local index {contact_local_idx} (t={target_contact_t})")
    for k in keyframes:
        hcx, hcy = k["hipCenter"]
        print(
            f'  {{ t: {k["t"]}, hipCenter: [{hcx}, {hcy}], '
            f'trunk: {k["trunk"]}, neck: {k["neck"]},\n'
            f'    uArmL: {k["uArmL"]}, fArmL: {k["fArmL"]}, '
            f'uArmR: {k["uArmR"]}, fArmR: {k["fArmR"]},\n'
            f'    thighL: {k["thighL"]}, shinL: {k["shinL"]}, '
            f'thighR: {k["thighR"]}, shinR: {k["shinR"]},\n'
            f'    racket: {k["racket"]}, racketLen: {k["racketLen"]} }},'
        )


if __name__ == "__main__":
    main()
