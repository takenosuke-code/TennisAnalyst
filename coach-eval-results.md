# Coach Eval — Haiku 4.5 vs Sonnet 4.6 (per-shot prompt)

Generated 2026-05-01T00:27:32.676Z

Both models run via `messages.stream` (no tool_use). Same system prompt + user prompt. `max_tokens=600`. Identical biomechanics reference + auto-detected mistakes block.

---

## Fixture: Clean forehand (advanced player)

**Angle summary input:**
```
preparation: elbow_R=115° elbow_L=160° shoulder_R=50° knee_R=145° hip_rot=40° trunk_rot=85°
loading: elbow_R=95° elbow_L=145° shoulder_R=110° knee_R=135° hip_rot=55° trunk_rot=100°
contact: elbow_R=125° elbow_L=145° shoulder_R=95° knee_R=160° hip_rot=58° trunk_rot=50°
follow-through: elbow_R=140° elbow_L=130° shoulder_R=70° knee_R=170° hip_rot=70° trunk_rot=20°
finish: elbow_R=120° elbow_L=140° shoulder_R=45° knee_R=170° hip_rot=75° trunk_rot=-10°
```

### claude-sonnet-4-6

*TTFT: 931 ms · total: 12740 ms · output: 475 tok*

## What You're Doing Well

Your shoulder turn in the backswing is really solid. You're getting a full coil behind the ball and that's where a lot of players lose free power before they even start swinging. Your elbow position at contact is also in a great spot. Nice and relaxed, not locked out, exactly where you want it to generate that whip through the ball.

## 3 Things to Work On

**1. Let your hips finish the job**
Your hips are loading up nicely in the backswing but they kind of stall out right around contact. By the time you finish the swing they haven't rotated nearly as far as they should. That's leaving a lot of power on the table. Think "belt buckle to the fence on the other side." Lead with your hips through the whole swing, not just the first half. Try hitting five balls where you consciously feel your hips turning all the way through until your belly button faces the net.

**2. Keep the trunk rotating past contact**
Your shoulders are actually ahead of your hips through the swing, which is backwards. Hips should fire first and pull the shoulders through. Right now your trunk is slowing down right around contact instead of continuing to unwind. Think of it like cracking a whip. If you stop the handle early, the tip dies. On your next ball, focus on letting your chest keep turning even after you feel the hit.

**3. Don't let your legs go quiet at the finish**
You've got a nice knee bend loaded up in the backswing but by contact your legs are pretty much straight. You want to feel yourself pushing up and through the ball, not just rotating around stiff legs. Try this: on your next few forehands, consciously drive up off the ground as you swing. Think "push the floor away." You should feel your body rising slightly through the shot.

## Your Practice Plan

Load your hips and feel your belt buckle finish pointing at the net on every single forehand. Keep your chest turning even after the ball leaves the strings. Drive up through your legs at contact like you're trying to lift off the ground just a little.

### claude-haiku-4-5

*TTFT: 702 ms · total: 7590 ms · output: 571 tok*

## What You're Doing Well

Your preparation is solid—you're getting a nice shoulder turn, and your non-dominant arm is staying extended for balance. Your elbow stays relaxed through the loading phase too, which means you're not muscling the shot. That's a good foundation to build on.

## 3 Things to Work On

**1. Load your legs deeper into the shot**

Right now your back knee is sitting around 135 degrees during loading when you want it down closer to 130. That might not sound like much, but it's the difference between feeling like you're sitting into the shot versus standing too tall. Deeper legs mean more push coming into the ball. Try this: on your next 10 forehands, exaggerate the sit—really settle down into your back leg like you're sitting into a low chair. You'll feel the ground push you up through the shot. That's power.

**2. Let your hips lead the rotation sequence**

Here's what I'm seeing: your hips and shoulders are firing pretty close together, when ideally your hips should turn first and pull your shoulders through. Your hips are at 55 degrees of rotation during loading, but your shoulders are already at 110 degrees. That separation is where your free power lives. Think "lead with your belt buckle"—your hips turn toward the target before your shoulders follow, like cracking a whip. Next practice, hit some shadow swings and freeze at the loading position. Check: are your hips visibly ahead of your shoulders? That's the feel you want.

**3. Keep rotating through the ball**

I notice your trunk rotation drops sharply after contact—you're at 50 degrees at contact but then it collapses to 20 degrees by follow-through. That means you're decelerating into the shot instead of accelerating through it. Your follow-through tells the story. Swing all the way through to your opposite shoulder and hold the finish. Try 10 forehands where you exaggerate the follow-through wrap—let the racket travel completely around. That finish position is your checkpoint that you actually accelerated.

## Your Practice Plan

1. Hit 10 forehands with a deep knee bend in the loading phase—sit first, then explode up through the ball.
2. Do 5 shadow swings where you freeze at loading and check that your hips are visibly turned more than your shoulders.
3. Hit 15 forehands focusing only on the follow-through—freeze at the finish and make sure the racket is over your opposite shoulder.

---

## Fixture: Forehand with cramped elbow + locked knees (intermediate, mistakes flagged)

**Angle summary input:**
```
preparation: elbow_R=110° elbow_L=160° shoulder_R=45° knee_R=175° hip_rot=10° trunk_rot=30°
loading: elbow_R=85° elbow_L=145° shoulder_R=95° knee_R=178° hip_rot=20° trunk_rot=45°
contact: elbow_R=82° elbow_L=140° shoulder_R=85° knee_R=178° hip_rot=25° trunk_rot=20°
follow-through: elbow_R=130° elbow_L=125° shoulder_R=70° knee_R=175° hip_rot=30° trunk_rot=10°
finish: elbow_R=125° elbow_L=140° shoulder_R=50° knee_R=175° hip_rot=35° trunk_rot=5°
```

**Pre-flagged mistakes:**
- Cramped elbow at contact: You're hitting too close to your body — reach out and meet the ball in front.
- Locked knees: Straight legs are dead legs. Bend and feel the ground push you into the shot.

### claude-sonnet-4-6

*TTFT: 847 ms · total: 12149 ms · output: 467 tok*

## What You're Doing Well

Your shoulder turn through the backswing is solid. You're getting a good coil going, and that's the foundation everything else builds on. Your follow-through arm position also shows you're committing to the shot and not quitting early.

## 3 Things to Work On

**1. Reach out and meet the ball in front**
Right now you're letting the ball get too close to your body before contact. Your elbow is really tucked in tight when you hit, almost like you're pushing the ball away rather than swinging through it. That cramped position kills your leverage and takes a ton of pace off the shot. On your next ball, think "shake hands with the net." Reach forward to meet the ball, arm extended comfortably in front of your hip, and feel the difference in how clean the contact is.

**2. Bend those knees and feel the ground**
Your legs are staying almost completely straight through your whole swing, from loading all the way to the finish. Straight legs are dead legs. You're leaving a huge amount of free power sitting on the table because your legs aren't driving you into the shot at all. Try this: before the ball even crosses the net, sit down a little into your back leg like you're lowering yourself into a chair. Then push up and through the ball as you swing. Feel the ground push you into the shot. That's where your real power comes from.

**3. Fire your hips before your arm**
Your hips aren't rotating nearly enough through the swing. Your belt buckle needs to turn toward your target before your arm starts pulling through. Right now your arm is doing most of the work on its own. Try hitting ten forehands where you exaggerate it, where you think "hips first" and almost feel your belly button point at the net before your racket comes through.

## Your Practice Plan

Bend and sit into every shot before you swing, and feel your legs push you up into the ball. Meet every ball out front like you're reaching to shake someone's hand. Turn your hips toward the target before your arm starts moving.

### claude-haiku-4-5

*TTFT: 553 ms · total: 7593 ms · output: 555 tok*

## What You're Doing Well

You've got a solid shoulder turn in your backswing, and your racket lag is there when you load up. That tells me you understand how to coil the upper body. Your follow-through is completing nicely too, which means you're not decelerating early.

## 3 Things to Work On

**1. Load into your back leg — your knees are too straight right now**

I'm watching your legs stay locked at around 175 to 178 degrees from preparation all the way through loading. That's like trying to jump off concrete. You want to sit into the shot, bend those knees down to maybe 140 to 150 degrees, and then drive up and through the ball. Your legs are your engine. Try this: on your next 10 forehands, exaggerate dropping your weight into your back leg before you swing. Feel the ground push you up into the shot. That's where all your free power lives.

**2. Turn your hips — they're not doing the work yet**

Your hips are barely rotating. You're only getting to about 25 degrees of hip rotation at contact when you should be around 40 to 50. That's a huge difference. Your hips need to fire first, before your shoulders follow. Think of it like cracking a whip. Lead with your belt buckle toward the target, then let everything else follow. On the next ball, really feel your hips open up first before your arm even starts forward.

**3. Meet the ball out in front — you're hitting too close to your body**

Your elbow is bent to about 82 degrees at contact, and it's happening right beside you instead of in front. You want that elbow extended more, around 110 to 120 degrees, with the contact point way out front like you're shaking someone's hand. Reach for the ball. When your arm is cramped in tight like that, you lose all your leverage and whip. Try 10 forehands where you reach out and meet the ball early. Check yourself: is your arm extended? Is the ball in front of your hip?

## Your Practice Plan

1. Hit 15 forehands focusing only on bending your knees deeper and feeling the push from the ground.

2. Do 15 more where you exaggerate the hip turn first, then let the rest follow.

3. Finish with 15 where you reach out front and catch the ball way ahead of your body.

---

## Fixture: Serve with focus question on knee bend

**Angle summary input:**
```
preparation: elbow_R=160° elbow_L=160° shoulder_R=30° knee_R=165° hip_rot=20° trunk_rot=20°
loading: elbow_R=70° elbow_L=170° shoulder_R=170° knee_R=162° hip_rot=15° trunk_rot=40°
contact: elbow_R=175° elbow_L=80° shoulder_R=175° knee_R=175° hip_rot=10° trunk_rot=10°
follow-through: elbow_R=140° elbow_L=70° shoulder_R=140° knee_R=170° hip_rot=5° trunk_rot=-10°
finish: elbow_R=120° elbow_L=80° shoulder_R=80° knee_R=175° hip_rot=0° trunk_rot=-20°
```

**Pre-flagged mistakes:**
- Insufficient knee bend on serve: You're loading too shallow — drop into the back leg before you go up.

**User focus question:** "is my knee bend deep enough on the serve?"

### claude-sonnet-4-6

*TTFT: 1251 ms · total: 12729 ms · output: 466 tok*

Honestly, your knee bend is a little shallow right now. You want to sit down into that back leg a lot more before you go up, and right now you're staying pretty tall through the loading phase.

## What You're Doing Well

Your elbow position during the loading phase looks solid. You've got a nice relaxed bend there, which protects your shoulder and keeps you ready to accelerate through the ball. Your arm extends well toward contact too, so you're reaching up and out the way you want to on a serve.

## 3 Things to Work On

**1. Sit deeper into your legs before you launch**
You're barely bending your knees during that trophy position. Think of it like sitting into a low chair before you explode up. Right now you're almost standing straight up, and that means your legs have nothing to push off from. All that leg power just disappears. Try this: before your next serve, feel your weight drop into your back leg until your thighs are really loaded, then drive up like you're jumping to touch the ceiling.

**2. Let your hips rotate through the shot**
Your hips are barely moving through the swing, and by contact they've almost stopped completely. Your hips should be pulling your whole upper body through the ball. Try feeling your belt buckle turn toward the target as you go up to hit. On your next serve, cue yourself with "hips first" as you start your upward drive.

**3. Keep your trunk rotating all the way through**
Your trunk is rotating backward through the follow-through, which tells me your body is stopping instead of driving forward through the ball. You want to feel your chest swinging through and finishing facing the court. Think "swing through the wall" and let everything follow the ball over the net.

## Your Practice Plan

Load your legs deep at the trophy position every single serve, feel that sit-and-explode feeling before you go up. Cue "hips first" as you drive up so your lower body leads the shot. Finish every serve with your chest facing the net and your racket wrapping around toward your off shoulder.

### claude-haiku-4-5

*TTFT: 608 ms · total: 7335 ms · output: 507 tok*

## What You're Doing Well

Your arm position in the trophy position looks solid. Your elbow is staying bent and relaxed at about 70 degrees on the hitting side, which is exactly what you want. You're also getting good shoulder external rotation in the layback phase, so you've got that coil built in.

## 3 Things to Work On

**1. Load deeper into your legs before you explode up**

Your knees are staying pretty straight through the loading phase. Right now you're at about 162 degrees of knee bend when you want to sit down into it more like 140 to 150 degrees. That shallow loading kills your leg drive. Think "sit and explode" on your next serve. Drop your butt down into your back leg like you're settling into a chair, then launch up to the ball like you're trying to dunk a basketball. You'll feel way more power coming from the ground instead of just your arm.

**2. Let your hips and core do more of the work**

I'm noticing your hips and trunk aren't rotating much through the swing. Your hip rotation drops from 20 degrees in loading all the way down to almost nothing at contact. That means your arm is doing all the heavy lifting instead of your body. On your next set of serves, focus on turning your pocket toward the back fence during your loading. Feel your belt buckle rotating first, then let your shoulders and arm follow through naturally. Your hips lead the dance, everything else just follows.

**3. Keep your legs driving through contact**

Your knees are almost locked straight by contact at around 175 degrees. You want to feel your legs still pushing you up and into the ball at that moment. The ground should be helping you all the way through. It's like the power starts in the ground and travels up through your body. Right now you're stopping your leg drive too early.

## Your Practice Plan

Hit 20 serves focusing on dropping into a deep knee bend before you explode upward, and feel the ground pushing you up to the ball. Throw a basketball instead of hitting serves for five minutes to remind your body that the power comes from sitting and driving, not your arm. On your final 10 serves, freeze at contact and check that your legs are still pushing and your hips have rotated forward.

