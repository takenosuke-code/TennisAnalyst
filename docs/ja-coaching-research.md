# Japanese Tennis Coaching — Research Reference

Engineering reference for Swingframe's Japanese-mode coaching cues. Goal: feed this to Claude as few-shot examples so generated JA cues feel like a Japanese tennis coach wrote them, not Google-translated American copy. **Not a final spec — a JA-fluent reviewer must vet before launch.**

---

## Section 1: Overview of JA Tennis Coaching Tone

Based on roughly twelve articles surveyed (tennisbear.net, tennis-gakko.info, note.com tennis writers, sposuru, dream-coaching.com, JTA materials, plus several practitioner blogs), Japanese tennis coaching prose lands in a noticeably different register than American instructional copy:

- **Less imperative, more invitational.** Where an English coach writes "load into your back leg," the JA equivalent typically softens to ~してみてください ("try doing X"), ~しましょう ("let's do X"), or ~を意識する ("be aware of X"). Bare imperatives like 〜しろ or 〜せよ would sound harsh and are essentially absent from coaching content. The casual imperative 〜して is acceptable but rare in writing.
- **More descriptive / image-driven.** Coaches lean heavily on visual metaphors: 窓拭きをするイメージ ("imagine wiping a window") for the wiper finish, バランスボールを持つようなイメージ ("imagine holding a balance ball") for spacing, ラケットをしならせる ("flex the racket like a whip") for relaxation. Felt-sense language (感覚をつかむ — "grab the sensation") shows up constantly.
- **Polite-casual register, not 敬語.** The dominant verb endings are です/ます with occasional plain form (~のです、~んだ、~から). Pure 敬語 (お~になる, ご~ください) reads as customer service and is avoided. Pure plain-form (~だ、~だぞ) reads as too rough except in YouTube banter.
- **Diagnose then prescribe.** Articles routinely lead with the symptom ("打点が近いので窮屈そう" — "your contact point is too close, you look cramped") before offering the fix. Pure praise-first sandwich is less common than in US sports coaching; the structure is more "here's what's happening → here's why → try this."
- **Hedged corrections.** Mistakes are softened with の・ようです、〜と思います、〜のではないでしょうか — "it seems," "I think," "isn't it the case that…" Direct "you're wrong" framing is rare; even firmly-worded blogs use 〜は誤解されやすい ("this is easily misunderstood") rather than 〜は間違い outright.
- **Outcome framing.** Cues often end with the resulting feeling: 〜が打てるようになります ("you'll come to be able to hit X"), 〜が安定します ("X will stabilize"). The reward is named.

Sources for this overview: tennisbear.net articles 926/1983, tennis-gakko.info ball-fh, sposuru forehand article, note.com tenniszero 085, tennis-goodspeed body-opens, jh-tennis.jp split-step, sports.yahoo.co.jp 体重移動 column.

---

## Section 2: Terminology Mini-Glossary (EN → JA)

Compiled from tennisbear glossary, tennis-goodspeed glossary, noahis.com beginner words, and contextual usage in coaching articles. Register notes are based on observed usage.

| English | JA term used | Type | Register notes |
|---|---|---|---|
| forehand | フォアハンド | katakana | universal |
| backhand | バックハンド | katakana | universal |
| swing | スイング | katakana | universal; 振り also used in compounds (素振り = shadow swing) |
| follow-through | フォロースルー / 振り抜き | mixed | フォロースルー in technical writing; 振り抜き ("swing-through") in casual speech |
| contact point | 打点 (だてん) | native compound | universal, the dominant term |
| weight transfer | 体重移動 (たいじゅういどう) | native compound | universal across formal and casual |
| hip rotation | 腰の回転 / 腰を回す | native | "腰を回す" is the verb form; coaches debate whether it's literal or metaphorical |
| trunk / core rotation | 体幹の回転 / 体の回転 | native | 体幹 reads slightly more technical than 体 |
| loading / coil | タメ (ため) | native (often katakana) | very common, casual; means "the loaded coiled state" |
| footwork | フットワーク | katakana | universal |
| split step | スプリットステップ | katakana | universal |
| shoulder turn | 肩を入れる / ショルダーターン | mixed | 肩を入れる ("set the shoulder in") is the natural coach phrase |
| unit turn | ユニットターン | katakana | technical writing; less common in casual speech |
| takeback | テイクバック | katakana | universal |
| racket face | ラケット面 (めん) / フェイス | mixed | 面 is more common in spoken coaching |
| grip | グリップ / 握り (にぎり) | mixed | グリップ for grip type; 握り for the act of gripping |
| stance (open/closed/square) | スタンス (オープン/クローズド/スクエア) | katakana | universal |
| topspin | スピン / トップスピン | katakana | universal |
| slice | スライス | katakana | universal |
| volley | ボレー | katakana | universal |
| serve | サーブ | katakana | universal |
| return | リターン | katakana | universal |
| ready position | 構え (かまえ) / レディーポジション | mixed | 構え more idiomatic |
| axis foot / pivot foot | 軸足 (じくあし) | native | universal coaching term |
| step in / drive forward | 踏み込み (ふみこみ) | native | the forward weight commitment |
| wrist | 手首 (てくび) | native | universal |
| elbow | 肘 (ひじ) | native | universal |
| knee bend | 膝を曲げる (ひざをまげる) | native | universal |
| finish position | フィニッシュ | katakana | universal |
| body opens (early) | 体が開く (からだがひらく) | native | very common diagnostic phrase, slightly negative connotation |

Note on **wrist lag**: no clean single JA term surfaced in the surveyed sources. Coaches describe it functionally — 手首の角度を一定に保つ ("keep the wrist angle constant"), ラケットが遅れて出てくる ("the racket comes through late"). Flag for native speaker review.

---

## Section 3: Example Coaching Cues

All phrases below are verbatim from the sources cited at the end. EN equivalent and tone notes added by Claude.

| # | JA cue (verbatim) | EN equivalent / situation | Tone |
|---|---|---|---|
| 1 | 左手を意識してみましょう。 | "Try to feel your left hand." Forehand non-dominant hand cue. | Soft imperative, encouraging |
| 2 | 身体の前にスペースを作る。 | "Make space in front of your body." Spacing fix. | Descriptive, principle-stating |
| 3 | しっかりターンする。 | "Turn fully." Pre-swing prep. | Casual imperative |
| 4 | 窓拭きをするイメージを持ちましょう。 | "Hold the image of wiping a window." Wiper-finish cue. | Imagery, encouraging |
| 5 | バランスボールを持つようなイメージで。 | "Like you're holding a balance ball." Front-of-body spacing. | Imagery |
| 6 | 体が開かないように打つ直前で回転を止める。 | "Stop your rotation right before contact so your body doesn't open early." | Corrective, mechanical |
| 7 | 打点が近いので窮屈そう。 | "Your contact point is too close, you look cramped." | Diagnostic, gentle |
| 8 | 手打ちはダメ。 | "Don't arm it." Common diagnosis when player swings only with the arm. | Casual, blunt |
| 9 | 力まずにフォアハンドを打つ。 | "Hit the forehand without tensing up." | Principle |
| 10 | 肘を支点に振り抜きます。 | "Swing through using your elbow as the pivot." Wiper finish. | Descriptive instruction |
| 11 | 肘を口につけるくらいのイメージです。 | "Imagine bringing your elbow up to your mouth." Finish-height cue. | Imagery, casual |
| 12 | 体に巻きつけるように振り抜く。 | "Swing through wrapping around your body." Finish path. | Imagery |
| 13 | 踏み込み足のヒザを柔らかく使い、前に重心が移動するようにします。 | "Keep the front knee soft so your weight moves forward." Weight-transfer cue. | Instructional |
| 14 | 体重が後ろにかかった状態でスイングすればするほど、後傾になる。 | "The more you swing with your weight back, the more you lean back." Diagnosis. | Diagnostic |
| 15 | スプリットステップは大きく飛んではダメです。体を沈み込ませる感じです。 | "Don't jump big on the split step — it's more of a sink." | Corrective + imagery |
| 16 | 相手が打つ時に着地できるようにジャンプを開始しましょう。 | "Start your jump so you land when the opponent strikes." Split-step timing. | Soft imperative |
| 17 | 上体の捻りを加え、タメるのです。 | "Add upper-body coil — that's the loaded position." Serve setup. | Emphatic explanation |
| 18 | 軸足にしっかり乗ってからスイングする。 | "Load fully onto the axis foot before swinging." | Sequencing cue |
| 19 | ラケットをしならせるように全体の力を抜くことで振りは速くなります。 | "Relax everything so the racket whips — that's how the swing speeds up." | Principle + outcome |
| 20 | 「ビュン」といった音が上手くいっている証拠です。 | "A whoosh sound means you got it." | Encouraging feedback |
| 21 | ラケット面は地面と垂直にしましょう。 | "Keep the face perpendicular to the ground." | Soft imperative |
| 22 | 同じスイングをもう一回。 | "Same swing, one more." Repeat-good-rep cue. | Casual affirmation (Claude-generated from EN equivalent — verify) |
| 23 | 今のいい、それ続けて。 | "That was good — keep doing that." | Casual affirmation (Claude-generated — verify) |
| 24 | 打ち終わった後に体が回転するように。 | "Let the body rotate AFTER you finish the strike." Sequencing. | Descriptive instruction |

Items 22 and 23 are Claude-generated affirmation patterns based on the casual register observed elsewhere; **flag for native speaker review.** All other items are verbatim from cited sources.

---

## Section 4: Patterns to Encode in the LLM Prompt

1. **Use 〜してみて / 〜しましょう / 〜を意識する for suggestions, not 〜しろ.** Because bare imperatives sound harsh in JA writing; observed 0 instances of 〜しろ across surveyed coaching content.
2. **Lead with the symptom before the fix when correcting.** Because JA articles consistently follow diagnose→explain→prescribe order, not US-style fix-first. Example pattern: "[observation], [だから], [try this]."
3. **Prefer image-based cues over biomechanical jargon when the cue is short.** Because 窓拭き ("wiping a window"), バランスボール, ラケットをしならせる are the dominant memorable cues; coaches treat imagery as the primary delivery vehicle.
4. **Avoid 敬語 (お~になる, ご~ください) entirely; stay in です/ます with occasional plain-form.** Because 敬語 reads as customer-service register and breaks the coach-to-player relationship.
5. **Soften corrections with hedges (〜のようです, 〜と思います, 〜かもしれません).** Because direct "you are wrong" framing is rare even in opinionated blog posts; 〜は誤解されやすい is the typical strong critique.
6. **End cues with the outcome the player will get (〜できるようになります、〜が安定します).** Because reward-naming is a recurring structural pattern across tennisbear, sposuru, e-toprun.
7. **Use katakana for instrument/technique nouns (フォアハンド, スプリットステップ, タメ) and native compounds for body actions (体重移動, 腰の回転, 軸足, 打点).** Because mixing matches actual coach speech; over-using either side reads as off.
8. **Affirmations should be short and use casual sentence-final particles (〜だね、〜それ、もう一回).** Because the English-side affirmations ("Clean — repeat that.", "Lock that in.") map to brief, particle-light JA.
9. **Avoid em dashes, en dashes, and connector hyphens — same rule as English mode.** Because TTS and natural JA use 、and 。 not — anyway; consistency with the existing prompt rule.
10. **One sentence, max ~50 JA characters when possible.** Because JA renders ~1.5–2x more characters per English word but the audio-cue use case is the same length. Stay terse.

---

## Section 5: Few-Shot Examples for livePrompt.ts JA Mode

Hypothetical angle summaries paired with the JA cue Claude should produce. Designed to mirror the existing English livePrompt.ts shape (one sentence, feel-based, optional one parenthetical measurement).

### Example 1 — Hip rotation closed at contact
**Angle summary (invented):** Hip rotation peaked at 28° on three of four swings; trunk rotation lagged hip by 40 ms.
**EN-mode cue (reference):** "Hip rotation looked closed (28°). Open up earlier."
**JA-mode cue:** 腰の開きが少し遅れてる(28°)、もう一テンポ早く開いてみて。

### Example 2 — Clean reps, advanced player
**Angle summary:** Knee bend, hip rotation, trunk rotation all within baseline. No outliers across four swings.
**EN-mode cue:** "Trust that swing, do it again."
**JA-mode cue:** いい感じ、その同じスイングをもう一回。

### Example 3 — Late finish on forehand
**Angle summary:** Peak frame consistently 60 ms after expected; elbow finishes below shoulder on all swings.
**EN-mode cue:** "Finish is late — let the racket pass your hip first."
**JA-mode cue:** フィニッシュがちょっと遅れてる、ラケットを腰の前に通すイメージで振り抜いて。

### Example 4 — Inconsistent across the batch
**Angle summary:** First two swings: trunk rotation 75°. Last two: 50°. Hip rotation similar drop.
**EN-mode cue:** "First two were tighter than the last two — keep that turn."
**JA-mode cue:** 最初の二球の方がしっかりターンできてた、その感覚をキープ。

### Example 5 — Weight stuck on back foot
**Angle summary:** Knee bend deep on back leg, minimal extension forward; trunk leans back through contact.
**EN-mode cue:** "Weight stayed back. Drive into the front foot through contact."
**JA-mode cue:** 体重が後ろに残ってる、前足にしっかり乗せてインパクトしてみて。

All five JA cues are Claude-drafted in the patterns observed in Section 3 sources. **Native speaker review required before shipping** — particularly for naturalness of 〜てる contractions, particle choice, and whether 〜してみて lands as encouraging vs. patronizing in this context.

---

## Section 6: Sources Consulted

- https://www.tennisbear.net/blog/926 — forehand technique, 6 verbatim coaching phrases extracted (left-hand awareness, racket-pulling imagery, window-wiping)
- https://www.tennisbear.net/blog/1983 — forehand spacing, 11 verbatim phrases (balance-ball image, fixed elbow position, no-tension principle)
- https://www.tennisbear.net/blog/826 — terminology glossary, used for confirming katakana adoption of common terms
- https://tennis-gakko.info/technique/ball-fh — power forehand, 10 phrases (face-feel, wrist-angle stability, "ビュン" sound feedback, racket-flex relaxation)
- https://tennis-gakko.info/technique/split — split step (text excerpt only, used for terminology confirmation)
- https://www.jh-tennis.jp/entry/split-step/ — split step purpose, 3 conversational phrases (体を沈み込ませる, casual だよ ending). Thinner than expected on direct cues.
- https://www.dream-coaching.com/magazine/tennis/tennis-article150/ — split step article (403 forbidden, could not access)
- https://thedigestweb.com/tennis/detail/id=12882 — footwork basics, 7 verbatim cues (small-jump split, wide stance, cross-step recovery)
- https://sposuru.com/contents/sports-quest/tennis-forehand/ — forehand essentials, 10 short cue-style phrases (raise the racket, swing through, knee bend, etc.)
- https://sports.yahoo.co.jp/column/detail/201806280017-spnavido — weight transfer column, 7 verbatim phrases (front-knee softness, weight-back diagnosis, sensation-grabbing language)
- https://note.com/mr_service_drt/n/nee7fd9889e37 — follow-through control, 7 phrases (don't force the finish, situation-dependent, です/ます register)
- https://note.com/tenniszero/n/n8a20e02a7b6c — body usage misconception, 10 phrases (手打ちはダメ, conversational/analytical voice, のです endings)
- https://note.com/tenniszero/n/n81b94472167b — follow-through importance, 9 phrases (rhetorical-question lede, から/ね particles)
- https://tennis-goodspeed.com/body-opens/ — hip rotation, 8 phrases (体が開く diagnostic, "腰を回せ" framed as misconception)
- https://tennis-goodspeed.com/glossary/ — terminology, used for cross-checking which English concepts have native vs. katakana JA equivalents
- https://noahis.com/beginner/words — beginner glossary, used for stance/grip/swing-phase terms and 雁行陣/平行陣 doubles formations
- https://e-toprun.com/news/hiketsu/hiketsu-vol-01/ — forehand basics, 11 phrases (stance width, knee angle in degrees, 7割の力 partial-power cue)
- https://tofutennis.com/forehand_follow-through/ — finish positions, 6 phrases (shoulder-carry finish, body-wrap imagery, elbow-pivot description)
- https://improving-tennis.com/サービスでタメを作る方法 — serve タメ, 6 phrases (loading vs. knee-bend distinction, casual self-reflective register)
- https://www.jta-tennis.or.jp/Portals/0/PDF/jrhs_guidance.pdf — JTA middle-school instruction guide. **Could not extract** — PDF binary format defeated text extraction. A native speaker with the printed/searchable version would significantly upgrade Section 1's authoritative sourcing.

**Gaps where research was thin:**
- YouTube transcripts: the search returned channel listings (窪田テニス教室, Star Tennis Academy, 小野田/鈴木 channels, etc.) but actual transcript mining was not feasible without YouTube transcript API access. Spoken JA coaching has more 〜だよ、〜じゃん、〜ね energy than the written sources here capture.
- JTA official materials: the public PDF was unreadable in this environment; the formal JA register from official curricula is underrepresented.
- "Wrist lag" and several micro-finish terms: no clean JA equivalent surfaced — coaches describe these functionally, not lexically.
- Affirmation/silence-replacement phrases (the "Clean — repeat that." cluster): not directly sourced; the four examples in Section 3 (#22, #23) and Section 5 are Claude-extrapolated from the casual register patterns and need native speaker confirmation.

**Recommendation:** before launch, have a JA-fluent recreational or competitive tennis player (not just a fluent translator) read Section 5 examples and the affirmation set. Tennis coach voice is a subculture register, not just textbook JA.

---

## Section 7: Gap-Fill Research (Pass 2)

Pass 1 (Sections 1–6) flagged three areas as thin: courtside spoken register, affirmation/silence cues, and wrist-lag/micro-finish terminology. This section appends sourced corpus targeting each. Sources are listed in 7.4; sources from Pass 1's bibliography are not re-cited. Where a phrase is Claude-extrapolated rather than sourced verbatim, it is tagged `verify with native speaker`.

### 7.1 Courtside Spoken Register — Sourced Phrases

The biggest finding: written tennis blogs underrepresent how blunt and how short coaches actually speak in person. Several written-source coaches describe lessons or quote themselves; once you find that quoted speech, the casual register *is* genuinely distinct from the polite-instructional written voice the first pass captured. Below are verbatim phrases that two coaches and one student-blogger reported being said on court.

| # | JA phrase (verbatim) | Context / situation | Source |
|---|---|---|---|
| C1 | ラインより下がらない。ライジングでとるようにして。 | Coach instructing student not to back up; take ball on the rise. Note bare 〜して imperative. | tennismama 2020/02 |
| C2 | ちゃんと下がって、ボールとの距離をとって。ストロークの打ち方がおかしくなってるよ。 | A *different* coach correcting same student. Note 〜よ ending and 〜てる contraction; diagnostic + corrective combined. | tennismama 2020/02 |
| C3 | 君さ、返事は良いからさ。ちゃんと走ってやる気を見せてよ。 | Coach calling out a student for low effort. Very casual: 君さ、〜からさ、〜てよ. This is the bluntest sourced register in the corpus. | ameblo greenyy8 |
| C4 | Aさん以外、返事がないぞ‼ | Coach addressing the group; uses rough 〜ぞ ending — only male, only group context. | ameblo greenyy8 |
| C5 | とりあえずこうやって握って打ってみようか。 | Coach offering a grip change with low pressure. 〜ようか pattern — invitational, casual, very common. | note coach_raku |
| C6 | サーブ、なかなかいいじゃないですか。 | Trial-lesson coach to adult student. Mixed register: polite frame, casual じゃないですか tag. | mikawa-kyosei (excerpt via search) |
| C7 | 中級でばっちりですよ! | Same coach, post-lesson placement assessment. Casual ばっちり + polite ですよ. | mikawa-kyosei |
| C8 | 〇〇さんナイスショット | Standard pattern: name + ナイスショット. The name-prefix is named explicitly as a relationship-building technique. | thedigestweb 55985, officeyamamoto |
| C9 | 〇〇さんドンマイ | Same name-prefix pattern after a miss. | thedigestweb 55985, officeyamamoto |
| C10 | 今のナイスだったよ | Doubles partner / coach voicing approval after the point. 〜だったよ past-tense affirmation. | chiebukuro q14265229104 |
| C11 | ドンマイドンマイ、次頑張ろう | Doubled ドンマイ + 次頑張ろう. Standard between-points pickup; the doubled form is more natural than single. | chiebukuro q14265229104 |
| C12 | OK、OK、気にしないでもう一度ボールに集中しよう。 | Reset cue after a missed point. Note OK loanword + 〜しよう volitional. | feeling-tennis advice-2 |
| C13 | OK、OK、終わった事は忘れて、次のボールに集中しよう。 | Variant of C12 — same writer recommends both. | feeling-tennis advice-2 |

**Register patterns observed (across these 13 phrases):**

- **Sentence-final 〜よ / 〜ね / 〜さ** dominate over 〜です. Written sources hide this; spoken corpus surfaces it.
- **〜てる contraction** (打ち方がおかしくなってる, 残ってる) is normal coach-to-player, even from coaches who write in 〜ています.
- **Sentence fragments and trailing 〜て imperatives** ("〜して。" without 〜ください) are the courtside default. Bare 〜して reads as *neither* harsh nor soft — it's the working register.
- **Name-prefix is the primary politeness marker** in groups. "〇〇さんナイス" lets the coach drop other politeness scaffolding without sounding rude.
- **Doubled affirmations** (ドンマイドンマイ、ナイスナイス) are more natural than singletons in fast-pace contexts.

Two extrapolated cues to consider, both `verify with native speaker`:
- C-ext-1: いい感じ、もう一回 ("Good feel, one more.")
- C-ext-2: それそれ、それキープ ("That, that — keep that.")

### 7.2 Affirmation Cue Corpus

This is where the corpus is genuinely thinnest. The Japanese coach-voice affirmation lexicon is smaller than English's. The dominant on-court affirmations across all sources:

**Sourced (used verbatim in cited sources):**
| EN target | JA equivalent | Source / register note |
|---|---|---|
| "Nice." | ナイス / ナイスショット | universal; chiebukuro, officeyamamoto, feeling-tennis |
| "Nice watch." (defensive read) | ナイスウォッチ | doubles-specific; chiebukuro |
| "Good line." | ナイスコース | chiebukuro |
| "Nice try." | ナイストライ | chiebukuro |
| "That was nice." | 今のナイスだったよ | chiebukuro |
| "Don't worry, next one." | ドンマイ / ドンマイドンマイ、次 | chiebukuro |
| "OK, refocus." | OK、OK、〜に集中しよう | feeling-tennis |
| "Lucky!" | ラッキー / シャーラッキー | chiebukuro (informal exclamation, post-net-cord etc.) |
| "Just barely." | ジャストでしたね | chiebukuro |
| "So close." | おしかったです | chiebukuro |
| "Nice!" (volleyball-adjacent) | ナイス / ナイスファイト | agevolley (cross-sport corpus) |
| "Go for it / take it." | 思い切って / 攻めていいよ | agevolley |

**Extrapolated for English-side affirmations the corpus does not cover (all `verify with native speaker`):**

| EN target | Proposed JA | Rationale |
|---|---|---|
| "Clean — repeat that." | きれいなスイング、もう一回。 | "きれい" is the natural JA evaluative for a clean technical execution; もう一回 is corpus-attested. |
| "Trust that swing." | 今のスイング、信じて。 / そのスイングでいいよ。 | 信じて imports English coaching directly; そのスイングでいい matches feeling-tennis register. |
| "Same swing again." | 同じスイングでもう一回。 | Direct compositional translation; matches Pass 1 #22. |
| "Lock that in." | その感覚、覚えといて。 / それキープ。 | 感覚を覚える is corpus-attested (sports.yahoo "感覚をつかむ"); キープ from racket-face source. |
| "That's it." | それそれ。 / そう、それ。 | Idiomatic JA approval echo; not directly sourced in tennis but pan-sport. |
| "Stay with it." | そのまま、そのまま。 | Doubled-imperative pattern parallels ドンマイドンマイ. |
| (silence-affirmation: nod, no words) | 〔無言で頷く / OK サインのみ〕 | JA coaches lean on physical OK signs and head-nods more than English coaches; per agevolley + feeling-tennis, silence is acceptable when the action speaks. Worth treating as a valid output, not a missing one. |

**Important observation for the LLM prompt:** when the EN-side cue is purely affirmative and contains no diagnostic information ("Lock it in." after a clean rep), the most natural JA output may be a *non*-sentence: a fragment, an interjection, or a doubled word. Forcing a complete です/ます sentence here over-translates and reads as condescending. This is a structural difference Pass 1 did not surface.

### 7.3 Wrist-Lag / Racket-Face / Contact-Point Terminology

Pass 1 noted no clean JA term for "wrist lag." Pass 2 surfaces a small but real lexicon — coaches do have terms; they just don't map 1:1 to English.

**Wrist lag, racket-head delay:**
| English concept | JA expression(s) | Source |
|---|---|---|
| (general) racket head lag | ラケットヘッドが遅れて出てくる / ラケットヘッドが落ちる | tennistsu, ryoji.site racket-head-down |
| grip-end leads, head trails | グリップエンド先行 (でインパクトを迎える) | improving-tennis forehand-inpact2 |
| late wrist release | リストの解放タイミング(が早い/遅い) | improving-tennis forehand-inpact2 |
| relax the wrist (so lag happens) | 手首をリラックスさせる / 手首は意識的に使わない | tennistsu |
| don't manipulate the wrist | 手首をこねない | tennispirits, tofutennis |
| stabilize the wrist | 手首を固める / 手首をコックする | tofutennis |
| dorsiflexion / palmar flexion (technical) | 背屈 / 掌屈 | tofutennis |
| arm-only swing (anti-pattern) | 手打ち | Pass 1 + tennispirits (re-confirmed) |

**Racket face:**
| English | JA | Source |
|---|---|---|
| keep face vertical | ラケット面を垂直に保つ | tennisnavi racket-face |
| close the face | 面を伏せる (下向きにする) | tennisnavi racket-face |
| open face | 面が開いている | tennisnavi racket-face |
| make/build the face | 面を作る | search snippet (tennis-jotatsu) |
| face control (general) | ラケット面のコントロール | tennisnavi racket-face |

**Contact-point nuance:**
| English | JA | Source |
|---|---|---|
| contact point is forward | 打点は前 | jh-tennis contact-point (cites this as the conventional cue, *then critiques it*) |
| body's "side" rather than "front" | 体の横 (で打つ) | jh-tennis contact-point |
| pull the ball in / wait for it | ボールを引き付ける / ボールをしっかり引き付ける | jh-tennis contact-point |
| meet impact with grip end leading | グリップエンド先行でインパクトを迎える | improving-tennis forehand-inpact2 |

**Practical takeaway for prompt engineering:** the cleanest single coaching cue for "wrist lag" is **「ラケットヘッドを遅らせて」** or **「グリップエンド先行で」** — but coaches more often *describe the cause* (relaxed wrist, body rotation) than the *symptom* (lag itself). A faithful JA cue for "you're losing wrist lag" is closer to **「手首に力が入ってる、もう少しリラックスして」** than to a literal "wrist lag" translation.

### 7.4 New Sources Consulted (Not in Pass 1's bibliography)

- https://www.tennistsu.com/stroke_racket-drop.html — racket-head drop, wrist-relaxation principles. Polite ます register.
- https://ryoji.site/racket-head-down/ — racket-head-down with caveats; short cues "ラケットを寝かす", "手の力を抜く".
- https://tennisnavi.com/technique/forehand-stroke/racket-face/ (formerly xn--zckzap9321bz4p.com) — racket-face control, 面を伏せる/開く lexicon.
- https://www.jh-tennis.jp/entry/contact-point-for-ground-strokes/ — contact-point critique; "ボールを引き付ける" verbatim. (Different post from Pass 1's split-step entry on same site.)
- https://improving-tennis.com/forehand-inpact2 — "グリップエンド先行", "リストの解放タイミング" — the cleanest sourced wrist-lag terminology found.
- https://tofutennis.com/forehand_wrist/ — wrist terminology (背屈/掌屈, 手首をこねる, 手首を固める).
- https://tennispirits.com/blog/skillup/list-used/ — wrist usage; "手首をこねない" verbatim.
- https://www.xn--zckzap9321bz4p.com/forehand/how-to-use-wrist — wrist-use article (search snippet only).
- https://tennismama.hatenadiary.jp/entry/2020/02/09/... — student-blogger quoting two different coaches verbatim. C1, C2 above. **Highest-value new source for casual register.**
- https://ameblo.jp/greenyy8/entry-12845595427.html — coach blog quoting his own on-court speech. C3, C4 above (the blunt 〜さ / 〜ぞ register).
- https://note.com/coach_raku/n/n6bc0386fc5dc — coach reflecting on minimal-talking philosophy; one verbatim casual cue (C5).
- https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q14265229104 — Yahoo Q&A on doubles encouragement phrases. Single best source for the affirmation lexicon.
- https://feeling-tennis.com/advice-2/ — match advice + reset cues (C12, C13).
- https://thedigestweb.com/tennis/detail/id=55985 — manners article noting name + ナイスショット pattern. (Different post from Pass 1's footwork article on same site.)
- https://officeyamamoto-tennis.com/116-2/ — re-confirms name-prefix pattern.
- https://ameblo.jp/yoshimie1/entry-12724629706.html — student blog quoting coach (referenced for register confirmation; coaching content not extracted as a cue).
- https://sports-zip.com/blog/coach_tweet/101746 — senior-coach quotes; useful for register confirmation, not added as cues.
- https://agevolley-blog.com/how-to-support-with-voice/ — adjacent-sport (volleyball) affirmation corpus; ナイスファイト, 思い切って, 攻めていいよ.
- https://mikawa-kyosei.jp/blog/5532/ — search-snippet excerpt only; full page did not surface the quoted phrases on fetch (C6, C7 are from the search snippet, treat as medium-confidence).

### 7.5 What Still Needs Human Review

- **Section 7.1 phrases C6 and C7** were read from a search-result snippet, not the full page. A native reviewer should confirm the source uses them verbatim before they are quoted as such in shipping copy.
- **Section 7.2 extrapolations** (the seven proposed JA equivalents tagged `verify with native speaker`) are the highest-stakes items. Several map English idioms ("Lock that in.", "Trust that swing.") that have no clean JA equivalent; reviewer should choose between (a) the proposed near-equivalent, (b) a different phrasing entirely, or (c) silence/non-sentence output.
- **The 〜ぞ register (C4)** is masculine and group-context only. Do **not** generate this tone for one-on-one cues to an unknown user; flag in the prompt.
- **YouTube transcript mining** is still the largest unfilled gap. A bilingual reviewer with one hour and a YouTube transcript tool could pull 50+ verbatim courtside phrases from窪田 / Star Tennis Academy / ともやん channels. This would move 7.1 from "medium" to "high" confidence.
- **The silence-as-affirmation pattern** (suggested in 7.2) is a hypothesis based on coach-philosophy articles (note coach_raku explicitly advocates *less* talking). Worth validating with one coach interview before treating as a designed output.
