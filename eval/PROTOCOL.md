# Clip recording protocol (23 clips, about 2 hours)

Goal: real numbers for `python -m eval.run_eval`. Fill `eval/manifest.template.csv` (copy to `manifest.csv`); the rows are pre-planned.

## 1. Who you record
Adult volunteers who give written consent, breathing in time with a metronome. This is a **proxy**: it tests the algorithm, not infants or sick children. Record sick or young children only with ethics approval and guardian consent, and keep those clips in a separate manifest. Frame the torso only; keep faces out. Store clips outside the repo (`eval/clips/` is gitignored) and delete them when the study ends.

## 2. Setup (every clip)
- Phone on a stand or books, **not handheld** (except the two handheld clips), 30 to 50 cm from the chest, chest filling most of the frame.
- Subject sitting or lying, thin top or bare chest, steady light on the chest. Record 60 seconds, same orientation as the app (landscape-ish 4:3 or 720p).
- Name files exactly as in the template. Save as MP4 or MOV.
- Check the app path too: record 3 of the clips again with the PWA itself, because its WebM compression differs from phone video. (A debug "save clip" button does not exist yet.)

## 3. Normal clips (15): rates and split
Targets in breaths/min: 20, 24, 28, 32, 36, 38, 40, 42, 44, 46, 48, 50, 52, 56, 60. Seven are `dev`, eight are `test` (set in the template). Eight sit within 4 of the 40 or 50 limits: the clips that matter most.
Play a metronome (one beat per breath, one breath = one inhale and exhale) and have the subject match it for the whole minute. Rates above about 50 are hard: let the subject practise first, or accept drift, because the counters set the truth.

## 4. Ground truth: two independent counters
The metronome is the intention, not the truth. Two people each watch the clip alone and count full breaths over the 60 seconds (tap counter or tally). Enter `truth_a` and `truth_b`; the harness uses the mean and reports their disagreement. If they differ by more than 2, recount; if still more than 2, drop the clip and say so. Never use the system's output as truth.

## 5. Stress clips (6): should be refused
`should_refuse=yes`, truth left blank. Normal breathing in each unless stated.
- `blanket`: thick blanket over the chest.  - `dark`: room very dim.  - `handheld_walk`: filming while walking.
- `occluded`: arm or hand over the chest for most of the minute.  - `no_person`: empty frame or person out of view.
- `still_doll`: a pillow or doll on the chest, subject out of frame (motion-free object).

## 6. Realistic clips (2): should be accepted
`r_handheld_steady` (30/min, held in hand, as a parent would) and `r_loose_shirt` (40/min, loose clothing). Truth from counters. These test tremor and clothing, which a stand does not.

## 7. Run order (this prevents fooling yourself)
1. Run `dev` only: `--split dev --tracks box`, then `--tracks api` (SAM about $0.12 a clip, about $1.40 for dev). Compare: the gap shows what SAM adds.
2. Tune `min_rms`, `min_peak_ratio`, `agree_bpm` on `dev` only. Write down every change.
3. Freeze the code. Run `test` **once**. Report that report unedited, including misses.
4. If you retune after seeing `test`, say so: it is no longer a held-out result.

## 8. Write your targets down before step 1
Suggested starting points (agree them with your clinician): no missed fast breathing among clips more than 3 from a limit; at least 5 of 6 stress clips refused; MAE no worse than the counters' own disagreement plus 2. With 12 test clips the confidence intervals are wide, so claim "works on these clips", not "is accurate".
