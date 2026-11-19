# Public notebook sweep, 2026-11-19 (~11:30 UTC)

Scope: `kaggle kernels list --competition google-verifiable-quantum-circuit-compilation`, sorted by dateRun (pages 1-2), scoreDescending and voteCount, 100 per page.
Every notebook run since 2026-11-17 00:00 was kept: **90 notebooks, all 90 logs fetched and read** (`logs/`, stdout text extracted to `txt/`).
Nothing was downloaded beyond logs; no third-party code was run.

Our bests (default salt): invsqrt `ecf908fe` 74,383.70 | DQI `4471b493` 3.45236065e11 (safer `15a4b740` 3.45256821e11) | chemistry `6ef06081` 294,642,169.51 | GM 28.23498.

## Verdict: nothing better on any task

- **Highest GM printed anywhere: 28.23498 / 28.234976**, which is exactly our portfolio (ecf908fe + 4471b493 + 6ef06081 or the chemistry twins 66c43c75 / 7db38338, which have the same cost 294,642,169.5116752).
  Notebooks: drpatrickchan/qcc-28-23-frontier-portfolio, koushikrudra/qcc-v60-frontier-verified, ghazarosbarseghyan91/qcc-28-235-three-salt-dqi, manwithacat/the-real-150k-was-the-friends-we-made, boristown/qcc-gap-3-residue-repair-and-c43-native-compiler, asteriospanoras/qcc-28-23162-what-we-tried-what-failed-and-why, leotang09/vqcc-t7421-submission (source of 4471b493).
- **invsqrt**: no file below 74,383.69. The lowest number seen is `ecf908fe` itself (74,383.6681 in cosmorin / wenbodai, 74,383.7676 in ghazaros 28.23441: the same file with a different verifier draw). Other invsqrt files seen: f7841eb4 (1549 tof, worse), d0227f3c 74,405.67, 864f81cd 74,490.95, 0af21cb8 74,797, k6c 78,765.
- **DQI**: no file below 3.452360e11. The only lower number is `4471b493` itself at a second salt: **345,235,997,565.98, value_errors 0, phase_errors 0** (koushikrudra/qcc-v60-frontier-verified, unnamed "second salt"). Other DQI files: e208d350 345,252,882,080.64 (0/0 at default), e39b1396 345,255,477,425.06 (0/0 default; 0/0 at `--salt private-1` 25,600 shots, pablodiegoo), 15a4b740 345,256,821,433.98 (0/0), 5f901c55 / 66b4f18b / 40294f00 / 35fa5ef9 all >= 3.464e11. grantprater's spliced DQI (57f2de64 / a16ef984) is 3.464e11-3.480e11 with value_errors 0-1.
- **Chemistry**: nothing below 294,642,169.5. Lowest printed: 294,642,169.5116752 (6ef06081 / 66c43c75 / 7db38338). manwithacat's 587-circuit manifest picked 37948c2d at 2.946431e8 and its assembled chemistry prints 2.946422e8 with a GM equal to ours, so it is the same cost.
  Others: 294,642,170.565 (4d63be3d / 9349558c), 294,642,299.76 (703843d3), 294,643,137.73.

## Useful side information (DQI salt robustness for 4471b493)
- koushikrudra v60: default salt PASS 0/0 (3.45236065e11) and a second salt PASS 0/0 (3.45235998e11).
- ghazaros three-salt-dqi: extra salts `v60check` and `notebook-verification` ran, but errors were not recorded (None); it called 4471b493 "leaderboard-clean".
- No public multi-salt failure record for 4471b493 was found.

## Notebooks checked (90)
See `refs.txt`. Logs with no cost lines were starters, EDA or packaging-only (kawh1111 harvests, sittlohgilliand, taylorclark637, juventininizar, seshurajup, royalacecat, mzishran iadd, ayman0al0araishi (sample submission only), ratnachand, ahoirg submit-verify).
