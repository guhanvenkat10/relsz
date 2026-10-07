# RelSz: a causal, patient-relative seizure detector for scalp EEG

Guhan Venkat (Independent). Contact: guhan.venkatachalapathi@gmail.com

RelSz is a seizure detection algorithm for long-term scalp electroencephalography (EEG). It is packaged in the
format of the SzCORE benchmark (https://epilepsybenchmarks.com): it reads one EDF recording and writes one annotation
file. This repository contains the inference code, the trained weights and the container definition. It does not
contain training code.

The detector is causal. The alarm state at any second depends only on samples recorded up to that second, so
processing a stored recording gives the same output as processing it as a stream.

## Method

1. Input. A recording with the 19 electrodes of the international 10-20 system (unipolar), or with the 18 derivations
   of the longitudinal bipolar (double banana) montage, sampled at 256 Hz. Unipolar input is converted to the 18
   bipolar derivations.
2. Filtering. A causal fourth-order Butterworth band-pass filter, 0.5 to 70 Hz.
3. Spectral representation. For each channel, the log power spectrum of a 2 s Hann window is computed every second
   and averaged into 40 bands of 1 Hz width between 0.5 and 40.5 Hz.
4. Patient-relative normalisation. From each channel and band, the median over the preceding five minutes of the
   same channel and band is subtracted (computed on 10 s block means). The network therefore receives the deviation
   of the EEG from the recent baseline of the same recording, not its absolute power.
5. Networks. Six convolutional networks are evaluated and their output probabilities are averaged.
   - Three networks (82,161 parameters each) encode the 40-band spectrum of every channel with a shared encoder,
     apply causal dilated temporal convolutions per channel, pool over the channels that are present (mean and
     maximum), and apply further causal temporal convolutions to produce one logit per second.
   - Three networks (136,001 parameters each) have the same structure and a second branch. The second branch receives
     the difference between the relative spectra of the eight homologous left and right bipolar derivations.
   - The receptive field is 27 s. The ensemble has 654,486 parameters.
6. Alarm rule. No alarm is raised in the first 60 s of a recording, because the baseline is not yet defined. After
   that, an alarm starts when the 3 s trailing mean of the probability has been at or above 0.4 for 5 consecutive
   seconds, and ends when the trailing mean falls below 0.4. Alarm onsets are not moved backwards in time.

## Training data

| Dataset | Subjects | Montage | Use |
|---|---|---|---|
| Siena Scalp EEG Database (PhysioNet) | 14 adults, aged 20 to 71 years | 10-20 scalp EEG | training; selection of the alarm rule by cross-validation |
| SeizeIT2 (OpenNeuro) | 125 patients with focal epilepsy | two behind-the-ear channels | training |
| CHB-MIT Scalp EEG Database (PhysioNet), 7 patients | aged 3 to 14 years | bipolar scalp EEG | development (model selection); not used for training |
| CHB-MIT Scalp EEG Database (PhysioNet), 16 patients | aged 1.5 to 22 years | bipolar scalp EEG | held-out evaluation; not used for training or tuning (see the notes under the results table) |

No data from the Temple University Hospital EEG corpus were used. Each network was trained for 12,000 steps with
AdamW (learning rate 0.001, one-cycle schedule, weight decay 0.01) on 60 s windows, alternating between batches from
the two training datasets. For scalp recordings a random subset of 2 to 18 channels was drawn per batch. Three
random seeds were trained for each of the two network types.

The CHB-MIT database is a paediatric cohort. Its 23 patients were divided into the development and held-out groups by
a seeded random split, stratified by seizure count, that was fixed before any model was trained. The age of one
held-out patient is not recorded in the database.

## Evaluation

All results use the event-based scoring of SzCORE (`szcore-evaluation` 0.0.7, default settings) and are means over
subjects, as on the SzCORE leaderboard.

### Held-out patients

The 16 held-out CHB-MIT patients comprise 737.6 h of EEG and 153 reference seizure events.

| Detector | Causal | F1 | Sensitivity | Precision | False alarms per 24 h | Median latency | Seizures detected | False alarms |
|---|---|---|---|---|---|---|---|---|
| RelSz (this release) | yes | 0.493 | 0.566 | 0.623 | 1.13 | 24.5 s | 54 of 153 | 33 |
| RelSz, earlier three-network version | yes | 0.441 | 0.509 | 0.629 | 1.32 | 22.0 s | 47 of 153 | 37 |
| SeizureTransformer, published weights and settings | no | 0.303 | 0.592 | 0.311 | 26.8 | 6.0 s | 71 of 153 | 696 |

Differences in F1, with 95% confidence intervals from a paired bootstrap over patients (10,000 resamples):

- RelSz minus SeizureTransformer: +0.190 (+0.077 to +0.297).
- RelSz minus the earlier three-network version: +0.052 (+0.021 to +0.088). The F1 was higher in 8 patients and
  lower in none.

SeizureTransformer (Wu, Zhao and Yener, arXiv:2504.00336) is the highest-ranked entry of the 2025 SzCORE challenge.
It was run here from its public SzCORE container with its published weights, decision threshold and post-processing.

The following points are needed to interpret the table.

- The earlier three-network version consists of the three networks without the asymmetry branch, with an alarm when
  the probability has been at or above 0.8 for 2 s.
- The held-out patients were never used for training or tuning. They were used for evaluation on three occasions.
  The first scored the three-network version. The second tested a fusion of that version with SeizureTransformer,
  which was not adopted. The third compared the three-network and six-network versions, and the six-network version
  was retained because its F1 was higher and its false alarm rate was not higher. The value 0.493 is therefore the
  higher of two candidate scores.
- SeizureTransformer requires unipolar input, and CHB-MIT is distributed in a bipolar montage. Its input was
  reconstructed from the bipolar signals by least squares. On the Siena database, where both forms are available,
  this reconstruction lowered the F1 of SeizureTransformer by 0.059 (from 0.706 to 0.647). The comparison above
  therefore carries a bias of about 0.06 against SeizureTransformer.
- SeizureTransformer detected more seizures (71 against 54). The difference in F1 comes from the number of false
  alarms (696 against 33).
- SeizureTransformer is not causal. Its median latency of 6.0 s is obtained with access to the whole recording.
- RelSz detected none of the 17 held-out seizures shorter than 20 s, and no seizure in 3 of the 16 patients.

### Other populations

| Population | F1 | Seizures detected | False alarms | Note |
|---|---|---|---|---|
| CHB-MIT development group, 7 patients | 0.657 | 23 of 47 | 7 | used for model selection |
| Siena, 14 adults, cross-validated by patient | 0.669 | 27 of 47 | 3 | one random seed; the alarm rule was selected on these scores |
| Helsinki neonatal EEG dataset, 79 neonates | 0.095 | 22 of 277 | 3 | no neonatal data in training |

The detector is not suitable for neonatal EEG. On the adults of the Siena database the asymmetry networks did not
change performance measurably relative to the three-network version (F1 0.669 against 0.659). The improvement was
observed in the paediatric CHB-MIT cohort.

Performance on adult patients from a centre not represented in the training data has not been measured by the
author. The SzCORE benchmark evaluates submissions on such data.

## Use

With Docker, as the SzCORE benchmark runs it:

```
docker run --rm --network none \
  -v /path/to/recordings:/data -v /path/to/results:/output \
  -e INPUT=recording.edf -e OUTPUT=recording.tsv \
  ghcr.io/guhanvenkat10/relsz:0.2.0
```

From source (Python 3.10 or later):

```
pip install .
python -m relsz recording.edf recording.tsv
```

The output is a tab-separated annotation file in the SzCORE format. Processing one hour of 18-channel EEG takes about
5 s on a desktop CPU (median over 221 recordings, six threads, including file loading). No GPU and no network access
are required at run time.

## Limitations

- None of the 17 held-out seizures shorter than 20 s was detected.
- No seizure was detected in 3 of the 16 held-out patients.
- The median detection latency on the held-out patients was 24.5 s after the annotated onset.
- The detector expects the standard 10-20 scalp montage and has not been validated on other electrode layouts.
- The software is a research implementation. It is not a medical device and must not be used for clinical decisions.

## References

- Dan J, Pale U, Amirshahi A, et al. SzCORE: A Seizure Community Open-source Research Evaluation framework for the
  validation of EEG-based automated seizure detection algorithms. Epilepsia (2024). doi:10.1111/epi.18113.
- Dan J, Shahbazinia A, Kechris C, Atienza D. Quantifying the Generalization Gap in Seizure Detection: A Large-Scale
  Empirical Benchmark via the SzCORE Challenge. arXiv:2505.18191.
- Wu K, Zhao Z, Yener B. Large EEG-U-Transformer for Time-Step Level Detection Without Pre-Training.
  arXiv:2504.00336. (SeizureTransformer.)
- Detti P, Vatti G, Zabalo Manrique de Lara G. EEG Synchronization Analysis for Seizure Prediction: A Study on Data
  of Noninvasive Recordings. Processes 8(7):846 (2020). Siena Scalp EEG Database, PhysioNet.
- Shoeb A. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD thesis,
  Massachusetts Institute of Technology (2009). CHB-MIT Scalp EEG Database, PhysioNet.
- Bhagubai M, Chatzichristos C, Swinnen L, et al. SeizeIT2: Wearable Dataset Of Patients With Focal Epilepsy.
  arXiv:2502.01224.
- Stevenson NJ, Tapani K, Lauronen L, Vanhatalo S. A dataset of neonatal EEG recordings with seizure annotations.
  Scientific Data 6:190039 (2019).

## Licence

Code and weights are released under the Creative Commons Attribution-NonCommercial 4.0 International licence
(CC BY-NC 4.0). See `LICENSE`.
