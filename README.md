# MSAI-630 · Conditional GAN on CelebA blond hair

Sai Amar Nath, Chintha · University of the Cumberlands · Fall 2026 residency

Generative AI with Large Language Models (MSAI-630-M41). Residency project: build a conditional GAN, condition it on the faces dataset's blond-hair attribute, and change `train_step` so the generator and the critic receive the label in the shape each one accepts.

The network is the conditional WGAN-GP from David Foster, *Generative Deep Learning*, 2nd edition, notebook [`04_gan/03_cgan`](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/04_gan/03_cgan/cgan.ipynb), which follows Sayak Paul's Keras conditional-GAN tutorial.

## Question

Condition generation on CelebA `Blond_Hair`, and fix the label mismatch in `train_step`.

The generator takes a noise vector of width 32 concatenated with a 2-class one-hot. Class 0, not blond, is `[1, 0]`. Class 1, blond, is `[0, 1]`. The critic takes the 64×64 image concatenated with that same label copied onto every pixel, a map of shape `(64, 64, 2)`. Passing the generator's vector straight into the critic is the mismatch. The full question, the submission rules, and the 100-point rubric are in [`ASSIGNMENT.md`](ASSIGNMENT.md).

## Approach

`expand_label_to_map` broadcasts the one-hot across height and width inside `train_step`. The generator keeps the vector. The critic, including the gradient penalty, receives the map. Labels are joined to files by image id. Training is the notebook's schedule: 2,000 updates, batch 128, three critic steps, penalty weight 10, Adam at `5e-5` with betas `(0.5, 0.9)`, on all 202,599 aligned faces. The reasoning, the confound with gender and makeup, and how fidelity, diversity, and conditioning were measured are in [`APPROACH.md`](APPROACH.md).

## What to turn in

| Document | Role |
| --- | --- |
| [`MSAI-630_CGAN_Blond_Hair_Report.pdf`](MSAI-630_CGAN_Blond_Hair_Report.pdf) | The submission document. Code, shape-check output, samples from both classes, and the reading of accuracy. |
| [`cgan_blond_hair.ipynb`](cgan_blond_hair.ipynb) | The same work with the code executed. Summaries, the shape check, and the figures are stored in the notebook outputs. |
| [`cgan.py`](cgan.py) | Generator, critic, and `train_step`. |
| [`train.py`](train.py) | The 2,000-update run. |
| [`data_celeba.py`](data_celeba.py) | Filename join and the 64×64 cache. |
| [`evaluate.py`](evaluate.py) | Paired-noise measurements behind the reading. |
| [`output/`](output/) | Figures, loss history, shape audit, evaluation numbers. |
| [`models/generator.weights.h5`](models/generator.weights.h5) | Weights that produced the final samples. |

[`MSAI-630_CGAN_Submission.zip`](MSAI-630_CGAN_Submission.zip) packs the report, the notebook, the modules, and the figures.

## Rubric, and where each half is answered

The assignment is 100 points. Each half is 50. The standards are in [`ASSIGNMENT.md`](ASSIGNMENT.md).

**Code for the network and its output (50).** The generator and critic match the course notebook. `train_step` expands `(batch, 2)` into `(batch, 64, 64, 2)` before every critic call. The shape check in the report rejects the unexpanded vector and accepts the map, and both corners of the map equal the vector. The run is 2,000 updates. Samples of both classes, the paired label flip, and the loss curves are in the report and under `output/`.

**Observations about the output and its accuracy (50).** Section 4 of the report reads the pictures on the three questions the residency names.

- **Fidelity.** By step 2,000 the samples have a face layout and are still soft. Eyes, teeth, and hair strands are unresolved. Hair-band brightness landed close to the real class means (generated 0.345 vs real 0.376 for not blond; 0.502 vs 0.510 for blond). The loss curves recovered: the Wasserstein gap averages about −4.7 over the last 100 updates, and the gradient penalty averages about 0.04.
- **Diversity.** Pairwise distance is 0.359 for generated not-blond faces against 0.375 for real ones, and 0.320 against 0.362 for blond. The grids are not one face repeated.
- **Conditioning.** On all 64 paired noise vectors the blond label made the hair band brighter, by +0.163, and warmer. It moved the hair more than the center of the face (+0.095). The mean difference is an arch across the hair. A person still cannot name the color from one soft sample, and blond hair in CelebA is tied to gender and makeup. The report states that limit, and the evidence that would show the control is accurate.

## Run it again

CelebA is not in this repository. Its license is non-commercial research use, from the Multimedia Lab at the Chinese University of Hong Kong. Place the attribute file and the aligned images here:

```text
data/list_attr_celeba.txt
data/img_align_celeba/000001.jpg
```

A public copy of those two files is hosted at [Yuehao/celeba](https://huggingface.co/datasets/Yuehao/celeba).

```bash
pip install -r requirements.txt
python train.py --steps 2000
python evaluate.py
python build_report.py
```

`output/run_metadata.json` records the step count, the class balance, and the wall-clock time of the submitted run.
