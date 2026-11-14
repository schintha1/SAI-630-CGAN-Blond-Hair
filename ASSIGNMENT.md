# Assignment

**Course.** Generative AI with Large Language Models (MSAI-630-M41), full term.

**Institution.** University of the Cumberlands, PhD in IT, AI specialization.

**Event.** Fall 2026 residency, Day 1, Project 1.

**Title.** Building a GAN.

**Points.** 100. One attempt. SafeAssign.

## Question

Build a conditional generative adversarial network (CGAN) from the course notebook:

https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/04_gan/03_cgan/cgan.ipynb

Do both of the following.

1. Condition the CGAN on the blond-hair attribute of the faces dataset.
2. Change `train_step` so the label matches the input format of the generator and the input format of the critic.

The notebook is already a conditional WGAN-GP on CelebA. The label is already `Blond_Hair`, with two classes. The two networks do not take that label in the same shape.

- **Generator.** A noise vector concatenated with a 2-class one-hot vector. In the notebook the noise width is 32.
- **Critic.** The image, concatenated with that same label repeated across every pixel, so the label is a spatial map.
- **`train_step`.** This is where the vector becomes the map. Passing the generator's label straight into the critic is the mismatch.

## What to submit

- The code that was actually run, including the label expansion, written so it can be explained.
- Samples from both classes. A short residency run may not yet look like a face.
- A reading of the pictures. Did blond hair follow the label? Say what the pictures support, including a weak run.

## How a GAN is graded here

Look past "the images look realistic."

| Question | What it asks |
| --- | --- |
| Fidelity | Do the samples resemble real faces? |
| Diversity | Does the generator avoid mode collapse? |
| Conditioning | Does the blond-hair control actually work? |

The pair-discussion question is: what evidence would convince you the model learned the condition?

## Rubric

The assignment is 100 points, split evenly.

**Code for the network and its output. 50 points.**

| Score | Standard |
| ---: | --- |
| 0 | This part of the submission is missing completely. |
| 15 | The code for the network is buggy, and the code output makes little sense. |
| 33 | The code has some bugs. The output makes some sense, with inconsistencies. |
| 50 | The code for the network looks correct, and the code output makes sense. |

**Observations about the output and its accuracy. 50 points.**

This half is the reading of fidelity, diversity, and conditioning. A complete answer uses the pictures from the run, says what those pictures support, and does not treat a soft or unfinished sample as a photograph.
