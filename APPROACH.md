# Approach

The network is the conditional Wasserstein GAN with gradient penalty from David Foster, *Generative Deep Learning*, 2nd edition, notebook `04_gan/03_cgan`. That notebook follows Sayak Paul's Keras conditional-GAN tutorial. The label expansion in `train_step` is the change this assignment requires, and it is the step that was executed.

## Data

CelebA stores `Blond_Hair` as −1 or +1. The positive class is +1. Everything else is the negative class. Labels are joined to image files by filename (`000001.jpg` and the rest), so the tag stays on the face it was written for. A bare list of labels lines up with `image_dataset_from_directory` only when the directory walk happens to follow the attribute file.

| Class | Meaning | One-hot | Faces |
| --- | --- | --- | --- |
| 0 | not blond | `[1, 0]` | 172,616 |
| 1 | blond | `[0, 1]` | 29,983 |

All 202,599 aligned faces are used. Blond hair is 14.8% of them. Each image is resized to 64×64 with bilinear interpolation, the same resize the notebook's directory loader uses, and scaled to [−1, 1] to match `tanh`.

Blond hair is not an isolated tag. In this attribute file, 5.8% of blond faces are labeled male, against 47.9% of the other faces, and 67.5% of blond faces are labeled heavy makeup, against 33.7%. That confound is part of the accuracy reading. A lighter, smoother face can satisfy the label without the network having isolated hair.

## The two label shapes

The generator takes noise of width 32 concatenated with the 2-class one-hot, reshaped to 1×1×34, then transposed convolutions up to a 64×64×3 image. Its label input has shape `(batch, 2)`.

The critic takes the RGB image concatenated on the channel axis with a label map of shape `(batch, 64, 64, 2)`. The map is the same one-hot copied onto every pixel.

`expand_label_to_map` does that copy inside `train_step`. The generator is called with the vector. The critic is called with the map, for the real image, the generated image, and the gradient-penalty interpolation. One map for all three calls keeps the penalty on the same conditioned critic that produced the scores. Passing the vector straight into the critic raises a shape error. The executed shape check records that rejection, then the successful forward pass. Both corners of the map equal the vector.

## Training

The schedule is the notebook's. `EPOCHS = 20` and `epochs = EPOCHS * 100` with one batch per step is 2,000 generator updates. Batch size 128, three critic steps, gradient-penalty weight 10, Adam at `5e-5` with betas `(0.5, 0.9)`. The notebook assigns `beta_2` twice; 0.9 is the value that remains. The penalty's interpolation weight is the notebook's normal draw.

The notebook sets `run_eagerly=True` so a reader can step through the update. The arithmetic in `cgan.py` is that update. This run traced it, which is what let the 2,000 updates finish. They took 72.6 minutes and saw 256,000 images, about 1.26 passes over CelebA.

## How the output was read

Fidelity is the samples set next to real faces of the same label, plus whether training settled. The Wasserstein gap fell to about −175 at step 71 and then recovered. Over the last 100 updates it averages −4.71, and the gradient penalty averages 0.04.

Diversity is the spread across noise vectors of one class, compared with the spread of real faces of that class. Pairwise root-mean-square distance on pixels in [0, 1] is 0.359 for generated not-blond faces against 0.375 for real ones, and 0.320 against 0.362 for blond. The grids are not one face repeated.

Conditioning is judged on paired noise: one `z`, both one-hots. A single pair is too soft to name the hair color. Averaged over 64 shared noise vectors, the hair band was brighter under the blond label every time, by +0.163, and the shift is warm (red +0.199, green +0.170, blue +0.121). The center of the face was also brighter every time, by less, +0.095, and that shift is not yellow. The mean difference image is an arch across the hair. Real CelebA, measured with the same boxes, goes the same direction (+0.134 on the hair band, +0.068 on the face).

That is a directional effect on hair brightness. It is not yet an accurate hair-color switch. The samples do not resolve eyes, teeth, or strands, and the label is tied to gender and makeup. The evidence that would close the question is stated in the report: a person naming the color on recognizable faces, or a blond classifier trained only on real CelebA scoring the paired samples apart, with the difference staying on the hair.
