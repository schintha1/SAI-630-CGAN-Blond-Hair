"""Build the residency submission report from the executed run."""

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
PDF = ROOT / "MSAI-630_CGAN_Blond_Hair_Report.pdf"

NAVY = HexColor("#1B3A4B")
INK = HexColor("#1C1917")
MUTED = HexColor("#57534E")
RULE = HexColor("#D6D3D1")
CODE_BG = HexColor("#F5F4F1")
ROW = HexColor("#F7F6F3")
HEAD_BG = HexColor("#1B3A4B")

PAGE_W, PAGE_H = letter
LEFT = 0.72 * inch
RIGHT = 0.72 * inch
CONTENT_W = PAGE_W - LEFT - RIGHT


def styles():
    s = {}
    s["h1"] = ParagraphStyle(
        "h1",
        fontName="Times-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceAfter=4,
    )
    s["course"] = ParagraphStyle(
        "course",
        fontName="Times-Roman",
        fontSize=10,
        leading=13,
        textColor=MUTED,
        spaceAfter=2,
    )
    s["h2"] = ParagraphStyle(
        "h2",
        fontName="Times-Bold",
        fontSize=13,
        leading=16,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=6,
    )
    s["h3"] = ParagraphStyle(
        "h3",
        fontName="Times-Bold",
        fontSize=11.5,
        leading=14,
        textColor=INK,
        spaceBefore=9,
        spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body",
        fontName="Times-Roman",
        fontSize=10.5,
        leading=14.2,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
    )
    s["bullet"] = ParagraphStyle(
        "bullet",
        parent=s["body"],
        leftIndent=12,
        bulletIndent=0,
        spaceAfter=3,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        fontName="Times-Italic",
        fontSize=9,
        leading=11.5,
        textColor=MUTED,
        spaceBefore=2,
        spaceAfter=10,
    )
    s["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=7.4,
        leading=9.1,
        textColor=INK,
    )
    s["small"] = ParagraphStyle(
        "small",
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
        textColor=INK,
        alignment=TA_LEFT,
    )
    return s


S = styles()


def P(text, style="body"):
    return Paragraph(text, S[style])


def code_block(text):
    pre = Preformatted(text.rstrip() + "\n", S["code"])
    table = Table([[pre]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def figure(path, caption, max_width=None, max_height=340):
    max_width = CONTENT_W if max_width is None else max_width
    with PILImage.open(path) as im:
        width, height = im.size
    draw_w = max_width
    draw_h = draw_w * height / width
    if draw_h > max_height:
        draw_h = max_height
        draw_w = draw_h * width / height
    image = Image(str(path), width=draw_w, height=draw_h)
    return KeepTogether([image, P(caption, "caption")])


def table(headers, rows, col_widths):
    head = [
        Paragraph(f'<font color="white"><b>{cell}</b></font>', S["small"])
        for cell in headers
    ]
    body = [[Paragraph(str(cell), S["small"]) for cell in row] for row in rows]
    data = [head] + body
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, ROW]),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, NAVY),
        ("LINEBELOW", (0, 1), (-1, -2), 0.2, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 28, PAGE_W, 28, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawString(LEFT, PAGE_H - 18, "MSAI-630-M41  ·  University of the Cumberlands")
    canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 18, "Fall 2026 Residency")
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 26, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawString(LEFT, 10, "Conditional GAN conditioned on Blond_Hair")
    canvas.drawRightString(PAGE_W - RIGHT, 10, f"{doc.page}")
    canvas.restoreState()


EXPAND = '''def expand_label_to_map(one_hot_labels, image_size=IMAGE_SIZE):
    """Broadcast a class vector into the critic's spatial label map."""
    label_map = one_hot_labels[:, None, None, :]
    label_map = tf.repeat(label_map, repeats=image_size, axis=1)
    label_map = tf.repeat(label_map, repeats=image_size, axis=2)
    return label_map'''

TRAIN_STEP = '''def train_step(self, data):
    real_images, one_hot_labels = data

    # one_hot_labels is the generator format: (batch, 2).
    # The critic's second input is a map: (batch, 64, 64, 2).
    image_one_hot_labels = expand_label_to_map(one_hot_labels, IMAGE_SIZE)

    batch_size = tf.shape(real_images)[0]
    for _ in range(self.critic_steps):
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator([z, one_hot_labels], training=True)
            fake_predictions = self.critic(
                [fake_images, image_one_hot_labels], training=True
            )
            real_predictions = self.critic(
                [real_images, image_one_hot_labels], training=True
            )
            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                real_predictions
            )
            c_gp = self.gradient_penalty(
                batch_size, real_images, fake_images, image_one_hot_labels
            )
            c_loss = c_wass_loss + c_gp * self.gp_weight
        c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        )

    z = tf.random.normal(shape=(batch_size, self.latent_dim))
    with tf.GradientTape() as tape:
        fake_images = self.generator([z, one_hot_labels], training=True)
        fake_predictions = self.critic(
            [fake_images, image_one_hot_labels], training=True
        )
        g_loss = -tf.reduce_mean(fake_predictions)
    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(
        zip(gen_gradient, self.generator.trainable_variables)
    )'''

def _wrap_audit(text, width=88):
    """Keep the shape-check printout inside the code box."""
    lines = []
    for raw in text.splitlines():
        while len(raw) > width:
            cut = raw.rfind(" ", 0, width)
            if cut < 40:
                cut = width
            lines.append(raw[:cut])
            raw = raw[cut:].lstrip()
        lines.append(raw)
    return "\n".join(lines)


SHAPE = _wrap_audit((OUT / "shape_audit.txt").read_text().rstrip())


def chunk_source(lines, limit=48):
    """Break the listing on blank lines so a box never splits a statement."""
    chunks = []
    buf = []
    for index, line in enumerate(lines):
        buf.append(line)
        upcoming = lines[index + 1] if index + 1 < len(lines) else None
        at_boundary = line.strip() == "" or (
            upcoming is not None and (upcoming.startswith("def ") or upcoming.startswith("class "))
        )
        if len(buf) >= limit and at_boundary:
            chunks.append("\n".join(buf).strip("\n"))
            buf = []
    if any(line.strip() for line in buf):
        chunks.append("\n".join(buf).strip("\n"))
    return chunks


def build():
    story = []
    story.append(Spacer(1, 8))
    story.append(P("Conditional GAN conditioned on blond hair", "h1"))
    story.append(
        P(
            "Generative AI with Large Language Models (MSAI-630-M41) · Full term<br/>"
            "University of the Cumberlands · Fall 2026 residency",
            "course",
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        P(
            "This report is the submission for the residency GAN assignment. "
            "It contains the network that was trained, the change made to "
            "<font face='Courier' size='9'>train_step</font>, the output of that run, "
            "and a reading of the samples. The generator is conditioned on the "
            "CelebA <font face='Courier' size='9'>Blond_Hair</font> attribute. "
            "The architecture is the conditional Wasserstein GAN with gradient penalty "
            "from David Foster, <i>Generative Deep Learning</i>, 2nd edition, notebook "
            "<font face='Courier' size='9'>04_gan/03_cgan</font>, which follows Sayak Paul's "
            "Keras conditional-GAN tutorial. The label expansion below is the step this "
            "assignment asks for, and it is the step that produced the figures.",
        )
    )

    story.append(P("What the assignment asked for", "h2"))
    story.append(
        table(
            ["Requirement", "Where it is answered"],
            [
                [
                    "Condition generation on the faces dataset's blond-hair attribute.",
                    "Section 1. Class 1 is CelebA Blond_Hair = +1. Both classes are sampled.",
                ],
                [
                    "Modify train_step so the generator and critic receive the label in the shape each one accepts.",
                    "Section 2. The one-hot vector is broadcast to a 64×64 map before every critic call.",
                ],
                [
                    "Submit the code and its output.",
                    "Sections 2 and 3, and the appendix. The shape check, the loss curves, and the samples are from the executed run.",
                ],
                [
                    "Interpret the observations about the output and its accuracy.",
                    "Section 4. Fidelity, diversity, and whether the blond-hair label controlled the image.",
                ],
            ],
            [2.35 * inch, CONTENT_W - 2.35 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(P("1. Conditioning on Blond_Hair", "h2"))
    story.append(
        P(
            "CelebA's attribute file stores each tag as −1 or +1. "
            "<font face='Courier' size='9'>Blond_Hair</font> equal to +1 is the positive class. "
            "Every other value, including −1, is the negative class. Labels are joined to "
            "images by filename, so the tag stays on the face it was written for. "
            "The one-hot follows the course notebook."
        )
    )
    story.append(
        table(
            ["Class", "CelebA label", "One-hot passed to the generator", "Faces"],
            [
                ["0", "Blond_Hair ≠ +1", "[1, 0]", "172,616"],
                ["1", "Blond_Hair = +1", "[0, 1]", "29,983"],
            ],
            [0.7 * inch, 1.5 * inch, 2.5 * inch, CONTENT_W - 4.7 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "The training set is all 202,599 aligned faces. Blond hair is 14.8% of them. "
            "Images are resized to 64×64 with bilinear interpolation, the resize used by "
            "<font face='Courier' size='9'>image_dataset_from_directory</font>, and scaled "
            "to [−1, 1] to match the generator's tanh output. The two rows below are real "
            "faces drawn from those labels. They are the standard the samples are judged against."
        )
    )
    story.append(
        figure(
            OUT / "real_not_blond.png",
            "Figure 1. Real CelebA faces with Blond_Hair ≠ +1. Hair in this class is brown, black, or otherwise not the blond tag.",
            max_height=150,
        )
    )
    story.append(
        figure(
            OUT / "real_blond.png",
            "Figure 2. Real CelebA faces with Blond_Hair = +1. The tag is a light, yellow hair color, and it co-occurs with other attributes.",
            max_height=150,
        )
    )
    story.append(
        P(
            "The tag is not an isolated hair switch. In this attribute file, 5.8% of blond "
            "faces are labeled male, against 47.9% of the remaining faces. Heavy makeup is "
            "labeled on 67.5% of blond faces and on 33.7% of the others. A model can pick up "
            "a lighter, smoother face from those correlates. That matters in Section 4, when "
            "the question is whether the network learned hair."
        )
    )

    story.append(P("2. The network, and the change to train_step", "h2"))
    story.append(P("Two different shapes for one label", "h3"))
    story.append(
        P(
            "The generator and the critic both see the class, and they do not see it in the same tensor."
        )
    )
    story.append(
        P(
            "<b>Generator.</b> A noise vector of width 32 is concatenated with the 2-class "
            "one-hot, reshaped to 1×1×34, and expanded by transposed convolutions: "
            "4×4×128, 8×8×128, 16×16×128, 32×32×64, then a 64×64×3 tanh image. "
            "The generator has 729,859 parameters. Its label input is rank 1 per example: shape (2,)."
        )
    )
    story.append(
        P(
            "<b>Critic.</b> The 64×64 RGB image is concatenated, on the channel axis, with "
            "a label map of shape 64×64×2. The stack of five channels is convolved down "
            "through 32×32, 16×16, 8×8, and 4×4, then a valid 4×4 convolution to a single score. "
            "The critic has 662,977 parameters. Its label input is a spatial map, one copy of "
            "the one-hot at every pixel."
        )
    )
    story.append(
        P(
            "Passing the generator's vector straight into the critic is a shape error. "
            "The critic was built for (batch, 64, 64, 2). A tensor of shape (batch, 2) has "
            "no height and width. <font face='Courier' size='9'>train_step</font> is the place "
            "that vector becomes the map. The expansion copies the same one-hot onto every pixel. "
            "It does not draw a new label. The real image, the generated image, and the "
            "interpolated image inside the gradient penalty all receive that one map, so the "
            "penalty regularizes the same conditioned critic that produced the scores."
        )
    )
    story.append(P("Label expansion", "h3"))
    story.append(code_block(EXPAND))
    story.append(Spacer(1, 8))
    story.append(P("train_step after the change", "h3"))
    story.append(
        P(
            "The generator is called with <font face='Courier' size='9'>one_hot_labels</font>. "
            "The critic is called with <font face='Courier' size='9'>image_one_hot_labels</font>. "
            "The loss is the course notebook's Wasserstein objective: the critic minimizes the "
            "mean fake score minus the mean real score, plus ten times the gradient penalty. "
            "The generator minimizes the negative mean fake score. Three critic updates run for "
            "each generator update."
        )
    )
    story.append(code_block(TRAIN_STEP))
    story.append(Spacer(1, 8))
    story.append(
        P(
            "The full module that was executed, including both network definitions, is "
            "<font face='Courier' size='9'>cgan.py</font>. It is printed in the appendix. "
            "Training is <font face='Courier' size='9'>train.py</font>. "
            "The executed notebook <font face='Courier' size='9'>cgan_blond_hair.ipynb</font> "
            "reprints this step, rebuilds both networks, and stores the figures."
        )
    )

    story.append(P("3. Output of the code", "h2"))
    story.append(P("Shape check", "h3"))
    story.append(
        P(
            "Before the long run, one real batch was passed through the two formats. "
            "The unexpanded vector was given to the critic on purpose. It was rejected. "
            "The expanded map was accepted, and both corners of the map matched the vector. "
            "This is the printout from that check."
        )
    )
    story.append(code_block(SHAPE))
    story.append(Spacer(1, 8))
    story.append(P("Training record", "h3"))
    story.append(
        P(
            "The schedule is the course notebook's. <font face='Courier' size='9'>EPOCHS = 20</font> "
            "and a fit call of <font face='Courier' size='9'>EPOCHS × 100</font> steps, one batch "
            "per step, is 2,000 generator updates. The notebook sets "
            "<font face='Courier' size='9'>run_eagerly=True</font> so a reader can step through "
            "the update. The arithmetic above is that update. This run traced it, which is what "
            "let the 2,000 updates finish. The penalty's interpolation weight is the notebook's "
            "normal draw. The optimizer is Adam at 5×10<sup>−5</sup>, with β<sub>1</sub> = 0.5 "
            "and β<sub>2</sub> = 0.9. The notebook assigns β<sub>2</sub> twice; 0.9 is the value "
            "that remains."
        )
    )
    story.append(
        table(
            ["Setting", "Value"],
            [
                ["Faces / blond faces", "202,599 / 29,983 (14.80%)"],
                ["Image size, noise width", "64×64 RGB, z = 32"],
                ["Batch, critic steps, penalty weight", "128, 3, 10"],
                ["Generator updates", "2,000 (256,000 images seen, 1.26 passes)"],
                ["Critic updates", "6,000"],
                ["Wall clock", "72.6 minutes"],
                ["Seed", "42"],
            ],
            [2.7 * inch, CONTENT_W - 2.7 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        figure(
            OUT / "loss_curves.png",
            "Figure 3. Training curves. The Wasserstein estimate fell to −175 at step 71 and then returned to a small negative gap. The gradient penalty settled near zero. The generator loss came down from its early peak.",
            max_height=220,
        )
    )
    story.append(
        P(
            "Over the last 100 updates the Wasserstein gap averages −4.71 and the gradient "
            "penalty averages 0.04. A penalty near zero means the critic's input gradients "
            "are near unit length, which is the constraint this loss is there to hold. "
            "The generator's raw loss is a weak quality number: the critic has no squashing "
            "activation, so its score can sit far from zero while the gap between real and "
            "fake stays modest. The gap, and the pictures, are the record of the run."
        )
    )

    story.append(P("Samples from both classes", "h3"))
    story.append(
        P(
            "Each paired figure uses one noise vector per column. The top row is class 0, "
            "one-hot [1, 0]. The bottom row is class 1, one-hot [0, 1]. Nothing else changes."
        )
    )
    story.append(
        figure(
            OUT / "paired_step_0000.png",
            "Figure 4. Step 0, before any update. Both labels are a flat gray field. Batch normalization is still at its zero start, and tanh of zero is gray.",
            max_height=175,
        )
    )
    story.append(
        figure(
            OUT / "paired_step_0400.png",
            "Figure 5. Step 400. Skin-colored blobs with a darker mass along the top. The two rows are still difficult to separate by eye.",
            max_height=175,
        )
    )
    story.append(
        figure(
            OUT / "paired_step_1000.png",
            "Figure 6. Step 1,000. A face layout is present: a warmer oval, a hair region, a lighter surround. Features are unresolved.",
            max_height=175,
        )
    )
    story.append(
        figure(
            OUT / "paired_step_2000.png",
            "Figure 7. Step 2,000, the submitted run, same ten noise vectors as Figure 4. The blond row is lighter and softer. Pose stays in the column.",
            max_height=175,
        )
    )
    story.append(
        figure(
            OUT / "final_not_blond.png",
            "Figure 8. Sixteen fresh class-0 samples from the saved generator. Not blond, one-hot [1, 0].",
            max_height=200,
        )
    )
    story.append(
        figure(
            OUT / "final_blond.png",
            "Figure 9. Sixteen fresh class-1 samples from the saved generator. Blond, one-hot [0, 1]. Softer and lighter than Figure 8.",
            max_height=200,
        )
    )
    story.append(
        figure(
            OUT / "final_paired.png",
            "Figure 10. Eight shared noise vectors, label flipped. This is the comparison used to judge the condition.",
            max_height=200,
        )
    )
    story.append(
        figure(
            OUT / "real_vs_generated.png",
            "Figure 11. Real photographs beside generated samples of the same label. The top two rows are class 0. The bottom two rows are class 1.",
            max_height=320,
        )
    )

    story.append(P("4. Observations about the output and its accuracy", "h2"))
    story.append(
        P(
            "The figures in Section 3 are the 2,000-update run. Three questions have to stay "
            "separate, because a GAN can pass one and fail the others. Fidelity asks whether "
            "the samples look like faces. Diversity asks whether the generator collapsed onto "
            "one picture. Conditioning asks whether the blond-hair label controls the sample. "
            "Accuracy, for this assignment, is the third question answered against the pictures."
        )
    )

    story.append(P("Fidelity", "h3"))
    story.append(
        P(
            "At step 0 both classes are a blank gray square. By step 400 the samples are "
            "skin-colored blobs with a darker cap. By step 1,000 a face layout is visible, "
            "and at step 2,000 that layout is steadier. Eyes, teeth, ears, and individual "
            "hair strands are unresolved. Set beside the real photographs in Figure 11, "
            "the samples read as face-shaped color fields. They would not be taken for pictures of people."
        )
    )
    story.append(
        P(
            "The curves say the softness is unfinished learning. The critic pulled far ahead "
            "early, the penalty spiked, and both quantities then settled into the band "
            "described in Section 3. The run did not diverge."
        )
    )
    story.append(
        P(
            "Brightness in the hair region did land close to the data, even while the "
            "structure inside that region did not. Luminance is measured on pixels scaled "
            "to [0, 1], inside the yellow box in Figure 12."
        )
    )
    story.append(
        table(
            ["Class", "Generated hair luminance", "Real hair luminance"],
            [
                ["Not blond", "0.345", "0.376"],
                ["Blond", "0.502", "0.510"],
            ],
            [2.2 * inch, 2.4 * inch, CONTENT_W - 4.6 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "The hair region of a generated blond sample is about as bright as the hair "
            "region of a real blond face. The pixels in that region are still a blur."
        )
    )

    story.append(P("Diversity", "h3"))
    story.append(
        P(
            "Figures 8 and 9 are not one image copied across the row. Head placement, "
            "how dark the cap is, and the background all change from column to column. "
            "The average root-mean-square distance between distinct samples, again on "
            "pixels in [0, 1], sits close to the same distance measured on real faces."
        )
    )
    story.append(
        table(
            ["Class", "Generated pairwise RMSE", "Real pairwise RMSE"],
            [
                ["Not blond", "0.359", "0.375"],
                ["Blond", "0.320", "0.362"],
            ],
            [2.2 * inch, 2.4 * inch, CONTENT_W - 4.6 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Not-blond samples are about as spread out as real not-blond faces. Blond "
            "samples cover about 88% of the real blond spread, and they look softer. "
            "That fits a class that is 14.8% of the data. Mode collapse would push the "
            "pairwise distance toward zero and make every column the same picture. "
            "Neither class does that."
        )
    )

    story.append(P("Conditioning, and the accuracy of the blond-hair control", "h3"))
    story.append(
        P(
            "A single column of Figure 7 or Figure 10 is a weak place to call the hair blond. "
            "The bottom image keeps the pose of the top image, which is what a conditional "
            "model should do, and it looks like the same blur with the contrast lowered. "
            "A person asked to name the hair color from one sample would have little to go on. "
            "On that test, the control is not yet accurate."
        )
    )
    story.append(
        P(
            "The label is doing something systematic. It shows up when the same comparison "
            "is averaged over noise. Sixty-four noise vectors were drawn once and passed "
            "through both one-hots. In every one of the 64 pairs, the hair band was brighter "
            "under the blond label. The mean gap is +0.163. The center of the face was also "
            "brighter in every pair, by less: +0.095. Figure 14 is that pair of distributions. "
            "Both sit entirely to the right of zero, and the hair-band mass sits to the right "
            "of the face-center mass."
        )
    )
    story.append(
        figure(
            OUT / "measurement_boxes.png",
            "Figure 12. Measurement boxes on a real blond face. Yellow is the hair band (rows 10–26). Blue is the center of the face (rows 30–46), used as the control region.",
            max_width=2.5 * inch,
            max_height=220,
        )
    )
    story.append(
        figure(
            OUT / "label_difference_map.png",
            "Figure 13. Mean paired difference, blond minus not blond, over the 64 shared noise vectors. Red is a pixel the blond label made brighter. The bright pixels form an arch across the hair. The face changes less. The side background moves slightly the other way.",
            max_height=250,
        )
    )
    story.append(
        figure(
            OUT / "label_luminance_gap.png",
            "Figure 14. For each of 64 noise vectors, blond luminance minus not-blond luminance. The hair band (gold) shifts further than the face center (blue). Every value is positive.",
            max_height=230,
        )
    )
    story.append(
        P(
            "The extra hair brightness is warm. Across the 64 pairs the hair band moves by "
            "+0.199 red, +0.170 green, and +0.121 blue. The face center moves by +0.085 red, "
            "+0.114 green, and +0.088 blue. The hair shift leans yellow. The face shift does not."
        )
    )
    story.append(
        P(
            "Figure 13 is the spatial version of those numbers. The average change is an arch "
            "across the hair, a smaller change on the face, and a slight darkening at the "
            "sides. A label that only raised a global brightness knob would be a flat field. "
            "This difference has the shape of hair."
        )
    )
    story.append(
        P(
            "Real CelebA, measured with the same boxes, goes the same direction. Blond faces "
            "are +0.134 brighter in the hair band and +0.068 brighter in the face center. "
            "The generator's gaps, +0.163 and +0.095, are a bit larger and in the same order. "
            "The hair-band means in the fidelity table already sit within a few hundredths of "
            "the real class means."
        )
    )
    story.append(
        P(
            "Taken together, the control moved hair luminance in the blond direction on every "
            "latent that was tested, and it moved hair more than it moved the face. That is "
            "partial evidence that the condition was learned. It is weak evidence that "
            "blond-hair control is accurate. The samples are too soft for the color to be "
            "named from one picture, and two facts about the label keep the claim bounded."
        )
    )
    story.append(
        P(
            "Blond is 14.8% of the set, so most critic updates are the negative class, and "
            "the blond grid is the blurrier of the two. Blond is also tied to gender and makeup, "
            "as Section 1 recorded. A network can score well by drifting toward a lighter, "
            "smoother face. The hair-shaped difference in Figure 13 is the reason to think "
            "the model caught more than that correlate. The face-center shift of +0.095, "
            "together with the makeup and gender imbalance, is the reason to treat the "
            "result as a directional effect on hair brightness rather than as isolated, "
            "accurate hair control. The dataset tag itself is coarse at the boundary: "
            "light-brown hair is often stored as not blond, and the generator cannot be "
            "cleaner than the label it was given."
        )
    )

    story.append(P("What would show that the condition had been learned", "h3"))
    story.append(
        P(
            "The comparison that counts uses one noise vector and both labels, so pose is "
            "held still, and then asks for something sharper than a brightness change."
        )
    )
    story.append(
        P(
            "• On recognizable faces, a person who has not seen the label should be able "
            "to name the hair color clearly above chance."
        )
    )
    story.append(
        P(
            "• A classifier trained only on real Blond_Hair labels should score the generated "
            "blond samples well above the not-blond samples that share their noise."
        )
    )
    story.append(
        P(
            "• The paired difference should stay on the hair. A difference that lights up "
            "the jaw, the makeup, or the whole frame is the correlate, not the attribute."
        )
    )
    story.append(
        P(
            "This run meets a weak form of the third test. The average paired difference is "
            "a warmer arch on the hair, and the hair gap is larger than the face gap. It does "
            "not meet the first test. The samples are too blurred for a person to call the "
            "color from one image, and no external classifier was used. The pictures support "
            "a directional, partial effect. They do not yet support a claim that the "
            "blond-hair control is accurate."
        )
    )

    story.append(P("5. Conclusion", "h2"))
    story.append(
        P(
            "The code builds the conditional WGAN-GP from the course notebook, conditions it "
            "on CelebA <font face='Courier' size='9'>Blond_Hair</font>, and expands the "
            "generator's one-hot into the critic's label map inside "
            "<font face='Courier' size='9'>train_step</font>. The shape check shows the "
            "unexpanded vector being rejected and the map being accepted, with every tested "
            "pixel equal to the vector. Two thousand updates produced a stable run. The "
            "samples have a face layout, they have not collapsed, and the blond label shifts "
            "hair luminance in the blond direction on every paired noise vector, by more than "
            "it shifts the face. The samples are soft, and the label is a partial control "
            "of hair brightness rather than an accurate hair-color switch."
        )
    )
    story.append(
        P(
            "Files that accompany this report: <font face='Courier' size='9'>cgan.py</font> "
            "(the network and <font face='Courier' size='9'>train_step</font>), "
            "<font face='Courier' size='9'>train.py</font> (the 2,000-update run), "
            "<font face='Courier' size='9'>data_celeba.py</font> (the filename join and the "
            "64×64 cache), <font face='Courier' size='9'>evaluate.py</font> (the measurements "
            "in Section 4), <font face='Courier' size='9'>cgan_blond_hair.ipynb</font> "
            "(the same code with executed output), and the figures under "
            "<font face='Courier' size='9'>output/</font>. Generator weights are "
            "<font face='Courier' size='9'>models/generator.weights.h5</font>."
        )
    )

    story.append(PageBreak())
    story.append(P("Appendix. Network source that was executed", "h2"))
    story.append(
        P(
            "The listing is <font face='Courier' size='9'>cgan.py</font>. "
            "Comments that record the course notebook's choices are left in, because those "
            "choices are part of the run: eager mode is off so the traced step can finish, "
            "and the penalty's alpha is the notebook's normal draw."
        )
    )
    for block in chunk_source((ROOT / "cgan.py").read_text().splitlines()):
        story.append(code_block(block))
        story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=letter,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=0.62 * inch,
        bottomMargin=0.55 * inch,
        title="MSAI-630 Conditional GAN on Blond Hair",
        author="MSAI-630-M41 residency submission",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"wrote {PDF} ({PDF.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
