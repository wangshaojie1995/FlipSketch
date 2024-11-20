# FlipSketch: Flipping assets Drawings to Text-Guided Sketch Animations


<span style="display: inline-flex; align-items: center; gap: 5px;">
  <a href="https://arxiv.org/pdf/2411.10818v1">
    <img src="https://img.icons8.com/material-outlined/24/000000/pdf.png" alt="PDF" style="vertical-align: middle;" />
    PDF
  </a>
</span>
<span style="display: inline-flex; align-items: center; gap: 5px; margin-left: 15px;">
  <a href="http://hmrishavbandy.github.io/flipsketch-web/">
    <img src="https://img.icons8.com/ios-glyphs/24/000000/globe.png" alt="Website" style="vertical-align: middle;" />
    Website
  </a>
</span>

<br>
<br>

<div align="center">
<video autoplay controls muted loop width="100%">
  <source src="./assets/videos/teaser.mp4" type="video/mp4">
</video>
</div>

<h1 style="text-align: center;">
  Code Coming Soon
</h1>

## Generation Pipeline

<div align="center">
<img src="./assets/images/model_1.png" alt="Generation Pipeline" width="75%">
<p>We use a T2V model fine-tuned on sketch animations, and condition it to follow an input sketch.</p>
<img src="./assets/images/model_2.png" alt="Noise Refinement" width="75%">
<p>We perform attention composition with reference noise from the input sketch.</p>
</div>


## Sketch+Text → Animation

### Examples:


  <div align="center">
  <video autoplay muted loop playsinline>
    <source src="./assets/videos/cat_vid.m4v" type="video/mp4">
  </video>
  </div>
  <div align="center">
  <video autoplay muted loop playsinline>
    <source src="./assets/videos/eagle_vid.m4v" type="video/mp4">
  </video>
  </div>
  <div align="center">
  <video autoplay muted loop playsinline>
    <source src="./assets/videos/gazelle_vid.m4v" type="video/mp4">
  </video>
  </div>

---

### Frame Extrapolation

<div align="center">
<video autoplay muted loop playsinline>
  <source src="./assets/videos/ballerina_extra.m4v" type="video/mp4">
</video>
<p>We extrapolate videos by using the last frame of one video as the first frame of another.</p>
</div>

---

### Real-World Video Generation

<div align="center">
<video autoplay muted loop playsinline>
  <source src="./assets/videos/real_vids.m4v" type="video/mp4">
</video>
</div>

---

## BibTeX

```bibtex
@misc{bandyopadhyay2024flipsketch,
  title={FlipSketch: Flipping assets Drawings to Text-Guided Sketch Animations}, 
  author={Hmrishav Bandyopadhyay and Yi-Zhe Song},
  year={2024},
  eprint={2411.10818},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2411.10818}, 
}
