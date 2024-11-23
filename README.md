
<div align="center" id = "user-content-toc">
<img align="left" width="70" height="70" src="https://github.com/user-attachments/assets/c61cec76-3c4b-42eb-8c65-f07e0166b7d8" alt="">
  
  # FlipSketch: Flipping assets Drawings to <br/> Text-Guided Sketch Animations
[Hmrishav Bandyopadhyay](https://hmrishavbandy.github.io/) . [Yi-Zhe Song](https://personalpages.surrey.ac.uk/y.song/)
</div>



https://github.com/user-attachments/assets/de8f2cef-3123-4a98-90e0-af59631a00f4

<div align="center"> 

## Sketch + Text &#8594; Animation 
</div>

![flipsketch_web_1](https://github.com/user-attachments/assets/8c04e9a7-2dd3-44d8-8a47-f2b3b81b83e7)

## App for Inference
- Install conda environment: 
```
conda env create -f flipsketch.yml
```
- Download T2V LoRA Model from HuggingFace
```
git clone https://huggingface.co/Hmrishav/t2v_sketch-lora
```
- Run app
```
python app.py
```


## How it works?

<div align="center">
<img src="./assets/images/model_1.png" alt="Generation Pipeline" width="75%">
<p>We use a T2V model fine-tuned on sketch animations, and condition it to follow an input sketch.</p>
<img src="./assets/images/model_2.png" alt="Noise Refinement" width="75%">
<p>We perform attention composition with reference noise from the input sketch.</p>
</div>

## Acknowledgements
- [Live-Sketch](https://github.com/yael-vinker/live_sketch)
- [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning)
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
