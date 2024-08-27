
# ***STEREO***: Towards Adversarially Robust Concept Erasing from Text-to-Image Generation Models


<p align="center">
 <a href="https://koushiksrivats.github.io/">Koushik Srivatsan</a>
 <a href="https://fahadshamshad.github.io/">Fahad Shamshad</a>
 <a href="https://muzammal-naseer.com/">Muzammal Naseer</a>
 <a href="https://scholar.google.com.pk/citations?user=2qx0RnEAAAAJ&hl=en">Karthik Nandakumar</a>
 <br>
    <span style="font-size:1em; "><strong> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), UAE</strong>.</span>
</p>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>


<p align="center">
  <img src="docs/static/images/teaser.jpg" align="center" width="95%">
</p>

<p align="center">
  <a href="" target='_blank'>
      <img src="https://img.shields.io/badge/arXiv-Paper-brown.svg">
  </a>

  <a href="https://koushiksrivats.github.io/robust-concept-erasing/" target='_blank'>
      <img src=https://img.shields.io/badge/Project-Website-87CEEB">
  </a>
</p>

# :rocket: Release
* **(August 28, 2024)**
  * Paper uploaded on arXiv.
 
## Abstract

**<p align="justify">** The rapid proliferation of large-scale text-to-image generation (T2IG) models has led to concerns about their potential misuse in generating harmful content. Though many methods have been proposed for erasing undesired concepts from T2IG models, they only provide a  false sense of security, as recent works demonstrate that concept-erased models (CEMs) can be easily deceived to generate the erased concept through adversarial attacks. The problem of adversarially robust concept erasing without significant degradation to model  utility (ability to generate benign concepts) remains an unresolved challenge, especially in the white-box setting where the adversary has access to the CEM. To address this gap, we propose an approach called ***STEREO*** that involves two distinct stages. The first   stage **S**earches **T**horoughly **E**nough (**STE**) for strong and diverse adversarial prompts that can regenerate an erased concept from a CEM, by leveraging robust optimization principles from adversarial training. In the second **R**obustly **E**rase **O**nce (**REO**) stage, we introduce an anchor-concept-based compositional objective to robustly erase the target concept at one go, while attempting to minimize the degradation on model utility. By benchmarking the proposed ***STEREO*** approach against four state-of-the-art concept erasure methods under three adversarial attacks, we demonstrate its ability to achieve a better robustness vs. utility trade-off.
 

## Highlights
Large-scale diffusion models for text-to-image generation are susceptible to adversarial attacks that can regenerate harmful concepts despite erasure efforts. We introduce ***STEREO***, a robust approach designed to prevent this regeneration while preserving the model's ability to generate benign content.

<p align="center">
  <img src="docs/static/images/pipeline.png" align="center" width="95%">
</p>


**<p align="justify">**
***Overview of STEREO***. 
We propose a novel two-stage framework for adversarially robust concept erasing from pre-trained text-to-image generation models without significantly affecting the utility for benign concepts. 

**Stage 1 (top)**: Search Thoroughly Enough (STE) follows the robust optimization framework of Adversarial Training and formulates concept erasing as a min-max optimization problem, to discover strong adversarial
prompts that can regenerate target concepts from erased models. Note that,  the core novelty of our approach lies in the fact that we employ AT not as a final solution, but only as an intermediate step to search thoroughly enough for strong adversarial prompts.

**Stage 2 (bottom)**: Robustly Erase Once fine-tunes the model using an anchor concept and the set of strong adversarial prompts from Stage 1 via a compositional objective, maintaining high-fidelity generation of benign concepts while robustly erasing the target concept.
 

## 🔜 Code and Models Coming Soon !!



## Citation
If you find our work and this repository useful, please consider giving our repo a star and citing our paper as follows:
```bibtex
@article{srivatsan2024stereo,
  title={STEREO: Towards Adversarially Robust Concept Erasing from Text-to-Image Generation Models},
  author={Koushik, Srivatsan and Shamshad, Fahad and Naseer, Muzammal and Nandakumar, Karthik},
  journal={arXiv preprint arXiv:xxxxxxx},
  year={2024}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at koushiksrivatsan.ofcl@gmail.com.

## Acknowledgement :pray:
Our code is built on top of the [ESD](https://github.com/rohitgandikota/erasing) repository. We thank the authors for releasing their code.