
# ***STEREO***: Towards Adversarially Robust Concept Erasing from Text-to-Image Generation Models

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>


> [**STEREO: Towards Adversarially Robust Concept Erasing from Text-to-Image Generation Models**]()<br>
> [Koushik Srivatsan](https://koushiksrivats.github.io/), [Fahad Shamshad](https://fahadshamshad.github.io/), [Muzammal Naseer](https://muzammal-naseer.com/), and [Karthik Nandakumar](https://scholar.google.com.pk/citations?user=2qx0RnEAAAAJ&hl=en)
>
> **Mohamed bin Zayed University of AI**
>
> [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
>
> ---
>
> # :rocket: Release
* **(August 28, 2024)**
  * Paper uploaded on arXiv.
 
## Abstract

The rapid proliferation of large-scale text-to-image generation (T2IG) models has led to concerns about their potential misuse in generating harmful content. Though many methods have been proposed for erasing undesired concepts from T2IG models, they only provide a false sense of security, as recent works demonstrate that concept-erased models (CEMs) can be easily deceived to generate the erased concept through adversarial attacks. The problem of adversarially robust concept erasing without significant degradation to model utility (ability to generate benign concepts) remains an unresolved challenge, especially in the white-box setting where the adversary has access to the CEM. To address this gap, we propose an approach called ***STEREO*** that involves two distinct stages. The first stage **S**earches **T**horoughly **E**nough (**STE**) for strong and diverse adversarial prompts that can regenerate an erased concept from a CEM, by leveraging robust optimization principles from adversarial training. In the second **R**obustly **E**rase **O**nce (**REO**) stage, we introduce an anchor-concept-based compositional objective to robustly erase the target concept at one go, while attempting to minimize the degradation on model utility. By benchmarking the proposed ***STEREO*** approach against four state-of-the-art concept erasure methods under three adversarial attacks, we demonstrate its ability to achieve a better robustness vs. utility trade-off.

<p align="center">
  <img src="pipeline.png" align="center" width="85%">
</p>

If you find our work and this repository useful, please consider giving our repo a star and citing our paper as follows:
```bibtex
@article{shamshad2024makeup,
  title={Makeup-Guided Facial Privacy Protection via Untrained Neural Network Priors},
  author={Shamshad, Fahad and Naseer, Muzammal and Nandakumar, Karthik},
  journal={arXiv preprint arXiv:2408.12387},
  year={2024}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at koushik.srivatsan@mbzuai.ac.ae.


## 🔜 Code and Models Coming Soon !!
