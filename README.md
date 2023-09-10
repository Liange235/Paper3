**Status:** Archive (code is provided as-is, no updates expected)

# VAE-DGP

Code, models and data from the paper ["Industrial Data Modeling with Low-Dimensional Inputs and High-Dimensional Outputs
Supplementary Material"](https://ieeexplore.ieee.org/document/10093135).

We have also [released a dataset] in the folder Data for researchers to study their behaviors.

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with GPT-2.

For basic information, see our [model card](./model_card.md).

### Some caveats

- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

### Work with us

Please [let us know](mailto:languagequestions@openai.com) if you’re doing interesting research with or working on applications of GPT-2!  We’re especially interested in hearing from and potentially working with those who are studying
- Potential malicious use cases and defenses against them (e.g. the detectability of synthetic text)
- The extent of problematic content (e.g. bias) being baked into the models and effective mitigations

## Development

See [DEVELOPERS.md](./DEVELOPERS.md)

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)

## Citation

Please use the following bibtex entry:
```
@ARTICLE{10093135,
  author={Tang, Jiawei and Lin, Xiaowen and Zhao, Fei and Chen, Xi},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Industrial Data Modeling with Low-Dimensional Inputs and High-Dimensional Outputs}, 
  year={2023},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/TII.2023.3264631}}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.

## License

[Modified MIT](./LICENSE)

