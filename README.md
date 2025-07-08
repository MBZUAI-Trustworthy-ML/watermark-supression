# Mitigating Watermark Stealing Attacks in Language Models via Multi-Key Watermarking

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository implements **multi-key watermarking** as a defense against watermark stealing attacks in large language models. Our approach significantly reduces spoofing success rates by using multiple watermarking keys and advanced detection algorithms with proper statistical corrections.

> **Built upon**: This work extends the excellent [Watermark Stealing](https://watermark-stealing.org) codebase. We thank the original authors for their open-source contributions.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenAI API key for GPT-as-a-judge evaluation

### Setup
```bash
git clone <repository-url>
cd watermark-supression
bash setup.sh  # Installs Conda, creates 'ws' environment, dependencies
export OAI_API_KEY="your-openai-api-key"
```

### Basic Usage
```bash
python3 main.py configs/spoofing/selfhash/mistral_4keys.yaml
```

## 📊 Key Results

Our multi-key watermarking achieves:
- **89% reduction** in spoofing success rates (from 94% to 11% with 4 keys)
- **Dual protection**: Degrades attacker's ability to generate coherent harmful content
- **Robust defense** against adaptive attacks with key clustering

## 🏗️ Repository Structure

```
├── main.py                 # Main entry point
├── main_adaptive.py        # Key clustering attack experiments  
├── src/
│   ├── attackers/         # Watermark stealing attack implementations
│   ├── config/           # Pydantic configuration files
│   ├── models/           # Server, attacker, judge, and PSP models
│   ├── utils/            # Utilities (file handling, logging, GPT judge)
│   ├── watermarks/       # Watermark implementations
│   ├── evaluator.py      # Attack evaluation code
│   └── server.py         # Watermarked model server
├── configs/              # YAML configuration files
└── data/                # Static datasets
```

## 🔬 Methodology

### Three-Step Attack Pipeline
1. **Querying**: Attacker queries watermarked server to collect samples
2. **Learning**: Attacker learns watermarking patterns from responses  
3. **Generation**: Attacker attempts spoofing using learned patterns

### Our Defense: Multi-Key Detection

We implement three detection algorithms:

- **Algorithm 1** (Baseline): Highest z-score detection
- **Algorithm 2** (Ours): Exactly-one-key detection with Sidak correction
- **Algorithm 3** (Advanced): Secondary threshold with theoretical bounds
- **Algorithm 4** (Advanced): Joint probability detection

#### Statistical Corrections
- **Sidak Correction**: Properly handles multiple comparisons problem
- **Calibrated Thresholds**: Maintains desired family-wise error rates
- **Ethics Filtering**: Additional layer for harmful content detection

## 📈 Running Experiments

### Standard Multi-Key Watermarking
```bash
# 4-key watermarking with Mistral-7B
python3 main.py configs/spoofing/selfhash/mistral_4keys.yaml

# Single key configurations
python3 main.py configs/spoofing/selfhash/mistral_1key.yaml
```

### Adaptive Attacks
```bash
# Key clustering attack simulation
python3 main_adaptive.py configs/spoofing/adaptive/adaptive.yaml
```

### Multi-Config Watermarking
Set seeding scheme to: `[lefthash;gptwm;selfhash;hard-additive_prf-1-False-15485863]`
```bash
python3 main.py configs/spoofing/multi/mix.yaml
```

### Analysis
Use `main_result.ipynb` for comprehensive result analysis across different FPR settings.

## 📁 Data Setup

Download pre-computed server outputs:
- [Our processed data](https://drive.google.com/file/d/1Le0Fwpr0sbWee1gLUeAYlOLalbIAK9Ir/view)
- [Original author's data](https://drive.google.com/file/d/1UrPUAJ-ZyHiMdL3uL9WUG0h8e2hPQN8v/view)

Extract so that `out_mistral/`, `out_llama/`, `out_llama13b/` are in project root.

For adaptive experiments, copy subset of `base/` from `out_mistral/` to `out_adaptive_attacker/{num_samples}/ours/`.

## 🔍 Advanced Features

### Context Suppression
*Coming soon*: Additional defense mechanism to reduce spoofing success through context-aware suppression.

### Joint Probability Detection
Alternative detection method that's less sensitive than secondary threshold approaches.

## 📊 Reproducing Paper Results

1. Run experiments with different key counts (1, 2, 3, 4)
2. Use `main_result.ipynb` for analysis  
3. Compare Algorithm 1 (baseline) vs Algorithm 2 (ours)
4. Evaluate at FPR@1e-2, FPR@1e-3, FPR@3e-5

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## 📧 Contact

**Toluwani Aremu**  
📧 toluwani.aremu@mbzuai.ac.ae

## 📜 Citation

### Multi-Key Watermarking
```bibtex
@article{aremu2024multikey,
  title={Mitigating Watermark Stealing Attacks in Language Models via Multi-Key Watermarking},
  author={Aremu, Toluwani and others},
  journal={arXiv preprint},
  year={2024}
}
```

### Original Watermark Stealing Work
```bibtex
@inproceedings{jovanovic2024watermarkstealing,
  author = {Jovanović, Nikola and Staab, Robin and Vechev, Martin},
  title = {Watermark Stealing in Large Language Models},
  journal = {ICML},
  year = {2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This work advances the state-of-the-art in watermark security for large language models, providing practical defenses against sophisticated stealing attacks.*