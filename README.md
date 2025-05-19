# Mitigating Watermark Stealing Attacks in Language Models via Multi-Key Watermarking

This repository is built upon the codebase accompanying the 'Watermark Stealing' paper: **[watermark-stealing.org](https://watermark-stealing.org)**. We thank the authors for their open-source contributions. This repository is to reproduce our results on detecting the attacks proposed by the original authors.

## Basic Setup

To set up the project, clone the repository and execute the steps from `setup.sh`. Running all steps will install Conda, create the `ws` environment, install the dependencies listed in `env.yaml`, install Flash attention, etc. On top of this, make sure to set the `OAI_API_KEY` environment variable to your OpenAI API key (to use GPT-as-a-judge evaluation).

## Repository Structure 

The project structure is as follows.
- `main.py` is the main entry point for the code.
- `main_adaptive.py` is the main entry point for the key clustering attack we proposed.
- `src/` contains the rest of the code, namely:
    - `src/attackers` contains all our attacker code for all 3 steps of the watermark stealing attack (see below under "Running the Code").
    - `src/config` contains definitions of our Pydantic configuration files. Refer to `ws_config.py` for detailed explanations of each field.
    - `src/models` contains model classes for our server, attacker, judge, and PSP models. 
    - `src/utils` contains utility functions for file handling, logging, and the use of GPT as a judge.
    - `src/watermarks` contains watermark implementations to be used on the server. 
    - `evaluator.py` implements all evaluation code for the attacker; we are primarily interested in the `targeted` evaluation mode. 
    - `gradio.py` contains the (experimental) Gradio interface used for debugging; this is not used in our experiments.
    - `server.py` contains the code for the server, i.e., the watermarked model.
- `configs/` contains YAML configuration files (corresponding to `src/config/ws_config.py`) for our main experiments reported in the paper. 
- `data/` holds static data files for some datasets used in the experiments.

## Running the Code

Our code can be run by providing a path to a YAML configuration file. For example:

```
python3 main.py configs/spoofing/selfhash/mistral_4keys.yaml
```

This example will run watermarking stealing with `Mistral-7B` as the watermarked server model using the `KGW2-SelfHash` scheme (with 4 different keys chosen at random), and `Mistral-7B` as the attacker model, evaluated on a _spoofing_ attack. If `use_neptune` is set to true the experiment will be logged in neptune.ai; to enable this, set the `NEPTUNE_API_TOKEN` environment variable and replace `ORG/PROJ` in `src/config/ws_config.py` with your project ID to set it as default, or add it to the config file for each of your runs.

This executes the following three key steps, also visible in each config file:

1) `querying`: The attacker queries the watermarked server with a set of prompts and saves the resulting responses as `json` files. This step can be skipped by downloading all watermarked server outputs used in our experimental evaluation from [this link](https://drive.google.com/file/d/1Le0Fwpr0sbWee1gLUeAYlOLalbIAK9Ir/view?usp=sharing) or [original author's link](https://drive.google.com/file/d/1UrPUAJ-ZyHiMdL3uL9WUG0h8e2hPQN8v/view?usp=sharing), and setting `skip: true` in the relevant section of the config file (done by default). Extract the archive such that `out_mistral`, `out_llama` and `out_llama13b` are in the root of the project.
2) `learning`: The attacker loads the responses and uses our algorithm to learn an internal model of the watermarking rules.
3) `generation`: The attacker mounts a _spoofing_ attack using the logit processors defined in `src/attackers/processors.py`. The `evaluator` section of the config file defines the relevant evaluation setting.

To obtain the results reported in the paper, use the `main_result.ipynb` notebook which will give you the results for different FPR settings (we reported 1e-2). 
> Algorithm 1 denotes the baseline detector's results
> Algorithm 2 denotes our method's results
> Algorithm 3 is a different version we didn't report yet. Algorithm 3 basically uses a secondary threshold based on a theoretic estimation regularized by $\tau$, well unless K is large enough for which $\tau$ should be set to 0.
> We also propose a joint probability detection algorithm which won't be as harsh and sensitive as Algorithm 3 above.
> You can also run the multi-config watermarking by setting seeding scheme to [lefthash;gptwm;selfhash;hard-additive_prf-1-False-15485863']. This can be changed based on the available algorithms. Also, one can combine multiple configs and multiple keys to improve the defense and detection.
> To run the adaptive experiment, ensure that you copy a subset of base from the `out_mistral` directory to the directory of the adaptive files i.e., `out_adaptive_attacker/5000/ours`. We are simulating an attacker with unrestricted access.
> Note, we are also working on a method, "Context Suppression" (which is basically the title of the repository), as a way to reduce spoofing success. We will be using this same repository for this defense too.

## Contact

Toluwani Aremu, toluwani.aremu@mbzuai.ac.ae<br>

## Citation

If you use our code please cite the following:
> Multi-Key Detection:
```
arXiv version coming soon.
```

> Watermark Stealing:
```
@inproceedings{jovanovic2024watermarkstealing,
    author = {Jovanović, Nikola and Staab, Robin and Vechev, Martin},
    title = {Watermark Stealing in Large Language Models},
    journal = {{ICML}},
    year = {2024}
}
```