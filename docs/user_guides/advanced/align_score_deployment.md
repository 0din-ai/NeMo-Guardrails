# AlignScore Deployment

In order to deploy an AlignScore server, follow these steps:

1. Install the `alignscore` package from the GitHub repository:

```bash
git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore
pip install .
```

2. Download the Spacy `en_core_web_sm` model:

```bash
python -m spacy download en_core_web_sm
```

3. Download the one or both of the AlignScore checkpoints:
```
curl -OL https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt
curl -OL https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt
```

4. Set the `ALIGN_SCORE_PATH` environment variable to point to the path where the checkpoints have been downloaded.

5. Set the `ALIGN_SCORE_DEVICE` environment variable to `"cpu"` to run the AlignScore model on CPU, or to the corresponding GPU device, e.g. `"cuda:0"`.
```bash
export ALIGN_SCORE_PATH=<path/to/folder_containing_ckpt>
export ALIGN_SCORE_DEVICE="cuda:0"
```

6. Start the AlignScore server.

```bash
python -m nemoguardrails.library.factchecking.align_score.server --port 5000 --models=base
```

By default, the AlignScore server listens on port `5000`. You can change the port using the `--port` option. Also, by default, the AlignScore server loads only the base model. You can load only the large model using `--models=large` or both using `--models=base --models=large`.
