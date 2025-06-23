# Finetune Dia TTS

## Prepare dataset

1. The dataset undergone silent trimming and generate permutation voice conversion from [mesolitica/Malaysian-Emilia-Audio-Tokens](https://huggingface.co/datasets/mesolitica/Malaysian-Emilia-Audio-Tokens), where this is done in [../sesame-tts](../sesame-tts/).

2. Convert to DAC speech tokens,

```bash
python3 convert_dac.py
```

3. Sort the permutation voice conversion and multipack for Encoder-Decoder, [merge-dia.ipynb](merge-dia.ipynb).

DAC tokens and sorted multipacking pushed to [mesolitica/Malaysian-Emilia-Audio-Tokens](https://huggingface.co/datasets/mesolitica/Malaysian-Emilia-Audio-Tokens).

## Finetuning

1. Run script,

```bash
bash finetune_dia_multipacking_v2.sh
```

The checkpoint weight saved as prefix `model`,

```
class TTS(nn.Module):
    def __init__(self):
        super(TTS, self).__init__()

        dia_cfg = DiaConfig.load('config.json')
        ckpt_file = hf_hub_download('nari-labs/Dia-1.6B', filename="dia-v0_1.pth")
        model = DiaModel(dia_cfg)
        model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
        self.model = model
```

So make sure you trimmed first the prefix before load using Dia-TTS library.
