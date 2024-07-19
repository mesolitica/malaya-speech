# Speech Dataset

## [Ambient](ambient)

Simple ambients gathered from Youtube.

## [Cloud-TTS](cloud-tts)

### [Azure-TTS](cloud-tts/azure-tts)

Semisupervised Malay TTS dataset from Azure TTS cloud.

### [GCP-TTS](cloud-tts/gcp-tts)

Semisupervised Malay TTS dataset from GCP TTS cloud.

## [Common Voice](common-voice)


## [Corpus](corpus)

Corpus we use to train KenLM and Neural CausalLM.

## [Emotion](emotion)

Speech emotion dataset used by Malaya-Speech for speech emotion detection.

## [IMDA](imbda)

Mirror link for IMDA dataset, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus, only downloaded PART 3 and SST dataset.

- 16000 sample rate.
- supervised approximate 2024 hours.

## [Language](language)

Language detection dataset used by Malaya-Speech for speech language detection.

- Gather youtube urls for hyperlocal language detection from speech {malay, manglish}.
- Use Common Voice to gather {english, mandarin, others}.

## [mixheadset](mixheadset)

Script to download mixheadset dataset.

## [noise](noise)

Simple noises gathered from Youtube for augmentation purpose.

## [self-record](self-record)

### [IIUM](self-record/iium)

Read random sentences from IIUM Confession.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/) and [Shafiqah Idayu](https://www.facebook.com/shafiqah.ayu).
- Heavily speaking in Selangor dialect.
- Recorded using low-end tech microphone.
- 44100 sample rate, split by end of sentences.
- approximate 2.4 hours.
- Still on going recording.

```bibtex
@misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Speech Dataset from IIUM Confession texts,
  author = {Husein, Zolkepli},
  title = {Malay-Dataset},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/iium}}
}
```

### [IIUM-Clear](self-record/iium-clear)

Read random sentences from IIUM Confession, cleaner version.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/).
- Heavily speaking in Selangor dialect.
- Recorded using mid-end tech microphone.
- 44100 sample rate, random 7 - 11 words window.
- approximate 0.1 hours.

```bibtex
@misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Speech Dataset from IIUM Confession texts,
  author = {Husein, Zolkepli},
  title = {Malay-Dataset},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/iium}}
}
```

### [news](https://github.com/huseinzol05/malaya-speech/tree/master/data/news)

Read random sentences from bahasa news.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/).
- Heavily speaking in Selangor dialect.
- Recorded using mid-end tech microphone, suitable for text to speech.
- 44100 sample rate, random 7 - 11 words window.
- approximate 3.01 hours.
- Still on going recording.

### [Sebut perkataan](https://github.com/huseinzol05/malaya-speech/tree/master/data/sebut-perkataan)

Read random words from malay dictionary started with 'tolong sebut <word>'.

- `sebut-perkataan-man` voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/)
- `tolong-sebut` voice by [Khalil Nooh](https://www.linkedin.com/in/khalilnooh/)
- `sebut-perkataan-woman` voice by [Mas Aisyah Ahmad](https://www.linkedin.com/in/mas-aisyah-ahmad-b46508a9/)
- Recorded using low-end tech microphones.

### [wattpad](https://github.com/huseinzol05/malaya-speech/tree/master/data/wattpad)

Read random sentences from bahasa wattpad.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/).
- Heavily speaking in Selangor dialect.
- Recorded using mid-end tech microphone, suitable for text to speech.
- 44100 sample rate, random 7 - 11 words window.
- approximate 0.15 hours.
- Still on going recording.

### [Wikipedia](https://github.com/huseinzol05/malaya-speech/tree/master/data/wikipedia)

Read random sentences from Bahasa Wikipedia.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/).
- Heavily speaking in Selangor dialect.
- Recorded using low-end tech microphone.
- 44100 sample rate, 4 words window.
- approximate 3.4 hours.
- Still on going recording.

## [Semisupervised](semisupervised)

### [audiobook](semisupervised/audiobook)

Semisupervised malay audiobooks from Nusantara Audiobook using Google Speech to Text.

- 44100 sample rate, super clean.
- semisupervised approximate 45.29 hours.
- windowed using Malaya-Speech VAD, each atleast 5 negative voice activities.

### [malay](semisupervised/malay)

Semisupervised malay youtube videos using Google Speech to Text, after that corrected by human.

- 16000 sample rate.
- semisupervised approximate 1804 hours.
- random length between 2 - 20 seconds, windowed using google VAD.
- supervised 768 samples, approximate 1.3 hours.

### [manglish](semisupervised/manglish)

Semisupervised manglish youtube videos using Google Speech to Text.

- 16000 sample rate.
- semisupervised approximate 107 hours.
- random length between 2 - 20 seconds, windowed using google VAD.

### [whisper-stt](semisupervised/whisper-stt)

Semisupervised 25k youtube videos using Whisper Large v2.

## [STT](stt)

### [Mixed STT](mixed-stt)

Malay, Singlish and Mandarin STT dataset in TFRecord format. Included scripts how to load using `torch.dataset`.

## [youtube](youtube)

Semisupervised transcription and Unsupervised Speaker Diarization on 5k malay speakers youtube videos.