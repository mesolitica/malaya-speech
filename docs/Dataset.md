# Speech Dataset

## How we gather dataset?

1. For semisupervised transcript, we use Google Speech to Text, after that verified / corrected by human.
2. We recorded using our own microphones.

## License

Malay-Speech dataset is available to download for research purposes under a Creative Commons Attribution 4.0 International License.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Dataset

### [Ambient](https://github.com/huseinzol05/malaya-speech/tree/master/data/ambient)

Simple ambients gathered from Youtube.

### [Audiobook](https://github.com/huseinzol05/malaya-speech/tree/master/data/audiobook)

Gather Youtube urls for indonesian, english and low quality english audiobooks only.

### [IIUM](https://github.com/huseinzol05/malaya-speech/tree/master/data/iium)

Read random sentences from IIUM Confession.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/) and [Shafiqah Idayu](https://www.facebook.com/shafiqah.ayu).
- Heavily speaking in Selangor dialect.
- Recorded using low-end tech microphone.
- 44100 sample rate, split by end of sentences.
- approximate 2.4 hours.
- Still on going recording.

### [IIUM-Clear](https://github.com/huseinzol05/malaya-speech/tree/master/data/iium-clear)

Read random sentences from IIUM Confession, cleaner version.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/).
- Heavily speaking in Selangor dialect.
- Recorded using mid-end tech microphone.
- 44100 sample rate, random 7 - 11 words window.
- approximate 0.1 hours.

### [IMDA](https://github.com/huseinzol05/malaya-speech/tree/master/data/imda)

Mirror link for IMDA dataset, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus, only downloaded PART 3 and SST dataset.

- 16000 sample rate.
- supervised approximate 2024 hours.

### [language](https://github.com/huseinzol05/malaya-speech/tree/master/data/language)

Gather youtube urls for hyperlocal language detection from speech {malay, indonesian, manglish, english, mandarin}.

Check hyperlocal language detection models at https://malaya-speech.readthedocs.io/en/latest/load-language-detection.html

### [news](https://github.com/huseinzol05/malaya-speech/tree/master/data/news)

Read random sentences from bahasa news.

- voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/).
- Heavily speaking in Selangor dialect.
- Recorded using mid-end tech microphone, suitable for text to speech.
- 44100 sample rate, random 7 - 11 words window.
- approximate 3.01 hours.
- Still on going recording.

### [noise](https://github.com/huseinzol05/malaya-speech/tree/master/data/noise)

Simple noises gathered from Youtube.

### [Sebut perkataan](https://github.com/huseinzol05/malaya-speech/tree/master/data/sebut-perkataan)

Read random words from malay dictionary started with 'tolong sebut <word>'.

- `sebut-perkataan-man` voice by [Husein Zolkepli](https://www.linkedin.com/in/husein-zolkepli/)
- `tolong-sebut` voice by [Khalil Nooh](https://www.linkedin.com/in/khalilnooh/)
- `sebut-perkataan-woman` voice by [Mas Aisyah Ahmad](https://www.linkedin.com/in/mas-aisyah-ahmad-b46508a9/)
- Recorded using low-end tech microphones.

### [Semisupervised audiobook](https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-audiobook)

Semisupervised malay audiobooks from Nusantara Audiobook using Google Speech to Text.

- 44100 sample rate, super clean.
- semisupervised approximate 45.29 hours.
- windowed using Malaya-Speech VAD, each atleast 5 negative voice activities.

### [Semisupervised malay](https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-malay)

Semisupervised malay youtube videos using Google Speech to Text, after that corrected by human.

- 16000 sample rate.
- semisupervised approximate 1804 hours.
- random length between 2 - 20 seconds, windowed using google VAD.
- supervised 768 samples, approximate 1.3 hours.

### [Semisupervised manglish](https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-manglish)

Semisupervised manglish youtube videos using Google Speech to Text.

- 16000 sample rate.
- semisupervised approximate 107 hours.
- random length between 2 - 20 seconds, windowed using google VAD.

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

### [youtube](https://github.com/huseinzol05/malaya-speech/tree/master/data/youtube)

Gathered Jeorogan, malay, malaysian, the thirsty sisters, richroll podcasts.

## Contribution

Contact us at husein.zol05@gmail.com or husein@mesolitica.com if want to contribute to speech bahasa dataset.