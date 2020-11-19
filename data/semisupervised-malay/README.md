## how-to

1. Download Malay youtube videos, [download-videos.ipynb](download-videos.ipynb).

2. Run semisupervised using Google Speech, [semisupervised-googlespeech.py](semisupervised-googlespeech.py).

16k sample rate, atleast 90% voice activity, 93 hours.

Download at https://f000.backblazeb2.com/file/malay-dataset/speech/semisupervised-malay.tar.gz

```
All the videos, songs, images, and graphics used in the video belong to their respective owners and I does not claim any right over them.

Copyright Disclaimer under section 107 of the Copyright Act of 1976, allowance is made for "fair use" for purposes such as criticism, comment, news reporting, teaching, scholarship, education and research. Fair use is a use permitted by copyright statute that might otherwise be infringing.
```

3. Run [transcribe.ipynb](transcribe.ipynb) to correct output from googlespeech.

## Download

1. [label-semisupervised-malay.tar.gz](label-semisupervised-malay.tar.gz)

  - 16000 sample rate.
  - random length between 2 - 20 seconds, windowed using google VAD.
  - 700 over 57895 done, approximate 1.14 hours.