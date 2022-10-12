## how-to train KNN

Read more about training KNN using MFCC features, https://github.com/facebookresearch/fairseq/tree/v0.12.0/examples/hubert/simple_kmeans

```
cd examples/hubert/simple_kmeans/

python3 dump_mfcc_feature.py /home/husein/ssd1/speech-bahasa/wav2vec2-malay train 1 0 /home/husein/ssd1/mfcc
python3 learn_kmeans.py /home/husein/ssd1/mfcc train 1 /home/husein/ssd1/hubert-knn/train.km 100 --percent -1 --max_iter 1 --max_no_improvement 5

python3 dump_mfcc_feature.py /home/husein/ssd1/speech-bahasa/wav2vec2-malay valid 1 0 /home/husein/ssd1/mfcc
python3 learn_kmeans.py /home/husein/ssd1/mfcc valid 1 /home/husein/ssd1/hubert-knn/valid.km 100 --percent -1 --max_iter 10 --max_no_improvement 5
```