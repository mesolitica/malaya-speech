## how-to train KNN

Read more about training KNN using MFCC features, https://github.com/facebookresearch/fairseq/tree/v0.12.0/examples/hubert/simple_kmeans

```bash
cd examples/hubert/simple_kmeans/

python3 dump_mfcc_feature.py /home/husein/ssd1/speech-bahasa/wav2vec2-malay train 1 0 /home/husein/ssd1/mfcc
python3 learn_kmeans.py /home/husein/ssd1/mfcc train 1 /home/husein/ssd1/hubert-knn/train.km 100 --percent -1 --max_iter 1 --max_no_improvement 5
python3 dump_km_label.py /home/husein/ssd1/mfcc train /home/husein/ssd1/hubert-knn/train.km 1 0 /home/husein/ssd1/mfcc-label

python3 dump_mfcc_feature.py /home/husein/ssd1/speech-bahasa/wav2vec2-malay valid 1 0 /home/husein/ssd1/mfcc
python3 learn_kmeans.py /home/husein/ssd1/mfcc valid 1 /home/husein/ssd1/hubert-knn/valid.km 100 --percent -1 --max_iter 10 --max_no_improvement 5
python3 dump_km_label.py /home/husein/ssd1/mfcc valid /home/husein/ssd1/hubert-knn/valid.km 1 0 /home/husein/ssd1/mfcc-label

export n_clusters=100
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> /home/husein/ssd1/mfcc-label/dict.km.txt

cp /home/husein/ssd1/mfcc-label/valid_0_1.km /home/husein/ssd1/mfcc-label/valid.km
cp /home/husein/ssd1/mfcc-label/train_0_1.km /home/husein/ssd1/mfcc-label/train.km
```

```bash
fairseq-hydra-train \
--config-dir /home/husein/fairseq/examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/home/husein/ssd1/speech-bahasa/wav2vec2-malay \
task.label_dir=/home/husein/ssd1/mfcc-label \
task.labels='["km"]' model.label_rate=100
```

```bash
fairseq-hydra-train \
--config-dir /home/husein/fairseq/examples/hubert/config/pretrain \
--config-name small \
task.data=/home/husein/ssd1/speech-bahasa/wav2vec2-malay \
task.label_dir=/home/husein/ssd1/mfcc-label \
task.labels='["km"]' model.label_rate=100
```