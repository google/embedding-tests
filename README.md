DISCLAIMER: This is not an officially supported Google product.

# Analysis in Embeddings
This repository contains implementations on word embedding and
quick-thought sentence embedding models. 
## Word Embedding
### 1. Data
A subset of wikipedia articles: [enwiki9](http://mattmahoney.net/dc/enwik9.zip).
Preprocessing script is at:
`data/wiki9.py`.

```python
from data.wiki9 import write_wiki9_articles
write_wiki9_articles()
```

In total 140,000 articles. For experiments, 70,000 were used for training word
embedding models.

### 2. Models
Word embedding models are usually lookup tables that map a word to a low
-dimension vector. Available models includes Word2Vec, FastText and Glove. 
Models were trained using public library
[gensim](https://radimrehurek.com/gensim/models/word2vec.html) and
[glove-python](https://github.com/maciejkula/glove-python).

Hyper-parameters were kept as in the original papers. To launch training for
Word2Vec for example, run: `shell python train_word_embedding.py --model=w2v`
model option can be w2v for Word2Vec, ft for FastText and glove for Glove.

A tensorflow implementation of Word2Vec is also available, and has an option for
training Word2Vec with differential privacy. For training with DP, run:

```shell
CUDA_VISIBLE_DEVICES="0" python train_word_embedding_dp.py --dpsgd \
--noise_multiplier=0.1 --l2_norm_clip=0.25 --batch_size=512
```

Explanation for the options is detailed in
[tensorflow-privacy](https://github.com/tensorflow/privacy).

Trained models are evaluated using standard evaluation
[questions](https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt).
Commandline for evaluation: `python eval_word_embedding.py --model=w2v`

## Sentence Embedding
### 1. Data
A collection of books crawled using scripts from
<https://github.com/soskek/bookcorpus>. After preprocessing, there are more than
14,000 books and 30,000,000 sentences. Preprocessing script is at:
`data/bookcorpus.py`.

```python
from data.bookcorpus import preprocess_pipeline
preprocess_pipeline()
```

### 2. Model
Sentence embedding models are usually neural networks that takes a sequence of 
words as input and output a low-dimension vector. We train 
[QuickThought](https://arxiv.org/pdf/1803.02893.pdf) locally on half of all books. 
The model is trained by predicting the sentences before and after
given a input sentence. Implementation is based on <https://github.com/lajanugen/S2V> 
with slight modification. To train QuickThought on books, run:

```shell
CUDA_VISIBLE_DEVICES="0" python train_quick_thought.py ----batch_size=500 \
--emb_dim=620 --encoder_dim=1200 --cell_type=LSTM --epochs=1
```

Evaluation of QuickThought is done by using the model as feature extractor for
downstream sentence classification tasks. Currently supports evaluation on
TREC and MSRP. Run evaluation with:

```shell
CUDA_VISIBLE_DEVICES="0" python eval_quick_thought.py --eval_data=trec
```
