import tensorflow as tf
from transformers import TFGPT2Model
import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
import pandas as pd



train_df = pd.read_csv('models/data_in/train_filter.csv')

TOKENIZER_PATH = 'models/gpt_ckpt/gpt2_kor_tokenizer.spiece'

tokenizer = SentencepieceTokenizer(TOKENIZER_PATH)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(TOKENIZER_PATH,
                                               mask_token=None,
                                               sep_token='<unused0>',
                                               cls_token=None,
                                               unknown_token='<unk>',
                                               padding_token='<pad>',
                                               bos_token='<s>',
                                               eos_token='</s>')


class TFGPT2Classifier(tf.keras.Model):
    def __init__(self, dir_path, num_class):
        super(TFGPT2Classifier, self).__init__()

        self.gpt2 = TFGPT2Model.from_pretrained(dir_path)
        self.num_class = num_class

        self.dropout = tf.keras.layers.Dropout(self.gpt2.config.summary_first_dropout)
        self.classifier = tf.keras.layers.Dense(self.num_class,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                    stddev=self.gpt2.config.initializer_range),
                                                name="classifier")

    def call(self, inputs):
        outputs = self.gpt2(inputs)
        pooled_output = outputs[0][:, -1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

optimizer = tf.keras.optimizers.Adam(learning_rate=6.25e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

load_model = TFGPT2Classifier(dir_path='models/gpt_ckpt', num_class=6)
load_model.load_weights('models/saved_model/02-0.96427.ckpt')
load_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
