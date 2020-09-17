import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="namuwikitext_20200302.train.processed",
    model_prefix="spm-8000-tokenizer",
    vocab_size=8000,
    shuffle_input_sentence=True,
    input_sentence_size=1 * 1000 * 1000,
)
