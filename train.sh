#!/bin/bash

TRAIN_FILE=data/sentences_with_age-ordered.txt
TOKENIZER=childs_tokenizer/tokenizer.json
EPOCHS=10
BATCH_SIZE=64
MAX_SEQ_LEN=128

# NoLimit (baseline, no ALiBi)
python src/NoLimit.py \
  --train_file ${TRAIN_FILE} \
  --tokenizer_file ${TOKENIZER} \
  --output_dir output/NoLimit \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --max_seq_length ${MAX_SEQ_LEN}

# StaticLimit (static ALiBi)
python src/StaticLimit.py \
  --train_file ${TRAIN_FILE} \
  --tokenizer_file ${TOKENIZER} \
  --output_dir output/StaticLimit \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --max_seq_length ${MAX_SEQ_LEN}

# DynamicLimit-Linear (dynamic ALiBi with linear schedule)
python src/DynamicLimit-Linear.py \
  --train_file ${TRAIN_FILE} \
  --tokenizer_file ${TOKENIZER} \
  --output_dir output/DynamicLimit-Linear \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --max_seq_length ${MAX_SEQ_LEN}

# DynamicLimit-Exp (dynamic ALiBi with exponential decay)
python src/DynamicLimit-Exp.py \
  --train_file ${TRAIN_FILE} \
  --tokenizer_file ${TOKENIZER} \
  --output_dir output/DynamicLimit-Exp \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --max_seq_length ${MAX_SEQ_LEN}
