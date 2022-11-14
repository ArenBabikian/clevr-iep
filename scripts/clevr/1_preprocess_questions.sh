python scripts/preprocess_questions.py \
  --input_questions_json /mnt/d/Documents/Education/Datasets/CLEVR_v1.0/CLEVR_v1.0/questions/CLEVR_val_questions.json \
  --output_h5_file data/questions/val_questions_0_3000.h5 \
  --output_vocab_json data/questions/val_vocab_0_3000.json \
  --min_index 0 \
  --max_index 3000 \
  --encode_unk 1
