python scripts/extract_features.py \
  --input_image_dir /mnt/d/Documents/Education/Datasets/CLEVR_v1.0/CLEVR_v1.0/images/val \
  --use_gpu 0 \
  --get_every_layer 1 \
  --model resnet101 \
  --model_stage 3 \
  --batch_size 1 \
  --min_index 0 \
  --max_index 3000 \
  --output_h5_dir data \
  --output_h5_prefix val_features
  