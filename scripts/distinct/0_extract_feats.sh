
python scripts/extract_features.py \
  --input_image_dir /mnt/c/git/clevr-dataset-gen/clevr-distinct/images \
  --use_gpu 0 \
  --get_every_layer 1 \
  --model resnet101 \
  --model_stage 3 \
  --batch_size 1 \
  --output_h5_dir data/distinct-feats/ \
  --output_h5_prefix k-resnet101
  
  # --output_h5_dir data/distinct \
