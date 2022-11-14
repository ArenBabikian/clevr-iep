### LEARNED
# python scripts/answer_qs_from_features.py \
#   --models_dir models/CLEVR \
#   --use_gpu 0 \
#   --questions_interval 10 \
#   --questions_file /mnt/c/git/clevr-iep/data/questions/val_questions_0_3000.h5 \
#   --image_features /mnt/c/git/Clevr-Relational/_data/features_features_0_3000.h5 \
#   --output_dir data/answers/learned

### RANDOMWITHOUTREPLACEMENT
# python scripts/answer_qs_from_features.py \
#   --models_dir models/CLEVR \
#   --use_gpu 0 \
#   --questions_interval 10 \
#   --questions_file /mnt/c/git/clevr-iep/data/questions/val_questions_0_3000.h5 \
#   --image_features /mnt/c/git/Clevr-Relational/_data/random1_features_0_3000.h5 \
#   --output_dir data/answers/random1

### RANDOMWITHREPLACEMENT
# python scripts/answer_qs_from_features.py \
#   --models_dir models/CLEVR \
#   --use_gpu 0 \
#   --questions_interval 10 \
#   --questions_file /mnt/c/git/clevr-iep/data/questions/val_questions_0_3000.h5 \
#   --image_features /mnt/c/git/Clevr-Relational/_data/randomwithreplacement1_features_0_3000.h5 \
#   --output_dir data/answers/randomwithreplacement1

### RESNET101
# python scripts/answer_qs_from_features.py \
#   --models_dir models/CLEVR \
#   --use_gpu 0 \
#   --questions_interval 1 \
#   --num_images 300 \
#   --questions_file /mnt/c/git/clevr-iep/data/questions/val_questions_0_3000.h5 \
#   --image_features /mnt/c/git/Clevr-Relational/_data/iepvqa/val_features_0_3000.h5 \
#   --output_dir data/answers/resnet101

### LEARNED FROM ANS CE
# python scripts/answer_qs_from_features.py \
#   --models_dir models/CLEVR \
#   --use_gpu 0 \
#   --questions_interval 1 \
#   --num_images 300 \
#   --questions_file /mnt/c/git/clevr-iep/data/questions/val_questions_0_3000.h5 \
#   --image_features /mnt/c/git/Clevr-Relational/_data/features_ansce_features_0_300.h5 \
#   --output_dir data/answers/learned_from_ans_9k_ce

### LEARNED FROM ANS MSE
python scripts/answer_qs_from_features.py \
  --models_dir models/CLEVR \
  --use_gpu 0 \
  --questions_interval 1 \
  --num_images 300 \
  --questions_file /mnt/c/git/clevr-iep/data/questions/val_questions_0_3000.h5 \
  --image_features /mnt/c/git/Clevr-Relational/_data/features_ansmse_features_0_300.h5 \
  --output_dir data/answers/learned_from_ans_9k_mse

  
### RANDOMMORE (OBSOLETE)
# python scripts/answer_qs_from_features.py \
#   --models_dir models/CLEVR \
#   --use_gpu 0 \
#   --questions_interval 10 \
#   --questions_file /mnt/c/git/clevr-iep/data/questions/val_questions_0_3000.h5 \
#   --image_features /mnt/c/git/Clevr-Relational/_data/randommore1_features_0_3000.h5 \
#   --output_dir data/answers/randommore1

  