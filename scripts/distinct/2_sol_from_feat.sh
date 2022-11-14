SUFFIXES="-obj-cnt -obj-ex -rel"
# SUFFIXES="-rel"
# SUFFIXES="-obj-ex -obj-cnt"
APPROACHES="resnet101 GatAllClevrRand1 GatAllClevrRandWRep1 GatAllClevrReal GatDisClevrRand1 GatDisClevrRandWRep1 GatDisClevrReal"
# APPROACHES="GatAllClevrRand1 GatAllClevrRandWRep1 GatAllClevrReal GatDisClevrRand1 GatDisClevrRandWRep1 GatDisClevrReal"
# APPROACHES="resnet101 GatDisClevrReal"
APPROACHES="resnet101"

for SUFFIX in $SUFFIXES
do
  QUESTIONS="questions"${SUFFIX}
  ANSWERS="answers"${SUFFIX}
  OUTPUT="data/distinct2"
  for APPROACH in $APPROACHES
  do
    python scripts/answer_qs_from_features.py \
      --models_dir models/CLEVR \
      --use_gpu 0 \
      --questions_interval 1 \
      --num_images 1000 \
      --questions_file /mnt/c/git/clevr-iep/${OUTPUT}/${QUESTIONS}.h5 \
      --image_features /mnt/c/git/clevr-iep/data/distinct-feats/k_${APPROACH}_0_1000.h5 \
      --output_dir ${OUTPUT}/${ANSWERS}/${APPROACH}
  done
done
