SUFFIXES="-obj-cnt -obj-ex -rel"
# SUFFIXES="-rel"
# SUFFIXES="-obj-cnt"
OUTPUT="data/distinct2"
for SUFFIX in $SUFFIXES
do

  QUESTIONS="questions"${SUFFIX}
  ANSWERS="answers"${SUFFIX}

  python scripts/measure_prediction_distance.py \
    --encoder_dir ${OUTPUT} \
    --images_min 0 \
    --images_max 100 \
    --vocab_json data/archive/questions/val_vocab_0_3000.json \
    --questions_json /mnt/c/git/clevr-dataset-gen/clevr-distinct/${QUESTIONS}.json \
    --answers_name ${ANSWERS}
done

# --images_max depends on the SUFFIXES, and what questions have been answere in 2_sol_from_feat
  