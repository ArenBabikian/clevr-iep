ENCODERS="resnet101"
DECODERS="9k 18k 700k_strong"

SUFFIXES="obj-cnt obj-ex rel-cnt rel-ex"
SUFFIXES="rel-ex rel-cnt"
# SUFFIXES="obj-ex obj-cnt"

# TEMPORARY
# SUFFIXES="rel-ex"
# SUFFIXES="obj-ex rel-ex"
# DECODERS="18k"

OUTPUT="data/distinct2"
SRCDIR="/mnt/c/git/clevr-iep"
WORKDIR=${SRCDIR}"/"${OUTPUT}

for SUFFIX in $SUFFIXES
do
  for DECODER in $DECODERS
  do
    for ENCODER in $ENCODERS
    do

      QUESTIONS="questions-"${SUFFIX}
      if [[ $SUFFIX == rel* ]]
        then
          DECODERPATH=${DECODER}"_0_100__1"
          MAXIMG=100
        else
          DECODERPATH=${DECODER}"_0_1000__1"
          MAXIMG=1000 # This can be 100 or 1000
      fi
      ANSWERS="answers-"${SUFFIX}

      python scripts/analyse_image_subsets.py \
        --encoder_dir ${OUTPUT} \
        --images_min 0 \
        --images_max ${MAXIMG} \
        --num_random_plots 500 \
        --vocab_json ${WORKDIR}/default-vocab.json \
        --questions_json /mnt/c/git/clevr-dataset-gen/clevr-distinct/${QUESTIONS}.json \
        --data_json ${SRCDIR}/data \
        --figs_path ${WORKDIR}/figsPostRel2 \
        --save_filename ${SUFFIX}-${MAXIMG}/${DECODER} \
        --encoder ${ENCODER} \
        --decoder_name ${DECODERPATH} \
        --answers_name ${ANSWERS} \
        --shape_analysis 10 \
        # --rankings_path ${WORKDIR}/rankings \
        # --features_dir ${SRCDIR}/data/distinct-feats
    done
  done
done
  