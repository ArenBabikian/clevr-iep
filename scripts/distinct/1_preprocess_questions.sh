ALLQUESTIONS="questions-obj-cnt questions-obj-ex questions-rel-ex questions-rel-cnt"

# OPTIONAL
# ALLQUESTIONS="questions-obj-cnt questions-obj-ex"
ALLQUESTIONS="questions-rel-ex questions-rel-cnt"
# ALLQUESTIONS="questions-rel-ex"
# ALLQUESTIONS="questions-obj-ex"
OUTPUT="data/distinct2"

for QUESTIONS in $ALLQUESTIONS
do
  python scripts/preprocess_questions.py \
    --input_questions_json /mnt/c/git/clevr-dataset-gen/clevr-distinct/${QUESTIONS}.json \
    --input_vocab_json /mnt/c/git/clevr-iep/data/distinct2/default-vocab.json \
    --expand_vocab 1 \
    --output_h5_file ${OUTPUT}/${QUESTIONS}.h5 \
    --output_vocab_json ${OUTPUT}/${QUESTIONS}-vocab.json \
    --encode_unk 1 \
    --mode prefix
done
