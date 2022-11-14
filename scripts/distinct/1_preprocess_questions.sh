ALLQUESTIONS="questions-obj-cnt questions-obj-ex questions-rel"
ALLQUESTIONS="questions-rel"
OUTPUT="data/distinct2"

for QUESTIONS in $ALLQUESTIONS
do
  python scripts/preprocess_questions.py \
    --input_questions_json /mnt/c/git/clevr-dataset-gen/clevr-distinct/${QUESTIONS}.json \
    --output_h5_file ${OUTPUT}/${QUESTIONS}.h5 \
    --output_vocab_json ${OUTPUT}/${QUESTIONS}-vocab.json \
    --encode_unk 1 \
    --mode prefix
done
