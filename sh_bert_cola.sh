export GLUE_DIR=data/glue_data/
export CUDA_VISIBLE_DEVICES="0"
export TASK_NAME=CoLA
python ./examples/run_glue_visual-tfidf_att.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 1e-5 \
    --eval_all_checkpoints \
    --save_steps 500 \
    --max_steps 5336 \
    --warmup_steps 320 \
    --image_dir data/train.lc.norm.tok.en \
    --image_embedding_file data/train-resnet50-avgpool.npy \
    --num_img 3 \
    --tfidf 5 \
    --image_merge att-gate \
    --stopwords_dir data/stopwords-en.txt \
    --output_dir experiments/CoLA_bert_wwm