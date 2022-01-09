# AppleLM

Which Apple Keeps Which Doctor Away? Colorful Word Representations with Visual Oracles (TASLP 2022)

### Setup

This implementation is based on [Transformers](https://github.com/huggingface/transformers). 

### Preparation

1. Download GLUE datasets

   The datasets can be downloaded automatically. Please refer to https://github.com/nyu-mll/GLUE-baselines

   ```
   git clone https://github.com/nyu-mll/GLUE-baselines.git
   python download_glue_data.py --data_dir glue_data --tasks all
   ```

   It is recommended to put the folder **glue_data** to data/. The architecture looks like:

   ```
   AppleLM
   └───data
   │   └───glue_data
   │       │   CoLA/
   │       │   MRPC/
   │       │   ...
   ```


2. Visual Features

   Pre-extracted visual features can be [downloaded from Google Drive](https://drive.google.com/drive/folders/1I2ufg3rTva3qeBkEc-xDpkESsGkYXgCf?usp=sharing) borrowed from the repo [Multi30K](https://github.com/multi30k/dataset).

   The features are used in image embedding layer for indexing. Extract **train-resnet50-avgpool.npy** and put it in the data/ folder.

### Training & Evaluate

```
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
```

### Reference

Please kindly cite this paper in your publications if it helps your research:

```
@ARTICLE{zhang2022which,
  author={Zhang, Zhuosheng and Yu, Haojie and Zhao, Hai and Utiyama, Masao},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Which Apple Keeps Which Doctor Away? Colorful Word Representations With Visual Oracles}, 
  year={2022},
  volume={30},
  number={},
  pages={49-59},
  doi={10.1109/TASLP.2021.3130972}
}
```
