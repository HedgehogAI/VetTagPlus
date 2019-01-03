# VetTag

## Introduction

This is the official cleaned repo we used to train and evaluate for VetTag paper. 

Please feel free to contact `yuhui-zh15@mails.tsinghua.edu.cn` if you have any problem using these scripts. 

## Usage

- Unsupervised Learning

Please create a json file in `hypes/` with the following format. `data_dir` represents for dataset directory, `encoder_path` represents for the dictionary which maps word to index, `sage` represents for dataset name, `label_size` represents for the label size.

```json
sage.json
{
  "data_dir": "/home/yuhuiz/Transformer/data/sage/",
  "encoder_path": "/home/yuhuiz/Transformer/data/sage/encoder_bpe_50000.json",
  "sage": "sage_ascii_oneline_bpe",
  "label_size": 0
}
```

Please save data in `/home/yuhuiz/Transformer/data/sage/sage_ascii_oneline_bpe_train.tsv`, `/home/yuhuiz/Transformer/data/sage/sage_ascii_oneline_bpe_valid.tsv` and `/home/yuhuiz/Transformer/data/sage/sage_ascii_oneline_bpe_test.tsv` for training, validation and test. The file should only contain one line for text.

Please save encoder in `/home/yuhuiz/Transformer/data/sage/encoder_bpe_50000.json`. It is a json file, like `{'hello': 0, 'world': 1}`.

Then use the following command to train and save the model in `exp/sage`

`python trainer.py --outputdir exp/sage  --train_emb --corpus sage --hypes hypes/sage.json --batch_size 5 --bptt_size 600 --model_type transformer`


- Supervised Learning

Please create a json file in `hypes/` with the following format. `data_dir` represents for dataset directory, `encoder_path` represents for the dictionary which maps word to index, `csu` represents for dataset name, `label_size` represents for the label size.

```json
csu.json
{
  "data_dir": "/home/yuhuiz/Transformer/data/csu/",
  "encoder_path": "/home/yuhuiz/Transformer/data/sage/encoder_bpe_50000.json",
  "csu": "csu_bpe_finegrained",
  "label_size": 4577
}
```

Please save data in `/home/yuhuiz/Transformer/data/csu/csu_bpe_finegrained_train.tsv`, `/home/yuhuiz/Transformer/data/csu/csu_bpe_finegrained_valid.tsv` and `/home/yuhuiz/Transformer/data/csu/csu_bpe_finegrained_test.tsv` for training, validation and test. The format should be `text\tlabel label label\n` for each line.

Please save encoder in `/home/yuhuiz/Transformer/data/sage/encoder_bpe_50000.json`. It is a json file, like `{'hello': 0, 'world': 1}`.

Then use the following command to train and save the model in `exp/csu/`

`python trainer.py --outputdir exp/csu/ --corpus csu --hypes hypes/csu_finegrained.json --batch_size 5 --model_type lstm/transformer --cut_down_len 600 --train_emb --hierachical`



