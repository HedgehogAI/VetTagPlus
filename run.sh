# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm/model-3.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/lstm.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm_auxiliary/model-10.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/lstm_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm_pretrain/model-8.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/lstm_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm_auxiliary_pretrain/model-10.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/lstm_auxiliary_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer/model-8.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/transformer.npy
CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer_auxiliary/model-9.pickle
mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/transformer_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer_pretrain/model-8.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/transformer_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer_auxiliary_pretrain/model-8.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/bpe/transformer_auxiliary_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm/model-3.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/lstm.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm_auxiliary/model-10.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/lstm_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm_pretrain/model-8.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/lstm_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/lstm_auxiliary_pretrain/model-10.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/lstm_auxiliary_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer/model-8.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/transformer.npy
CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer_auxiliary/model-9.pickle
mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/transformer_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer_pretrain/model-8.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/transformer_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/bpe/transformer_auxiliary_pretrain/model-8.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/bpe/transformer_auxiliary_pretrain.npy

# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm/model-4.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/lstm.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm_auxiliary/model-10.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/lstm_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm_pretrain/model-5.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/lstm_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm_auxiliary_pretrain/model-6.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/lstm_auxiliary_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer/model-10.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/transformer.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer_auxiliary/model-10.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/transformer_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer_pretrain/model-9.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/transformer_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/csu --corpus csu --hypes hypes/csu_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer_auxiliary_pretrain/model-10.pickle
# mv exp/analysis/csu/preds-1.npy exp/analysis/csu/transformer_auxiliary_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm/model-4.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/lstm.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm_auxiliary/model-10.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/lstm_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm_pretrain/model-5.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/lstm_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/lstm_auxiliary_pretrain/model-6.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/lstm_auxiliary_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer/model-10.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/transformer.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer_auxiliary/model-10.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/transformer_auxiliary.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer_pretrain/model-9.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/transformer_pretrain.npy
# CUDA_VISIBLE_DEVICES=3 python trainer.py --outputdir exp/analysis/pp --corpus pp --hypes hypes/pp_word2vec.json --batch_size 10 --proj_head 4 --cut_down_len 600 --inputdir exp/nonbpe/transformer_auxiliary_pretrain/model-10.pickle
# mv exp/analysis/pp/preds-1.npy exp/analysis/pp/transformer_auxiliary_pretrain.npy