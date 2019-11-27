# LSTM language model

```
python convert.py --output sentences.npz --word_num 10 --word_stateseq word_stateseq.npz --word_durations word_durations.npz --using_iteration -1
```
initialize model and training
```
python train.py --init_model --model model.ckpt --train_data sentences.npz --train_iter 100 --hidden_node 128
```
continue training
```
python train.py --model model.ckpt --train_data sentences.npz
```
generate
```
python generate.py --model model.ckpt --train_data sentences.npz
```
