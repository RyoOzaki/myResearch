# NPB-DAA offline training samples

NPB-DAAの学習プログラムサンプルを一部改変してnpzファイルから学習を行えるように変更したもの．

---

## Quick start
コンフィグファイルの展開
```
python unroll_default_config.py
```

学習と結果解析
```
python train.py --train_data sample_datas/datas.npz
python summary.py --phn_label sample_datas/phn.npz --wrd_label sample_datas/wrd.npz
```

複数試行をまとめて実行
```
sh runner.sh -t sample_datas/datas.npz -p sample_datas/phn.npz -w sample_datas/wrd.npz -l sample_result
# or
sh watchdog_runner.sh -t sample_datas/datas.npz -p sample_datas/phn.npz -w sample_datas/wrd.npz -l sample_result
```

---

npzファイルの中身は各発話の音響特徴量列をそれぞれ保存したもの．<br>
shape of each data must be (T, D): Tは時系列長，Dは次元数
