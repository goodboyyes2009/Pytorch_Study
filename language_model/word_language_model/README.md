# Train

## 使用CUDA在Wikitext-2数据上训练LSTM
```shell
python main.py --cuda --epochs 6
```

## 使用CUDA在Wikitext-2数据上训练Transformer
```shell
python main.py --cuda --epochs 2 --model Transformer --lr 5
```