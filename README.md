# Pytorch_Study

## 使用onehot embedding编码迭代100个epoch,在新闻标签训练集上的训练情况为:
```shell
cd scripts
./train_text_cnn_zh.sh
```
epoch 99 | train loss 1.45 | evaluate loss 1.48
Accuracy: 0.483607
Accuracy: 0.459016

torch.nn.embedding 加参数: padding_idx=0, max_norm=5.0
Accuracy: 0.295082

tencent embedding
Accuracy: 0.385246

tencent embedding freeze
Accuracy: 0.344262