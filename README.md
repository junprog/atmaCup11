# atmaCup#11 16th 解法

## step.1 Simple Siamese NetでResNet34, EfficientNet_b0を自己教師あり学習

```bash
$ python tarin_simsiam.py --data-dir [dataset dir path] --arch resnet34
$ python tarin_simsiam.py --data-dir [dataset dir path] --arch efficientnet_b0
```

## step.2 事前学習済モデルで初期化したResNet34, EfficientNet_b0 補助タスク付きモデルを学習

```bash
$ python train_multitask.py --data-dir [dataset dir path] --arch resnet34 --init-weight-path [step.1で保存した学習済モデルpath]
$ python train_multitask.py --data-dir [dataset dir path] --arch efficientnet_b0 --init-weight-path [step.1で保存した学習済モデルpath]
```

## step.3 ResNet34, EfficientNet_b0 補助タスク付きモデルのフュージョンモデルを学習

```bash
$ python train__multitask.py --data-dir [dataset dir path] --arch fusion ## engine/multi_task_trainer.py 150, 151行目にstep.2で学習したモデルパスを指定
```

## 推論

```bash
$ python test_multitask.py --data-dir [dataset dir path] --arch fusion --res-dir [step.3で保存した学習済モデル]
```