# 16位 解法 (atmaCup#11)

public: 0.6802, private: 0.6686

## Models
* fusion (ResNet34 + EfficinetNet_b0 -> 3 branches MLP)
    * ResNet34 (simsiam -> multitask)
    * EfficinetNet_b0 (simsiam -> multitask)

## Loss
* multi task loss = MSE + material_BCEWithLogits + technique_BCEWithLogits
    * MSE(output1, target)
    * material_BCEWithLogits(output2, materials_class)
    * technique_BCEWithLogits(output3, techniques_class)

* materils_class
    * 25種類　-> 最頻クラス上位5クラスピックアップ + 残り20クラスをotherクラス = 6クラスマルチラベル分類
    * materialsが存在しない画像はotherクラスとする

* techniques_class
    * 10種類　-> 最頻クラス上位2クラスピックアップ + 残り20クラスをotherクラス = 3クラスマルチラベル分類
    * techniquesが存在しない画像はotherクラスとする

## Optimizer
* Adam(lr = 0.01)

## Settings
* StratifiedKFold(k=5)
* Epoch: 400
* Augmentation:
    * RandomResizedCrop(256),
    * RandomHorizontalFlip(p=0.5),

* Fusionモデルにおいてはfeatures部分をfreeze
* 推論時はinput sizeを352

## Demo

### Environment
* python 3.6
    * torch
    * torchvision
    * lightly
    * numpy
    * pandas
    * sklearn
    * tqdm

### step.1 
* Simple Siamese NetでResNet34, EfficientNet_b0を自己教師あり学習

```bash
$ python tarin_simsiam.py --data-dir [dataset dir path] --arch resnet34
$ python tarin_simsiam.py --data-dir [dataset dir path] --arch efficientnet_b0
```

### step.2
* 事前学習済モデルで初期化したResNet34, EfficientNet_b0 補助タスク付きモデルを学習

```bash
$ python train_multitask.py --data-dir [dataset dir path] --arch resnet34 --init-weight-path [step.1で保存した学習済モデルpath]
$ python train_multitask.py --data-dir [dataset dir path] --arch efficientnet_b0 --init-weight-path [step.1で保存した学習済モデルpath]
```

### step.3
* ResNet34, EfficientNet_b0 補助タスク付きモデルのフュージョンモデルを学習

```bash
$ python train__multitask.py --data-dir [dataset dir path] --arch fusion ## engine/multi_task_trainer.py 150, 151行目にstep.2で学習したモデルパスを指定
```

### 推論

```bash
$ python test_multitask.py --data-dir [dataset dir path] --arch fusion --res-dir [step.3で保存した学習済モデル]
```
