# atmaCup#11 コード

```bash
$ python tarin_simsiam.py --data-dir C:/Users/Junya/Desktop/dataset_atmaCup11/photos --batch-size 128
```

```bash
$ python train_material.py --data-dir C:/Users/Junya/Desktop/dataset_atmaCup11 --batch-size 128
```

```bash
$ python train_fusion.py --data-dir /home/junya/Documents/dataset_atmaCup11/ --batch-size 128 --init-weight-path simsiam_logs/exp02-0710-182433/300_ckpt.tar --mate-res-dir material_logs/exp02-0712-000307 --tech-res-dir technique_logs/exp02-0712-025133
```

* Materials
class: 25 -> 5 にする

* Techniques
class: 10 -> 2 にする