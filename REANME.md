# MKM-SR
Code for paper 'Incorporating User Micro-behaviors and Item Knowledge into Multi-task Learning for Session-based Recommendation'

Using PyTorch and PyTorch-Geometric.

Origin [code](https://github.com/ciecus/MKM-SR), origin [paper](https://arxiv.org/pdf/2006.06922.pdf).
## File Structure
```
MKM-SR
    --dataset
        --KKBOX
            dataset_info.pkl
            kg2id
            test_pro.pkl
            test.pkl
            train_pro.pkl
            train.pkl
        --KKBOX_raw
            train.csv
            test.csv
            songs.csv
        prepare.ipynb
    --src
        dataset.py
        main.py
        model.py
        parse.py
        utils.py
    README.md
```
## Data preparation


1. Download KKBOX dataset from  https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data
2. put three files to `\dataset\KKBOX_raw`
3. run `prepare.ipynb`


## Training and testing
```
python src/main.py
```

