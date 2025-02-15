# GNN-TDA

## Introduction

<center>
  <img src="asset/overview_framework.png" alt="drawing" width="700" height="800px">
</center>

## Data

Download raw data from [PeMS3_Evolve](https://drive.google.com/file/d/1XZ3_a2SQNl_Bk-y5VQnZkvqgo6ww6PMT/view?usp=sharing) and [PeMS4_Evolve](https://drive.google.com/file/d/1O3aDKlYcW1mepG4H4NgUU4WHcM6T1-_M/view?usp=sharing), unzip the file and put it in the `data` folder. We also provide another version of PeMS4 with different evolve frequency (evolve daily and weekly) [PeMS4_Evolve_Diff](https://drive.google.com/file/d/1qz6q4-3otxvTTVqev6pxdYjCKL69Ny8z/view?usp=sharing)

## Usage
Create conda environment by
```
conda env create -f trafficStream.yaml
```

To run CAST (retrain for each year)
```
python main.py --conf conf/cast_pems04.json --gpuid 1
```

To run TEAM (incremental training)
```
python main.py --conf conf/team_pems04.json --gpuid 1
```
