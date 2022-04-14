# Attentional Constellation Nets For Few-shot Learning (Updated with PACS)

This is the code repository for the reproducibility project for group 16. 
Follow the following steps (taken from the original repository) to run the model.

Use **PACS_pickle.py** to create the desired pickle files if you are downloading PACS from the original repository (or use the google drive link presented in the next sections for downloading the datasets ready to go).

### Environment Preparation
1. Set up a new conda environment and activate it.
   ```bash
   # Create an environment with Python 3.8.
   conda create -n constells python==3.8
   conda activate constells
   ```

2. Install required packages.
   ```bash
   # Install PyTorch 1.8.0 w/ CUDA 11.1.
   conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

   # Install yaml
   conda install -c anaconda pyyaml

   # Install tensorboardx.
   conda install -c conda-forge tensorboardx tqdm
   ```

### Code and Datasets Preparation
1. Clone the repo.
   ```bash
   git clone https://github.com/dajtmullaj/ConstellationNet.git
   cd ConstellationNet
   ```

2. Download datasets
   - [Mini-ImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
   - [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))
   - [FC100](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))
   - [PACS](https://drive.google.com/drive/folders/1xCVNME8avKj4OH-J5sMDXTC6Ztb6RWzM?usp=sharing) (The name of the dataset indicates which domain has ben used for testing)

   The code assumes datasets are saved according to the following structure:
   
```
 materials
├── mini-imagenet
│   ├── miniImageNet_category_split_test.pickle
│   ├── miniImageNet_category_split_train_phase_test.pickle
│   ├── miniImageNet_category_split_train_phase_train.pickle
│   ├── miniImageNet_category_split_train_phase_val.pickle
│   ├── miniImageNet_category_split_val.pickle
├── cifar-fs
│   ├── CIFAR_FS_test.pickle
│   ├── CIFAR_FS_train.pickle
│   ├── CIFAR_FS_val.pickle
├── fc100
│   ├── FC100_test.pickle
│   ├── FC100_train.pickle
│   ├── FC100_val.pickle
├── photo
│   ├── PHOTO_test.pickle
│   ├── PHOTO_train.pickle
│   ├── PHOTO_val.pickle
|
```
   
### Train
   The following commands provide an example to train the Constellation Net .
   ```bash
   # Usage: bash ./scripts/train.sh [Dataset (mini, cifar-fs, fc100)] [Backbone (conv4, res12)] [GPU index] [Tag]
   bash ./scripts/train.sh mini conv4 0 trial1
   ```

### Evaluate
   The following commands provide an example to evaluate the checkpoint after training.
   ```bash
   # Usage: bash ./scripts/test.sh [Dataset (mini, cifar-fs, fc100)] [Backbone (conv4, res12)] [GPU index] [Tag]
   bash ./scripts/eval.sh mini conv4 0 trial1
   ```
