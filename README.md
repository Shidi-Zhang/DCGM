<h1 align="center"><b>DCGM</b></h1>


## Folder Specification

- `data/input/` folder contains necessary data or scripts for generating data.
  - `idx2SMILES.pkl`: Drug ID (we use ATC-4 level code to represent drug ID) to drug SMILES string dictionary.
  - `substructure_smiles.pkl`: A list containing the smiles of all the substructures.
  - `ddi_mask_H.py`: The python script responsible for generating `ddi_mask_H.pkl` and `substructure_smiles.pkl`.
  - `processing.py`: Processing the mimic_iii dataset. 
- `data/output_iii/` This folder stores the output files generated from the MIMIC-III data processing scripts.
- `data/output_iii/` This folder stores the output files generated from the MIMIC-III data processing scripts.
- `src/` folder contains all the source code.
  - `modules/`: Code for model definition.
    - `gnn/`: Support files including GeoGNN and GAT networks.
    - `GeoGCT.py`: Main model file.
  - `utils.py`: Code for metric calculations and some data preparation.
  - `training.py`: Code for the functions used in training and evaluation.
  - `main.py`: Train or evaluate our MoleRec Model.

## Data processing
```shell
cd data
python processing.py
```

If you want to re-generate `ddi_matrix_H.pkl` and `substructure_smiles.pkl`, use the following command:
```shell
cd data
python ddi_mask_H.py
```


## Run the code
```shell
cd src
python main.py
```




