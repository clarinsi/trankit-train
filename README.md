# About
This repository contains code utilized for retraining and evaluating models based on [Trankit: A Light-Weight Transformer-based Python Toolkit for Multilingual Natural Language Processing](https://github.com/nlp-uoregon/trankit). Using this setup, we developed new Trankit models for Slovenian, trained on a more recent and considerably larger version of the Slovenian UD Treebanks than the default Trankit models (trained on UD v2.5). 

For a detailed understanding of the inner workings and Trankit library options, please refer to the [original documentation](https://github.com/nlp-uoregon/trankit). This repository serves as an illustration, demonstrating how to leverage the improved models developed during this project.

## Published models

The models were trained on the successive versions of the SSJ UD treebank of written Slovenian, the SST UD treebank of spoken Slovenian, and a combined dataset incorporating both. For production use, we recommend the latest model, [Trankit SSJ+SST-2.15](http://hdl.handle.net/11356/1997), which achieves state-of-the-art performance for both written and spoken Slovenian (.

| Release date              | Short name             | Training Data                                                                                                                                                                                                                                        | Model (CLARIN.SI repository)                                   |
|------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| 2023-09-29 | Trankit-SSJ-2.12 | [SSJ r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/86f832a8a0663d908fdaf5cded8c0567508fd7c0)                                                                                                                        | [zip](http://hdl.handle.net/11356/1870) | 
| 2024-01-17 | Trankit_SSJ+SST-2.12 | [SSJ r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/86f832a8a0663d908fdaf5cded8c0567508fd7c0) + [SST r2.12](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/9d67eb90ae9aa6f37a7097d03d9e8864996c0609) | [zip](http://hdl.handle.net/11356/1909) |
| 2024-08-29 | Trankit_SSJ-2.15 | [SSJ r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/r2.15) + [SST r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/r2.15)                                                                       | [zip](http://hdl.handle.net/11356/1963) |
| 2024-12-06 | Trankit_SST-2.15 | [SST r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/r2.15)                                                                       | [zip](http://hdl.handle.net/11356/1996) |
| 2024-12-06 | Trankit_SSJ+SST-2.15 | [SSJ r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/tree/r2.15) + [SST r2.15](https://github.com/UniversalDependencies/UD_Slovenian-SST/tree/r2.15)                                                                       | [zip](http://hdl.handle.net/11356/1997) - recommended|



# Usage example
Below, we provide a step-by-step guide on how to use our models with the trankit tool.

## Step 1: Initialization
```python
from trankit import Pipeline, trankit2conllu

# Initialize trankit
p = Pipeline(lang='customized', cache_dir='<PATH TO DOWNLOADED MODELS>', embedding='xlm-roberta-large')
```

## Step 2: Process Input
There are two options for processing input:

### Option 1 - Using Text Input:
```python
text = 'Example text!'
dict_output = p(text)
```

### Option 2 - Using a Pre-tokenized List:
```python
pretokenized_list = [['Example', 'pre-tokenized', 'list', '!']]
dict_output = p(pretokenized_list)
```

## Step 3: Convert Output to CONLLu Format
```python
# Convert output from dictionary to CONLLu format
conllu_output = trankit2conllu(dict_output)
```
