# Aspect Sentiment Quad Prediction (ASQP)

This repo contains the annotated data and code for our paper [Aspect Sentiment Quad Prediction as Paraphrase Generation](https://aclanthology.org/2021.emnlp-main.726.pdf) in EMNLP 2021.


## Short Summary 
- We aim to tackle the aspect sentiment quad prediction (ASQP) task: given a sentence, we predict all sentiment quads `(aspect category, aspect term, opinion term, sentiment polarity)`

## Data
- We release two new datasets, namely `rest15` and `rest16` under the `data` dir.
- Each data instance contains the original sentence, as well as a list of sentiment quads, separated by `####`. 
- The annotations are from the combination of the existing TASD data and ASTE data. We conduct further annotations to obtain the complete quad label for each sentence. 
- You can also access the ABSA triplet data from the repo [Generative-ABSA](https://github.com/IsakZhang/Generative-ABSA).


## Requirements

We highly recommend you to install the specified version of the following packages to avoid unnecessary troubles:
- transformers==4.0.0
- sentencepiece==0.1.91
- pytorch_lightning==0.8.1


## Quick Start

- Set up the environment as described in the above section
- Download the pre-trained T5-base model (you can also use larger versions for better performance depending on the availability of the computation resource), put it under the folder `T5-base`.
  - You can also skip this step and the pre-trained model would be automatically downloaded to the cache in the next step
- Run command `sh run.sh`, which runs the ASQP task on the `rest15` dataset.
- More details can be found in the paper and the help info in the `main.py`.


## Citation

If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{zhang-etal-2021-aspect-sentiment,
    title = "Aspect Sentiment Quad Prediction as Paraphrase Generation",
    author = "Zhang, Wenxuan  and
      Deng, Yang  and
      Li, Xin  and
      Yuan, Yifei  and
      Bing, Lidong  and
      Lam, Wai",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.726",
    pages = "9209--9219",
}
```
