
<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->





<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/heraclex12/VLSP2020-Fake-News-Detection">
  </a>

  <h3 align="center">VLSP2020: Fake News Detection</h3>

  <p align="center">
    Fine-tune a variety of pre-trained Transformer-based models to solve Vietnamese Reliable Intelligent Identification (ReINTEL) problem in VLSP2020 shared task.
    <br />
  </p>
</p>



<!-- ABOUT THE PROJECT -->
## About The Project
In this project, we utilize the effectiveness of the different pre-trained language models such as vELECTRA, vBERT, PhoBERT, Bert Multilingual Cased, XLM-RoBERTa to identify reliable information shared on social network sites.

We evaluate the different input length models, it includes 256, 512, and multiple 512 (long document)

### Prerequisites

To reproduce the experiment of our model, please install the requirements.txt according to the following instructions:
* huggingface transformer
* emoji
* vncorenlp
* nltk
* pytorch
* python3
```sh
pip install -r requirements.txt
```

### Data

The dataset is provided by VLSP2020 Organizers. Please access [this site](https://vlsp.org.vn/vlsp2020/eval/reintel) for more information. 

<!-- CONTACT -->
## Contact

Hieu Tran - heraclex12@gmail.com

Project Link: [https://github.com/heraclex12/VLSP2020-Fake-News-Detection](https://github.com/heraclex12/VLSP2020-Fake-News-Detection)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Hierarchical Transformers for Long Document Classification](https://arxiv.org/abs/1910.10781)
* [Improving Sequence Tagging for Vietnamese Text Using Transformer-based Neural Models](https://arxiv.org/abs/2006.15994)
* [PhoBERT: Pretrained language model for Vietnamese](https://github.com/VinAIResearch/PhoBERT)
* [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf)
