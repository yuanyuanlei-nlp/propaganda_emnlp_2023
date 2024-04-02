# Read Me

<br/>

**Paper:** Discourse Structures Guided Fine-grained Propaganda Identification<br/>
**Accepted:** The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang<br/>
**Paper Link:** https://aclanthology.org/2023.emnlp-main.23/

<br/>

## Task Description
Identifying propaganda content at sentence-level and token-level.

<br/>

## Data Description
* **Propaganda:** We used the propaganda dataset for the propaganda identification experiments (https://aclanthology.org/D19-1565/) (https://propaganda.qcri.org/fine-grained-propaganda-emnlp.html)<br/>
* **PDTB:** We used PDTB 2.0 dataset to train the discourse relations model (https://aclanthology.org/L08-1093/) (https://catalog.ldc.upenn.edu/LDC2008T05)<br/>
* **NewsDiscourseData:** We used the news discourse structure dataset to train the news discourse role model (https://github.com/prafulla77/Discourse_Profiling)<br/>

<br/>

## Code Description
* **sentence_identification_additional.py:** sentence-level propaganda identification with feature concatenation model (section 3.1)<br/>
* **sentence_identification_distill.py:** sentence-level propaganda identification with knowledge distillation model (section 3.2)<br/>
* **token_identification_additional.py:** token-level propaganda identification with feature concatenation model (section 3.1)<br/>
* **token_identification_distill.py:** token-level propaganda identification with knowledge distillation model (section 3.2)<br/>

<br/>

## Citation
If you are going to cite this paper, please use the form:

Yuanyuan Lei and Ruihong Huang. 2023. Discourse Structures Guided Fine-grained Propaganda Identification. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 331–342, Singapore. Association for Computational Linguistics.

```bibtex
@inproceedings{lei-huang-2023-discourse,
    title = "Discourse Structures Guided Fine-grained Propaganda Identification",
    author = "Lei, Yuanyuan  and
      Huang, Ruihong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.23",
    doi = "10.18653/v1/2023.emnlp-main.23",
    pages = "331--342",
    abstract = "Propaganda is a form of deceptive narratives that instigate or mislead the public, usually with a political purpose. In this paper, we aim to identify propaganda in political news at two fine-grained levels: sentence-level and token-level. We observe that propaganda content is more likely to be embedded in sentences that attribute causality or assert contrast to nearby sentences, as well as seen in opinionated evaluation, speculation and discussions of future expectation. Hence, we propose to incorporate both local and global discourse structures for propaganda discovery and construct two teacher models for identifying PDTB-style discourse relations between nearby sentences and common discourse roles of sentences in a news article respectively. We further devise two methods to incorporate the two types of discourse structures for propaganda identification by either using teacher predicted probabilities as additional features or soliciting guidance in a knowledge distillation framework. Experiments on the benchmark dataset demonstrate that leveraging guidance from discourse structures can significantly improve both precision and recall of propaganda content identification.",
}
```

<br/>
