.. _language_models:

Use Text as Node Features
=============================
Many real world graphs have text contents as nodes' features, e.g., the title and description of a product, and the questions and comments from users. To leverage these text contents, GraphStorm supports large language models (LLMs), i.e., HuggingFace BERT models, to embed text contents and use these embeddings in Graph models' training and inference.

There are two modes of using LLMs in GraphStorm:

* Embed text contents with pre-trained LLMs, and then use them as the input node features, without fine-tuning the LLMs. Training speed in this mode is fast, and memory consumption will be lower. However, in some cases, pre-trained LLMs may not fit to the graph data well, and fail to improve performance.

* Co-train both LLMs and GML models in the same training loop. This will fine-tune the LLMs to fit to graph data. This mode in general can improve performance, but co-train the LLMs will consume much more memory, particularly GPU memory, and take much longer time to complete training loops.

This tutorial will use OGB-arXiv data 