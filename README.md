# Chinese Word Segmentation with BiLSTM-CRF

## Task

Given a Chinese sentence, divide it into segmentations according to its syntactic and semantic meaning.

## Dataset

Sighan-2004 competition data

[site](http://sighan.cs.uchicago.edu/)

## Prerequisite

PyTorch 0.4+

numpy 1.2+

## Model

### Embedding

Word embeddings are trained on character-level. The characters can be categorized into four types: Chinese characters, marks, English characters, and numbers. For simplicity, I did some substitutions in the data. English characters and numbers are replaced with special marks according to their lengths. Punctuation marks are replaced with a special mark 'm'. The substitutions are shown as follows:

Type | Length | special mark 
----- | ----- | -----
Chinese Character | - | -
English Character | 1 | z
English Character | \>1 | y
Number | 1 | a
Number | 2 | b
Number | \>2 | c
Punctuation mark | - | m

After substitutions, characters are embedded into 300-dimensional vectors.

### BiLSTM

A multi-layer BiLSTM is fed with the embedded vectors. After searching, I found the best hyper-parameters are:

* embedding-dim: 300
* hidden-dim: 200
* number of layers: 3
* dropout rate: 0.6
* batch size: 128

### Fully-connected Layer

Two FC layers are used to reduce the dimension to 4, namely, S ( Start ), M ( Middle ), E ( End ), S ( Single ). 

### CRF ( Conditional Random Fields )

CRF is a powerful tool for structure prediction. I add two special tags, &lt;beg&gt; and &lt;end&gt; for CRF. 

As I will mention below, the initial values of transition matrix are set to 0 for some purposes.

## Experiment

In the training and validation, I separate the sentences into some segments with the length of 50. For convenience, some special marks are padded on the two ends of sentences. In the evaluations, I will subtract the number of padded marks from the true-positives.

In fact, the punctuation marks provide the model with a lot of information, and it is worth thinking that whether we should treat punctuation marks as ordinary characters. That is to say, should the segmentation related to the punctuation marks be counted into my results? For comparison, I list the results of both of them.

The number of tags is just 4 ( 6 if &lt;beg&gt; and &lt;end&gt; are taken into consideration ), so CRF may not help a lot. In the experiment, I test the models with and without CRF layer.

## Result

The f1 score and accuracy are listed here.

Model | f1 Score | Accuracy
---- | ---- | ----
BiLSTM-CRF-PUNC | 0.984 | 0.981
BiLSTM-PUNC | **0.986** | **0.984**
BiLSTM-CRF | 0.977 | 0.977
BiLSTM | 0.980 | 0.980

You can use random seed 9788 to reproduce my result. It takes 20 epochs approximately.

## Analysis

Obviously, the f1 score is higher with the help of punctuation marks.

It is quite interesting that after adding CRF layer, the accuracy drops some points. As a matter of fact, to make the learning process easier and save time, I trained the BiLSTM firstly and then added the CRF layer above the pretrained BiLSTM model. To figure out why CRF doesn't work, I repacked the features of LSTM to truncate the gradient. After a epoch of training, I printed the weights of transition matrix:

\- | B | M | E | S | &lt;beg&gt; | &lt;end&gt;
--- | --- | --- | --- |--- | --- | ---
B|0.0| 0.0|15.5|15.0|12.9| 0.0
M| 16.1|13.6| 0.0| 0.0| 0.9| 0.0
E| 15.0|16.1| 0.0| 0.0| 1.9| 0.0
S|0.0| 0.0|15.0|14.1|14.3| 0.0
&lt;beg&gt;|-14.4| -14.4| -14.4|-14.4 |-14.4 | -14.4
&lt;end&gt;|11.0| 9.7|11.6|14.3 |-14.4| 0.0


As the table shows, some weights of transition matrix haven't even been changed ( remember that I initiated the weights of matrix with 0's ). Look into the tables, it is reasonable that the weights towards &lt;beg&gt; are negative because there is no node can be followed with START_TAG. Also, it is correct that M cannot be followed with B, B cannot precede B itself and so on.

It is clear that the work of CRF has been done by BiLSTM network, thus this weights will not obtain any back-propagated gradient. As we can see, 3-layer BiLSTM is capable to capture the structure information of the labeling, and it may be redundant to add CRF layer.

## Future work

* Maybe train BiLSTM and CRF at the same time will get more information about the structure.
* Ensemble may work.
* Due to the limitation of the performance of my computer, I didn't search for the best hyper-parameters very carefully. In fact, the f1 score didn't converge at last.

Any pull requests are welcome!!
