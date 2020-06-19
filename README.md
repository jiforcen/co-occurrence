# Co-occurrences of Deep Convolutional Features

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of our ***co-occurrence*** representation method from convolutional neural networks activation maps in PyTorch. This ***co-occurrence*** representation method is presented in the paper [***Co-Occurrence* of Deep Convolutional Features for image search**](https://arxiv.org/abs/2003.13827).


## Formulation


We define that a *co-occurrence* happens when the value of two different activations, a(i,j,k) and a(u,v,w), are greater than a threshold, t, and both are spatially located inside of a given region. (See Figure 1. left). The aggregation of the activation values when a *co-occurrence* occur produce the *co-occurrence* tensor. This *co-occurrence* formulation can be implemented using the convolution operator and filters. (See Figure 1. Right).




![](/figures/deep_cooc.png)

**Figure 1:** In the left an ilustration of a *co-occurrence* in an activation tensor. In the right a graphical example of how can be implemented using the convolution operaton.

## *Direct co-occurrences* & *Trainable co-occurrences*

One great advantage of this implementation is that *co-occurrence* filter can be learned inside a trainable pipeline, and this can lead to improved *co-occurrence* representations so we consider two tipes of *co-occurrence* representations.
* ***Direct co-occurrences*:** activation are aggregated using a fixed *co-occurrences* filter.
* ***Trainable co-occurrences*:** *co-occurrence* filter is learned during CNN training.


## *Co-occurrences* code

In [cooccurrences.py](./cooccurrence/cooccurrences.py) is the torch implementation to obtain the *co-occurrence* tensor from an activation tensor. This representation can be obtained with the next method called *calc_spatial_cooc*.

```
cooc_tensor = calc_spatial_cooc(tensor, cooc_filter, cooc_r)
```

##### Arguments

* **tensor:** activation 4D tensor with shape: `(batch_size, rows, cols, channels)`.
* **cooc_filter:** activation 4D tensor with shape: `(channels, channels, 2 * cooc_r + 1, 2 * cooc_r + 1)`. This *cooc_filter* can be initialized using the function called *ini_cooc_filter(channels, cooc_r)* also in [cooccurrences.py](./cooccurrence/cooccurrences.py).
* **cooc_r:** is the radious considered for the co-occurrence window

##### Output

* **cooc_tensor:** activation 4D tensor with same shape than the input activation 4D tensor: `(batch_size, rows, cols, channels)`.




## Example

In script [cooc_example.py](./cooc_example.py) is implemented an example of *co-occurrence* calculation from an image and its activation map. In this example are obtained the *co-occurrence* representation of *Direct co-occurrences* and *Trainable co-occurrences* previously trained with *retrieval-SfM-120k* dataset. (More details in our paper ***Co-Occurrence* of Deep Convolutional Features for image search**)


```
PYTHONPATH=. python cooc_example.py --test_image ./data/all_souls_000117.jpg --cooc_r 4
```
* **test_image:** path to the image to be tested.
* **cooc_r:** is the radious considered for the *co-occurrence* window.

In the next figure is shown a graphical representation of the *co-occurrence* tensors otbtained in the previous script. We can see that *Direct co-occurrence* representation emphasize regions with features, and *Trainable co-occurrence* representation emphasize buildings, because was trained in a buildings dataset.

![](/figures/cooc_example_image.png)

**Figure 2:** In this figure is represented for a given image its activation map, *Direct co-occurrence* representation and *Trainable co-occurrence* representation. (For representation purposes all channels are aggregated).



## Other samples

Also other examples are shown comparing *Direct co-occurrences* representation and *Trainable co-occurrences* representation.



![](/figures/Samples.png)


## Requirements
* Python 3
* Numpy
* torch > 0.4.1.post2
* matplotlib
* PIL
