---
title: "Using separate `Transforms` for your training and validation sets in fastai2"
summary: "Applying a Transform to specific subsets of your Dataset using fastai2's low level APIs"
layout: post
branch: master
toc: true
categories: [tutorials]
comments: true
category: blog
---

```python
%config autocompleter.use_jedi = False
```


```python
import fastai2
from fastai2.data.all import *
from fastai2.vision.all import *
```

## Applying a `Transform` to specific subsets of your Dataset using `fastai2`'s low level APIs

### Preliminaries:

**What we will be discussing:** We'll learn one very specific thing about the fastai2 library: how to restrict some transforms to work only on specific subsets of your data. For example, you may want a particular image augmentation to only run on your training data and not your validation data.<br>
**What we are not going to do:** We are not going to look at the source code in this blog and understand how to achieve the above objective. I've done all that homework so you don't have to, and we'll simply be looking into how we can write code to achieve this. Along the way, we'll get some insights on why things work the way they do. I believe that digging through the source code is a great exercise in itself and I encourage you to do the same to get to the core of what we will discuss today<br>
**What knowledge is assumed of the reader:** This blog is meant for users who have some familiarity with how to use the `Datasets` functionality (and it's relatives) in `fastai2`. Usually, these readers will be fairly comfortable at a high level with the `fastai2` library although it isn't required. To get the necessary background, I recommend you to work through the first of the FastAI V2 walkthroughs that Jeremy Howard posted a few months ago here: [Video](https://www.youtube.com/watch?v=44pe47sB4BI). Some of the names of components in the library have changed since then, most notably:<br>
1. DataSource -> Datasets
2. Databunch -> Dataloaders


From this point onwards, I will be assuming that you have the necessary prerequisites to follow through this blog. I encourage you to spin up a Jupyter notebook and run all the code below.

# Let's begin!

First, we need to get some items ready to work on ...


```python
data = untar_data(URLs.MNIST_SAMPLE)
# Dataset consisting of MNIST Images of classes `3` and `7`
```


```python
items = get_image_files(data)
```


```python
items[-10:], items[:10]
```




    ((#10) [Path('/Users/akash/.fastai/data/mnist_sample/train/3/10847.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/49926.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/25630.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/23241.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/24248.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/54977.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/23527.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/48386.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/28472.png'),Path('/Users/akash/.fastai/data/mnist_sample/train/3/27741.png')],
     (#10) [Path('/Users/akash/.fastai/data/mnist_sample/valid/7/9294.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/1186.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/6825.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/4767.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/6170.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/6164.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/9257.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/4773.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/8175.png'),Path('/Users/akash/.fastai/data/mnist_sample/valid/7/6158.png')])



We'll begin by defining define some transforms for our data


```python
@Transform
def imopen(o:Path): return PILImage.create(o)
# Simply creates an Image.Image from a Path
```


```python
img_tfms = [imopen, ToTensor()] # Open image, convert to tensor
cat_tfms = [parent_label] # Label each image using the parent folder name of it's path
splitter = RandomSplitter() # Splits items randomly into train and validation set
splits   = splitter(items) # Get split indices
```


```python
splits
```




    ((#11548) [10633,2570,13625,8068,1511,12874,1824,14429,10383,5628...],
     (#2886) [11197,954,6177,4220,2384,13223,5007,12005,11552,11120...])



So, here we define a `Datasets` object in the usual way, passing in the items and the transforms for each component of our data. We pass in `splits` to specify the split between training and validation sets. Or rather to be more specific, we do this to specify the `subset(0)` and the `subset(1)` of our dataset.

What does component above mean: Let's say each training sample of your data is of the form: (Image, label). In this case, `Image` is your first component, and `label` is your second component.


```python
ds = Datasets(items=items, tfms=[img_tfms, cat_tfms], splits=splits)
```

Sanity checks:


```python
ds[0], ds.train[0], ds.valid[0]
# The above is the same as: ds[0], ds.subset(0)[0], ds.subset(1)[0]
```




    ((TensorImage([[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
      
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
      
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8),
      '7'),
     (TensorImage([[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
      
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
      
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8),
      '3'),
     (TensorImage([[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
      
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
      
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8),
      '3'))



Now let's say, hypothetically, that we want to apply our `imopen` transform only to items in the `ds.train` dataset. We'll investigate how we might go about achieving that.

So, in `fastai2`, every split of your dataset is linked to a `split_idx`. In the above example, the `split_idx` for `ds.train` and `ds.valid` is `0` and `1` respectively. So, `split_idx` is essentially the *index* of a split in the Dataset.

In `fastai2`, each Pipeline that is assigned to a split of your data has a matching `split_idx`. For example, here are the values for our train and validation sets in `ds`:

Note: `tls[0]` is the Pipeline for opening our image and `tls[1]` is the Pipeline for setting our labels.


```python
ds.train.tls[0].split_idx, ds.train.tls[1].split_idx
```




    (0, 0)




```python
ds.valid.tls[0].split_idx, ds.valid.tls[1].split_idx
```




    (1, 1)




```python
# For completeness' sake: each ds.tls itself has splits inside of it.
# We can confirm this:
assert ds.tls[0].train == ds.train.tls[0]
# I prefer using ds.train.tls[0] because I believe it makes things a bit more readable
# and so that's what we'll be using throughout the rest of the blog.
```

As you can see, the Pipelines are automatically assigned the appropriate `split_idx`s depending on what subsets of your data they're meant for.<br>
Now, each `Transform` can have a `split_idx` too. But how does that help?<br>
If a `Transform` has a split_idx of `0`, it will only be active if it's inside a Pipeline which also has a `split_idx` of `0`. So basically, the `split_idx`s of the Pipeline and a transform inside it need to match if the transform is to be active, otherwise, it will just pass on the input without doing any operation on it. One thing to note here is that if a `Transform` does not have a `split_idx` (i.e. `split_idx = None`), it will be applied to all the inputs it gets regardless of the Pipeline's `split_idx`

The code examples ahead will help clarify this a bit more.<br>
We'll redefine everything with a few changes


```python
@Transform
def imopen(o:Path): return PILImage.create(o)
imopen.split_idx = 0
# Now, we should expect this to work only for the training set
```


```python
img_tfms = [imopen, ToTensor()] # Open image, convert to tensor
cat_tfms = [parent_label] # Label each image using the parent folder name of it's path
splitter = RandomSplitter() # Splits items randomly into train and validation set
splits   = splitter(items) # Get split indices
```


```python
ds = Datasets(items=items, tfms=[img_tfms, cat_tfms], splits=splits, split_idx=2)
```

So, at this point, we have three datasets in `ds`:
1. `ds` : the full dataset
2. `ds.train` : the first split, i.e. `ds.subset(0)`
2. `ds.valid` : the second split, i.e. `ds.subset(1)`

And for all these datasets, a separate set of `Pipelines` (or rather specifically, `TfmdLists`) has been created using the same transforms we passed above. Let's verify that: <br>
(we should expect each `TfmdList` to have two Pipelines for two of our components)


```python
len(ds.tls), len(ds.train.tls), len(ds.valid.tls)
```




    (2, 2, 2)



Looking at each Pipeline in the `TfmdLists`:


```python
ds.tls[0].tfms, ds.tls[1].tfms
```




    (Pipeline: imopen -> ToTensor, Pipeline: parent_label)



Now, here are the `split_idx` values for all of them:

Note: `split_idx` for `ds` itself is what we passed in when we created the `Datasets` object


```python
ds.split_idx, ds.train.split_idx, ds.valid.split_idx
```




    (2, 0, 1)



`split_idx` for each `TfmdLists` and for the `Pipeline`s contained in them for all our datasets:


```python
print(ds.tls[0].split_idx, ds.tls[1].split_idx) 
print(ds.tls[0].tfms.split_idx, ds.tls[1].tfms.split_idx)
```

    2 2
    2 2



```python
print(ds.train.tls[0].split_idx, ds.train.tls[1].split_idx)
print(ds.train.tls[0].tfms.split_idx, ds.train.tls[1].tfms.split_idx)
```

    0 0
    0 0



```python
print(ds.valid.tls[0].split_idx, ds.valid.tls[1].split_idx)
print(ds.valid.tls[0].tfms.split_idx, ds.valid.tls[1].tfms.split_idx)
```

    1 1
    1 1


As we can see, all of the `TfmdLists` and `Pipeline`s above have been assigned the appropriate `split_idx` values depending on what dataset they belong to. This is as expected

Here's the interesting bit: we'll now look at the `split_idx` values of the `imopen` function in each of these `TfmdLists`:


```python
ds.tls[0].fs[0], ds.tls[0].fs[0].split_idx
```




    (imopen: (Path,object) -> imopen , 0)




```python
ds.train.tls[0].fs[0], ds.train.tls[0].fs[0].split_idx
```




    (imopen: (Path,object) -> imopen , 0)




```python
ds.valid.tls[0].fs[0], ds.valid.tls[0].fs[0].split_idx
```




    (imopen: (Path,object) -> imopen , 0)



So, regardless of what `Pipeline` it belongs to, the `imopen` function has retained it's `split_idx` value. Now let's see how it affects things going ahead. We should expect `imopen` to work only for our train dataset.


```python
ds[0]
```




    (Path('/Users/akash/.fastai/data/mnist_sample/valid/7/9294.png'), '7')




```python
ds.train[0]
```




    (TensorImage([[[0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              ...,
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0]],
     
             [[0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              ...,
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0]],
     
             [[0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              ...,
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0],
              [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8),
     '7')




```python
ds.valid[0]
```




    (Path('/Users/akash/.fastai/data/mnist_sample/train/7/10408.png'), '7')



So the code above just confirms everything we just discussed.

Now, if we put everything into a `Dataloader`, we'll get the expected errors where items from `ds` and `ds.valid` will casuse issues when being put into a batch of tensors since you can't do that with `Path` objects:


```python
dl0 = DataLoader(dataset=ds, bs=4)
dl1 = DataLoader(dataset=ds.train, bs=4)
dl2 = DataLoader(dataset=ds.valid, bs=4)
```


```python
dl0.split_idx, dl1.split_idx, dl2.split_idx 
```




    (2, 0, 1)




```python
it0, it1, it2 = iter(dl0), iter(dl1), iter(dl2)
```


```python
next(it0)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-67-6a3290243c2c> in <module>
    ----> 1 next(it0)
    

    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in __iter__(self)
         95         self.randomize()
         96         self.before_iter()
    ---> 97         for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
         98             if self.device is not None: b = to_device(b, self.device)
         99             yield self.after_batch(b)


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/dataloader.py in __next__(self)
        343 
        344     def __next__(self):
    --> 345         data = self._next_data()
        346         self._num_yielded += 1
        347         if self._dataset_kind == _DatasetKind.Iterable and \


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/dataloader.py in _next_data(self)
        383     def _next_data(self):
        384         index = self._next_index()  # may raise StopIteration
    --> 385         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        386         if self._pin_memory:
        387             data = _utils.pin_memory.pin_memory(data)


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         32                 raise StopIteration
         33         else:
    ---> 34             data = next(self.dataset_iter)
         35         return self.collate_fn(data)
         36 


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in create_batches(self, samps)
        104         self.it = iter(self.dataset) if self.dataset is not None else None
        105         res = filter(lambda o:o is not None, map(self.do_item, samps))
    --> 106         yield from map(self.do_batch, self.chunkify(res))
        107 
        108     def new(self, dataset=None, cls=None, **kwargs):


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in do_batch(self, b)
        125     def create_item(self, s):  return next(self.it) if s is None else self.dataset[s]
        126     def create_batch(self, b): return (fa_collate,fa_convert)[self.prebatched](b)
    --> 127     def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
        128     def to(self, device): self.device = device
        129     def one_batch(self):


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in create_batch(self, b)
        124     def retain(self, res, b):  return retain_types(res, b[0] if is_listy(b) else b)
        125     def create_item(self, s):  return next(self.it) if s is None else self.dataset[s]
    --> 126     def create_batch(self, b): return (fa_collate,fa_convert)[self.prebatched](b)
        127     def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
        128     def to(self, device): self.device = device


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in fa_collate(t)
         44     b = t[0]
         45     return (default_collate(t) if isinstance(b, _collate_types)
    ---> 46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
         47             else default_collate(t))
         48 


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in <listcomp>(.0)
         44     b = t[0]
         45     return (default_collate(t) if isinstance(b, _collate_types)
    ---> 46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
         47             else default_collate(t))
         48 


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in fa_collate(t)
         45     return (default_collate(t) if isinstance(b, _collate_types)
         46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
    ---> 47             else default_collate(t))
         48 
         49 # Cell


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
         79         return [default_collate(samples) for samples in transposed]
         80 
    ---> 81     raise TypeError(default_collate_err_msg_format.format(elem_type))
    

    TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'pathlib.PosixPath'>



```python
next(it1)
```




    (TensorImage([[[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]],
     
     
             [[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]],
     
     
             [[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]],
     
     
             [[[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]],
     
              [[0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               ...,
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0],
               [0, 0, 0,  ..., 0, 0, 0]]]], dtype=torch.uint8),
     ('7', '3', '7', '3'))




```python
next(it2)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-69-58c7b9b7f185> in <module>
    ----> 1 next(it2)
    

    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in __iter__(self)
         95         self.randomize()
         96         self.before_iter()
    ---> 97         for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
         98             if self.device is not None: b = to_device(b, self.device)
         99             yield self.after_batch(b)


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/dataloader.py in __next__(self)
        343 
        344     def __next__(self):
    --> 345         data = self._next_data()
        346         self._num_yielded += 1
        347         if self._dataset_kind == _DatasetKind.Iterable and \


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/dataloader.py in _next_data(self)
        383     def _next_data(self):
        384         index = self._next_index()  # may raise StopIteration
    --> 385         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        386         if self._pin_memory:
        387             data = _utils.pin_memory.pin_memory(data)


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         32                 raise StopIteration
         33         else:
    ---> 34             data = next(self.dataset_iter)
         35         return self.collate_fn(data)
         36 


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in create_batches(self, samps)
        104         self.it = iter(self.dataset) if self.dataset is not None else None
        105         res = filter(lambda o:o is not None, map(self.do_item, samps))
    --> 106         yield from map(self.do_batch, self.chunkify(res))
        107 
        108     def new(self, dataset=None, cls=None, **kwargs):


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in do_batch(self, b)
        125     def create_item(self, s):  return next(self.it) if s is None else self.dataset[s]
        126     def create_batch(self, b): return (fa_collate,fa_convert)[self.prebatched](b)
    --> 127     def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
        128     def to(self, device): self.device = device
        129     def one_batch(self):


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in create_batch(self, b)
        124     def retain(self, res, b):  return retain_types(res, b[0] if is_listy(b) else b)
        125     def create_item(self, s):  return next(self.it) if s is None else self.dataset[s]
    --> 126     def create_batch(self, b): return (fa_collate,fa_convert)[self.prebatched](b)
        127     def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
        128     def to(self, device): self.device = device


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in fa_collate(t)
         44     b = t[0]
         45     return (default_collate(t) if isinstance(b, _collate_types)
    ---> 46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
         47             else default_collate(t))
         48 


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in <listcomp>(.0)
         44     b = t[0]
         45     return (default_collate(t) if isinstance(b, _collate_types)
    ---> 46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
         47             else default_collate(t))
         48 


    ~/Google Drive/Programming/Deep learning/Learn DL/FASTAI/FASTAI2/fastai2/fastai2/data/load.py in fa_collate(t)
         45     return (default_collate(t) if isinstance(b, _collate_types)
         46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
    ---> 47             else default_collate(t))
         48 
         49 # Cell


    ~/anaconda3/envs/fastai2/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
         79         return [default_collate(samples) for samples in transposed]
         80 
    ---> 81     raise TypeError(default_collate_err_msg_format.format(elem_type))
    

    TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'pathlib.PosixPath'>


# A very short summary:

1. Each `TfmdLists` (and the `Pipeline` inside it) in a `Datasets` object gets assigned appropriate `split_idx` values depending on what split of your data it is meant for.
2. You can assign a `split_idx` attribute to your `Transform` to specify what subsets of your dataset it will operate on.

If you look at the source code of `TfmdDL` and see what gets called just before we get our actual dataloader which is going to be used for the model, you'll see that it ends up overriding the `split_idx` of all these `Pipeline`s: `after_item`,`before_batch`,`after_batch`:
```python
# _batch_tfms = ('after_item','before_batch','after_batch')
def before_iter(self):
        super().before_iter()
        split_idx = getattr(self.dataset, 'split_idx', None)
        for nm in _batch_tfms:
            f = getattr(self,nm)
            if isinstance(f,Pipeline): f.split_idx=split_idx
```

I was initially confused as to why it would override any existing `split_idx` values for the pipelines. But in the context of what we just discuseed, my understanding is that doing this merely sets a *context* for the `Transforms` to operate in. Meaning that each `Transform` in the pipeline would know exactly what subset of the data it is working on when it gets the items, and if the `Transform` has a specific `split_idx` set already, it will only operate in the context where there's an exact match of the `split_idx` values.<br>

Please leave a comment if you thought this helped you!


```python

```
