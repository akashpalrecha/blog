---
layout: page
title: About Me
permalink: /about/
---
> Note: This needs to be updated significantly. It's been a while since I did that.
# Hi, I'm Akash.

🌐 [github.com/akashpalrecha](https://github.com/akashpalrecha) <br><br>
[E-mail](mailto:akash@pixxel.co.in) • [Twitter](https://twitter.com/akashpalrecha98) • [Github](https://github.com/akashpalrecha) • [LinkedIn](https://www.linkedin.com/in/akashpalrecha/) • [+919879990466](#)


[Resume.pdf]({{ site.baseurl }}/files/Updated_TA_Resume.pdf)

---

# 👨🏻‍🎓 Background

---
I work as an **AI Researcher** in a Space-tech startup ([Pixxel](http://pixxel.co.in)) that is building a constellation of earth-imaging small satellites 🛰 to provide an entirely new kind of dataset of the earth that today’s satellites aren’t capable of.

I am currently pursuing an **MSc. Mathematics** degree in **Birla Institute of Technology and Science, Pilani** as a pre-final year student. My best work primarily centres around making AI algorithms for solving computer vision problems such as ***classification and segmentation***. My research interests include *transfer learning, optimizers, interpretability* and everything that lowers the number of resources required to do great, world-class work and *makes deep learning more accessible*. I like reading new research papers every week and look forward to getting into NLP.

*I like tackling big, high impact problems. One such problem I believe in is **making AI accessible**. I am greatly influenced by [Jeremy Howard](https://www.fast.ai/about/#jeremy)'s (Co-Founder, [Fast.ai](https://www.fast.ai/)) approach to AI and my belief in making AI accessible stems from Fast.ai's core philisophy of making neural nets uncool again.* 

---

**In my free time**, I like helping out college students to get into deep learning. I also play drums with my college band in BITS Pilani's Music Club. We make new songs all the time and have a great time jamming together every day!

---

# Experience

---

- **AI Lead, Pixxel (February 2019 - January 2020)**
    - Currently leading teams for: Infrastructure Monitoring, Land Cover Classification, Crop classification.
    - Manage, recuirt for the AI team
    - Setup labeling procedures for Proprietary Data
    - Managed creation of a new Proprietary Dataset for Indian Roads in Satellite Imagery.
    - Built a SoTA Drought Detection model (using satellite imagery)
    - Planned and executed creation of some internal data pre-processing tools.
- **AI Researcher, Pixxel (September 2018 - February 2019)**
    - Training period: reading latest CV Research, getting comfortable with the cloud
    - Internal literature review on road segmentation from Satellite imagery
    - Built initial LinkNet model for Road segmentatation.

---

> ***A day in my life*** <br>
    - College classes. <br>
    - Finding new, impactful Research papers to send to my Kindle (Twitter's AI Community helps). <br>
    - Work in Pixxel AI, and on personal research projects. <br>
    - Binge-surfing [forums.fast.ai](https://forums.fast.ai). <br>
    - Go Run! <br>
    - Jam/Compose with my band post dinner (I play drums!) <br>
    - Learn, learn, learn! <br>
    
---

# 👨🏻‍💻Working On / Learning

---

### Research

1. A modification to [Batchnorm](https://arxiv.org/abs/1502.03167) layers that doesn't require mean or variance parameters and gives comparable or better performance. (personal research)
2. A general method to classify images using deep models by taking assitance from query images. 

### Learning

1. FastAI V2
2. Part 2 (lecture 12) of FastAI's [Deep Learning Course](https://course.fast.ai/part2)
3. (Week 8) [CS50's Web Programming with Python and Javascript](https://courses.edx.org/courses/course-v1:HarvardX+CS50W+Web/course/)

### Blogs/Tutorials:

- **[An Inquiry Into Matplotlib's Figures](https://matplotlib.org/matplotblog/posts/an-inquiry-into-matplotlib-figures/) : published on Matplotlib's Official Blog!**

    Talks in depth about some of matplotlib's foundational concepts involving `Figures`, `Axes`, and `GridSpec`

- **[(Tutorial+Code) Using your own pre-trained models in FastAI](https://github.com/akashpalrecha/Custom-Pretrained-Models-in-Fastai)**

    Shows how to use your own pretrained models in the FastAI library along with pre-written functions to allow you to do so in just 3 lines of code.

More blogs coming soon

---

# 💻 Projects / Small Code Sprints

---

### Training Deep Models:

- **Lookahead Optimizer Analysis**

    [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

    Closely simulates experiments run by Geoffrey Hinton in the paper linked to above. 

    The paper talks about a new optimizer called "Lookahead" which can essentially wrap around any existing state-of-the-art optimizers and increase their performance.

    The results I've achieved very closely agree with those in the paper. Specifically, the proposed optimizer (`Lookahead+SGD`) gives the `lowest validation loss` (as compared to `AdamW`, `SGD`, `RMSProp` in each of: (Using a `resnet34`)

    1. `CIFAR10`
    2. `CIFAR100`
    3. `Imagenette` (A representative subset of Imagenet created by Jeremy Howard to accelerate research)

    💻 Interactive Report: **[Lookahead Optimizer Project](https://www.notion.so/Lookahead-Optimizer-Project-913e45b63e9a4528bee56a588e477f9f)** 

    [akashpalrecha/Lookahead](https://github.com/akashpalrecha/Lookahead)

- **Drought Detection Challenge: #1 on Weights and Biases Leaderboard**

    ![files/leaderboard.png]({{ site.baseurl }}/images/about/leaderboard.png)
		

    This is a small part of my work in Pixxel that is public. This challenge provides a dataset to detect areas of the earth affected by drought using satellite imagery. I trained the model for this challenge using PyTorch and am currently (as of 13 Jan, 2020) **#1** on the leaderboard since September 2019.

    [akashpalrecha/drought-watch](https://github.com/akashpalrecha/drought-watch)

- **Toxic Comments Classification using ULMFiT: Kaggle Competition**

    -a very quick model built to solve the Kaggle Toxic Comments Classification Challenge. 

    A lot of time has not been spent on training the model to a great accuracy.*The objcetive here was to learn how to build multi-label language classification models using Fast.ai’s ULMFiT (Universal Language Model Fitting) approach.*

    With a few hours of training, I was able to get an `accuracy` of `99.26%` and an `ROC-AUC` score of `98.7%` on `10%` of the training data kept aside for validation.

    **Github Repository: [Toxic Comments Classification : Kaggle](https://github.com/akashpalrecha/Kaggle-Toxic-Comments-Classification)**

    **Github Gist: [Training Code](https://gist.github.com/akashpalrecha/1f636e3a82f6e7f9802656d31c96f477)**

- **Human Protein Atlas Classification Challenge: Kaggle Competition**

    This project is my attempt at the [Human Protein Atlas Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification) competition on Kaggle. To make the model work better, I’ve used multiple approaches and optimizations such as:

    1. Multilabel stratification
    2. 4 channel Resnets
    3. Making transfer learning work for 4 channels
    4. Progressive resizing

    At the time of taking part in this competition, I had to suddenly stop working on this project due to some unforeseen circumstances. As a result I couldn’t train the model for a significant amount of time. 

    I managed to get an `accuracy` of about **`88%`** on the multilabel classification task invloving `29 classes`. The best `fbeta` score was **`0.412`**.

    **This is an incomplete / dropped project**

### From Scratch (Utilities, Practice):

**I believe that being able to build things for yourself when you can't find open-source code (libraries) is an important skill. This is a small subset of the things that I've needed to build over time:**

- **Java-ML (I am proud of this)**

    Implemented a neural network library with PyTorch like interface in Java from scratch without using any third-party libraries

    - Rudimentary `Autograd` to allow for arbitrary stacking of Neural network layers and activation functions.
    - Binary matrix operations `(x, +, -, ÷)`
    - Random matrix generators.
    - Initialization (`Kaiming He`, `Gaussian`, `Random`)
    - Optimizers: `SGD` (Full bactch, Mini-batch). Support for other optimizers such as `AdamW`, `RMSProp`, etc.
    - Activation Functions: `ReLu`, `TanH`, `TanSigmoid`, `Sigmoid`. Includes support for others.
    - Loss Functions: `SoftmaxCrossEntropy`, `CrossEntropy` . Includes support for others
    - Supports binary and multi-label classification. Easily extended for regression.

    Objective: better understand the depth of the code that goes into making a neural network work.

    Complete Report with code examples: **[Java-ML](https://www.notion.so/Java-ML-b49282c45e404aaab99afa75597e1769)** 

    [akashpalrecha/Java-ML](https://github.com/akashpalrecha/java-ML)

- **Image Augmentation: Custom Package**

    Implemented customized Data Augmentation package for images. Compatible with all popular deep learning frameworks (PyTorch, Tensorflow, etc).

    Provides Image augmentation functions such as:

    1. `Horizontal Flip`
    2. `Vertical Flip`
    3. `Gaussian Blur`
    4. `Color Jitter`
    5. `Brightness/Contrast`
    - Can take transforms from other popular libraries

    Usage:
```python
    from ImageReader import *
    tfms = [Horizontal_flip(), Vertical_flip(), Gaussian_blur(3),
            Crop_and_resize(do_crop=False, sz=(300, 300))]
    imr = ImageReader(PATH_TO_IMAGES, transforms=tfms)
    image = imr.read_image_random() # or imr.read_image_by_id(name)
```
   
   [akashpalrecha/ImageReader_w_data_augmentation](https://github.com/akashpalrecha/ImageReader_w_data_augmentation)

   I do not use/mainatin this repository anymore. Similar functionality is provided by other more managed/complete open source projects. The core objective here was just getting used to write boiler-plate code for data science work flows when you do not have open source to rescue you.

- **Efficiently Oversampling Image Classification Datasets**

    This package makes it easy to use oversampling in image classification datasets. Datasets with an imbalance between the number of data points per category are pretty common and oversampling the less frequent classes often gives improved results. This package gives an oversampled representation of the data that works well with FastAI’s Datablock API.

    This package helps you very quickly increase the proportion of any given class in your dataset in a few lines of code without actually copying the data .

    Head over to the repository to see how it works: 

    [akashpalrecha/classification_oversampling](https://github.com/akashpalrecha/classification_oversampling)

    I do not use/mainatin this repository anymore. Similar functionality is provided by other more managed/complete open source projects. The core objective here was just getting used to write boiler-plate code for data science work flows when you do not have open source to rescue you.

- **Fast Callbacks**

    Custom Callbacks to extend the FastAI library’s functionality when training models with large datasets where each epoch takes a very long time. 

    I had such an issue with when working with a large Satellite dataset in Pixxel where each epoch was taking more than an hour. To monitor and save progress every `N` iterations as opposed to the usual `N` epochs which most popular libraries implement, I had to write my own callbacks. 

    1. Gradient Accumulation: Emulates a higher batch size when you cannot increase batch size due to memory constraints. Having a low batch-size (`<16`) while using batchnorm at the same time generally results in unstable training. This callback helps accumulate gradients over `N` batches while training any model using the FastAI library.
    2. Skip `N` iterations of training: for when the server crashes in the middle of a huge epoch and you need to restart training at the last `iteration` you saved your model. Which brings us to:
    3. Saving the model after every `N` iterations: self-explanatory.
    4. Stop training in `N` iterations: self-explanatory.
    5. Show results every `N` iterations: self-explanatory.

    [akashpalrecha/custom-fastai-callbacks](https://github.com/akashpalrecha/custom-fastai-callbacks)

- **Multichannel Resnets**

    Creates Resnets with an arbitrary number of input channels along with approximate Imagenet pre-training benefits in 2 lines of code.
    ```python
    from multichannel_resnet import get_arch as Resnet
    #returns a callable that you can pass to libraries like fastai.
    #Usage: Resnet(encoder_depth, number_of_desired_input_channels)
    resnet34_4_channel = Resnet(34, 4)
    # use resnet34_4_channels(False) to get a non pretrained model
    model = resnet34_4_channel(True)
    ```

    It's a very common requirement when working with Satellite data to need to have `Resnets` with more than 3 channels (multi-spectral, hyperspectral imagery). There wasn't an off-the-shelf compononent available online so I had to write a utility for myself.

    I have found that simply copying the weights of the kernels in the first layer into the additional kernels added (to increase channels) still gives much of the Imagenet-pretraining benefits.  

    [akashpalrecha/Resnet-multichannel](https://github.com/AKASHPALRECHA/RESNET-MULTICHANNEL)


### Development

- 📖 **Book Review Website**

    This is a small, very minimal flask-powered webapp that allows the user to register, login, and search and review thousands of books. (Motivation: Because a Data scientist that just knows data science isnt' enough). See it in action here:

    [https://boiling-gorge-23297.herokuapp.com/](https://boiling-gorge-23297.herokuapp.com/)

    [akashpalrecha/books-reviews-website](https://github.com/akashpalrecha/books-reviews-website)

---

# Skills:

---

- Expert
    1. PyTorch
    2. FastAI
    3. Classification, Segmentation
    4. NumPy, Pandas, Matplotlib, Jupyter
    5. Python
- Intermediate
    1. Sklearn
    2. OpenCV
    3. Webapps using Flask+CSS+HTML+Javascript
- Basic
    1. Tensorflow
    2. AWS, GCP (Managing VMs for AI Research / Model Training
    3. Language Models using ULMFiT
    4. GANs
- Learning
    1. Computation Linear Algebra by FastAI
    2. Javascript, Django
    3. Docker
