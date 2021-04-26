---
layout: post
title: The 8 Commandments for Coding Your Research
tags: [research]
mathjax: true
---

This article is in a different flavor from the other posts in this publication. This is because I have been reading Roman Vershynin’s "High Dimensional Probability" for the last few days, and between that and visa formalities, I didn’t get a chance to check out new papers. I do plan to write an article on new methods for object detection (such as RCNN, Faster RCNN, and YOLO) sometime next month.

I have only been a researcher for a couple of years now but during this period I have gained valuable insights on how to structure a research project. When I started out with research back in 2016, I was too eager to obtain results, a mistake that most beginner researchers make. In my eagerness, I used to cut corners with my code structure and take several liberties, especially because there was no review process. I made several mistakes during the project (on relation classification: [link to Github repo](https://github.com/desh2608/crnn-relation-classification)), some of which I list here:

* **Lack of planning:** I did not have a research plan to begin with, which showed in my code. It is true that in applied machine learning, much of the progress is dictated by experimental results, but a broad outline still helps. At the beginning, I just read papers, cloned their repositories, and ran them on my GPU. Sometimes this used to eat up a lot of precious time since some repositories had several dependencies.
* **Haphazard code format:** This stemmed from the first issue. Since I had not planned in advance, I would work with every dataset differently, depending upon how it was available. Some of the code would be in IPython notebooks, while some would be Shell scripts. I would use plain Tensorflow for some training and a Keras wrapper for others.
* **Not caring about reproducibility:** This is perhaps the biggest crime an ML researcher can commit. Although my code is legitimate and publicly available, I highly doubt that anyone could reproduce it (not easily, in any case). This is because at that time, all I cared about was getting a publication (which I did, in the end). I did not have a good README file, nor instructions on how to reproduce the results.

Based on these and several other mistakes, I have come up with some guidelines on how to write good code for a research project. I ascribe much of my learning to working on [Kaldi](https://github.com/kaldi-asr/kaldi) for the last few months. Here are the 8 commandments, along with examples in Python.

#### 1. Define and validate data types at the outset

Define data structures which will hold your input and output data. Since Python allows using data structures without declaring them implicitly, this can be done by having validation functions which are invoked whenever the data structure is used. This would ensure that the structure is consistent throughout the project. An example of a data structure validation for an “object” type (which is a dict with just one key) is below.

```python
def validate_object(x):
    """This function validates an object x that is supposed to represent an object inside an image, and throws an exception on failure. Specifically it is checking that:
      x['polygon'] is a list of >= 3 integer (x,y) pairs representing the corners of the polygon in clockwise or anticlockwise order.
    """
    if type(x) != dict:
        raise ValueError('dict type input required.')

    if 'polygon' not in x:
        raise ValueError('polygon object required.')

    if not isinstance(x['polygon'], (list,)):
        raise ValueError('list type polygon object required.')

    points_list = x['polygon']
    if len(points_list) < 3:
        raise ValueError('More than two points required.')

    for x, y in points_list:
        if type(x) != int or type(y) != int:
            raise ValueError('integer (x,y) pairs required.')

    return
```

#### 2. Write data loader scripts for all your datasets

Now that wehave common data structures to use with our model(s), we need to convert all our datasets to that format. There are 2 ways to achieve this:

* Preprocess the dataset to the required structure and save in a serialized file (e.g. Pickle in Python).
* Have a data loader class to read the dataset from source at the time of running and return in the desired format.

*When should you use the second method?* When the dataset itself is large, or we need to have several additional elements in the structure, such as mask data (for an image), or associate word vectors (for text data).

Additionally, if you have a decent processor and parallelizable script, the compute time in method 2 should be low enough such that the total runtime in 1 becomes larger due to greater I/O time.

#### 3. Put common methods in a shared library

Since all the datasets are in a common structure, several transformation methods may be applicable to many of them. So it would make sense to have these methods in a global shared library and link to this library inside each of the local dataset directories. This achieves 2 things:

* Reduces clutter and reduplication in the directory.
* Allows for ease in making modifications to the shared functions.

#### 4. Write unit tests for utility functions

Instead of writing a test file and modifying it for testing the utility functions, it would be better to use the **[unittest](https://docs.python.org/3/library/unittest.html)** package in Python, or analogous packages in other languages. For example, in an object detection project, there may be utilities to visualize the object with a mask, or to compress the image. The unit test file may then look like this.

```python
import unittest


class ImageUtilsTest(unittest.TestCase):
    """Testing image utilities: visualization and compression
    """
    def setUp(self):
        """This method sets up objects for all the test cases.
        """
        <code for loading data>


    def test_visualize_object(self):
        """Given a dictionary object as follows
        x['img']: numpy array of shape (height,width,colors)
        x['mask']: numpy array of shape (height,width), with every element categorizing it into one of the object ids
        The method generates an image overlaying a translucent mask on the image.
        """
        visualize_mask(self.test_object)


    def test_compress_object(self):
        """Given a dictionary object x, the method compresses the object and prints the original and compressed sizes.
        It also asserts that the original size should be greater than the compressed size.
        """
        y = compress_image_with_mask(self.test_object,self.c)
        x_mem = sys.getsizeof(self.test_object)
        y_mem = sys.getsizeof(y)
        self.assertTrue(y_mem <= x_mem)


if __name__ == '__main__':
    unittest.main()
```

#### 5. Prepare installation and run scripts

This is key for reproducibility. It is very arrogant to assume that readers would clone your repository, install several dependencies one by one, then download the datasets, preprocess the data using some script in your repo, and only then be able to start training. All of these steps can and should be automated using simple Bash shell scripts, so that the users can just run an **install.sh** or a **run.sh** file with certain parameters to get things done.

If you have built a small library providing some functionality, say a text classification library, it would be best if you package it and make it available for download via a manager such as **pip** so that the package can be used directly in other projects.

In any case, installation and run instructions should be documented elaborately in a README file.

#### 6. Put parameter tuning options as command line arguments

In continuation with #5, the user should never be expected to open your training script to tune hyperparameters, or provide path to data directories, or other similar stuff. Python has the **argparse** library which facilitates parsing command line arguments, and it is insanely simple to use. Bash has the parser available by default and the arguments can be accessed using the numbered variables $0, $1, and so on. Similar functionalities are available for almost every programming language.

```python
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
```

#### 7. Have a defined naming convention for saved models and result files

Your saved model and result file names should indicate the important hyperparameters that were used in that instance of training. This simplifies testing with several available models later during analysis.

#### 8. Get your code reviewed before merging

I can’t stress this enough. Regardless of how sincere you have been in your coding, your commits would still be flawed in some way. Having a reviewer always helps, even if it is to point out some pep8 naming convention.

In my undergrad thesis project, I was the sole contributor, and so there used to be several weeks in which I didn’t push any code, arguing that it was all there in my local system anyway. I would think of Git as an additional time-consuming formality, instead of the immensely useful tool it is. Don’t make that mistake!

*****
 
*I hope these guidelines are useful to some researcher who is just starting out on her first project. Yes, it would take some time to get used to following all these rules, but trust me, your research would only be the better for it in the long run!*