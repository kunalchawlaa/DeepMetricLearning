###  AI Meets Beauty Challenge
#### ACM Multimedia 2018
Perfect Half Million Beauty Product Image Recognition Challenge at ACM Multimedia 2018 [Link](https://challenge2018.perfectcorp.com)

Repo contains the scripts used for downloading the dataset and evaluation of our submission for the challenge [Team: VCA (Visual Commerce Assistant)]

#### Usage
* Download the whole retrieval dataset consisting of 500K images. 
```
$ python crawl.py 0
```
* Download [Validation Set](http://s3.bj.bcebos.com/challenge2018/public_testset/val.zip), [Validation Ground Truth](http://s3.bj.bcebos.com/challenge2018/public_testset/val.csv) and [Testset](http://s3.bj.bcebos.com/challenge2018/public_testset/testset_v1.zip)

* Modify testbed.cfg file and run the following command to evaluate (Model files are not shared)
```
$ python -u testbed.py --p deploy.caffemodel -i 0 | tee LOG
```
 

