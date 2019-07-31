# Multi Class Image Classification of Yoga postures using Watson Studio and Deep Learning As A Service

Computer vision usability is on the rise these days and there could be scenarios where a machine has to classify images based on their class to aid the decision making process. In this pattern, we will demonstrate a methodology to do multi class classification (with 3 classes) using Watson Studio. We will be using yoga postures data to identify the class given an image. This methodology can be applied to any domain and dataset which requires multiple classes of images to be classified accurately which can be extended for further analysis. Some of the advantages of computer vision are reliability, accuracy, cost reduction, wide range of use and simpler processes. We will demonstrate how to use Python scripts & drag n drop GUI for achieving the objective.

## How can IBM technologies help? 

**Deep Learning As A Service**

We understand that, to solve this problem there's a need to use deep learning techniques to achive state of the art results. But how? Can we automate the process of hyper parameters optimization which is the key aspect to achieve great results & use GPU's for quick computation? The answer is Yes!

IBM's Deep Learning as a Service enables organizations to overcome the common barriers to deep learning deployment: skills, standardization, and complexity. It embraces a wide array of popular open source frameworks like TensorFlow, Caffe, PyTorch and others, and offers them truly as a cloud-native service on IBM Cloud, lowering the barrier to entry for deep learning. It combines the flexibility, ease-of-use, and economics of a cloud service with the compute power of deep learning. With easy to use REST APIs, one can train deep learning models with different amounts of resources per user requirements, or budget.

Training of deep neural networks, known as deep learning, is currently highly complex and computationally intensive. It requires a highly-tuned system with the right combination of software, drivers, compute, memory, network, and storage resources.To realize the full potential of this rising trend, we want this technology to be more easily accessible to developers and data scientists so they can focus more on doing what they do best –concentrating on data and its refinements, training neural network models with automation over these large datasets, and creating cutting-edge models.

In this pattern, we demonstrate the creation and deployment of deep learning models using Jupyter Notebook (using CPU) in Watson Studio environment and create deep learning `Experiments` (using GPU) with hyper parameters optimization using Watson Studio GUI for monitoring different runs and select the best model for deployment. 

## What is CNN?

A convolutional neural network (CNN or Convnets) is one of the most popular algorithms for deep learning, a type of machine learning in which a model learns to perform classification tasks directly from images, video, text, or sound.

CNNs are particularly useful for finding patterns in images to recognise & classify persons, objects, faces, and scenes. They learn directly from image data, using patterns to classify images and eliminating the need for manual feature extraction.

## Advantages of using CNN

CNNs eliminate the need for manual feature extraction—the features are learned directly by the CNN.
CNNs produce state-of-the-art recognition results.
CNNs can be retrained for new recognition & classification tasks, enabling you to build on pre-existing networks.


When the reader has completed this code pattern, they will understand how to:

* Preprocess the images to get them ready for model building.
* Access images data from cloud object storage and write the predicted output to cloud object storage.
* Create a step by step deep learning model (code based) which includes flexible hyper parameters to classify the images accurately.
* Create experiments in Watson Studio (GUI based) for deploying state of the art models with hyper parameters optimization. 
* Create visualizations for better understanding of the model predictions.
* Interpret the model summary and generate predictions using the test data.
* Analyze the results for further processing to generate recommendations or taking informed decisions.

## Prerequisites

* Python programming is required to understand and modify the scripts as and when needed.
* Knowledge of Keras, tensorflow and computer vision is a plus.

## Architecture Diagram

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/architecture.png)

## Flow
1. User uploads images data to IBM Cloud Storage.
2. User accesses the data in Jupyter notebook.
3. User runs the baseline model notebook which has the deep learning convnets model along with tunable hyper parameters.
4. Notebook will train on the sample images from train and validation datasets and classifies the test data images using the deep learning model.
5. User can classify images into different classes using a REST client.
6. User can write the predicted output to cloud object storage in csv format which can be downloaded for further analysis.


## Included components

* [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio): Analyze data using RStudio, Jupyter, and Python in a configured, collaborative environment that includes IBM value-adds, such as managed Spark.

* [IBM Watson Machine Learning](https://www.ibm.com/in-en/cloud/machine-learning): IBM Watson Machine Learning helps data scientists and developers work together to accelerate the process of moving to deployment and integrate AI into their applications.

* [IBM Deep Learning As A Service](https://www.ibm.com/blogs/watson/2018/03/deep-learning-service-ibm-makes-advanced-ai-accessible-users-everywhere/): Making Deep Learning More Accessible, and Easier to Scale.

* [IBM Cloud Object Storage](https://console.bluemix.net/catalog/services/cloud-object-storage): An IBM Cloud service that provides an unstructured cloud data store to build and deliver cost effective apps and services with high reliability and fast speed to market. This code pattern uses Cloud Object Storage.

* [Jupyter Notebooks](http://jupyter.org/): An open-source web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text.

## Featured technologies

* [Data Science](https://developer.ibm.com/code/technologies/data-science/): Systems and scientific methods to analyze structured and unstructured data in order to extract knowledge and insights.
* [Artificial Intelligence](https://www.ibm.com/in-en/services/artificial-intelligence): Create systems that accelerate, enhance, and scale the human expertise.
* [Analytics](https://developer.ibm.com/code/technologies/analytics/): Analytics delivers the value of data for the enterprise.
* [Python](https://www.python.org/): Python is a programming language that lets you work more quickly and integrate your systems more effectively.


# Steps

Follow these steps to setup and run this code pattern. The steps are
described in detail below.

1. [Create an account with IBM Cloud](#1-create-an-account-with-ibm-cloud)
1. [Create a new Watson Studio project](#2-create-a-new-watson-studio-project)
1. [Create the notebook](#3-create-the-notebook)
1. [Add the data](#4-add-the-data)
1. [Insert the credentials](#5-insert-the-credentials)
1. [Run the notebook](#6-run-the-notebook)
1. [Analyze the results](#7-analyze-the-results)
1. [Access cloud object storage bucket](#8-access-cloud-object-storage-bucket)
1. [Run the notebook and publish it to Watson Machine Learning](#9-run-the-notebook-and-publish-it-to-watson-machine-learning)
1. [Create experiments using GPU for hyper parameters optimization](#10-create-experiments-using-gpu-for-hyper-parameters-optimization)

## 1. Create an account with IBM Cloud

Sign up for IBM [**Cloud**](https://console.bluemix.net/). By clicking on create a free account you will get 30 days trial account.

## 2. Create a new Watson Studio project

Sign up for IBM's [Watson Studio](http://dataplatform.ibm.com/). 

Click on New project and select Data Science as per below.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/new_project.png)

Define the project by giving a Name and hit 'Create'.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/define_project.png)

By creating a project in Watson Studio a free tier ``Object Storage`` service will be created in your IBM Cloud account.

## 3. Create the notebook

* Open [IBM Watson Studio](https://dataplatform.ibm.com).
* Click on `Create notebook` to create a notebook.
* Select the `From URL` tab.
* Enter a name for the notebook.
* Optionally, enter a description for the notebook.
* Enter this Notebook URL: https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/notebooks/Image-classification_baseline_model.ipynb
* Select the runtime (16 vCPU and 64 GB RAM)
* Click the `Create` button.
* Repeat the above steps to import the remaining notebooks which are in the notebooks folder into the project.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/create_notebook.png)

## 4. Add the data

`The images have been sourced from Google search and is being used for research activities. The images have been used as part of Fair Use policies for demonstration purpose only.`

[Clone this repo](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service)
Navigate to [images data](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/tree/master/data/images) and save the zip file on the disk. The sample data has been extracted and renamed from the original dataset.

`If you are using a mac machine and creating new zip folder of images, the compression of new image files creates additional file which should be deleted. On command prompt, go to the compressed file location and run the below command & then upload the zip file into cloud object storage. This activity is not needed if you use the sample trained_Data folder which is available in this repository.

* zip -d filename.zip \__MACOSX/\\*`

To create your own dataset, follow the below naming structure for each type of image


\----trained_data
     
     +----train
    
         + image 1.jpg
         
         + image 2.jpg
         
    +----test
    
         + image 1.jpg
         
         + image 2.jpg
         
    +----validation
    
         + image 1.jpg
         
         + image 2.jpg
    

Use `Find and Add Data` (look for the `10/01` icon)
and its `Files` tab. From there you can click
`browse` and add the images data file from your computer.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/add_file.png)

Note: The images data file is in the `data/images` directory

## 5. Insert the credentials

Select the cell below `Read the Data` section in the notebook.

Use `Find and Add Data` (look for the `10/01` icon) and its `Files` tab. You should see the file names uploaded earlier. Make sure your active cell is the empty one created earlier. Select `Insert to code` (below your file name). Click `Insert StreamingBody Object` & `Insert credentials` from drop down menu as specified in the notebook.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/insert.png)

## 6. Run the notebook

When a notebook is executed, what is actually happening is that each code cell in
the notebook is executed, in order, from top to bottom.

Each code cell is selectable and is preceded by a tag in the left margin. The tag
format is `In [x]:`. Depending on the state of the notebook, the `x` can be:

* A blank, this indicates that the cell has never been executed.
* A number, this number represents the relative order this code step was executed.
* A `*`, this indicates that the cell is currently executing.

There are several ways to execute the code cells in your notebook:

* One cell at a time.
  * Select the cell, and then press the `Play` button in the toolbar.
* Batch mode, in sequential order.
  * From the `Cell` menu bar, there are several options available. For example, you
    can `Run All` cells in your notebook, or you can `Run All Below`, that will
    start executing from the first cell under the currently selected cell, and then
    continue executing all cells that follow.
    
## 7. Analyze the results

In this Section, we will generate predictions on the test data which is not seen by the model. The format will be per below where the true filename & predicted filename are listed side by side. We will send these results in the form of csv to the cloud object storage where we can download the csv file for further analysis. 

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/results-from-model.png)

This will help us validate the prediction accuracy on the test data. In this case, the test data accuracy is 84% (5 out of 6 images have been classified accurately) if we use the images in the jpg format as input.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/results-from-wml.png)

If we preprocess the images (resize it to 224/224) and convert it to pickle format, the test data accuracy will be 100% because the images are converted into pixel array with input and target variables in the pickle files and the model is able to learn the pattern better than the raw images in jpg format. The models have been fine tuned in such a way that with less computation & less data time state of the art results are achieved. We have provided the notebooks for both methodologies for you to explore more as per your requirement. A tutorial on image preprocessing will be released soon which will cover many aspects with regards to preprocessing the images.

## 8. Access cloud object storage bucket

Log in to [IBM Cloud](https://cloud.ibm.com/login) and click on Dashboard-Storage-cloud-object-storage-he which will display the bucket created for our project. Click on the bucket to view the files inside it and hit the three dots towards the right side of the file to download the file object onto the local machine for evaluating the results.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/download_file.png)

## 9. Run the notebook and publish it to Watson Machine Learning

Follow the steps 1 to 7 with the below changes.

* In step number 3, enter the below notebook URL to create and import the notebook.

* Enter this Notebook URL: https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/notebooks/Image_classification_WML_model_Deployment.ipynb

* In step number 4, add the data files by name train.pkl, test.pkl & validation.pkl to the cloud object storage. These files will be ingested into the notebook which is created per above step. You are free to use any other methodology as per your comfort to generate pickle files. 

* In Step number 5, insert credentials as specified in the notebook and run the notebook as per step 6. The scoring URL would be generated which will be used to predict the class for test data.

* In step number 7, to analyze the results or generate predictions, use the scoring URL in the notebook and provide the test data in the required format as a JSON file to generate predictions. We can either run the last few cells in the notebook to generate predictions or use a separate notebook and provide the test_data.json to get the predictions. 

### Deploy the model

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/deploy_the_model.png)

### Generate scoring URL

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/deployment_url.png)

## 10. Create experiments using GPU for hyper parameters optimization

In this section, we will see how to create `experiments using Deep Learning As A Service(DLAAS) for hyper parameters optimization and deploy the best model with highest accuracy as a REST API for real-time scoring.`

First, we need to save the scripts & artefacts for running the experiment to the local file system or cloud object storage.
Navigate to [scripts-for-experiments](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/tree/master/scripts-for-experiments) and download the zip file onto the system. `This zip file is critical for creating and running experiments successfully. If you want to modify the model parameters or create a new model, then it has to be done in image_classify.py file and then zip it to be uploaded for experiments.`

Next step is to launch the Watson Studio interface and choose the project that we are working on and go to `Assets` tab. Under experiments, click on `New Experiment`.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/new-experiment.png)

Give a name for the experiment under `Define Experiment` Details. Select Machine Learning Service Instance from the dropdown (create one by following the instructions if it is not there). Select Cloud Object Storing Bucket for training data from the dropdown (create one by following the instructions if it is not there) and select Cloud Object Storage bucket for storing results (create one by following the instructions if it is not there). It is good to keep two separate buckets for training data and for storing results data.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/define-experiment.png)

We need to add a `Training Definition` by clicking on Add Training Definition. Give it a name and click on browse to select your training source code which is the zip file which was downloaded earlier. 

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/add-trng-defn.png)

`After selecting the training source code, select the framework, execution command & the Compute Configuration per below screenshot.`

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/create-trng-defn.png)

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/select-code-frmwrk.png)

We need to add `Hyperparameter optimization` method and other details per below. Number of optimizer steps can be reduced or increased as per the requirement, i have gone ahead with 30. The objective is to maximize the validation accuracy which needs to be selected per below.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/select-paramtr-opt.png)

The next step is to click on `Add Hyperparameter` and update the parameters per below. We can add more parameters like number of filters and layers if required using the same methodology. Give a name to the hyper parameter and select Values & Data type per below & hit Create. The hyper parameter name should be same as what is mentioned in the script image_classify.py.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/lr.png)

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/bs.png)

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/epochs.png)

After the `hyper parameters` are created, hit Create and then click on Create and run. The training run will be submitted for processing on IBM cloud. First, it will be in Queued state for about 2 minutes and then it will move to In progress where you can compare the training runs of different combinations of hyper parameters. 

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/trng-status.png)

`Compare` different training runs

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/compare-runs.png)

Check the `accuracy` & `loss` of training runs.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/check-accuracy.png)

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/check-loss.png)

We need to select which training run has given highest accuracy for training & validation data and then save the model per below. For ex :- if `Exp_1_1` has given highest accuracy then we have to click on three dots on the right side and click on Save model, give a name and hit Save. In this case, we can either select Exp_1_6 or Exp_1_28 which has low validation loss and high validation accuracy and save the model for deployment.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/save-model.png)

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/save-deplyd-mdl.png)

We will see a message that `Model successfully saved`. View model details here. Click on it and we will be directed to the model details per below.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/view-model.png)

Click on Deployments and select `Add Deployment`. Give a name for the deployed model and select Web Service radio button as we are deploying it as REST API for online scoring and hit Save.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/create-dplymt.png)

After a couple of seconds, the deployed model will have the Status as `DEPLOY_SUCCESS`. We have successfully deployed the model as a web service.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/deploy-success.png)

Click on `Deployments` to find the instance we have deployed and select the instance by clicking on it. 

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/view-saved-model.png)

Test the model : Click on deployed model to see the Overview, Implementation & Test attributes. Under Implementation we can find the Python code to do real time scoring or use the Test attribute and copy paste the contents from test_data_json file in this repository and hit Predict to generate predictions. We can also use the `scoring` notebook for generating the predictions and accuracy.

![](https://github.com/IBM/create-a-predictive-system-for-image-classification-using-deep-learning-as-a-service/blob/master/doc/source/images/predict.png)

This completes the section of creating experiments for hyper parameters optimization and deploying the optimized model as a REST API for realtime scoring. 

### Watch the Video for creating experiments & deploying the model

`Will be uploaded soon`


# License

This code pattern is licensed under the Apache Software License, Version 2.  Separate third party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the Developer [Certificate of Origin, Version 1.1 (DCO)](https://developercertificate.org/) and the [Apache Software License, Version 2](http://www.apache.org/licenses/LICENSE-2.0.txt).

ASL FAQ link: http://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN
