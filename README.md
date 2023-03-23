# Document-Classification-LayoutLMv3

## Model Training and Inference

In order to perform model inference please download the finetuned model and config from [here](https://drive.google.com/drive/folders/1zu6P7S-CDduk326grC90nej8DGbvLjCK?usp=sharing). Create a directory called 'checkpoints' inside the 'document_classification' directory and place the 'pytorch_model.bin' and 'config.json' files inside the 'checkpoints' directory. The directory structure should look like as follows:

```
Document-Classification-LayoutLMv3
|
|--document_classifiation
|	|-checkpoints
|	|-Dataset
|	|-all the .py files
|
|
|--test_imgs
|--Dockerfile
```

Once the checkpoints have been placed in the correct directories, the docker image must be built using the following command.

```
docker build -t zainkhan97/dl_assignment_px_zainullah_khan:latest .
```

After the docker image has been built, it can be used for inference by first running the docker using the following command:

```
docker run --name document_classification --gpus all -itd -v d:/zain_dev/python_dev/Document-Classification-LayoutLMv3/document_classification:/document_classification -v d:/zain_dev/python_dev/Document-Classification-LayoutLMv3/test_imgs:/images_dir zainkhan97/dl_assignment_px_zainullah_khan:latest
```

Note that two volumes are mounted here. The first volume contains all the code required to run the mode, the second volume is a directory where all the test images must need to be placed.
The docker is made detachable and interactable.

### Training
Once the docker has been been run, the `docker exec` command is used to execute commands. In this case the training script is executed using the python interpretter using the following command.

```
docker exec document_classification python train.py
```

The model checkpoints are stored inside the `checkpoints` directory.

When running for the first time the base model will be downloaded, which will be fine tuned on custom datasetl.

### Inference
The `docker exec` command is used to make inference on files that have been placed inside the `test_imgs` directory. The name of the image is specified using an environment variable called `TEST_IMG_NAME`. The model inference is given below:

```
docker exec -e "TEST_IMG_NAME=doc_000051.png" document_classification python inference.py
```

The model prediction is logged to the screen.