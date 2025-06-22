# Lemon Disease Detector

A Bi-model Deep Learning Architecture to detect the disease of Lemon Leaves using `Convolutional Neural Networks`.

The models are also available on [HuggingFace Models](https://huggingface.co/IshaqueJunejo/Lemon-Disease-Detector).

## Demo

![Demo Usage](Images/Usage-Demo.gif)

## Architecture

Architecture is based on 2 `Convolutional Neural Networks`.

First one is a `Binary Classification Model` that can tell if a given image is a *Lemon Leaf* or *Not*. The Image is passed to second model only if it is a *Lemon Leaf*.

Second model is a `Multi-Class Classifier` that tells the likelihood of the given image of Lemon Leaf having each *disease* and of being *healthy*.

Both models are based of `MobileNetV2` for Transfer Learning, as there was limited number of images to train a model from scratch.

## How to use

To use this project, you will have to run it locally on your machine, as there is no publically available Inference API endpoint.

- First clone this repository.
``` bash
git clone https://github.com/IshaqJunejo/Lemon-Disease-Detector.git
```
- Go to the `Lemon-Disease-Detector/api/` directory and run the API.
``` bash
cd Lemon-Disease-Detector/api/
pip install -r requirements.txt
python app.py
```
- Open a new terminal session, and host a server for the frontend-client.
``` bash
cd Lemon-Disease-Detector/web/
python -m http.server
```
- Open a Browser, and run the frontend to test the project by opening `http://127.0.0.1:8000`, upload an image, and click **Analyze**.

#### Optional

You can change the `web/static/script.js`, by replacing `127.0.0.1:7860/predict` to `<your-ip-address>:7860/predict`, if you want to try it on more than one device in your local network.

## Contributions

Any and every contribution you wish to make for this project is highly appreciated.

- Just fork this repository.
- Clone your fork.
- Create a new branch, make changes, and push to your fork.
- Open a Pull Request.

## Dataset

- [Lemon Leaf Disease Dataset](https://www.kaggle.com/datasets/mahmoudshaheen1134/lemon-leaf-disease-dataset-lldd) on Kaggle
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) on Kaggle
- [Natural Images Dataset](https://www.kaggle.com/datasets/prasunroy/natural-images) on Kaggle

Images from `PlantVillage` and `Natural Images` dataset were used as *Not Lemon Leaves* for training and testing the `Binary Classification Model`.

## License

This project is Licensed Under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE) due to the inclusion of [Plantvillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and [Natural Images Dataset](https://www.kaggle.com/datasets/prasunroy/natural-images) which are also License under **CC BY-NC-SA 4.0**. 