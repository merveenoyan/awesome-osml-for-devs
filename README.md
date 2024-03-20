# Awesome Open-source Machine Learning for Developers

List of resources, libraries and more for developers who would like to build with open-source machine learning off-the-shelf.

**Motivation**: Developers are often building machine learning with closed-source models behind gated APIs. These models can change by time without developers knowing, companies are giving away their data during inference and have no control over the model nor data.

There are a ton of open-source models out there that can be deployed by developers, but reducing barrier of entry to use these models and making developers aware of them are necessary, so I created this repository to do so.

Using the resources here, you can learn to find the model you need and serve it on the platform of your choice using the tools given here.

Hint: Take a look at foundation models section for one-model-fits-all type of models.

_Note_: To contribute, send a pull request to this repository. Note that this repository is focused on open-source machine learning.

## Table of Contents
<!-- MarkdownTOC depth=4 -->

- [Resources](#resources)
- [Libraries, Platforms and Development Platform-specific Resources](#libraries-platforms-and-development-platform-specific-resources)
  - [Platforms](#platforms)
  - [Development Platform](#development-platform)
    - [Web](#web)
    - [Mobile](#mobile)
    - [Edge](#edge)
    - [Cloud Deployment](#cloud-deployment)
    - [Serving](#serving)
    - [Game Development](#game-development)
- [Modalities and Tasks](#modalities-and-tasks)
  - [Foundation Models](#foundation-models)
  - [LLMs](#llms)
    - [Tools](#tools)
  - [Multimodal Models](#multimodal-models)
    - [Models and Demos](#models-and-demos)
    - [Understanding Image and Text](#understanding-image-and-text)
    - [Document AI](#document-ai)
  - [Generative AI](#generative-ai)
    - [Models and Demos](#models-and-demos)
  - [Computer Vision](#computer-vision)
    - [Models and Demos](#models-and-demos)
  - [Natural Language Processing](#natural-language-processing)
  - [Audio](#audio)
- [Advanced](#advanced)

<!-- /MarkdownTOC -->

## Resources

- [Tasks](https://huggingface.co/tasks): A documentation project to let developers build their first machine learning based product using models off-the-shelf.
- [Open-source AI Cookbook](https://huggingface.co/learn/cookbook/en/index): Recipes and notebooks using open-source models and libraries.

## Libraries, Platforms and Development Platform-specific Resources

### Platforms

- [Hugging Face Hub](https://huggingface.co/): Collaborative platform for machine learning. Discover hundreds of thousands of open-source models able to work off-the-shelf in [/models](https://huggingface.co/models).
  
### Development Platform

- [ONNX Runtime](https://onnxruntime.ai/): Platform agnostic model runtime to use ML models.

### Web

- [Transformers.js](https://huggingface.co/docs/transformers.js/en/index): A library to run cutting edge models directly in-browser.
- [huggingface.js](https://huggingface.co/docs/huggingface.js/en/index): A library to play with models on Hugging Face Hub through javascript.

### Mobile

- [TensorFlow Lite](https://www.tensorflow.org/lite): A library to deploy models on mobile and edge devices.
- [Mediapipe](https://developers.google.com/mediapipe): A framework that has prebuilt and customizable ML solutions, ready to deploy on Android, iOS.
- [ExecuTorch](https://pytorch.org/executorch/): A library for enabling on-device ML in mobile/edge devices for PyTorch models.
- [huggingface.dart](https://github.com/shivance/huggingface.dart): A Dart SDK to interact with the models on Hugging Face Hub.
- [flutter-tflite](https://github.com/tensorflow/flutter-tflite): TensorFlow Lite Flutter plugin provides an easy, flexible, and fast Dart API to integrate TFLite models in flutter apps across mobile and desktop platforms.

### Edge

- [TensorFlow Lite](https://www.tensorflow.org/lite): A library to deploy models on mobile and edge devices.
- [ExecuTorch](https://pytorch.org/executorch/): A library for enabling on-device ML in mobile/edge devices for PyTorch models.

### Cloud Deployment

### Serving

- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index): Toolkit to serve large language models.
- [Text Embedding Inference](https://huggingface.co/docs/text-embeddings-inference/index): Toolkit to serve text embeddings.
- [TorchServe](https://pytorch.org/serve/): Flexible, easy to use and scalable inference server.

### Game Development

- #### Unity 

  - [MediaPipe Unity Plugin](https://github.com/homuler/MediaPipeUnityPlugin): Unity plugin to run MediaPipe. This approach may sacrifice performance when you need to call multiple APIs in a loop, but it gives you the flexibility to use MediaPipe instead.
  - [Hugging Face API for Unity 🤗](https://github.com/huggingface/unity-api): This Unity package provides an easy-to-use integration for the Hugging Face Inference API, allowing developers to access and use Hugging Face AI models within their Unity projects.

## Modalities and Tasks

This section contains powerful models that can generalize well and can be used out-of-the-box.

### Foundation Models

The following resources are on zero-shot models: These models take in an image or text and possible classes in those images or texts.

- [Zero-shot Object Detection](https://huggingface.co/tasks/zero-shot-object-detection)
- [Zero-shot Image Classification Resources](https://huggingface.co/tasks/zero-shot-image-classification)
- [Zero-shot Text Classification Resources](https://huggingface.co/tasks/zero-shot-classification)

_Note_: The foundation model can be found under their associated task.

## LLMs

### Tools

- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index): Toolkit to serve large language models.
- [Text Embedding Inference](https://huggingface.co/docs/text-embeddings-inference/index): Toolkit to serve text embeddings.
- [TGI Benchmark](https://www.youtube.com/watch?v=jlMAX2Oaht0): Understanding throughput of inference servers, with TGI example.
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): A leaderboard to find the open-source LLM for your use and budget.

## Multimodal Models

- [Awesome Open-source Foundation and Multimodal Models](https://github.com/SkalskiP/awesome-foundation-and-multimodal-models): A curated list of models that have high zero-shot capabilities and the models that can take in two different modalities (e.g. a model that can take in image and text and output text).

### Models and Demos

- [Kosmos-2](https://huggingface.co/spaces/merve/kosmos2): Demo for Kosmos-2 model by Microsoft, that can understand image and text, answer questions about images, caption images, detect objects on images and gives answer without hallucinating.
- [Fuyu-8B](https://huggingface.co/spaces/adept/fuyu-8b-demo): Demo for Fuyu-8b by Adept, that can understand image and text and answer questions about images and caption images.

## Understanding Image and Text

### Document AI

- [Document AI Collection](https://huggingface.co/collections/merve/awesome-document-ai-65ef1cdc2e97ef9cc85c898e): A collection on demos and models for document AI.

## Generative AI

### Models and Demos

- [Stable Cascade](https://huggingface.co/stabilityai/stable-cascade): An app based on state-of-the-art image generation model Stable Cascade. (as of March '24)
- [Stable Diffusion XL Inpainting](https://huggingface.co/spaces/diffusers/stable-diffusion-xl-inpainting): An application that can do inpainting when given a text prompt and an image.

## Computer Vision

### Models and Demos

- [OWL](https://huggingface.co/collections/merve/owl-series-65aaac3114e6582c300544df): A curation about OWL model released by Google, the most powerful zero-shot object detection model. (as of March '24)
- [Segment Anything](https://huggingface.co/collections/merve/segment-anything-model-6585835fc76915aa14e2bcbd): A curation about Segment Anything model released by Meta, the most powerful zero-shot image segmentation model. (as of March '24)
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything): A highly practical solution for robust monocular depth estimation by training on a combination of 1.5M labeled images and 62M+ unlabeled images.

## Natural Language Processing

## Audio

- [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper): A CLI to shrink and serve Whisper on-device.
- [Open TTS Tracker](https://github.com/Vaibhavs10/open-tts-tracker): An awesome repository to keep a track of open-source text-to-speech models.

## Advanced

## Other

- [Raycast](https://github.com/raycast/extensions) Automate commands on macOS apps with a local ollama LLM, with Raycast extensions.
