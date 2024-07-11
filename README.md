# 🚀 Shakespeare: Unleashing the Power of PyTorch for Object Identification

**🇬🇧 English** | [🇹🇷 Türkçe](README_TR.md)

Shakespeare is a lightweight PyTorch project designed to help you train a model to visually identify objects using popular architectures like ResNet, EfficientNet, VGG, DenseNet, or MobileNet.

<p align="center">
    <img src="https://github.com/DogukanUrker/Shakespeare/assets/62756402/76d2fe03-1c5e-474f-99ed-f8683ec66a97" alt="logo">
</p>

## 📹 Tutorial Video

Check out my [Tutorial Video](https://youtu.be/8xC1-un_sjA) for a step-by-step guide!

## 📂 Installation

Clone the repository:

```bash
git clone https://github.com/DogukanUrker/Shakespeare.git
cd Shakespeare
```

Run setup.py to create necessary folders and install required modules:

```bash
python3 setup.py
```

## ⚡️ Getting Started

1. 💿 **Prepare Your Data**:

   - Place your images in the following default directories:
     - `data/object/`: Contains images of objects you want to classify.
     - `data/notObject/`: Contains images that do not belong to the object class.
     - `data/test/`: Additional images for testing the trained model.

2. ⚙️ **Configure the Model**:

   - Open `defaults.py` and set `MODEL_NAME` to the desired model architecture (`resnet`, `efficientnet`, `vgg`, `densenet`, `mobilenet`).

3. 🏋️ **Train the Model**:

   - Start training and create a `.pkl` file by running:
     ```bash
     python3 main.py
     ```

4. 📝 **Testing**:
   - After training, evaluate your model's performance using:
     ```bash
     python3 test.py
     ```

## 🎨 Customization

- Modify `defaults.py` or `train.py` to adjust hyperparameters or experiment with different model architectures.
- Extend functionality by adding preprocessing steps or data augmentation in `train.py`.

## 💞 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to add features, please submit a pull request.

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🌟 Support Our Project!

If you find Shakespeare helpful in your projects and would like to support its development and maintenance, you can contribute to my project's sustainability.

### How You Can Help:

- **Give Us a Star on GitHub**: Show your appreciation by starring my [GitHub repository](https://github.com/DogukanUrker/Shakespeare). It helps me reach more developers like you!

- **Visit my [donation page](https://dogukanurker.com/donate)** to choose from multiple platforms and support my work directly.

Your support means a lot and helps us continue improving Shakespeare for the community. Thank you for considering!

Created by [Doğukan Ürker](https://dogukanurker.com)
