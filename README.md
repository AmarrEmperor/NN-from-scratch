# Neural Networks from Scratch

A hands-on study of neural network fundamentals through binary classification, data augmentation, and experiments with pretrained weights. The project builds a multilayer perceptron using PyTorch tensors while implementing the forward pass, loss, backpropagation, and parameter updates explicitly.

The work covers two classification problems: distinguishing two Iris species and learning a circular decision boundary in a synthetic dataset. Supporting work includes a C++ data generator and a Manim animation of common activation functions.

## What has been implemented

### Manual neural network

The notebooks implement a fully connected **2 → 3 → 2 → 1** network with sigmoid activations at every layer. Weights start as normally distributed random tensors and biases start at zero.

The training implementation includes:

- Matrix-based forward propagation through two hidden layers and one output layer.
- Binary cross-entropy, with probabilities clamped to avoid taking the logarithm of zero.
- Explicit gradients for every weight matrix and bias vector using the chain rule.
- Full-batch gradient descent with direct parameter updates.
- Binary predictions using a probability threshold of `0.5`.
- Training loss logging, test evaluation, and saved model artifacts.

PyTorch provides tensor operations; the training code does not rely on `nn.Module`, automatic differentiation, or a built-in optimizer. The main training runs use **5,000 epochs** and a **learning rate of 0.1**.

### Iris preparation and classification

The Iris workflow loads the 150-sample dataset from scikit-learn, converts it into tensors, shuffles it, and splits it into 120 training samples and 30 test samples before selecting the binary task.

PCA-based augmentation creates ten training sets: the original training set plus nine modified copies. The code perturbs the third and fourth principal directions with Gaussian noise and reconstructs the samples in the original feature space. This produces 1,200 training samples across the original three classes. The test samples remain unaugmented.

The binary dataset then retains **versicolor and virginica**, remaps their labels to `0` and `1`, and uses **petal length and petal width** as the two inputs. The classifier notebook trains the manual network, reports individual predictions, and saves both a parameter dictionary and the complete model.

![Original and PCA-augmented Iris samples](Iris_neural_net/images/original%26augmented_samples.png)

### Synthetic circular classification

The synthetic experiments use `Synthetic/data/synthetic_data.csv`, containing **1,000 samples** with columns `X1`, `X2`, and `output`. The C++ generator samples coordinates uniformly from `[-1.5, 1.5]` and assigns labels using:

```text
output = 1 if X1² + X2² < 1.4324, otherwise 0
```

This creates a circular boundary with a radius of approximately `1.1968`. The experiment notebooks use the first 800 rows for training and the last 200 for testing.

Three approaches are implemented:

1. **Random initialization:** train a new network on the synthetic data.
2. **Pretrained Iris weights:** evaluate the Iris-trained model directly on the synthetic test set.
3. **Fine-tuned Iris weights:** initialize from the Iris model and train it on the synthetic training set.

Each experiment saves a model and a Joblib experiment dictionary containing accuracy, precision, recall, F1 score, loss history, a confusion matrix, and a classification report.

### Activation-function visualization

`activation_functions/activation_function_plt.py` defines a Manim scene that draws **sigmoid, ReLU, and tanh**. A rendered video is included at [ActivationFunctionPlot.mp4](activation_functions/media/videos/activation_function_plt/1080p60/ActivationFunctionPlot.mp4).

## Recorded results

These values come from the outputs already saved in the notebooks. They have not been regenerated as part of this documentation update.

| Dataset / experiment | Test samples | Test accuracy | Test binary cross-entropy |
| --- | ---: | ---: | ---: |
| Iris binary classifier | 23 | 100.0% | Not printed |
| Synthetic: random initialization | 200 | 64.5% | 0.644987 |
| Synthetic: pretrained Iris weights | 200 | 48.0% | 2.475302 |
| Synthetic: fine-tuned Iris weights | 200 | 45.5% | 0.689089 |

In the recorded synthetic runs, random initialization performed best. Direct reuse of the Iris model predicted class `0` for all 200 test samples; fine-tuning reduced its test loss but did not improve classification accuracy over direct reuse. These runs illustrate the difficulty of transferring weights between tasks with different input distributions and decision boundaries.

The Iris score applies to a small saved test split. Random seeds are not fixed, and the augmentation notebook currently fits PCA on the full Iris dataset, including test samples. Consequently, these results should be treated as exploratory observations rather than a leakage-free, reproducible benchmark.

## Project structure

```text
NN from scratch/
├── README.md
├── Iris_neural_net/
│   ├── notebook/
│   │   ├── data_augmentation.ipynb
│   │   ├── building_dataset.ipynb
│   │   └── iris_mlpclassifier.ipynb
│   ├── data/                    # Original, augmented, and binary tensors; weights
│   ├── models/                  # Saved Iris model
│   └── images/                  # PCA and augmentation illustrations
├── Synthetic/
│   ├── notebookes/
│   │   ├── load_&_visualize_data.ipynb
│   │   ├── random_initialization.ipynb
│   │   ├── trained_weights.ipynb
│   │   ├── trained_weights_fine-tuned.ipynb
│   │   └── comparison.ipynb
│   ├── data/                    # Two synthetic CSV datasets
│   ├── models/                  # Saved models for the three experiments
│   └── dictionary/              # Serialized experiment summaries
├── activation_functions/
│   ├── activation_function_plt.py
│   └── media/                   # Rendered animation and intermediate assets
├── temp/rand_val_gen/           # C++ generator and Visual Studio project
└── temp.cpp                    # Earlier random-number generation experiment
```

The `notebookes` spelling above matches the existing directory. Temporary compiler outputs, IDE files, and rendering intermediates are also present in the project folder.

## Getting started

### Python environment

Create an environment from the project root. For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch numpy pandas matplotlib scikit-learn joblib jupyterlab
jupyter lab
```

The project does not currently include a dependency lockfile or pinned package versions. The command above installs the libraries imported by the classification notebooks; compatibility has not been verified against a fresh installation.

### Iris notebooks

Run the notebooks in this order, executing cells from top to bottom:

1. `Iris_neural_net/notebook/data_augmentation.ipynb`
2. `Iris_neural_net/notebook/building_dataset.ipynb`
3. `Iris_neural_net/notebook/iris_mlpclassifier.ipynb`

**Path adjustment required:** with the kernel working directory set to `Iris_neural_net/notebook`, the first two notebooks use `../../data/...`, which points outside `Iris_neural_net`. Change those references to `../data/...` to use the included data directory. The classifier notebook already uses `../data/...` and `../models/...`.

The prepared tensors are included, so you can start directly with the classifier notebook if you do not need to regenerate the dataset. Rerunning preparation or training overwrites the corresponding saved artifacts.

### Synthetic notebooks

Use `Synthetic/notebookes` as the kernel working directory so the existing relative paths resolve correctly.

1. Open `load_&_visualize_data.ipynb` to inspect the data and circular boundary.
2. Run `random_initialization.ipynb` for the baseline.
3. Run `trained_weights.ipynb` to evaluate the saved Iris model without additional training.
4. Run `trained_weights_fine-tuned.ipynb` to train the Iris model on the synthetic task.

The two pretrained experiments load `Iris_neural_net/models/iris_nn_model.pt`. Their notebooks define the `IrisNeuralNetwork` class before loading the serialized model; execute that definition first. Whole-model checkpoints use `weights_only=False` and depend on the Python class definition. Load only trusted checkpoints.

`comparison.ipynb` is an unfinished comparison notebook. It references `../data/synthetic_data.pt`, which is not included, and does not yet implement an aggregate comparison of the saved experiments.

### Optional supporting tools

To regenerate the activation animation, install Manim and its required rendering dependencies, including LaTeX for the scene's `Tex` labels. From `activation_functions`:

```powershell
python -m pip install manim
manim -pqh activation_function_plt.py ActivationFunctionPlot
```

To regenerate synthetic samples, build `temp/rand_val_gen/rand_val_gen.cpp` using a C++ compiler or the included Visual Studio project. The executable writes `random_data.csv` to its working directory. Copy that CSV to `Synthetic/data/synthetic_data.csv` if you want the notebooks to use the new samples.

## Current scope and next steps

The completed work establishes an explicit neural network training pipeline, demonstrates PCA-based augmentation, and records a comparison of random initialization and transferred weights on a nonlinear task.

Useful extensions include:

- Correcting and centralizing notebook data paths.
- Fitting PCA on training samples only and using seeded, stratified splits.
- Completing the comparison notebook using the saved experiment dictionaries.
- Repeating experiments across multiple seeds and reporting variation in results.
- Exploring feature scaling, alternative initializations, hidden-layer sizes, and activation functions.
- Extracting shared network code into a reusable module and checking manual gradients against an independent reference.

The repository currently serves as an educational, notebook-based experiment collection; it does not include a packaged training application, an automated test suite, or a dependency specification.
