import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MLP Example

    In this notebook, we will explore how to load a custom model, specifically a Multi-Layer Perceptron (MLP), using the configuration settings defined in our project.

    *Note: A bit of this notebook was inspired by [this tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html).*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Local Modules
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    import subprocess


    def get_project_root() -> Path:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=".",
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode()
        )


    project_root = get_project_root()
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))

    try:
        from utils.seed_utils import set_seed
    except ImportError:
        raise ImportError("Cannot import module. Make sure that the project is on the path")

    SEED = 42
    set_seed(SEED)
    return Path, SEED


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup logs and config

    Here's the configuration used:

    ```yaml
    logs:
      _target_: schemas.log_schema.LogSchema
      sink: ${path_config:run_log}
      level: INFO
      format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {thread.name} | {name}:{function}:{line} - {message}"
      rotation: 10 MB
      retention: 7 days
      colorize: true

    model:
      _target_: models.smol_lm3.smol_lm3_model.SmolLM3Model
    ```

    But we want to use an MLP model instead:

    ```yaml
    model:
      _target_: models.mlp.mlp_model.MLPModel
    ```

    So here we can override the model configuration when loading the settings, as shown below.
    """)
    return


@app.cell
def _(Path):
    from configs.app_config import AppConfig
    from utils.log_utils import setup_logs


    def setup(
        config_dir: str | Path | None = None,
        config_file: str = "settings.yaml",
        overrides: list[str] | None = None,
    ):
        print(f"{config_dir=}, {config_file=}, {overrides=}")
        AppConfig.load(
            config_dir=config_dir,
            config_file=config_file,
            overrides=overrides,
        )
        setup_logs(config=AppConfig.settings["logs"])


    setup(overrides=["model._target_=models.mlp.mlp_model.MLPModel"])
    return (AppConfig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load MLP Model
    """)
    return


@app.cell
def _(AppConfig):
    from utils.log_utils import log

    mlp = AppConfig.settings["model"].model
    log(message=mlp)
    return log, mlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load dataset
    """)
    return


@app.cell
def _():
    import torch
    import torchvision
    import torchvision.transforms as transforms

    from configs.path_config import path_config


    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST(
        path_config.data, train=True, transform=transform, download=True
    )
    validation_set = torchvision.datasets.FashionMNIST(
        path_config.data, train=False, transform=transform, download=True
    )

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=4, shuffle=False
    )

    # Class labels
    classes = (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    )

    # Report split sizes
    print("Training set has {} instances".format(len(training_set)))
    print("Validation set has {} instances".format(len(validation_set)))
    return (
        classes,
        path_config,
        torch,
        torchvision,
        training_loader,
        validation_loader,
    )


@app.cell
def _(classes, torchvision, training_loader):
    import matplotlib.pyplot as plt
    import numpy as np

    # Helper function for inline image display
    def matplotlib_imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap='Greys')
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
    _dataiter = iter(training_loader)
    _images, _labels = next(_dataiter)
    img_grid = torchvision.utils.make_grid(_images)
    matplotlib_imshow(img_grid, one_channel=True)
    # Create a grid from the images and show them
    print('  '.join((classes[_labels[j]] for j in range(4))))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup training MLP
    """)
    return


@app.cell
def _(Path, log, mlp, path_config, torch):
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    lr = 0.001
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    num_workers = 2 if device.type == 'cuda' else 0
    # Hyperparameters
    log(message=f'Using device: {device}')
    log(message=torch.cuda.is_available())
    model = mlp.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      # safe default for notebooks
    def run_epoch(model, loader, optimizer=None):
        is_train = optimizer is not None
        model.train(is_train)
        total_loss, total_correct, total = (0.0, 0, 0)
    # Model
        for x, y in loader:
            x, y = (x.to(device), y.to(device))
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss = total_loss + loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct = total_correct + (preds == y).sum().item()
            total = total + x.size(0)
        avg_loss = total_loss / total
        acc = total_correct / total
        return (avg_loss, acc)
    best_val_acc = 0.0
    save_dir = Path(path_config.models) / 'mlp_fashionmnist'  # MLP has an internal Flatten
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / 'checkpoint.pth'
    tensorboard_log_path = Path(path_config.tensorboard) / 'mlp_fashionmnist'
    # Setup TensorBoard
    summary_writer = SummaryWriter(log_dir=tensorboard_log_path)
    return (
        best_val_acc,
        checkpoint_path,
        device,
        epochs,
        model,
        optimizer,
        run_epoch,
        summary_writer,
        tensorboard_log_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train MLP on FashionMNIST
    """)
    return


@app.cell
def _(
    AppConfig,
    SEED,
    best_val_acc,
    checkpoint_path,
    epochs,
    log,
    model,
    optimizer,
    run_epoch,
    summary_writer,
    tensorboard_log_path,
    torch,
    training_loader,
    validation_loader,
):
    from tqdm import tqdm
    from utils.tensorboard_utils import launch_tensorboard
    launch_tensorboard(log_dir=tensorboard_log_path)
    for epoch in tqdm(range(1, epochs + 1), desc='Training Progress'):
        train_loss, train_acc = run_epoch(model, training_loader, optimizer)
        val_loss, val_acc = run_epoch(model, validation_loader, optimizer=None)
        log(message=f'Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}')
        summary_writer.add_scalars(main_tag='accuracy', tag_scalar_dict={'train': train_acc, 'val': val_acc}, global_step=epoch)
        summary_writer.add_scalars(main_tag='loss', tag_scalar_dict={'train': train_loss, 'val': val_loss}, global_step=epoch)
        if val_acc > best_val_acc:
            best_val_acc_1 = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'config': {**AppConfig.settings['model'].config, 'seed': SEED}}, checkpoint_path)
            log(message=f'\tSaved new best (val_acc={best_val_acc_1:.4f})')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict with loaded checkpoint
    """)
    return


@app.cell
def _(
    AppConfig,
    checkpoint_path,
    classes,
    device,
    log,
    torch,
    validation_loader,
):
    loaded = AppConfig.settings['model'].model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    loaded.load_state_dict(state['model_state_dict'])
    _dataiter = iter(validation_loader)
    _images, _labels = next(_dataiter)
    _images, _labels = (_images.to(device), _labels.to(device))
    log(message=f'Ground truth: {', '.join((classes[_labels[j]] for j in range(4)))}')
    outputs = loaded(_images)
    _, preds = torch.max(outputs, 1)
    log(message=f'Predicted: {', '.join((classes[preds[j]] for j in range(4)))}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Here, the rest of your code. Enjoy !
    """)
    return


if __name__ == "__main__":
    app.run()
