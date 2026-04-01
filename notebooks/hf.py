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
    # Hugging Face example

    In this notebook, we will explore how to load a Hugging Face model, specifically the SmolLM3-3 using the configuration settings defined in our project.
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
    return (Path,)


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
        AppConfig.load(config_dir=config_dir, config_file=config_file, overrides=overrides)
        setup_logs(config=AppConfig.settings["logs"])


    setup()
    return (AppConfig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load SmolLM3-3B Model
    """)
    return


@app.cell
def _(AppConfig):
    from utils.log_utils import log

    log(message=AppConfig.settings["model"].model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Playground
    """)
    return


@app.cell
def _(AppConfig, mo):
    tokenizer = AppConfig.settings["model"].tok_or_proc
    model = AppConfig.settings["model"].model

    def ai_answer(prompt: str):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=40)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    mo.ui.chat(
        ai_answer,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Here, the rest of your code. Enjoy !
    """)
    return


if __name__ == "__main__":
    app.run()
