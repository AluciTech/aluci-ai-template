from pathlib import Path
from fire import Fire
from configs.app_config import AppConfig
from utils.log_utils import log, setup_logs


def run(
    config_dir: str | Path | None = None,
    config_file: str = "settings.yaml",
    overrides: list[str] | None = None,
):
    # Setup config and logs
    AppConfig.load(config_dir=config_dir, config_file=config_file, overrides=overrides)
    setup_logs()

    # Playground
    tokenizer = AppConfig.settings["model"].tok_or_proc
    model = AppConfig.settings["model"].model

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    log(message=tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))


if __name__ == "__main__":
    Fire(run)
