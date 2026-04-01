# Aluci's AI Project Template

## Overview

This repository provides a reusable template for creating python projects that follow Aluci's engineering and documentation standards. It is intended for internal teams and developers who want to quickly set up a new project with a consistent structure and best practices.

## Setup

### Default Setup (Nvidia GPU and ARM)

```bash
uv sync
```

### AMD Setup

```bash
uv sync --extra amd --no-group torch
```

## README Template

You can use the following as a starting point for your project's README file. Make sure to fill in the placeholders with relevant information about your project.

````markdown
# Name of the project

In case of early development, please include a warning:

> [!WARNING]
> This project is in early development and is not yet ready for production use.

## Overview

This project is a [brief description of the project]. It aims to [describe the main goal or functionality of the project].

## Setup

### Requirements

If there are any requirements, please list them here.

### Recommended Setup (using Miniconda)

```bash
./setup.sh --env-name MyEnv --python-version 3.11
```

### Venv

```bash
./setup.sh --env-name MyEnv --python-version 3.11 --venv
```

## Usage

Always include a usage section, even if it's just a single command.

## License

This project is licensed under the Apache License (Version 2.0).

See the [LICENSE](LICENSE) file for details.

## AI Usage Transparency

This project occasionally uses AI tools such as ChatGPT to assist with development.
For more details, see [AI_USAGE.md](./AI_USAGE.md).
````

## License

This project is licensed under the Apache License (Version 2.0).

See the [LICENSE](LICENSE) file for details.

## AI Usage Transparency

This project occasionally uses AI tools such as ChatGPT to assist with development.
For more details, see [AI_USAGE.md](./AI_USAGE.md).
