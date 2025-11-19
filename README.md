<h1>
  Query Type Classification Models<br>
  Training & Inference
</h1>

[![UI screenshot](/readme/interface.png)](/readme/interface.png)

## Overview

This service provides tools for training, evaluating, and using models that classify search queries into two categories: ORG (organizations) and LOC (locations).

**`ORG`** includes shops, cafes, restaurants, banks, and other businesses.  
**`LOC`** includes cities, streets, squares, rivers, lakes, and other geographic locations.

This classification can be useful in a search system similar to Google Maps, where the system must decide whether to show an organization card or a map point for each query.

## How to run

1. Copy env.example to .env and set the HF_TOKEN value
2. Run `docker compose up --build` in the project directory
3. Open http://localhost:8000 in a browser

## How it works

The service provides a simple web GUI for training and running classification models. You can choose any of the available models, adjust their training settings, start training, track progress, and run inference directly from the interface.

New models can be added by implementing the same interfaces used by the existing ones. Once a model is registered, the GUI automatically includes it in the list of available options. It also inspects the model configuration, detects the training parameters it supports, and renders the corresponding controls on the page without any extra setup.

## Tests

Tests use pytest and automatically include any new model you implement. This ensures that custom models are integrated into the unified interface and validated for correct behavior.
