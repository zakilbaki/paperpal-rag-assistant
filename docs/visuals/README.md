# Documentation visuals

The SVGs in `../assets/` are generated from `visual_spec.json`. Diagram geometry is explicit and editable. Recorded-result plots include their source and evaluation scope.

```bash
python docs/visuals/render.py
```

Python 3.10+ is required. Result plots use Matplotlib 3.10.9 (`python -m pip install matplotlib==3.10.9`); no training or data download is performed. The diagram renderer uses the standard library.

Colors distinguish stored data (teal), models (violet), processing (white), and outputs (blue). Every SVG has an explicit background and accessible title.
