# AGUS Explainer

This is a strictly static, single-page explainer for the AGUS benchmark.
It uses plain HTML, CSS, and vanilla JS. It has no external dependencies, and uses system fonts to maximize performance and aesthetics.

## How to run locally

Because it fetches local data from `data.json`, it must be served over a local HTTP server (simply opening `index.html` via `file://` may trigger CORS/origin restrictions in modern browsers).

Run the following command from this directory:

```bash
python -m http.server 8000
```

Then visit [http://localhost:8000](http://localhost:8000) in your browser.
