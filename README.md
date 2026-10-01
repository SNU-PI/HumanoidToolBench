# HumanoidToolBench project page

This branch holds the static site served at
<https://snu-pi.github.io/HumanoidToolBench/>. The code lives on
[`main`](https://github.com/SNU-PI/HumanoidToolBench).

- `index.html` is the whole page and is meant to be edited by hand.
- `static/css/style.css` and `static/js/main.js` hold the styling and the
  small amount of script (clip playback, the standard/decoy switch, table
  tooltips, copy button, and the GitHub star and ToolBook like counts,
  which are read from the public APIs when the page loads).
- `static/img/` holds the figures, `static/video/` the overview video and
  `static/video/clips/` the short clips with their poster images.

A cell in the result tables is `<td style="--v:76">76</td>`: `--v` is the
value in percent and sets the shade. To edit the author list or add the BibTeX
entry, search `index.html` for `AUTHORS` and for `id="citation"`.

Preview locally with `python3 -m http.server` in this directory.

Fonts are self-hosted Inter under the SIL Open Font License (`static/fonts/OFL.txt`).
Images and clip posters are WebP; `og.jpg` and the icons stay JPEG/PNG for link previews.
