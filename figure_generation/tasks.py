from invoke import task
from pathlib import Path

basepath = Path.cwd()

open_cmd = "open"

fig_names = {
    "overview": "figure_data_overview",
    "sorting": "figure_spike_sorting",
    "annotations": "figure_annotations_overview",

}

@task
def convertpngpdf(c, fig):
    _convertsvg2pdf(c, fig)
    _convertpdf2png(c, fig)


########################################################################################
# Helpers
########################################################################################
@task
def _convertsvg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.svg")
    for path in pathlist:
        c.run(f"inkscape {str(path)} --export-pdf={str(path)[:-4]}.pdf")


@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.pdf")
    for path in pathlist:
        c.run(
            f'inkscape {str(path)} --export-png={str(path)[:-4]}.png -b "white" --export-dpi=600'
        )