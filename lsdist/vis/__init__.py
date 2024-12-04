import io
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors


def init_mpl(font_path: Path, use_latex: bool = False, color_cycle: str = None):
    # https://github.com/garrettj403/SciencePlots
    try:
        import scienceplots

        style_lst = ["science", "nature"]
        if color_cycle is not None:
            style_lst.append(color_cycle)
        if not use_latex:
            style_lst.append("no-latex")
        plt.style.use(style_lst)
    except ImportError as err:
        warnings.warn(
            f"Please run 'conda install -c conda-forge scienceplots' "
            f"to enable SciencePlots, now disabled."
        )
    default_color_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = [mcolors.to_rgb(color["color"]) for color in default_color_cycle]

    font_path = Path(font_path).expanduser()
    # https://zhuanlan.zhihu.com/p/501395717
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_name,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    mpl.rcParams.update(
        {
            "figure.figsize": (3, 1.5),
            "figure.dpi": 150,
        }
    )

    # The font above needs to be installed in the system
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
        }
    )

    return default_colors, font_prop


def dump_mpl_img(fig: plt.Figure, format: str = "pdf", pad_inches: float = 0) -> bytes:
    with io.BytesIO() as f:
        fig.savefig(f, format=format, transparent=True, pad_inches=pad_inches)
        img = f.getvalue()
    return img
