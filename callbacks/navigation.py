"""
Callback de navegación entre páginas — 4 pestañas SEMMA.
"""
from dash import Input, Output, callback, html
from layout.home import create_home_layout
from layout.segmentation import create_segmentation_layout
from layout.model_evaluation import create_model_eval_layout
from layout.xai_simulator import create_xai_simulator_layout


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    """Enruta la URL a la página correspondiente."""
    pages = {
        "/": create_home_layout,
        "/segmentation": create_segmentation_layout,
        "/assessment": create_model_eval_layout,
        "/xai-simulator": create_xai_simulator_layout,
    }

    page_func = pages.get(pathname, create_home_layout)
    return page_func()
