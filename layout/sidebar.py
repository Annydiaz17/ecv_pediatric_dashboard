"""
Sidebar de navegación lateral del dashboard.
Usa íconos profesionales MDI vía Dash Iconify.
"""
from dash import html, dcc
from utils.icons import icon, SIDEBAR_NAV, LOGO_ICON


def create_sidebar():
    """Crea la barra de navegación lateral."""
    nav_links = []
    for item in SIDEBAR_NAV:
        nav_links.append(
            dcc.Link(
                html.Div([
                    html.Span(
                        icon(item["icon"], size=20),
                        className="nav-icon",
                    ),
                    html.Span(item["label"]),
                ]),
                href=item["page"],
                className="nav-link",
                id=f"nav-{item['page'].replace('/', '') or 'home'}",
            )
        )

    sidebar = html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span(
                    icon(LOGO_ICON, size=32, color="#8bb5d8"),
                    className="logo-icon",
                ),
                "ECV Pediátrico"
            ], className="sidebar-logo"),
            html.Div("Sistema de Apoyo Clínico", className="sidebar-subtitle"),
        ], className="sidebar-header"),

        # Navigation
        html.Div([
            html.Div("Navegación", className="nav-section-title"),
            *nav_links
        ], className="sidebar-nav"),

        # Footer
        html.Div([
            html.P("© 2026 Ana Díaz · Yeison Ramírez"),
            html.P("Fundación Universitaria de Popayán"),
            html.P("Ingeniería de Sistemas"),
            html.P("Todos los derechos reservados",
                   style={"marginTop": "6px", "fontSize": "9px",
                          "opacity": "0.7"}),
        ], className="sidebar-footer"),

    ], className="sidebar")

    return sidebar
