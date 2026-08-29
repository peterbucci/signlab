import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";

import { navigationItems } from "./routeDefinitions";

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const [menuOpen, setMenuOpen] = useState(false);

  const closeMenu = () => {
    setMenuOpen(false);
  };

  return (
    <div className="site-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <header className="site-header">
        <div className="header-bar">
          <NavLink className="brand" to="/" end aria-label="SignLab overview" onClick={closeMenu}>
            <span className="brand-mark" aria-hidden="true">
              SL
            </span>
            <span>
              <strong>SignLab</strong>
              <small>Gesture recognition research</small>
            </span>
          </NavLink>

          <span className="prototype-label">Research prototype</span>

          <button
            className="menu-toggle"
            type="button"
            aria-controls="primary-navigation"
            aria-expanded={menuOpen}
            onClick={() => {
              setMenuOpen((open) => !open);
            }}
          >
            <span aria-hidden="true">{menuOpen ? "Close" : "Menu"}</span>
            <span className="visually-hidden">
              {menuOpen ? "Close navigation" : "Open navigation"}
            </span>
          </button>
        </div>

        <nav
          id="primary-navigation"
          className={menuOpen ? "primary-navigation is-open" : "primary-navigation"}
          aria-label="Primary navigation"
        >
          {navigationItems.map((item) => (
            <NavLink key={item.path} to={item.path} end={item.path === "/"} onClick={closeMenu}>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main id="main-content">{children}</main>

      <footer className="site-footer">
        <p>SignLab is a research prototype, not a sign-language translator.</p>
        <nav aria-label="Project information">
          <NavLink to="/privacy">Privacy</NavLink>
          <NavLink to="/limitations">Limitations</NavLink>
        </nav>
      </footer>
    </div>
  );
}
