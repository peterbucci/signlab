import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { App } from "./App";

const routeExpectations = [
  { path: "/", heading: "Five gestures, tested honestly.", title: "Overview" },
  { path: "/live", heading: "Live recognition", title: "Live demo" },
  { path: "/replay", heading: "Deterministic replay", title: "Replay" },
  { path: "/results", heading: "Research results", title: "Results" },
  { path: "/methodology", heading: "How the research is built", title: "Methodology" },
  { path: "/feedback", heading: "Feedback stays local by default", title: "Feedback" },
  { path: "/privacy", heading: "Designed for on-device processing", title: "Privacy" },
  {
    path: "/limitations",
    heading: "What SignLab does—and does not—claim",
    title: "Limitations",
  },
] as const;

describe("static application routes", () => {
  it.each(routeExpectations)(
    "renders $path without making a network request",
    ({ path, heading, title }) => {
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      render(
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>,
      );

      expect(screen.getByRole("heading", { level: 1, name: heading })).toBeInTheDocument();
      expect(screen.getByRole("navigation", { name: "Primary navigation" })).toBeInTheDocument();
      expect(document.title).toBe(`${title} | SignLab`);
      expect(fetchSpy).not.toHaveBeenCalled();

      fetchSpy.mockRestore();
    },
  );

  it("opens and closes the compact navigation", () => {
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );

    const menuButton = screen.getByRole("button", { name: "Open navigation" });
    expect(menuButton).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(menuButton);
    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });

  it("marks only the current route in navigation", () => {
    render(
      <MemoryRouter initialEntries={["/live"]}>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByRole("link", { name: "SignLab overview" })).not.toHaveAttribute(
      "aria-current",
    );
    expect(screen.getByRole("link", { name: "Live demo" })).toHaveAttribute("aria-current", "page");
  });

  it("renders a useful not-found page", () => {
    render(
      <MemoryRouter initialEntries={["/missing"]}>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByRole("heading", { level: 1, name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Return to the overview" })).toHaveAttribute(
      "href",
      "/",
    );
  });
});
