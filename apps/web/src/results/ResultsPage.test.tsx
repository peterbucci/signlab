import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResultsPage } from "./ResultsPage";

const evidenceCommit = "5fb187c5a36678f868b14200452fcdc7c8650f94";

describe("research results page", () => {
  it("publishes bounded, pinned evidence without human media", () => {
    const { container } = render(<ResultsPage />);

    expect(screen.getByRole("heading", { level: 1, name: "Research results" })).toHaveFocus();
    expect(screen.getByText("No headline candidate accuracy is published.")).toBeInTheDocument();
    expect(screen.getByText("Signer-held-out logistic baseline")).toBeVisible();
    expect(screen.getByText("0.725")).toBeVisible();
    expect(screen.getByRole("table", { name: "Aggregate test errors by prompt" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Runtime equivalence evidence" })).toBeVisible();
    expect(screen.getByText(/session count is not defined/i)).toBeVisible();
    expect(screen.getByText(/5 warmups followed by 50 measured calls/)).toBeVisible();
    for (const heading of [
      "Exact-candidate locked-test performance",
      "Natural other movement",
      "Natural continuous false activations",
      "Fairness",
      "Robustness",
      "Population performance",
    ]) {
      expect(screen.getByRole("heading", { level: 3, name: heading })).toBeVisible();
    }
    const links = screen.getAllByRole("link");

    expect(
      links.every((link) => link.getAttribute("href")?.includes(`/blob/${evidenceCommit}/docs/`)),
    ).toBe(true);
    links.forEach((link) => expect(link.title).toMatch(/^sha256:[a-f0-9]{64}$/));
    expect(container.querySelector("img, video")).toBeNull();
  });
});
