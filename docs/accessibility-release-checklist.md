# Public demo accessibility release check

Story: #48  
Scope: public overview, live demo, privacy, and not-found routes

## Automated checks

- [x] Axe WCAG A/AA and best-practice smoke passes on representative routes.
- [x] `color-contrast` and `link-in-text-block` are checked in a real browser; axe documents
      them as unreliable in jsdom.
- [x] Web tests, type checks, lint, formatting, and production builds pass.

## Keyboard and focus

- [x] Skip link is visible on focus, moves focus to main, and does not change the hash route.
- [x] Route changes focus the new page heading.
- [x] Start/Stop and Pause/Resume work by keyboard and retain focus when labels change.
- [x] Menu, links, camera choices, mirror option, retry, and diagnostics have visible focus.
- [x] Focus order follows the visual reading order with no trap.

## Visual and responsive checks

- [x] Text and focus indicators meet WCAG 2.2 AA contrast.
- [x] Meaning does not depend on color alone.
- [x] Pages reflow at 320 px and 390 px without horizontal scrolling or hidden controls.
- [x] Pages remain usable at 200% browser zoom and at a 1440×900 desktop viewport.
- [x] Reduced-motion preference removes meaningful motion.

## Semantics and announcements

- [x] Browser accessibility tree exposes landmarks, headings, labels, and control groups clearly.
- [x] Readiness, failures, and completed predictions use bounded polite status updates.
- [x] Camera preview has an accurate name; no canvas or landmark overlay exists in this release.
- [x] Narrator or NVDA listen-through confirms understandable names, focus, and announcement pace.

## Evidence

- Commit: tracked by Story #48 and its pull request.
- Automated run: 107 tests; axe found zero violations on four representative routes.
- Browser matrix: Chromium at 320, 390, 640 zoom-equivalent, and 1440×900; no overflow.
- Contrast: primary text 4.85:1; focus ring 3.50:1 or better; gradient label 7.55:1.
- Screen-reader walkthrough: Windows Narrator, user-confirmed 2026-08-31.
