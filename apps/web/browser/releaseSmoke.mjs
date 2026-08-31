import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium, firefox, webkit } from "playwright";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = resolve(appRoot, "dist");
const types = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".wasm", "application/wasm"],
]);
const routes = [
  ["/", "Five gestures, tested honestly."],
  ["/live", "Live recognition"],
  ["/replay", "Deterministic replay"],
  ["/results", "Research results"],
  ["/methodology", "How the research is built"],
  ["/feedback", "Feedback stays local by default"],
  ["/privacy", "Designed for on-device processing"],
  ["/limitations", "What SignLab does—and does not—claim"],
  ["/missing", "Page not found"],
];

function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function startServer() {
  check((await stat(resolve(distRoot, "index.html"))).isFile(), "browser.smoke.build_missing");
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", "http://localhost");
      const relative = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);
      const path = resolve(distRoot, `.${relative}`);
      check(path.startsWith(`${distRoot}${sep}`), "browser.smoke.path_invalid");
      const bytes = await readFile(path);
      response.writeHead(200, {
        "Content-Type": types.get(extname(path)) ?? "application/octet-stream",
      });
      response.end(bytes);
    } catch {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not found");
    }
  });
  await new Promise((accept, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", accept);
  });
  const address = server.address();
  check(typeof address === "object" && address !== null, "browser.smoke.server_invalid");
  return { origin: `http://127.0.0.1:${address.port}`, server };
}

function installBrowserMocks() {
  const media = new WeakMap();
  const state = {
    channels: [],
    constraints: [],
    denyNext: false,
    lastTrack: null,
    stopCount: 0,
  };
  Object.defineProperty(globalThis, "__signlabSmoke", { value: state });
  Object.defineProperty(HTMLMediaElement.prototype, "srcObject", {
    configurable: true,
    get() {
      return media.get(this) ?? null;
    },
    set(value) {
      media.set(this, value);
    },
  });
  const mediaDevices = {
    async enumerateDevices() {
      return [
        { deviceId: "mock-camera", groupId: "mock", kind: "videoinput", label: "Mock camera" },
      ];
    },
    async getUserMedia(constraints) {
      state.constraints.push(constraints);
      if (state.denyNext) {
        state.denyNext = false;
        throw new DOMException("Mock permission denial", "NotAllowedError");
      }
      const events = new EventTarget();
      const track = {
        addEventListener: events.addEventListener.bind(events),
        enabled: true,
        getSettings: () => ({ deviceId: "mock-camera" }),
        removeEventListener: events.removeEventListener.bind(events),
        stop() {
          if (this.stopped) return;
          this.stopped = true;
          state.stopCount += 1;
        },
        stopped: false,
      };
      state.lastTrack = track;
      return { getTracks: () => [track], getVideoTracks: () => [track] };
    },
  };
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: mediaDevices });
  Object.defineProperty(navigator, "sendBeacon", {
    configurable: true,
    value: () => {
      state.channels.push("beacon");
      return false;
    },
  });
  if (typeof EventSource === "function") {
    globalThis.EventSource = new Proxy(EventSource, {
      construct(target, argumentsList) {
        state.channels.push("eventsource");
        return Reflect.construct(target, argumentsList);
      },
    });
  }
}

async function runEngine(name, launcher, origin) {
  const browser = await launcher.launch({ headless: true });
  let context;
  try {
    context = await browser.newContext();
    await context.addInitScript(installBrowserMocks);
    const page = await context.newPage();
    const errors = [];
    const requests = [];
    const sockets = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(`console: ${message.text()}`);
    });
    page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
    page.on("request", (request) => requests.push([request.method(), request.url()]));
    page.on("websocket", (socket) => sockets.push(socket.url()));

    await page.goto(`${origin}/#/`);
    await page.getByRole("heading", { level: 1, name: routes[0][1] }).waitFor();
    for (const [path, heading] of routes.slice(1, -1)) {
      await page.locator(`nav[aria-label="Primary navigation"] a[href="#${path}"]`).click();
      await page.getByRole("heading", { level: 1, name: heading }).waitFor();
    }
    await page.goto(`${origin}/#${routes.at(-1)[0]}`);
    await page.getByRole("heading", { level: 1, name: routes.at(-1)[1] }).waitFor();

    await page.goto(`${origin}/#/live`);
    await page.getByRole("button", { name: "Start camera" }).click();
    await page
      .getByText("Camera is on. Raw video stays on this page and is not saved or uploaded.")
      .waitFor();
    await page.getByRole("button", { name: "Pause preview" }).click();
    check(
      (await page.evaluate(() => globalThis.__signlabSmoke.lastTrack.enabled)) === false,
      `${name}.camera.pause`,
    );
    await page.getByRole("button", { name: "Resume preview" }).click();
    check(
      (await page.evaluate(() => globalThis.__signlabSmoke.lastTrack.enabled)) === true,
      `${name}.camera.resume`,
    );
    await page.getByRole("button", { name: "Stop camera" }).click();
    check(
      (await page.evaluate(() => globalThis.__signlabSmoke.stopCount)) === 1,
      `${name}.camera.stop`,
    );
    const constraints = await page.evaluate(() => globalThis.__signlabSmoke.constraints);
    check(
      constraints.length === 1 &&
        constraints[0].audio === false &&
        typeof constraints[0].video === "object",
      `${name}.camera.constraints`,
    );

    await page.evaluate(() => {
      globalThis.__signlabSmoke.denyNext = true;
    });
    await page.getByRole("button", { name: "Start camera" }).click();
    await page
      .getByText(
        "Camera permission was denied. Allow camera access in site settings, then try again.",
      )
      .waitFor();

    const channels = await page.evaluate(() => globalThis.__signlabSmoke.channels);
    check(errors.length === 0, `${name}.browser.errors: ${errors.join(" | ")}`);
    check(sockets.length === 0 && channels.length === 0, `${name}.network.channel`);
    for (const [method, url] of requests) {
      check(
        method === "GET" && new URL(url).origin === origin,
        `${name}.network.request: ${method} ${url}`,
      );
    }
    process.stdout.write(
      `PASS ${name}: ${routes.length} routes, camera lifecycle, same-origin GET-only shell requests\n`,
    );
  } finally {
    await context?.close().catch(() => undefined);
    await browser.close().catch(() => undefined);
  }
}

const { origin, server } = await startServer();
try {
  for (const [name, launcher] of [
    ["Chromium", chromium],
    ["Firefox", firefox],
    ["WebKit", webkit],
  ]) {
    await runEngine(name, launcher, origin);
  }
} finally {
  await new Promise((accept, reject) =>
    server.close((error) => (error ? reject(error) : accept())),
  );
}
