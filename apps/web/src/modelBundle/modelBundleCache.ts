const CACHE = "signlab-model-bundles";
const ROOT = "https://model-bundle-cache.signlab.invalid/v1";
const POINTER = `${ROOT}/pointer`;
const FORMAT = "signlab-model-bundle-pointer/1";
const SHA256 = /^sha256:[0-9a-f]{64}$/;

type Storage = Pick<CacheStorage, "open">;
type Pointer = {
  readonly format: typeof FORMAT;
  readonly active: string;
  readonly previous: string | null;
};
type PointerRead = Pointer | "missing" | "unsupported";

export type ModelBundleCacheState = Pick<Pointer, "active" | "previous">;
type Activation = { state: ModelBundleCacheState | null; saved: boolean };
export interface CachedModelBundle {
  readonly identity: string;
  readonly manifest: Blob;
  readonly state: ModelBundleCacheState | null;
  asset(role: string): Promise<Blob | null>;
}

const url = (identity: string, name: string) =>
  `${ROOT}/bundles/${encodeURIComponent(identity)}/${encodeURIComponent(name)}`;

function validPointer(value: unknown): value is Pointer {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const pointer = value as Record<string, unknown>;
  return (
    pointer.format === FORMAT &&
    typeof pointer.active === "string" &&
    SHA256.test(pointer.active) &&
    (pointer.previous === null ||
      (typeof pointer.previous === "string" && SHA256.test(pointer.previous))) &&
    pointer.active !== pointer.previous
  );
}

async function readPointer(cache: Cache): Promise<PointerRead> {
  try {
    const response = await cache.match(POINTER);
    if (response === undefined) return "missing";
    const value: unknown = await response.json();
    return validPointer(value) ? value : "unsupported";
  } catch {
    return "unsupported";
  }
}

const stateOf = (value: PointerRead) => (typeof value === "string" ? null : value);

function nextPointer(current: ModelBundleCacheState | null, identity: string): Pointer {
  return {
    format: FORMAT,
    active: identity,
    previous: current?.active === identity ? current.previous : (current?.active ?? null),
  };
}

const putPointer = async (cache: Cache, pointer: Pointer) =>
  await cache.put(
    POINTER,
    new Response(JSON.stringify(pointer), { headers: { "content-type": "application/json" } }),
  );

export class ModelBundleCache {
  constructor(private readonly storage: Storage | undefined) {}

  async read(identity: string): Promise<CachedModelBundle | null> {
    if (!SHA256.test(identity)) return null;
    const cache = await this.open();
    if (cache === null) return null;
    return await this.entry(cache, identity, stateOf(await readPointer(cache)));
  }

  async saveAndActivate(bundle: {
    identity: string;
    manifest: Blob;
    bytesByRole: Readonly<Record<string, Blob>>;
  }): Promise<Activation> {
    return await this.store(bundle.identity, async (cache) => {
      await cache.put(
        url(bundle.identity, "manifest"),
        new Response(await bundle.manifest.arrayBuffer()),
      );
      for (const [role, bytes] of Object.entries(bundle.bytesByRole)) {
        await cache.put(url(bundle.identity, role), new Response(await bytes.arrayBuffer()));
      }
    });
  }

  async activate(identity: string): Promise<Activation> {
    return await this.store(identity, () => Promise.resolve());
  }

  private async store(
    identity: string,
    stage: (cache: Cache) => Promise<void>,
  ): Promise<Activation> {
    const cache = await this.open();
    if (cache === null || !SHA256.test(identity)) return { state: null, saved: false };
    const stored = await readPointer(cache);
    const current = stateOf(stored);
    if (stored === "unsupported") return { state: current, saved: false };
    try {
      await stage(cache);
      if (current?.active === identity) return { state: current, saved: true };
      const next = nextPointer(current, identity);
      await putPointer(cache, next);
      const obsolete = current?.previous;
      if (obsolete !== undefined && obsolete !== null && !Object.values(next).includes(obsolete)) {
        await this.remove(cache, obsolete).catch(() => undefined);
      }
      return { state: next, saved: true };
    } catch {
      return { state: current, saved: false };
    }
  }

  async rollback(identity: string): Promise<ModelBundleCacheState | null> {
    const cache = await this.open();
    if (cache === null) return null;
    const stored = await readPointer(cache);
    if (typeof stored === "string" || stored.previous !== identity) return null;
    const next: Pointer = { format: FORMAT, active: identity, previous: stored.active };
    try {
      await putPointer(cache, next);
      return next;
    } catch {
      return null;
    }
  }

  async readSelected(key: "active" | "previous"): Promise<CachedModelBundle | null> {
    const cache = await this.open();
    if (cache === null) return null;
    const pointer = await readPointer(cache);
    if (typeof pointer === "string" || pointer[key] === null) return null;
    return await this.entry(cache, pointer[key], pointer);
  }

  private async entry(
    cache: Cache,
    identity: string,
    state: ModelBundleCacheState | null,
  ): Promise<CachedModelBundle | null> {
    try {
      const response = await cache.match(url(identity, "manifest"));
      if (response === undefined) return null;
      return {
        identity,
        manifest: await response.blob(),
        state,
        asset: async (role) => {
          try {
            return (await cache.match(url(identity, role)))?.blob() ?? null;
          } catch {
            return null;
          }
        },
      };
    } catch {
      return null;
    }
  }

  private async open(): Promise<Cache | null> {
    try {
      return (await this.storage?.open(CACHE)) ?? null;
    } catch {
      return null;
    }
  }

  private async remove(cache: Cache, identity: string): Promise<void> {
    const prefix = `${ROOT}/bundles/${encodeURIComponent(identity)}/`;
    for (const request of await cache.keys()) {
      if (request.url.startsWith(prefix)) await cache.delete(request);
    }
  }
}
