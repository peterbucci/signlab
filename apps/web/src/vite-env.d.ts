/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SIGNLAB_MODEL_BUNDLE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
