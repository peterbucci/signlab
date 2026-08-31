/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SIGNLAB_MODEL_BUNDLE_URL?: string;
  readonly VITE_SIGNLAB_TRUSTED_MODEL_MANIFEST_SHA256?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
