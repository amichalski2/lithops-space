/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the Lithops API. Defaults to the local backend. */
  readonly VITE_API_URL?: string;
  /** Run the landing page opens for replay. Without it the landing offers a fresh run instead. */
  readonly VITE_DEMO_RUN_ID?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface Window {
  __LITHOPS_CONFIG__?: {
    demoRunId?: string | null;
  };
}
