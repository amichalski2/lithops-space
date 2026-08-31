import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // WSL cannot deliver inotify events for files on /mnt/c (drvfs), so the watcher never sees an
    // edit and HMR silently keeps serving the cached module. Polling is the only reliable option
    // for this checkout; drop it if the repo ever moves onto the Linux filesystem.
    watch: { usePolling: true, interval: 300 },
  },
});
