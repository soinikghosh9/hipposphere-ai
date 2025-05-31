import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ mode }) => {
  // Load any .env variables if you have them (like GEMINI_API_KEY)
  const env = loadEnv(mode, process.cwd(), "");

  // If we are in production (i.e. `npm run build`), use the GitHub Pages base.
  // Otherwise (dev mode), use the root "/" so localhost:5173 works normally.
  const isProduction = mode === "production";
  const basePath = isProduction ? "/hipposphere-ai/" : "/";

  return {
    base: basePath,
    plugins: [react()],
    resolve: {
      alias: {
        "@components": path.resolve(__dirname, "src/components"),
        "@assets": path.resolve(__dirname, "src/assets"),
        "@screens": path.resolve(__dirname, "src/screens"),
        "@services": path.resolve(__dirname, "src/services"),
      },
    },
    define: {
      "process.env": {
        GEMINI_API_KEY: JSON.stringify(env.GEMINI_API_KEY),
      },
    },
    server: {
      port: 5173,
    },
  };
});
