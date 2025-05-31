// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// If you are using Tailwind CSS, uncomment these two lines:
// import "./tailwind.css";
// import "./index.css";

// If you are not using Tailwind, but want to import your existing global CSS:
import "./index.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}
const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
