import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { initToken } from "./api";
import "./styles.css";

initToken();
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
