/** Entry point of the offline demo bundle: mock backend + the real page component. */
import { createRoot } from "react-dom/client";
import Home from "@/app/page";
import { installMockApi } from "./mockApi";

installMockApi();
const el = document.getElementById("root");
if (el) createRoot(el).render(<Home />);
