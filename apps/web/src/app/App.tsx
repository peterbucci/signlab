import { Route, Routes } from "react-router-dom";

import { AppShell } from "./AppShell";
import { staticPages } from "./routeDefinitions";
import { NotFoundPage, OverviewPage, StaticPage } from "./routes";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        {staticPages.map((page) => (
          <Route key={page.path} path={page.path} element={<StaticPage page={page} />} />
        ))}
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </AppShell>
  );
}
