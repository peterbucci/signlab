import { Route, Routes } from "react-router-dom";

import { AppShell } from "./AppShell";
import { FeedbackPage } from "../feedback/Feedback";
import { ResultsPage } from "../results/ResultsPage";
import { staticPages } from "./routeDefinitions";
import { LivePage, NotFoundPage, OverviewPage, StaticPage } from "./routes";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/live" element={<LivePage />} />
        {staticPages.map((page) => (
          <Route
            key={page.path}
            path={page.path}
            element={
              page.path === "/feedback" ? (
                <FeedbackPage />
              ) : page.path === "/results" ? (
                <ResultsPage />
              ) : (
                <StaticPage page={page} />
              )
            }
          />
        ))}
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </AppShell>
  );
}
