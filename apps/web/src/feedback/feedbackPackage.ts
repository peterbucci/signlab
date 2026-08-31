import type { FeedbackRecord } from "./feedbackStore";

const compareText = (left: string, right: string) => (left < right ? -1 : left > right ? 1 : 0);

function exportRecords(records: readonly FeedbackRecord[], includeLandmarks: boolean) {
  return [...records]
    .sort((left, right) => compareText(left.id, right.id))
    .map((record) => {
      const exported: Record<string, unknown> = { ...record };
      if (!includeLandmarks) {
        delete exported.landmarks;
        exported.consent = { ...record.consent, landmarksIncluded: false };
      }
      return exported;
    });
}

function summarizeExportedRecords(
  records: readonly Record<string, unknown>[],
  landmarksIncluded: boolean,
) {
  const bundleVersions = new Map<string, { id: string; version: string }>();
  for (const record of records) {
    const bundle = record.bundle as { id: string; version: string };
    bundleVersions.set(JSON.stringify([bundle.id, bundle.version]), bundle);
  }
  return {
    recordFormat: "signlab-feedback-record/1",
    recordCount: records.length,
    bundleVersions: [...bundleVersions.values()].sort(
      (left, right) => compareText(left.id, right.id) || compareText(left.version, right.version),
    ),
    fields: [...new Set(records.flatMap((record) => Object.keys(record)))].sort(compareText),
    localConsentScope: "local_feedback_only",
    landmarkRecordCount: records.filter((record) => Object.hasOwn(record, "landmarks")).length,
    landmarksIncluded,
  };
}

export function summarizeFeedbackPackage(
  records: readonly FeedbackRecord[],
  includeLandmarks: boolean,
) {
  return summarizeExportedRecords(exportRecords(records, includeLandmarks), includeLandmarks);
}

export async function createFeedbackPackage(
  records: readonly FeedbackRecord[],
  includeLandmarks: boolean,
  grantedAt = new Date().toISOString(),
  subtle = globalThis.crypto.subtle,
) {
  if (new Date(grantedAt).toISOString() !== grantedAt) throw new Error("Invalid export time");
  const exported = exportRecords(records, includeLandmarks);
  const payloadJson = JSON.stringify(exported);
  const bytes = new TextEncoder().encode(payloadJson);
  const digest = new Uint8Array(await subtle.digest("SHA-256", bytes));
  const payloadSha256 = `sha256:${[...digest]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("")}`;
  return {
    format: "signlab-feedback-package/1",
    manifest: {
      ...summarizeExportedRecords(exported, includeLandmarks),
      exportConsent: {
        statementVersion: "signlab-feedback-export-consent/1",
        granted: true,
        grantedAt,
        scope: "manual_research_review",
        trainingUse: "requires_review_and_approval",
      },
    },
    payloadJson,
    payloadSha256,
  } as const;
}

const browserDownloadTarget = {
  createObjectURL: (blob: Blob) => URL.createObjectURL(blob),
  revokeObjectURL: (url: string) => URL.revokeObjectURL(url),
  click: (url: string, filename: string) => {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  },
};

export async function downloadFeedbackPackage(
  records: readonly FeedbackRecord[],
  includeLandmarks: boolean,
  grantedAt = new Date().toISOString(),
  target = browserDownloadTarget,
) {
  const feedbackPackage = await createFeedbackPackage(records, includeLandmarks, grantedAt);
  const blob = new Blob([`${JSON.stringify(feedbackPackage, null, 2)}\n`], {
    type: "application/json",
  });
  const url = target.createObjectURL(blob);
  try {
    target.click(url, `signlab-feedback-${grantedAt.slice(0, 10)}.signlab-feedback.json`);
  } finally {
    target.revokeObjectURL(url);
  }
  return feedbackPackage;
}
