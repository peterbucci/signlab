export async function readBoundedResponse(
  response: Response,
  limit: number,
  onOversized: () => never,
): Promise<ArrayBuffer> {
  if (Number(response.headers.get("content-length")) > limit) onOversized();
  const reader = response.body?.getReader();
  if (reader === undefined) return new ArrayBuffer(0);
  const result = new Uint8Array(limit);
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return result.slice(0, total).buffer;
      if (total + value.byteLength > limit) {
        await reader.cancel().catch(() => undefined);
        onOversized();
      }
      result.set(value, total);
      total += value.byteLength;
    }
  } finally {
    reader.releaseLock();
  }
}
