/**
 * Shared helpers: loading state, JSON POST, disease upload.
 */
function setLoading(loaderEl, buttonEl, isLoading) {
  if (loaderEl) loaderEl.classList.toggle("visible", isLoading);
  if (buttonEl) buttonEl.disabled = isLoading;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.error || res.statusText || "Request failed";
    throw new Error(msg);
  }
  return data;
}

function showResult(container, html, isError) {
  if (!container) return;
  container.innerHTML = html;
  container.classList.toggle("error", !!isError);
}
