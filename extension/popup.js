chrome.runtime.sendMessage({ type: 'GET_STATS' }, (stats) => {
  if (!stats) return;

  document.getElementById('captured').textContent = stats.captured;
  document.getElementById('errors').textContent = stats.errors;

  if (stats.lastTs) {
    const d = new Date(stats.lastTs);
    document.getElementById('last').textContent =
      `Last capture: ${d.toLocaleTimeString()}`;
  }
});
