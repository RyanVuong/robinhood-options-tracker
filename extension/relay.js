/**
 * relay.js — runs in ISOLATED world (extension context)
 *
 * Bridges window.postMessage (from injector.js in MAIN world)
 * to chrome.runtime.sendMessage (to background service worker).
 *
 * Validates the __rh_tracker flag to avoid acting on unrelated messages.
 */
window.addEventListener('message', (event) => {
  // Only accept messages from the same window
  if (event.source !== window) return;
  if (!event.data || !event.data.__rh_tracker) return;

  chrome.runtime.sendMessage(event.data).catch(() => {
    // Service worker may not be running yet; safe to ignore
  });
});
