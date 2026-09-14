/*
  PIERBITE PWA — DEVELOPMENT SERVICE WORKER

  During development, always get PierBite pages from the network
  instead of serving old cached HTML.

  This prevents older versions of the homepage and pier navigation
  from reappearing while the PWA is being built.

  A production offline-cache system will be added after the
  PierBite pages and navigation are finalized.
*/

const PIERBITE_DEVELOPMENT_VERSION = "PierBite-PWA-Development-v2";

self.addEventListener("install", function () {
  self.skipWaiting();
});

self.addEventListener("activate", function (event) {

  event.waitUntil(

    caches.keys()

      .then(function (cacheNames) {

        return Promise.all(

          cacheNames.map(function (cacheName) {
            return caches.delete(cacheName);
          })

        );

      })

      .then(function () {
        return self.clients.claim();
      })

  );

});

self.addEventListener("fetch", function (event) {

  if (event.request.method !== "GET") {
    return;
  }

  const requestURL = new URL(event.request.url);

  /*
    Leave outside data sources such as PierBite's GitHub
    data.json and photos.json completely alone.
  */
  if (requestURL.origin !== self.location.origin) {
    return;
  }

  /*
    During development, always request the current file.
    Do not substitute an older PWA cache.
  */
  event.respondWith(
    fetch(event.request, {
      cache: "no-store"
    })
  );

});
