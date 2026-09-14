const PIERBITE_CACHE = "pierbite-pwa-test-v1";

const APP_FILES = [
  "./",
  "./index.html",
  "./PierBite-Web-App-Manifest.webmanifest",
  "./PierBite-App-Icon-192x192.png",
  "./PierBite-App-Icon-512x512.png",
  "./PierBite-Apple-Touch-Icon-180x180.png"
];

self.addEventListener("install", function (event) {
  event.waitUntil(
    caches.open(PIERBITE_CACHE).then(function (cache) {
      return cache.addAll(APP_FILES);
    })
  );

  self.skipWaiting();
});

self.addEventListener("activate", function (event) {
  event.waitUntil(
    caches.keys().then(function (cacheNames) {
      return Promise.all(
        cacheNames.map(function (cacheName) {
          if (cacheName !== PIERBITE_CACHE) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );

  self.clients.claim();
});

self.addEventListener("fetch", function (event) {
  if (event.request.method !== "GET") {
    return;
  }

  const requestURL = new URL(event.request.url);

  if (requestURL.origin !== self.location.origin) {
    return;
  }

  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request)
        .then(function (response) {
          const responseCopy = response.clone();

          caches.open(PIERBITE_CACHE).then(function (cache) {
            cache.put(event.request, responseCopy);
          });

          return response;
        })
        .catch(function () {
          return caches.match("./index.html");
        })
    );

    return;
  }

  event.respondWith(
    caches.match(event.request).then(function (cachedResponse) {
      const networkResponse = fetch(event.request)
        .then(function (response) {
          const responseCopy = response.clone();

          caches.open(PIERBITE_CACHE).then(function (cache) {
            cache.put(event.request, responseCopy);
          });

          return response;
        })
        .catch(function () {
          return cachedResponse;
        });

      return cachedResponse || networkResponse;
    })
  );
});
