const CACHE_NAME = 'ninga-cache-v4';
const urlsToCache = [
  './',
  './index.html',
  './logo.png',
  './manifest.json'
];

self.addEventListener('install', event => {
  self.skipWaiting(); // Forces the browser to activate this new version immediately[cite: 1]
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          // Deletes the old cache so returning users see the new update[cite: 1]
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim(); // Takes control of the webpage immediately[cite: 1]
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});