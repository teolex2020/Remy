const CACHE_NAME = "remy-v1.12";
const ASSETS_TO_CACHE = [
  "/",
  "/index.html",
  "/css/main.css",
  "/js/app.js",
  "/js/api-client.js",
  "/js/activity.js",
  "/js/approval.js",
  "/js/chat.js",
  "/js/history.js",
  "/js/knowledge.js",
  "/js/memory.js",
  "/js/tasks.js",
  "/js/stats.js",
  "/js/settings.js",
  "/js/graph.js",
  "/js/vendor/cytoscape.min.js",
  "/img/icon.svg",
  "/manifest.json"
];

self.addEventListener("install", (event) => {
  self.skipWaiting(); // Force active immediately
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;

  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request).catch(() => {
          // If offline and request is for page, return index or offline page
          if (event.request.mode === 'navigate') {
              return caches.match('/index.html');
          }
      });
    })
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      self.clients.claim(), // Take control of open clients immediately
      caches.keys().then((keyList) => {
        return Promise.all(
          keyList.map((key) => {
            if (key !== CACHE_NAME) {
              return caches.delete(key);
            }
          })
        );
      })
    ])
  );
});

// ============== Push Notifications ==============

self.addEventListener("push", (event) => {
  const data = event.data?.json() || { title: "Remy", body: "New notification" };
  event.waitUntil(
    self.registration.showNotification(data.title || "Remy", {
      body: data.body || "",
      icon: "/static/img/icon.svg",
      badge: "/static/img/icon.svg",
      data: { url: data.url || "/" },
    })
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: "window" }).then((list) => {
      for (const client of list) {
        if (client.url.includes(self.location.origin)) {
          return client.focus();
        }
      }
      return clients.openWindow(event.notification.data.url || "/");
    })
  );
});
