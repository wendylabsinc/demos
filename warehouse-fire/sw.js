self.addEventListener('push',event=>{
  const data=event.data?event.data.json():{title:'Wendy fire demo',body:'An alert arrived.'};
  event.waitUntil(self.registration.showNotification(data.title,{body:data.body,tag:`fire-${data.incident_id||'test'}`,data:{url:'/'},requireInteraction:true}));
});
self.addEventListener('notificationclick',event=>{
  event.notification.close();
  event.waitUntil(clients.matchAll({type:'window',includeUncontrolled:true}).then(windows=>{
    const target=windows.find(w=>new URL(w.url).origin===self.location.origin);
    return target?target.focus():clients.openWindow('/');
  }));
});
