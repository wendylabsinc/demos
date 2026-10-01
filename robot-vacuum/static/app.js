import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const viewport = $('viewport');
let home, state, renderer, scene, camera, controls, robot, brush, robotLight, dustMesh, dustShadowMesh;
let trailMesh, trailLine, labels = [], recording = null, viewMode = 'isometric', speed = 1;
let latestRobot = { x: .75, y: .72, yaw: 0 }, visualRobot = { ...latestRobot };
let lastPoll = 0, toastTimer, lastDustSignature = '', lastTrailSignature = '', pollInFlight = false, lastServerError = null;
const clock = new THREE.Clock();
const materials = new Map();
const trailMaterial = new THREE.MeshBasicMaterial({ color: '#839e67', transparent: true, opacity: .115, depthWrite: false, side: THREE.DoubleSide });
const trailLineMaterial = new THREE.LineBasicMaterial({ color: '#739056', transparent: true, opacity: .36, depthWrite: false });
const temp = new THREE.Object3D();
const pos = new THREE.Vector3();
const colors = { green: '#567560', cream: '#efe9d7', oak: '#b58959', dark: '#3a4943', metal: '#596360' };

function material(color, options = {}) {
  const key = color + JSON.stringify(options);
  if (!materials.has(key)) materials.set(key, new THREE.MeshStandardMaterial({ color, roughness: .8, ...options }));
  return materials.get(key);
}
function box(parent, w, h, d, x, y, z, color, options = {}) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), typeof color === 'string' ? material(color, options) : color);
  mesh.position.set(x, y, z); mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh;
}
function cylinder(parent, top, bottom, h, x, y, z, color, segments = 24) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(top, bottom, h, segments), material(color));
  mesh.position.set(x, y, z); mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh;
}
function sphere(parent, x, y, z, sx, sy, sz, color) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10), material(color));
  mesh.position.set(x, y, z); mesh.scale.set(sx, sy, sz); mesh.castShadow = true; parent.add(mesh); return mesh;
}
function group(parent, x = 0, y = 0, z = 0) { const g = new THREE.Group(); g.position.set(x, y, z); parent.add(g); return g; }
function leg(parent, x, z, h, color = '#866846', r = .037) { return cylinder(parent, r, r * .7, h, x, h / 2, z, color, 10); }
function hash(value) { let a = 2166136261; for (const c of String(value)) { a ^= c.charCodeAt(0); a = Math.imul(a, 16777619); } return (a >>> 0) / 4294967295; }
function floorTexture(color, type) {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const ctx = c.getContext('2d'); ctx.fillStyle = color; ctx.fillRect(0, 0, 512, 512);
  if (type === 'tile') {
    ctx.strokeStyle = '#778c7928'; ctx.lineWidth = 3;
    for (let p = 0; p <= 512; p += 128) { ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 512); ctx.moveTo(0, p); ctx.lineTo(512, p); ctx.stroke(); }
    for (let n = 0; n < 1800; n++) { ctx.fillStyle = n % 2 ? '#ffffff11' : '#51624e09'; ctx.fillRect(hash(n + 'x') * 512, hash(n + 'y') * 512, 1, 1); }
  } else {
    for (let n = 0; n < 8; n++) {
      ctx.fillStyle = n % 3 === 0 ? '#fff4c90d' : '#6a46200a'; ctx.fillRect(n * 64, 0, 64, 512);
      ctx.fillStyle = '#6e4f2b23'; ctx.fillRect(n * 64, 0, 1, 512);
      for (let line = 0; line < 28; line++) {
        ctx.strokeStyle = `rgba(119,83,43,${.025 + hash(n * 100 + line) * .03})`; ctx.lineWidth = 1;
        const p = n * 64 + hash(line + 'a' + n) * 64;
        ctx.beginPath(); ctx.moveTo(p, 0); ctx.bezierCurveTo(p - 5, 170, p + 5, 280, p, 512); ctx.stroke();
      }
      ctx.fillStyle = '#6a492426'; ctx.fillRect(n * 64, (n % 3) * 165 + 50, 64, 1);
    }
  }
  const texture = new THREE.CanvasTexture(c); texture.colorSpace = THREE.SRGBColorSpace; texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.anisotropy = 8; return texture;
}
function rugTexture(color) {
  const c = document.createElement('canvas'); c.width = c.height = 512; const ctx = c.getContext('2d');
  ctx.fillStyle = color; ctx.fillRect(0, 0, 512, 512);
  ctx.strokeStyle = '#f6ecd386'; ctx.lineWidth = 10; ctx.strokeRect(17, 17, 478, 478);
  ctx.strokeStyle = '#bca98138'; ctx.lineWidth = 3; ctx.strokeRect(32, 32, 448, 448);
  ctx.strokeStyle = '#75654309'; ctx.lineWidth = 1;
  for (let p = 0; p < 512; p += 4) { ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 512); ctx.moveTo(0, p); ctx.lineTo(512, p); ctx.stroke(); }
  const texture = new THREE.CanvasTexture(c); texture.colorSpace = THREE.SRGBColorSpace; texture.anisotropy = 8; return texture;
}
function mug(parent, x, y, z, color = '#f7eee0') {
  cylinder(parent, .045, .035, .075, x, y + .038, z, color, 16);
  cylinder(parent, .032, .032, .002, x, y + .077, z, '#6f4b34', 16);
  const handle = new THREE.Mesh(new THREE.TorusGeometry(.027, .009, 6, 12), material(color)); handle.position.set(x + .048, y + .045, z); handle.rotation.y = Math.PI / 2; parent.add(handle);
}
function book(parent, x, y, z, w, d, color, rotation = 0) { const g = group(parent, x, y, z); g.rotation.y = rotation; box(g, w, .025, d, 0, 0, 0, color); box(g, w - .008, .018, d - .007, 0, .02, 0, '#eae2c9'); box(g, w, .009, d, 0, .033, 0, color); }
function plant(parent, width, height) {
  const potH = height * .27, radius = width * .35;
  cylinder(parent, radius, radius * .7, potH, 0, potH / 2, 0, '#c29372');
  cylinder(parent, radius * .91, radius * .91, .01, 0, potH, 0, '#6b5541');
  for (let i = 0; i < 8; i++) {
    const angle = i * 2.4, h = height * (.55 + hash(i + 'leaf') * .38);
    const leaf = sphere(parent, Math.cos(angle) * width * .19, h * .75, Math.sin(angle) * width * .19, width * .14, h * .27, width * .08, i % 2 ? '#537753' : '#63815a');
    leaf.rotation.set(Math.sin(angle) * .35, angle, Math.cos(angle) * .35);
    cylinder(parent, .009, .012, h - potH, 0, (h + potH) / 2, 0, '#5a754b', 6);
  }
}
function buildFurniture(f) {
  const [w, d] = f.size, h = f.height, c = f.color || colors.oak;
  const g = group(scene, f.center[0], .038, f.center[1]);
  if (f.kind === 'sofa') {
    box(g, w, .22, d, 0, .21, 0, c);
    box(g, .22, h * .7, d, -w / 2 + .1, h * .6, 0, c);
    for (const z of [-d / 2 + .1, d / 2 - .1]) box(g, w, .25, .2, 0, .49, z, c);
    for (let i = 0; i < 3; i++) { box(g, w * .76, .12, (d - .45) / 3 - .02, .055, .38, (i - 1) * (d - .45) / 3, '#718d7a'); box(g, .13, .3, (d - .45) / 3 - .04, -.3, .55, (i - 1) * (d - .45) / 3, '#678570'); }
    const pillow = box(g, .2, .26, .28, .04, .55, -.66, '#d4ba8b'); pillow.rotation.x = .23; pillow.rotation.z = -.2;
    const pillow2 = box(g, .19, .26, .28, .07, .55, .65, '#e1dfc9'); pillow2.rotation.z = -.3;
    for (const x of [-w * .36, w * .36]) for (const z of [-d * .42, d * .42]) leg(g, x, z, .17, '#66533f');
  } else if (['coffee', 'dining', 'desk'].includes(f.kind)) {
    box(g, w, .065, d, 0, h - .035, 0, c);
    for (const x of [-w * .4, w * .4]) for (const z of [-d * .36, d * .36]) leg(g, x, z, h - .07);
    if (f.kind === 'coffee') { book(g, -.21, h + .006, .12, .31, .23, '#57736d', -.17); mug(g, .32, h, -.19); cylinder(g, .085, .085, .018, .14, h + .011, .14, '#dbc8a3'); }
    if (f.kind === 'dining') { cylinder(g, .16, .12, .07, 0, h + .04, 0, '#e6d6b7'); sphere(g, -.04, h + .1, 0, .06, .06, .06, '#b48a51'); sphere(g, .06, h + .1, .03, .06, .06, .06, '#808e54'); mug(g, -.46, h, -.19, '#c9d5c3'); }
    if (f.kind === 'desk') {
      box(g, .13, .03, .17, .03, h + .015, -.07, '#4f5a54'); box(g, .035, .2, .035, .03, h + .11, -.11, '#525b54');
      box(g, .65, .4, .037, .03, h + .38, -.16, '#434b48'); box(g, .59, .33, .008, .03, h + .39, -.136, '#a7c4b8');
      box(g, .26, .021, .006, -.09, h + .44, -.13, '#d4e2cb'); box(g, .39, .02, .006, .01, h + .4, -.13, '#bed6c7');
      box(g, .41, .025, .145, .03, h + .014, .15, '#e3e0ce'); mug(g, .7, h, .1, '#d0aa7b'); book(g, -.72, h + .008, -.03, .29, .34, '#82977b', .08);
    }
  } else if (f.kind === 'bed') {
    box(g, w, .22, d, 0, .2, 0, '#8f7256'); box(g, w * .97, .21, d * .96, 0, .4, 0, '#f0ede1');
    box(g, w + .06, .82, .13, 0, .45, d / 2 - .04, '#ad9276');
    box(g, w, .035, d * .65, 0, .526, -d * .14, c);
    for (const x of [-w * .25, w * .25]) { const pillow = box(g, w * .38, .115, .43, x, .565, d * .3, '#efede0'); pillow.rotation.x = -.06; }
    box(g, w + .025, .04, .45, 0, .558, -d * .3, '#718c78');
    for (let i = 0; i < 9; i++) box(g, .009, .001, d * .57, -w / 2 + .18 + i * w / 10, .548, -d * .17, '#c5d3b4');
  } else if (f.kind === 'counter') {
    box(g, w, h - .06, d, 0, (h - .06) / 2, 0, c); box(g, w + .04, .065, d + .03, 0, h - .023, 0, '#f3f0e5');
    const alongX = w > d;
    const count = Math.round(Math.max(w, d) / .7);
    for (let i = 0; i < count; i++) {
      const p = -Math.max(w, d) / 2 + (i + .5) * Math.max(w, d) / count;
      if (alongX) { box(g, Math.max(w, d) / count - .02, h - .16, .015, p, h * .47, d / 2 + .01, '#d1d7c9'); box(g, .1, .018, .024, p, h * .77, d / 2 + .026, '#798676'); }
      else { box(g, .015, h - .16, Math.max(w, d) / count - .02, w / 2 + .01, h * .47, p, '#d1d7c9'); box(g, .024, .018, .1, w / 2 + .026, h * .77, p, '#798676'); }
    }
    if (alongX) {
      box(g, .65, .012, .51, -w * .25, h + .017, 0, '#41524c');
      for (const x of [-.16, .16]) for (const z of [-.12, .12]) cylinder(g, .082, .082, .009, -w * .25 + x, h + .029, z, '#829087');
      box(g, .62, .017, .46, w * .22, h + .02, 0, '#9da8a1'); box(g, .51, .019, .35, w * .22, h + .023, 0, '#738f8a');
      cylinder(g, .012, .012, .17, w * .22, h + .1, -.26, '#bfc5ba', 8); box(g, .012, .012, .1, w * .22, h + .18, -.21, '#bfc5ba');
      const herb = group(g, w * .43, h, 0); plant(herb, .15, .3);
    } else { book(g, 0, h + .012, -.4, .23, .31, '#b1815c', .1); mug(g, 0, h, .2); }
  } else if (['chair', 'dining_chair', 'office_chair'].includes(f.kind)) {
    const seatH = f.kind === 'chair' ? .33 : h * .57;
    box(g, w, .09, d, 0, seatH, 0, c); box(g, w, h - seatH, .11, 0, (h + seatH) / 2, -d / 2 + .055, c);
    if (f.kind === 'chair') { box(g, .1, .19, d, -w / 2 + .05, .44, 0, c); box(g, .1, .19, d, w / 2 - .05, .44, 0, c); box(g, w * .7, .055, d * .8, 0, .408, 0, '#d59b79'); }
    if (f.kind === 'office_chair') { cylinder(g, .035, .035, seatH, 0, seatH / 2, 0, '#65716a'); for (let i = 0; i < 5; i++) { const spoke = box(g, .3, .022, .024, 0, .055, 0, '#69716a'); spoke.rotation.y = i * Math.PI / 5; } }
    else for (const x of [-w * .36, w * .36]) for (const z of [-d * .36, d * .36]) leg(g, x, z, seatH - .04);
    if (f.name?.includes('two')) g.rotation.y = Math.PI;
  } else if (f.kind === 'plant') { plant(g, w, h); }
  else if (f.kind === 'tv') {
    box(g, w, h, d, 0, h / 2, 0, c);
    for (let i = 0; i < 4; i++) box(g, w / 4 - .02, h - .09, .018, -w / 2 + (i + .5) * w / 4, h / 2, d / 2 + .01, '#84644b');
    box(g, 1.34, .74, .04, .07, h + .47, -.06, '#424c46'); box(g, 1.25, .65, .011, .07, h + .47, -.033, '#71837b');
    box(g, .038, .14, .045, -.37, h + .08, 0, '#4b534b'); box(g, .038, .14, .045, .46, h + .08, 0, '#4b534b');
    const tinyPlant = group(g, -.86, h, 0); plant(tinyPlant, .16, .3);
  } else if (f.kind === 'nightstand') {
    box(g, w, h, d, 0, h / 2, 0, c); box(g, w * .83, .2, .012, 0, h * .6, d / 2 + .01, '#bda17b'); box(g, .07, .018, .027, 0, h * .65, d / 2 + .024, '#7d6e51');
    cylinder(g, .05, .075, .025, 0, h + .015, 0, '#8f9a7d'); cylinder(g, .012, .012, .2, 0, h + .1, 0, '#b39a6f'); cylinder(g, .12, .18, .17, 0, h + .29, 0, '#eee6cb');
    book(g, -.15, h + .007, .14, .22, .15, '#77917e', .1);
  } else if (f.kind === 'wardrobe') {
    box(g, w, h, d, 0, h / 2, 0, c); for (let i = 0; i < 3; i++) { box(g, .015, h - .05, d / 3 - .025, -w / 2 - .008, h / 2, -d / 2 + (i + .5) * d / 3, '#c4b095'); box(g, .025, .18, .022, -w / 2 - .022, h / 2, -d / 2 + (i + .5) * d / 3 + .2, '#776d53'); }
  } else if (f.kind === 'bookcase') {
    for (const z of [-d / 2 + .035, d / 2 - .035]) box(g, w, h, .07, 0, h / 2, z, c);
    box(g, .045, h, d, w / 2 - .02, h / 2, 0, c);
    for (let row = 0; row < 4; row++) {
      const shelfY = .08 + row * h / 4; box(g, w, .04, d, 0, shelfY, 0, c);
      for (let i = 0; i < 12; i++) { const z = -d / 2 + .13 + i * (d - .25) / 12; box(g, .32, .14 + hash(i + row * 30) * .065, .055, -.05, shelfY + .12, z, ['#83937a', '#c69a7c', '#d4c29b', '#6e8481', '#c1b296'][i % 5]); }
    }
  } else box(g, w, h, d, 0, h / 2, 0, c);
}

function buildHome() {
  const center = new THREE.Vector3(home.width / 2, 0, home.height / 2);
  scene = new THREE.Scene();
  const floorBase = box(scene, home.width + .32, .2, home.height + .32, center.x, -.08, center.z, '#e3e0cd'); floorBase.receiveShadow = true;
  box(scene, home.width + .12, .06, home.height + .12, center.x, .005, center.z, '#eae4d2');
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.ShadowMaterial({ opacity: .13 })); ground.rotation.x = -Math.PI / 2; ground.position.y = -.2; ground.receiveShadow = true; scene.add(ground);
  home.rooms.forEach((room, i) => {
    const [x1, y1, x2, y2] = room.bounds, width = x2 - x1, depth = y2 - y1;
    const texture = floorTexture(room.color, room.floor); texture.repeat.set(width / 4, depth / 4);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), new THREE.MeshStandardMaterial({ map: texture, roughness: .97 }));
    mesh.rotation.x = -Math.PI / 2; mesh.position.set((x1 + x2) / 2, .038, (y1 + y2) / 2); mesh.receiveShadow = true; scene.add(mesh);
    const label = document.createElement('div'); label.className = 'room-label'; label.textContent = room.name; $('room-labels').append(label);
    // Labels sit in the open floor beside furniture so the rooms remain readable from above.
    const labelX = (x1 + x2) / 2, labelY = y1 + depth * .24;
    labels.push({ element: label, position: new THREE.Vector3(labelX, .06, labelY), room });
    const row = document.createElement('div'); row.className = 'room-row'; row.dataset.room = room.name; row.innerHTML = `<span class="room-color" style="background:${room.color}"></span><span>${room.name}</span>`; $('room-list').append(row);
  });
  (home.rugs || []).forEach(r => { const mesh = new THREE.Mesh(new THREE.BoxGeometry(r.size[0], .018, r.size[1]), new THREE.MeshStandardMaterial({ map: rugTexture(r.color), roughness: 1 })); mesh.position.set(r.center[0], .05, r.center[1]); mesh.receiveShadow = true; scene.add(mesh); });
  home.walls.forEach(wall => {
    const exterior = wall.center[0] < .1 || wall.center[0] > home.width - .1 || wall.center[1] < .1 || wall.center[1] > home.height - .1;
    const height = exterior ? .34 : .42;
    box(scene, wall.size[0], height, wall.size[1], wall.center[0], .04 + height / 2, wall.center[1], exterior ? '#e7e8da' : '#ececde');
    box(scene, wall.size[0] + .005, .025, wall.size[1] + .005, wall.center[0], height + .056, wall.center[1], '#f6f5e8');
    box(scene, wall.size[0], .04, wall.size[1] + .012, wall.center[0], .064, wall.center[1], '#bcc4ab');
  });
  home.furniture.forEach(buildFurniture);
  const dock = group(scene, home.dock[0], .045, home.dock[1]); dock.rotation.y = -(home.dock[2] || 0);
  box(dock, .17, .25, .38, -.17, .13, 0, '#eceee3'); box(dock, .045, .16, .32, -.06, .1, 0, '#647266');
  box(dock, .3, .014, .34, .005, .01, 0, '#a4af9b'); box(dock, .004, .015, .07, -.06, .2, 0, '#c1dfa0', { emissive: '#7aab56', emissiveIntensity: .5 });
  const dockLabel = document.createElement('div'); dockLabel.className = 'room-label'; dockLabel.textContent = 'Home base'; $('room-labels').append(dockLabel); labels.push({ element: dockLabel, position: new THREE.Vector3(home.dock[0] + .05, .07, home.dock[1] - .34), dock: true });
  buildRobot();
  const hemi = new THREE.HemisphereLight('#fffff2', '#9bab87', 2.5); scene.add(hemi);
  const sun = new THREE.DirectionalLight('#fff3db', 3.0); sun.position.set(-3, 14, 8); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); sun.shadow.camera.left = -12; sun.shadow.camera.right = 12; sun.shadow.camera.top = 12; sun.shadow.camera.bottom = -12;
  sun.shadow.normalBias = .035; sun.shadow.bias = -.00005; sun.shadow.radius = 4; sun.target.position.copy(center); scene.add(sun, sun.target);
  const fill = new THREE.DirectionalLight('#dbe9de', 1.3); fill.position.set(14, 9, -10); scene.add(fill);
  dustMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(.012, 6, 4), material('#af8954'), 8000); dustMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); dustMesh.count = 0; dustMesh.frustumCulled = false; scene.add(dustMesh);
  dustShadowMesh = new THREE.InstancedMesh(new THREE.CircleGeometry(.02, 8), new THREE.MeshBasicMaterial({ color: '#ad8754', transparent: true, opacity: .21, depthWrite: false }), 8000); dustShadowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); dustShadowMesh.count = 0; dustShadowMesh.frustumCulled = false; scene.add(dustShadowMesh);
  camera = new THREE.OrthographicCamera(-10, 10, 7, -7, .1, 100);
  camera.position.set(center.x + 12, 17, center.z + 14); camera.lookAt(center);
  controls = new OrbitControls(camera, renderer.domElement); controls.target.copy(center); controls.enableDamping = true; controls.dampingFactor = .075; controls.minZoom = .6; controls.maxZoom = 5; controls.maxPolarAngle = Math.PI / 2.2; controls.minPolarAngle = .025; controls.enablePan = true;
  controls.addEventListener('start', () => { if (viewMode === 'follow') { viewMode = 'free'; document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active')); } });
  resize();
}
function buildRobot() {
  robot = group(scene, home.dock[0], .042, home.dock[1]); const r = home.robot.radius;
  cylinder(robot, r, r, .086, 0, .061, 0, '#ecefe7', 64);
  cylinder(robot, r * 1.008, r * 1.008, .023, 0, .03, 0, '#48514c', 64);
  cylinder(robot, r * .95, r * .95, .009, 0, .108, 0, '#fafaf2', 64);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(r * .89, .0018, 5, 64), material('#b4bdb0')); ring.rotation.x = -Math.PI / 2; ring.position.y = .114; robot.add(ring);
  cylinder(robot, .047, .05, .025, -.048, .127, 0, '#c5cfbf', 32);
  cylinder(robot, .042, .042, .015, -.048, .148, 0, '#3c4a42', 32);
  cylinder(robot, .034, .034, .005, -.048, .158, 0, '#edf0e7', 24);
  robotLight = box(robot, .026, .005, .012, .068, .116, 0, '#a9c987', { emissive: '#82b756', emissiveIntensity: .4 });
  cylinder(robot, .009, .009, .003, .092, .117, .021, '#c0cab8', 12);
  for (const z of [-r * .85, r * .85]) { const wheel = cylinder(robot, .033, .033, .023, 0, .025, z, '#39443e', 16); wheel.rotation.x = Math.PI / 2; }
  box(robot, .035, .003, .025, .016, .118, 0, '#b6c1ad');
  brush = group(robot, .113, .017, .135);
  cylinder(brush, .015, .015, .006, 0, 0, 0, '#b49a6c', 12);
  for (let i = 0; i < 3; i++) { const b = group(brush); b.rotation.y = i * Math.PI * 2 / 3; box(b, .083, .003, .012, .055, 0, 0, '#8b805d'); for (let j = 0; j < 3; j++) box(b, .044, .002, .002, .092, 0, (j - 1) * .005, '#6b7658'); }
  const robotGlow = new THREE.Mesh(new THREE.RingGeometry(r * 1.06, r * 1.11, 48), new THREE.MeshBasicMaterial({ color: '#d4dca9', transparent: true, opacity: .32, depthWrite: false })); robotGlow.rotation.x = -Math.PI / 2; robotGlow.position.y = .005; robot.add(robotGlow);
}
function resize() {
  if (!renderer || !camera) return;
  const w = Math.max(1, viewport.clientWidth), h = Math.max(1, viewport.clientHeight), aspect = w / h;
  const height = Math.max(12.5, 17.7 / aspect);
  camera.left = -height * aspect / 2; camera.right = height * aspect / 2; camera.top = height / 2; camera.bottom = -height / 2;
  camera.updateProjectionMatrix(); renderer.setSize(w, h); renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
}
function roomAt(x, y) { return home.rooms.find(r => x >= r.bounds[0] && x <= r.bounds[2] && y >= r.bounds[1] && y <= r.bounds[3]); }
function updateDust(dust) {
  if (!Array.isArray(dust)) return;
  const signature = `${dust.length}:${dust[0]?.id}:${dust.at(-1)?.id}`;
  // The simulation only removes dust until reset, so its remaining count identifies a change.
  if (signature === lastDustSignature) return;
  lastDustSignature = signature;
  dustMesh.count = dustShadowMesh.count = Math.min(dust.length, 8000);
  for (let i = 0; i < dustMesh.count; i++) {
    const d = dust[i], x = Array.isArray(d) ? d[0] : d.x, y = Array.isArray(d) ? d[1] : d.y;
    const noise = hash(d.id ?? i), scale = .75 + noise * 1.1;
    temp.position.set(x, .076, y); temp.rotation.set(0, noise * 6.28, .3); temp.scale.set(scale * 1.5, .5, scale); temp.updateMatrix(); dustMesh.setMatrixAt(i, temp.matrix);
    temp.position.y = .061; temp.rotation.set(-Math.PI / 2, 0, 0); temp.scale.setScalar(scale); temp.updateMatrix(); dustShadowMesh.setMatrixAt(i, temp.matrix);
  }
  dustMesh.instanceMatrix.needsUpdate = dustShadowMesh.instanceMatrix.needsUpdate = true;
}
function updateTrail(trail) {
  if (!Array.isArray(trail)) return;
  const signature = `${trail.length}:${trail.at(-1)?.[0]}:${trail.at(-1)?.[1]}`;
  if (signature === lastTrailSignature) return;
  lastTrailSignature = signature;
  if (trailMesh) { scene.remove(trailMesh, trailLine); trailMesh.geometry.dispose(); trailLine.geometry.dispose(); }
  if (trail.length < 2) return;
  const points = [], ribbon = [], step = Math.max(1, Math.floor(trail.length / 9000));
  for (let i = 0; i < trail.length; i += step) points.push(trail[i]);
  if (points.at(-1) !== trail.at(-1)) points.push(trail.at(-1));
  const centerLine = new Float32Array(points.length * 3), radius = (home.robot.brush_radius || .24) * .8;
  for (let i = 0; i < points.length; i++) { centerLine[i * 3] = points[i][0]; centerLine[i * 3 + 1] = .065; centerLine[i * 3 + 2] = points[i][1]; }
  for (let i = 1; i < points.length; i++) {
    const [ax, az] = points[i - 1], [bx, bz] = points[i], dist = Math.hypot(bx - ax, bz - az); if (dist < .001 || dist > 1) continue;
    const nx = -(bz - az) / dist * radius, nz = (bx - ax) / dist * radius;
    ribbon.push(ax + nx, .063, az + nz, ax - nx, .063, az - nz, bx + nx, .063, bz + nz, ax - nx, .063, az - nz, bx - nx, .063, bz - nz, bx + nx, .063, bz + nz);
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(ribbon, 3));
  trailMesh = new THREE.Mesh(geo, trailMaterial); scene.add(trailMesh);
  const lineGeo = new THREE.BufferGeometry(); lineGeo.setAttribute('position', new THREE.BufferAttribute(centerLine, 3));
  trailLine = new THREE.Line(lineGeo, trailLineMaterial); scene.add(trailLine);
  trailMesh.visible = trailLine.visible = $('trail-toggle').checked;
}
function formatTime(value) { const seconds = Math.max(0, Math.floor(value || 0)); return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; }
function updateUI(s) {
  const coverage = Math.max(0, Math.min(100, Number(s.coverage ?? s.progress ?? 0)));
  $('coverage').textContent = coverage >= 99.95 ? '100' : coverage.toFixed(0);
  $('progress-bar').style.width = `${coverage}%`; $('ring-progress').style.strokeDashoffset = 138.23 * (1 - coverage / 100);
  $('elapsed').textContent = formatTime(s.time ?? s.elapsed); $('distance').textContent = Number(s.distance || 0).toFixed(1); $('battery').textContent = Math.round(s.battery ?? 100);
  const mode = s.status || 'idle', running = ['cleaning', 'returning'].includes(mode), fresh = (s.time ?? s.elapsed ?? 0) === 0 && (s.distance ?? 0) === 0, room = s.room || roomAt(s.robot?.x || 0, s.robot?.y || 0)?.name;
  const descriptions = { idle: ['Ready to clean', 'STANDBY', 'Ready to clean the virtual home.'], cleaning: ['Cleaning in progress', 'CLEANING', `${(s.cleaned || 0).toLocaleString()} dust particles collected`], returning: ['Heading home', 'RETURNING', 'Making its way to the dock.'], paused: ['Cleaning paused', 'PAUSED', 'Resume the cleaning mission.'], docked: [coverage > 90 ? 'Cleaning complete' : 'At the charging dock', 'DOCKED', coverage > 90 ? 'The cleaning mission is complete.' : 'Ready for a new cleaning mission.'], complete: ['Cleaning complete', 'COMPLETE', 'The cleaning mission is complete.'] };
  const text = descriptions[mode === 'paused' && fresh ? 'idle' : mode] || descriptions.idle; $('status-text').textContent = text[0]; $('mission-tag').textContent = text[1]; $('mission-detail').textContent = text[2];
  $('status-dot').classList.toggle('running', running);
  $('start-label').textContent = running ? 'Pause cleaning' : mode === 'paused' && !fresh ? 'Resume cleaning' : 'Start cleaning'; $('start-icon').innerHTML = `<use href="#i-${running ? 'pause' : 'play'}"/>`;
  $('robot-location').textContent = mode === 'idle' || mode === 'docked' || fresh ? 'At the charging dock' : mode === 'returning' ? 'Returning to home base' : room ? `In the ${room.toLowerCase()}` : 'Exploring your home';
  document.querySelectorAll('.room-row').forEach(row => row.classList.toggle('active', row.dataset.room === room && mode === 'cleaning'));
  robotLight.material.color.set(running ? '#aad586' : mode === 'paused' ? '#d3b077' : '#c7dfa2');
  if (s.speed && s.speed !== speed) selectSpeed(s.speed);
}
async function poll() {
  if (pollInFlight) return; pollInFlight = true;
  try {
    const response = await fetch('/api/state', { cache: 'no-store' }); if (!response.ok) throw new Error(`Simulation returned ${response.status}`);
    const s = await response.json(); state = s;
    if (s.robot) latestRobot = { x: Number(s.robot.x), y: Number(s.robot.y), yaw: Number(s.robot.yaw || 0) };
    updateDust(s.dust || []); updateTrail(s.trail || []); updateUI(s);
    if (s.error) {
      $('connection').classList.add('offline'); $('connection').innerHTML = '<span></span>Simulation stopped';
      $('status-text').textContent = 'Simulation stopped'; $('mission-detail').textContent = 'Restart the Wendy app to continue.'; $('status-dot').classList.remove('running');
      if (s.error !== lastServerError) toast(String(s.error), 9000);
    } else { $('connection').classList.remove('offline'); $('connection').innerHTML = '<span></span>Wendy connected'; }
    lastServerError = s.error || null;
  } catch (e) { $('connection').classList.add('offline'); $('connection').innerHTML = '<span></span>Reconnecting'; }
  finally { pollInFlight = false; }
}
function toast(message, duration = 3500) { clearTimeout(toastTimer); $('toast').textContent = message; $('toast').classList.add('visible'); toastTimer = setTimeout(() => $('toast').classList.remove('visible'), duration); }
async function control(action, extra = {}) {
  try { const r = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, ...extra }) }); if (!r.ok) throw new Error(await r.text()); await poll(); return true; }
  catch (e) { toast('The simulation is unavailable. Check that the Wendy app is running.'); return false; }
}
function selectSpeed(n) { speed = Number(n); document.querySelectorAll('[data-speed]').forEach(b => b.classList.toggle('active', Number(b.dataset.speed) === speed)); $('speed-label').textContent = speed === 1 ? 'Real time' : `${speed}× faster`; }
function setView(mode) {
  viewMode = mode; document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view === mode));
  const center = new THREE.Vector3(home.width / 2, 0, home.height / 2); controls.target.copy(center); camera.zoom = 1;
  if (mode === 'top') { camera.position.set(center.x + .001, 25, center.z + .001); camera.up.set(0, 0, -1); }
  else { camera.up.set(0, 1, 0); camera.position.set(center.x + 12, 17, center.z + 14); if (mode === 'follow') camera.zoom = 2.6; }
  camera.lookAt(controls.target); camera.updateProjectionMatrix(); controls.update();
}
function cinema(active) { document.body.classList.toggle('cinema', active); requestAnimationFrame(resize); }
function roundedRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }
function drawRecording() {
  if (!recording) return;
  const { ctx, canvas } = recording, w = canvas.width, h = canvas.height;
  const grad = ctx.createRadialGradient(w * .5, h * .45, 40, w * .5, h * .45, w * .65); grad.addColorStop(0, '#fcfbf3'); grad.addColorStop(1, '#e9edde'); ctx.fillStyle = grad; ctx.fillRect(0, 0, w, h);
  const source = renderer.domElement, scale = Math.min(w / source.width, (h - 50) / source.height), dw = source.width * scale, dh = source.height * scale;
  ctx.drawImage(source, (w - dw) / 2, (h - dh) / 2 + 25, dw, dh);
  ctx.fillStyle = '#34533e'; ctx.font = '600 54px Avenir, sans-serif'; ctx.fillText('sweep.', 70, 87); ctx.fillStyle = '#80916f'; ctx.font = '12px Avenir, sans-serif'; ctx.fillText((home.name || 'MAPLE HOUSE').toUpperCase() + '  /  A WENDY ROBOT SIMULATION', 73, 113);
  ctx.fillStyle = '#fffef4db'; roundedRect(ctx, w - 375, 46, 305, 113, 16); ctx.fill(); ctx.strokeStyle = '#dce2d0'; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = '#35533d'; ctx.font = '500 43px Avenir, sans-serif'; ctx.fillText(`${Math.round(state?.coverage || 0)}%`, w - 350, 102);
  ctx.fillStyle = '#859475'; ctx.font = '13px Avenir, sans-serif'; ctx.fillText('FLOOR CLEANED', w - 348, 130);
  ctx.fillStyle = '#6b835c'; ctx.font = '20px Avenir, sans-serif'; ctx.textAlign = 'right'; ctx.fillText(formatTime(state?.time ?? state?.elapsed), w - 95, 96); ctx.font = '11px Avenir, sans-serif'; ctx.fillText('RUN TIME', w - 95, 123); ctx.textAlign = 'left';
  ctx.fillStyle = '#7b8b6b'; ctx.font = '12px Avenir, sans-serif'; ctx.fillText('SWEEP ONE  ·  AUTONOMOUS CLEANING', 73, h - 43); ctx.textAlign = 'right'; ctx.fillText(`${speed}× SIMULATION SPEED`, w - 70, h - 43); ctx.textAlign = 'left';
}
function toggleRecording() {
  if (recording) { recording.recorder.stop(); return; }
  if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) { toast('This browser does not support video recording. Use Chrome, Edge, or Safari.'); return; }
  try {
    const canvas = document.createElement('canvas'); canvas.width = 1920; canvas.height = 1080;
    const stream = canvas.captureStream(30), chunks = [];
    const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
    const mimeType = types.find(t => MediaRecorder.isTypeSupported(t));
    if (!mimeType) throw new Error('No supported video encoder');
    const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 9_000_000 });
    recording = { canvas, ctx: canvas.getContext('2d'), stream, recorder, started: performance.now() };
    recorder.addEventListener('dataavailable', e => { if (e.data.size) chunks.push(e.data); });
    recorder.addEventListener('stop', () => {
      const blob = new Blob(chunks, { type: mimeType }), url = URL.createObjectURL(blob), a = document.createElement('a');
      a.href = url; a.download = `sweep-maple-house-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.${mimeType.includes('mp4') ? 'mp4' : 'webm'}`; a.click();
      stream.getTracks().forEach(t => t.stop()); setTimeout(() => URL.revokeObjectURL(url), 60_000); recording = null; $('record-button').classList.remove('recording'); $('record-button').querySelector('span').textContent = 'Record clip'; toast('Your 1080p cleaning clip is ready.');
    });
    recorder.addEventListener('error', () => { stream.getTracks().forEach(t => t.stop()); recording = null; $('record-button').classList.remove('recording'); $('record-button').querySelector('span').textContent = 'Record clip'; toast('Recording failed. Try another browser.'); });
    drawRecording(); recorder.start(1000); $('record-button').classList.add('recording'); $('record-button').querySelector('span').textContent = 'Stop & save'; toast('Recording a 1080p clip. Press Stop & save to download.');
  } catch (e) { recording = null; toast(`Could not start recording. ${e.message}`); }
}
function animate() {
  requestAnimationFrame(animate); const dt = Math.min(clock.getDelta(), .1), now = performance.now();
  const blend = 1 - Math.exp(-dt * 24);
  visualRobot.x += (latestRobot.x - visualRobot.x) * blend; visualRobot.y += (latestRobot.y - visualRobot.y) * blend;
  const yawDelta = Math.atan2(Math.sin(latestRobot.yaw - visualRobot.yaw), Math.cos(latestRobot.yaw - visualRobot.yaw)); visualRobot.yaw += yawDelta * blend;
  robot.position.set(visualRobot.x, .042, visualRobot.y); robot.rotation.y = -visualRobot.yaw;
  if (['cleaning', 'returning'].includes(state?.status) && !state?.error) brush.rotation.y -= dt * 21;
  if (viewMode === 'follow') {
    const target = new THREE.Vector3(visualRobot.x, .04, visualRobot.y); const delta = target.clone().sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 3)); controls.target.add(delta); camera.position.add(delta);
  }
  controls.update(); renderer.render(scene, camera);
  labels.forEach(label => { pos.copy(label.position).project(camera); const x = (pos.x + 1) / 2 * viewport.clientWidth, y = (1 - pos.y) / 2 * viewport.clientHeight; label.element.style.left = `${x}px`; label.element.style.top = `${y}px`; label.element.style.opacity = viewMode === 'follow' || pos.z > 1 || pos.z < -1 ? '0' : label.dock ? '.55' : '.75'; });
  drawRecording();
  if (now - lastPoll > 100) { lastPoll = now; poll(); }
}
async function init() {
  try {
    const response = await fetch('/api/home'); if (!response.ok) throw new Error(`Home returned ${response.status}`); home = await response.json();
    $('home-name').textContent = home.name || 'Maple House'; $('home-area').textContent = Math.round(home.width * home.height);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true }); renderer.setClearColor(0x000000, 0); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.08;
    viewport.prepend(renderer.domElement); buildHome(); latestRobot = visualRobot = { x: home.dock[0], y: home.dock[1], yaw: home.dock[2] || 0 };
    await poll(); $('loader').classList.add('hidden'); $('loader').setAttribute('aria-hidden', 'true'); animate();
    new ResizeObserver(resize).observe(viewport);
    $('start-button').addEventListener('click', () => control(['cleaning', 'returning'].includes(state?.status) ? 'pause' : 'start'));
    $('dock-button').addEventListener('click', () => control('dock'));
    $('reset-button').addEventListener('click', async () => { if (await control('reset')) { lastDustSignature = ''; lastTrailSignature = ''; visualRobot = { ...latestRobot }; toast('Mission reset. All dust restored.'); } });
    document.querySelectorAll('[data-speed]').forEach(b => b.addEventListener('click', async () => { const n = Number(b.dataset.speed); if (await control('speed', { speed: n })) selectSpeed(n); }));
    document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
    $('trail-toggle').addEventListener('change', e => { if (trailMesh) trailMesh.visible = trailLine.visible = e.target.checked; });
    $('cinema-button').addEventListener('click', () => cinema(!document.body.classList.contains('cinema'))); $('exit-cinema').addEventListener('click', () => cinema(false));
    $('record-button').addEventListener('click', toggleRecording);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') cinema(false); if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); $('start-button').click(); } });
  } catch (e) {
    console.error(e); $('loader').classList.add('error'); $('loader').querySelector('.loader-spinner').style.display = 'none'; $('loader').querySelector('strong').textContent = 'Your virtual home could not load.'; $('loader').lastElementChild.textContent = `${e.message}. Start the Wendy app and refresh this page. A browser with WebGL is required.`;
    $('connection').classList.add('offline'); $('connection').innerHTML = '<span></span>Not connected';
  }
}
init();
