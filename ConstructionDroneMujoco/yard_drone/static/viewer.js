import * as THREE from 'three';
import { OrbitControls } from '/three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(['viewport', 'loading-state', 'render-error', 'connection-dot', 'connection-label', 'live-badge', 'altitude', 'speed', 'rtf', 'flight-mode', 'mission-kind', 'baseline-check', 'benchmark-status', 'mission-status', 'mission-progress', 'progress-fill', 'fire-button', 'oil-button', 'follow-button', 'stage-status', 'sim-time', 'evidence-empty', 'evidence-empty-text', 'incident-list', 'incident-count', 'mic-button', 'voice-title', 'voice-help', 'command-input', 'toast'].map((id) => [id, $(id)]));
const controlButtons = [...document.querySelectorAll('[data-command], [data-worker], [data-incident]')];
let state = null;
let lastStateAt = 0;
let connected = false;
let transportConnected = false;
let physicsFault = false;
let faultMessage = '';
let sceneDescription = null;
let renderer, scene, camera, controls, drone, worker, targetMarker, scanCone, scanRing, flightLine;
let hazardMeshes;
let lastHazardTime=null;
let plannedLine;
let plannedPathSignature='';
let lastCameraFrame=null;
let cameraRequestPending=false;
let frameTime = performance.now();
let cameraMode = 'orbit';
let lastUiAt = 0;
let incidentSignature = '';
let trailMission = null;
const trail = [];
const propellers = [];
const threeClock = new THREE.Clock();
const materials = new Map();
const localUp = new THREE.Vector3(0, 0, 1);
let toastTimer;
let voiceReplies = false;
const microphoneSupported = Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
let recording = null;
let voiceProcessing = false;
let selectedIncident = null;

function material(color, options = {}) {
  const key = color + JSON.stringify(options);
  if (!materials.has(key)) materials.set(key, new THREE.MeshStandardMaterial({ color, roughness: 0.85, ...options }));
  return materials.get(key);
}

function box(parent, size, position, color, options = {}) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material(color, options));
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function cylinder(parent, top, bottom, height, position, color, options = {}) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(top, bottom, height, 20), material(color, options));
  mesh.rotation.x = Math.PI / 2;
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function beam(parent, start, end, radius, color) {
  const a = new THREE.Vector3(...start), b = new THREE.Vector3(...end);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, a.distanceTo(b), 8), material(color));
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.sub(a).normalize());
  mesh.castShadow = true;
  parent.add(mesh);
  return mesh;
}

function label(text, subtitle, position, scale = 1.25) {
  const canvas = document.createElement('canvas');
  canvas.width = 384;
  canvas.height = 132;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(244,246,229,.94)';
  ctx.beginPath();
  ctx.roundRect(4, 4, 376, 124, 15);
  ctx.fill();
  ctx.strokeStyle = '#b7c5a6';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#344e31';
  ctx.font = '600 38px system-ui';
  ctx.fillText(text, 192, 54);
  ctx.fillStyle = '#829273';
  ctx.font = '24px system-ui';
  ctx.fillText(subtitle, 192, 94);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: true }));
  sprite.scale.set(scale * 2.5, scale * 0.86, 1);
  sprite.position.set(...position);
  scene.add(sprite);
  return sprite;
}

function groundTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 512;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#c6c5ad';
  ctx.fillRect(0, 0, 512, 512);
  let seed = 38;
  const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  for (let i = 0; i < 28000; i++) {
    const gray = 140 + Math.floor(rand() * 67);
    ctx.fillStyle = `rgba(${gray + 6},${gray + 4},${gray - 10},${0.12 + rand() * 0.3})`;
    const r = 0.3 + rand() * 1.3;
    ctx.fillRect(rand() * 512, rand() * 512, r, r);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(5, 4);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function buildScene(description) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#e9eddf');
  scene.fog = new THREE.Fog('#e9eddf', 27, 85);
  camera = new THREE.PerspectiveCamera(39, 1, 0.05, 160);
  camera.up.copy(localUp);
  camera.position.set(21, -27, 21);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  ui.viewport.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.9);
  controls.enableDamping = true;
  controls.enabled=!selectedIncident;
  controls.dampingFactor = 0.06;
  controls.minDistance = 4;
  controls.maxDistance = 53;
  controls.maxPolarAngle = Math.PI / 2 - 0.03;
  controls.addEventListener('start', () => { if (cameraMode === 'follow') selectView('orbit'); });
  renderer.domElement.addEventListener('pointerup',placeSelectedIncident);
  scene.add(new THREE.HemisphereLight('#fff9df', '#7c8c69', 2.3));
  const sun = new THREE.DirectionalLight('#fff8e7', 3.2);
  sun.position.set(-9, -12, 25);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -19, right: 19, top: 19, bottom: -19, near: 1, far: 70 });
  sun.shadow.normalBias = 0.025;
  sun.shadow.bias = -0.0002;
  scene.add(sun);
  const ambientGround = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), material('#c6d0b1'));
  ambientGround.position.z = -0.04;
  ambientGround.receiveShadow = true;
  scene.add(ambientGround);
  const ground = new THREE.Mesh(new THREE.BoxGeometry(18.4, 16.4, 0.12), material('#ffffff', { map: groundTexture() }));
  ground.position.z = -0.06;
  ground.receiveShadow = true;
  scene.add(ground);
  box(scene, [3.1, 15.8, 0.014], [0, 0, -0.005], '#b4b7a4');
  for (let y = -7; y < 8; y += 1.25) box(scene, [0.045, 0.65, 0.002], [0, y, 0.004], '#e9e6cc');
  for (const [letter, zone] of Object.entries(description.zones)) {
    const [x, y] = zone.position;
    const pad = new THREE.Mesh(new THREE.PlaneGeometry(6.2, 5.6), material(letter === 'D' ? '#a9a184' : '#c1c1a7'));
    pad.position.set(x, y, 0.001);
    pad.receiveShadow = true;
    scene.add(pad);
    const corners = [[-3,-2.7],[-3,2.7],[3,-2.7],[3,2.7]];
    for (const [dx,dy] of corners) {
      box(scene, [.45,.045,.001], [x+dx+(dx<0?.2:-.2),y+dy,.005], '#e6dfbf');
      box(scene, [.045,.45,.001], [x+dx,y+dy+(dy<0?.2:-.2),.005], '#e6dfbf');
    }
    label(`ZONE ${letter}`, zone.name, [x, y + (y > 0 ? 2.6 : -2.6), 1.25], 0.64);
  }
  createFence();
  createOffice(description.obstacles.find((o) => o.name === 'site office'));
  createContainer(description.obstacles.find((o) => o.name === 'shipping container'));
  createMaterials();
  createEquipment();
  createSiteDetails(description.home);
  createDrone();
  createWorker();
  createHazards(description.hazards);
  targetMarker = new THREE.Mesh(new THREE.RingGeometry(0.34, 0.38, 48), new THREE.MeshBasicMaterial({ color:'#f19b54', side:THREE.DoubleSide, transparent:true, opacity:.8, depthWrite:false }));
  targetMarker.position.z = .05;
  scene.add(targetMarker);
  scanCone = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 48, 1, true), new THREE.MeshBasicMaterial({ color:'#86c5ad', transparent:true, opacity:.08, side:THREE.DoubleSide, depthWrite:false }));
  scanCone.rotation.x = Math.PI / 2;
  scene.add(scanCone);
  scanRing = new THREE.Mesh(new THREE.RingGeometry(.94, 1, 64), new THREE.MeshBasicMaterial({ color:'#579e82', transparent:true, opacity:.4, side:THREE.DoubleSide, depthWrite:false }));
  scene.add(scanRing);
  flightLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color:'#4e836a', transparent:true, opacity:.65 }));
  scene.add(flightLine);
  plannedLine=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineDashedMaterial({color:'#ad9252',dashSize:.18,gapSize:.12,transparent:true,opacity:.65}));
  scene.add(plannedLine);
  new ResizeObserver(resize).observe(ui.viewport);
  resize();
  animate();
}

function createFence() {
  for (const y of [-8.2,8.2]) {
    for (let x=-9.2;x<=9.3;x+=1.5) {
      cylinder(scene,.035,.035,1.05,[x,y,.52],'#687563');
    }
    for (const z of [.35,.75,1]) beam(scene,[-9.2,y,z],[9.2,y,z],.013,'#89977e');
  }
  for (const x of [-9.2,9.2]) {
    for (let y=-8.2;y<=8.3;y+=1.5) cylinder(scene,.035,.035,1.05,[x,y,.52],'#687563');
    for (const z of [.35,.75,1]) beam(scene,[x,-8.2,z],[x,8.2,z],.013,'#89977e');
  }
  const wire = new THREE.LineBasicMaterial({color:'#95a18a',transparent:true,opacity:.45});
  for (const y of [-8.2,8.2]) {
    const points=[];
    for(let x=-9.2;x<9.2;x+=.28) {points.push(new THREE.Vector3(x,y,.07),new THREE.Vector3(Math.min(x+.6,9.2),y,1));}
    scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),wire));
  }
}

function createOffice(obstacle) {
  if (!obstacle) return;
  const [x,y,z]=obstacle.position;
  box(scene,obstacle.size.map((v)=>v*2),[x,y,z],obstacle.color);
  box(scene,[2.6,4.2,.16],[x,y,2.28],'#627365');
  for(const wy of [-1.25,.35,1.25]) {
    box(scene,[.035,.68,.57],[x+1.215,y+wy,1.32],'#687b78',{roughness:.25});
    box(scene,[.044,.025,.59],[x+1.239,y+wy,1.32],'#dce3d9');
  }
  box(scene,[.045,.65,1.3],[x+1.22,y-.43,.7],'#a6b1a7');
  box(scene,[.06,.06,.06],[x+1.27,y-.21,.79],'#46584b');
  box(scene,[.5,1,.15],[x+1.44,y-.43,.075],'#9ea591');
  box(scene,[.6,.7,.035],[x,y,2.4],'#586e63');
}

function createContainer(obstacle) {
  if(!obstacle)return;
  const [x,y,z]=obstacle.position;
  box(scene,obstacle.size.map((v)=>v*2),[x,y,z],obstacle.color);
  box(scene,[2.1,4.08,.07],[x,y,2.035],'#ab6631');
  for(let cy=-1.85;cy<2;cy+=.22) box(scene,[.025,.045,1.9],[x-1.018,y+cy,1],'#b87036');
  for(const cx of [-.7,.7]) beam(scene,[x+cx,y-2.025,.12],[x+cx,y-2.025,1.87],.025,'#e0ba82');
  box(scene,[1.7,.025,.1],[x,y-2.03,1.8],'#d79753');
}

function createMaterials() {
  for(let k=0;k<4;k++) for(let j=0;j<5;j++) box(scene,[2.8,.17,.17],[-6,5.04+j*.2,.12+k*.18],k%2?'#b99a6a':'#c4ab7a');
  for (const x of [-4.04,-3.4]) for(const z of [.25,.7]) {
    const pipe=cylinder(scene,.22,.22,1.28,[x,5.4,z],'#a7afa2');
    pipe.rotation.x=0;
    const end = new THREE.Mesh(new THREE.CircleGeometry(.145,24),material('#62735f'));
    end.rotation.x=Math.PI/2;
    end.position.set(x,4.75,z);
    scene.add(end);
    const rim=new THREE.Mesh(new THREE.TorusGeometry(.18,.035,6,24),material('#b6bdae'));
    rim.rotation.x=Math.PI/2;
    rim.position.set(x,4.742,z);scene.add(rim);
  }
  const tank=cylinder(scene,.48,.48,1.9,[6,5.5,.7],'#c7cdc0',{roughness:.45,metalness:.3});
  tank.rotation.x=0;tank.rotation.z=Math.PI/2;
  for(const tx of [5.35,6.65]) box(scene,[.15,.85,.2],[tx,5.5,.1],'#6f7e69');
  cylinder(scene,.09,.09,.18,[6,5.5,1.28],'#6b7d68');
  for(const [x,y] of [[7,6.9],[6.2,6.9],[7.7,6.9]]) {
    cylinder(scene,.22,.22,.67,[x,y,.335],'#799080');
    for(const z of [.18,.51]) cylinder(scene,.232,.232,.035,[x,y,z],'#516c5b');
  }
}

function createEquipment() {
  const x=6.4,y=-5;
  for(const sx of [-.5,.5]) box(scene,[.28,1.85,.3],[x+sx,y,.22],'#465347');
  for(const sx of [-.5,.5]) for(let sy=-.7;sy<.8;sy+=.28) {
    const wheel=cylinder(scene,.12,.12,.3,[x+sx,y+sy,.21],'#637064');
    wheel.rotation.x=0;wheel.rotation.z=Math.PI/2;
  }
  box(scene,[1.25,1.2,.48],[x,y+.1,.65],'#d9ac48');
  box(scene,[.67,.72,.7],[x-.23,y+.24,1.19],'#dfb448');
  box(scene,[.05,.53,.45],[x-.57,y+.24,1.28],'#546d68',{roughness:.2});
  box(scene,[.51,.04,.44],[x-.23,y-.13,1.28],'#5c7770',{roughness:.2});
  box(scene,[.77,.81,.08],[x-.23,y+.24,1.57],'#374e3d');
  beam(scene,[x+.35,y-.4,.86],[x+.35,y-1.03,1.95],.1,'#d9ae43');
  beam(scene,[x+.35,y-1.03,1.95],[x+.35,y-1.75,.45],.078,'#d9ae43');
  beam(scene,[x+.35,y-.4,1.02],[x+.35,y-.84,1.85],.035,'#73806e');
  box(scene,[.57,.46,.29],[x+.35,y-1.76,.24],'#667361');
  for(let z=.08;z<.67;z+=.13) {
    for(const yy of [-.45,.45]) box(scene,[1.2,.14,.075],[3.8,-5.5+yy,z],'#b69a6b');
    for(const xx of [-.44,0,.44]) box(scene,[.17,1.4,.065],[3.8+xx,-5.5,z+.07],'#c3aa7b');
  }
  for(let i=0;i<5;i++) box(scene,[.82,.42,.4],[-6.65+i*.85,-4.8,.21],'#8f876d');
  box(scene,[3.8,1.35,.05],[-4.75,-4,.04],'#91886b');
  for(let i=0;i<7;i++) {
    const pile=new THREE.Mesh(new THREE.DodecahedronGeometry(.3+(i%3)*.15,0),material('#a69d7a'));
    pile.scale.set(1.5,1.2,.65);pile.position.set(-6.4+(i%3)*.6,-5.7+Math.floor(i/3)*.55,.2);pile.castShadow=true;scene.add(pile);
  }
}

function createSiteDetails(home) {
  const pad=cylinder(scene,1.05,1.05,.002,[home[0],home[1],.002],'#495f50');
  const rim=new THREE.Mesh(new THREE.RingGeometry(.87,.91,64),material('#ece5c9',{side:THREE.DoubleSide}));
  rim.position.set(home[0],home[1],.004);scene.add(rim);
  for(const dx of [-.22,.22]) box(scene,[.07,.58,.001],[home[0]+dx,home[1],.005],'#f1e6be');
  box(scene,[.5,.065,.001],[home[0],home[1],.005],'#f1e6be');
  label('DRONE 01','Launch pad',[home[0],home[1]-1.3,.7],.5);
  for(const [x,y] of [[-1.8,-6.7],[1.8,-6.7],[-7.9,2.5],[7.9,-2.5]]) {
    box(scene,[.32,.32,.05],[x,y,.025],'#5a6454');
    cylinder(scene,.04,.13,.4,[x,y,.25],'#e89a4e');
    cylinder(scene,.082,.1,.08,[x,y,.26],'#eee7c9');
  }
  for(const [x,y] of [[-12,7],[-13,-3],[12,8],[13,-7],[-11,-10],[-15,1],[11,13],[-6,12]]) {
    cylinder(scene,.07,.11,1.7,[x,y,.83],'#7f8061');
    const crown=new THREE.Mesh(new THREE.IcosahedronGeometry(1.2,1),material('#9caf82'));
    crown.position.set(x,y,2);crown.scale.set(1,1,1.2);crown.castShadow=true;scene.add(crown);
  }
  const crane=new THREE.Group();crane.position.set(-11,7,0);
  for(const dx of [-.23,.23]) for(const dy of [-.23,.23]) beam(crane,[dx,dy,0],[dx,dy,8],.034,'#bca25f');
  for(let z=.2;z<7.8;z+=.6) {
    beam(crane,[-.23,-.23,z],[.23,-.23,z+.6],.019,'#b6a165');
    beam(crane,[.23,.23,z],[-.23,.23,z+.6],.019,'#b6a165');
    beam(crane,[-.23,-.23,z],[-.23,.23,z+.6],.019,'#b6a165');
  }
  beam(crane,[-2,0,7.8],[6,0,7.8],.09,'#bba267');
  beam(crane,[0,0,8.5],[6,0,7.8],.018,'#7e8668');
  beam(crane,[5.5,0,7.8],[5.5,0,4.8],.013,'#5e6d54');
  scene.add(crane);
}

function createDrone() {
  drone=new THREE.Group();
  box(drone,[.28,.21,.1],[0,0,0],'#2d403e',{roughness:.4,metalness:.3});
  box(drone,[.2,.15,.035],[0,0,.07],'#f0c76d');
  const rotorPositions=[[-.26,-.26],[.26,-.26],[.26,.26],[-.26,.26]];
  rotorPositions.forEach(([x,y],i)=>{
    beam(drone,[0,0,.005],[x,y,.005],.022,'#4b5b53');
    cylinder(drone,.043,.043,.065,[x,y,.03],'#334e47');
    const rotor=new THREE.Group();rotor.position.set(x,y,.075);
    box(rotor,[.25,.026,.008],[0,0,0],'#5e7669',{transparent:true,opacity:.7});
    box(rotor,[.026,.25,.008],[0,0,0],'#5e7669',{transparent:true,opacity:.7});
    drone.add(rotor);propellers.push(rotor);
    cylinder(drone,.018,.018,.014,[x,y,.084],i<2?'#f0aa57':'#8abf94',{emissive:i<2?'#ae6229':'#396e45',emissiveIntensity:.5});
  });
  for(const y of [-.095,.095]) {
    for(const x of [-.085,.085]) beam(drone,[x,y,-.04],[x,y*1.5,-.13],.009,'#40544b');
    beam(drone,[-.16,y*1.5,-.13],[.16,y*1.5,-.13],.008,'#40544b');
  }
  box(drone,[.065,.075,.062],[.105,0,-.082],'#2b3b38');
  const lens=cylinder(drone,.024,.024,.03,[.143,0,-.08],'#426b66',{metalness:.7,roughness:.15});
  lens.rotation.x=0;lens.rotation.z=Math.PI/2;
  scene.add(drone);
}

function createWorker() {
  worker=new THREE.Group();
  for(const x of [-.075,.075]) {
    box(worker,[.11,.12,.36],[x,0,.24],'#526661');
    box(worker,[.13,.22,.08],[x,-.04,.08],'#354943');
  }
  box(worker,[.31,.18,.42],[0,0,.62],'#e4a557');
  box(worker,[.25,.015,.045],[0,-.098,.69],'#edebbc');
  for(const x of [-.18,.18]) beam(worker,[x*.8,0,.77],[x,-.01,.44],.055,'#dfb780');
  const head=new THREE.Mesh(new THREE.SphereGeometry(.115,16,12),material('#d7b58e'));
  head.position.z=.98;worker.add(head);
  const helmet=new THREE.Mesh(new THREE.SphereGeometry(.14,16,12,0,Math.PI*2,0,Math.PI/2),material('#f1bc55'));
  helmet.rotation.x=Math.PI/2;helmet.position.z=1.045;worker.add(helmet);
  cylinder(worker,.163,.163,.025,[0,0,1.04],'#e7ac3d');
  worker.scale.setScalar(1.55);
  scene.add(worker);
}

function createHazards() {
  const capacity=1152;
  function fieldMesh(geometry,roughness=0.8){
    const alpha=new THREE.InstancedBufferAttribute(new Float32Array(capacity),1);
    geometry.setAttribute('instanceAlpha',alpha);
    const mat=new THREE.MeshStandardMaterial({color:'#ffffff',roughness,transparent:true,depthWrite:false});
    mat.onBeforeCompile=(shader)=>{shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nattribute float instanceAlpha; varying float vInstanceAlpha;').replace('#include <begin_vertex>','#include <begin_vertex>\nvInstanceAlpha = instanceAlpha;');shader.fragmentShader=shader.fragmentShader.replace('#include <common>','#include <common>\nvarying float vInstanceAlpha;').replace('#include <color_fragment>','#include <color_fragment>\ndiffuseColor.a *= vInstanceAlpha;');};
    mat.customProgramCacheKey=()=>'yard-field-alpha';
    const mesh=new THREE.InstancedMesh(geometry,mat,capacity);mesh.setColorAt(0,new THREE.Color('#ffffff'));mesh.count=0;mesh.frustumCulled=false;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mesh.userData.alpha=alpha;scene.add(mesh);return mesh;
  }
  hazardMeshes={oil:fieldMesh(new THREE.BoxGeometry(1,1,1),.15),fire:fieldMesh(new THREE.CylinderGeometry(1,1,1,12).rotateX(Math.PI/2)),smoke:fieldMesh(new THREE.SphereGeometry(1,10,8))};
}

function updateHazardFields() {
  if(!hazardMeshes)return;
  const fields=state.environment?.hazardFields;
  if(!fields||fields.time===lastHazardTime)return;
  lastHazardTime=fields.time;
  const counts={oil:0,fire:0,smoke:0},transform=new THREE.Object3D(),color=new THREE.Color();
  const clip=(value,low,high)=>Math.max(low,Math.min(high,value));
  function place(kind,x,y,z,sx,sy,sz,r,g,b,alpha){const mesh=hazardMeshes[kind],i=counts[kind]++;if(i>=1152)return;transform.position.set(x,y,z);transform.scale.set(sx,sy,sz);transform.updateMatrix();mesh.setMatrixAt(i,transform.matrix);mesh.setColorAt(i,color.setRGB(r,g,b));mesh.userData.alpha.setX(i,alpha);}
  for(const cell of fields.cells||[]){
    const x=Number(cell.x),y=Number(cell.y),depth=Number(cell.oilDepth),temperature=Number(cell.temperatureK),smoke=Number(cell.smoke);
    if(!Number.isFinite(x)||!Number.isFinite(y))continue;
    if(depth>.00015){const halfHeight=Math.max(.003,depth/2);place('oil',x,y,halfHeight+.004,.492,.492,2*halfHeight,.035,.046,.07,clip(depth/.003,.45,.97));}
    if(temperature>510){const heat=clip((temperature-500)/600,0,1),height=.03+1.2*heat,radius=.5*(.18+.18*heat);place('fire',x,y,.015+height/2,radius,radius,height,1,.15+.6*heat,.025,.78+.2*heat);}
    if(smoke>.001){const radius=clip(.18+Math.sqrt(smoke)*2,.18,.55);place('smoke',x,y,.2+Number(cell.smokeHeight||0),radius,radius,radius,.18,.18,.2,clip(smoke*5,0,.22));}
  }
  for(const [kind,mesh] of Object.entries(hazardMeshes)){mesh.count=Math.min(counts[kind],1152);mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;mesh.userData.alpha.needsUpdate=true;}
  ui['fire-button'].classList.toggle('active',counts.fire>0);ui['oil-button'].classList.toggle('active',counts.oil>0);
}

function resize() {
  if(!renderer)return;
  const {width,height}=ui.viewport.getBoundingClientRect();
  renderer.setSize(width,height);
  camera.aspect=width/Math.max(height,1);camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  const now=performance.now(),dt=Math.min((now-frameTime)/1000,.1),t=threeClock.getElapsedTime();frameTime=now;
  if(state?.position) {
    drone.position.set(...state.position);
    if(state.quaternion?.length===4)drone.quaternion.set(...state.quaternion);
    if(state.workerPosition)worker.position.set(state.workerPosition[0],state.workerPosition[1],state.workerPosition[2]-.9);
    const rotorForces=state.rotors||[];
    if(connected)propellers.forEach((prop,i)=>{prop.rotation.z+=(i%2?-1:1)*dt*Math.min(200,Math.max(0,Number(rotorForces[i]||0))*85);});
    const altitude=Math.max(0,state.position[2]);
    const scanning=connected&&state.mission?.status==='running'&&altitude>.5;
    scanCone.visible=scanRing.visible=scanning;
    if(scanning){scanCone.scale.set(altitude*.38,altitude,altitude*.38);scanCone.position.set(state.position[0],state.position[1],altitude*.5);scanRing.scale.setScalar(altitude*.38);scanRing.position.set(state.position[0],state.position[1],.045);scanRing.material.opacity=.2+Math.sin(t*3)*.1;}
    targetMarker.visible=connected&&Array.isArray(state.target)&&altitude>.3;
    if(targetMarker.visible){targetMarker.position.set(state.target[0],state.target[1],.045);targetMarker.scale.setScalar(1+Math.sin(t*2)*.08);}
    if(cameraMode==='follow'){
      const desired=drone.position.clone().add(new THREE.Vector3(4,-6,4));camera.position.lerp(desired,1-Math.exp(-dt*2.5));controls.target.lerp(drone.position,1-Math.exp(-dt*4));
    }
    if(state.mission?.id!==trailMission){trailMission=state.mission?.id;trail.length=0;flightLine.geometry.dispose();flightLine.geometry=new THREE.BufferGeometry();}
    const point=drone.position.clone();
    if(connected&&altitude>.3&&(!trail.length||trail.at(-1).distanceTo(point)>.08)){trail.push(point);if(trail.length>2400)trail.shift();flightLine.geometry.dispose();flightLine.geometry=new THREE.BufferGeometry().setFromPoints(trail);}
  }
  controls.update();renderer.render(scene,camera);
}

function selectView(mode) {
  cameraMode=mode;
  ['orbit','top','follow'].forEach((m)=>$(`${m}-view`).classList.toggle('active',m===mode));
  if(!camera)return;
  if(mode==='orbit'){camera.position.set(21,-27,21);controls.target.set(0,0,.9);}
  if(mode==='top'){camera.position.set(0,-.01,31);controls.target.set(0,0,0);}
  controls.update();
}
['orbit','top','follow'].forEach((mode)=>$(`${mode}-view`).addEventListener('click',()=>selectView(mode)));

function setConnection(value) {
  transportConnected=value;
  connected=value&&!physicsFault;
  ui['connection-dot'].className=`status-dot ${connected?'connected':'disconnected'}`;
  ui['connection-label'].textContent=value?(physicsFault?'Physics stopped · connection open':'Simulator connected'):'Simulator disconnected';
  ui['live-badge'].classList.toggle('connected',connected);
  ui['live-badge'].textContent=physicsFault?'FAULT':value?'LIVE':'OFFLINE';
  document.querySelectorAll('.telemetry .status-dot, .stage-status .status-dot').forEach((dot)=>dot.classList.toggle('disconnected',!connected));
  controlButtons.forEach((button)=>{button.disabled=!connected;});
  document.querySelectorAll('.incident-card-foot button').forEach((button)=>{button.disabled=!connected;});
  $('scan-button').disabled=!connected||!state?.benchmark?.available;
  ui['mic-button'].disabled=!connected||!microphoneSupported||voiceProcessing;
  ui['command-input'].disabled=!connected;
  $('command-form').querySelector('button').disabled=!connected;
  if(!connected){cancelRecording();window.speechSynthesis?.cancel();heldWorkerDirections.clear();document.querySelectorAll('[data-worker]').forEach((button)=>button.classList.remove('held'));if(selectedIncident)selectIncident(null);}
  if(value)ui['loading-state'].style.display='none';
  else if(state)ui['stage-status'].textContent='Connection lost. Waiting for simulator.';
  if(physicsFault){
    ui['stage-status'].textContent=`Physics fault · Telemetry is stale. ${faultMessage}`;
    ui['flight-mode'].textContent='FAULT · STALE';
    ui['voice-title'].textContent='Flight commands paused';
    ui['voice-help'].textContent=faultMessage;
    ui['voice-help'].parentElement.classList.add('error');
  }
}

function receiveState(next) {
  const previousFault=physicsFault,previousMessage=faultMessage;
  state=next;lastStateAt=performance.now();
  physicsFault=Boolean(next.error)||next.ready===false;
  faultMessage=physicsFault?String(next.error||'The physics simulation stopped. Restart the simulator to resume flight.') : '';
  if(!transportConnected||physicsFault!==previousFault||faultMessage!==previousMessage)setConnection(true);
  if(physicsFault&&(!previousFault||faultMessage!==previousMessage))toast(`Physics fault: ${faultMessage}`,true);
  if(previousFault&&!physicsFault){
    ui['voice-title'].textContent=microphoneSupported?'Tell the drone what to do':'Type your flight request.';
    ui['voice-help'].textContent=microphoneSupported?'Audio is transcribed inside the WendyOS VM.':'This browser cannot record microphone audio. Use the text box or open Chrome on localhost.';
    ui['voice-help'].parentElement.classList.remove('error');
  }
  if(performance.now()-lastUiAt>110){lastUiAt=performance.now();updateUi();}
}

function updateUi() {
  if(!state)return;
  const position=state.position||[],velocity=state.velocity||[];
  ui.altitude.textContent=Number.isFinite(position[2])?position[2].toFixed(1):'--';
  ui.speed.textContent=velocity.length===3?Math.hypot(...velocity).toFixed(1):'--';
  const factor=state.telemetry?.realTimeFactor??state.telemetry?.rtf??state.telemetry?.realtimeFactor??state.telemetry?.real_time_factor??state.realTimeFactor;
  ui.rtf.textContent=Number.isFinite(factor)?`${factor.toFixed(2)}×`:'--';
  ui['flight-mode'].textContent=physicsFault?'FAULT · STALE':String(state.mode||'unknown').replaceAll('_',' ').toUpperCase();
  const mission=state.mission||{},benchmark=state.benchmark||{},scanned=mission.scannedZones||[];
  $('scan-button').disabled=!connected||!benchmark.available;
  $('scan-button').title=benchmark.available?'Compare the current flight with the benchmark':'Complete a baseline flight first';
  ui['mission-kind'].textContent=physicsFault?'FAULT':mission.status==='running'?(mission.kind==='baseline'?'BASELINE':'SURVEYING'):'STANDBY';
  ui['baseline-check'].textContent=benchmark.available?'✓':'○';
  ui['benchmark-status'].textContent=benchmark.available?`Benchmark ready · ${benchmark.viewCount||0} views · ${Math.round(benchmark.coveragePercent||0)}% coverage`:'No benchmark recorded yet';
  const title=mission.kind==='baseline'?'Baseline flight':'Inspection flight';
  ui['mission-status'].textContent=physicsFault?'Physics stopped. Mission paused.':mission.status==='running'?(mission.kind==='baseline'?'Recording baseline':'Comparing flight'):mission.status==='complete'?`${title} complete`:mission.status==='interrupted'?`${title} interrupted`:'Waiting for flight';
  const views=Number(mission.viewCount||0),coverage=Number(mission.coveragePercent??state.navigation?.coveragePercent??0);
  ui['mission-progress'].textContent=mission.kind==='current'?`${views} / ${benchmark.viewCount||0} views`:`${views} views · ${Math.round(coverage)}%`;
  ui['progress-fill'].style.width=`${Math.min(100,Math.max(0,Number(mission.progress||0)*100))}%`;
  document.querySelectorAll('[data-zone-progress]').forEach((item)=>item.classList.toggle('complete',scanned.includes(item.dataset.zoneProgress)));
  ui['follow-button'].classList.toggle('active',state.mode==='follow');
  const seconds=Math.floor(Number(state.time??state.elapsed??0));
  ui['sim-time'].textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
  ui['stage-status'].textContent=physicsFault?`Physics fault · Telemetry is stale. ${faultMessage}`:state.sensor?.error?`Camera fault · ${state.sensor.error}`:mission.status==='running'?`${title} · ${views} camera views · ${Math.round(coverage)}% yard coverage`:state.mode==='follow'?`Camera following · ${(state.perception?.tracks||[]).length} worker tracks`:state.mode==='landed'?'Drone on the launch pad. Ready for a mission.':`Drone ${String(state.mode||'ready').replaceAll('_',' ')} · Live physics state`;
  const incidents=state.incidents||[];
  ui['incident-count'].textContent=incidents.length?`${incidents.length} change${incidents.length===1?'':'s'} detected`:'No incidents reported';
  $('evidence-heading').textContent=incidents.length&&incidents.every((incident)=>incident.status==='resolved')?'Resolved changes from this flight':'Changes that need attention';
  ui['evidence-empty'].hidden=incidents.length>0;
  ui['evidence-empty'].style.display=incidents.length?'none':'flex';
  $('evidence-empty-title').textContent=mission.status==='running'?(mission.kind==='baseline'?'Recording the benchmark.':'Surveying the current yard.'):!benchmark.available?'A clear yard starts with a baseline.':mission.kind==='current'&&mission.status==='complete'?'No changes in the compared camera views.':'Benchmark ready for comparison.';
  if(mission.status==='running'&&mission.kind==='baseline')ui['evidence-empty-text'].textContent='Capturing camera views across the yard. Run a current mission afterward to compare the imagery.';
  else if(benchmark.available)ui['evidence-empty-text'].textContent=mission.status==='running'?'The drone is recording this mission. Changes will appear as camera views are compared.':mission.kind==='current'&&mission.status==='complete'?'This flight found no visual changes in the captured views compared with the benchmark.':'The benchmark is ready. Add an incident and scan the current mission to compare the imagery.';
  else ui['evidence-empty-text'].textContent='Record a benchmark flight, add a fire or oil leak, then scan again to see the evidence.';
  const signature=JSON.stringify(incidents);
  if(signature!==incidentSignature){incidentSignature=signature;renderIncidents(incidents);}
  updateCamera();
  updateNavigation();
  updateHazardFields();
}

function updateCamera() {
  const sensor=state.sensor||{},perception=state.perception||{},image=$('camera-feed'),placeholder=$('camera-placeholder');
  const age=sensor.ageSeconds==null?NaN:Number(sensor.ageSeconds),stale=!connected||!Number.isFinite(age)||age>2||Boolean(sensor.error);
  document.querySelector('.camera-panel').classList.toggle('stale',stale);
  $('camera-age').textContent=Number.isFinite(age)?`${age.toFixed(1)}s${stale?' · STALE':''}`:'WAITING';
  const tracks=Array.isArray(perception.tracks)?perception.tracks:[];
  const detections=Array.isArray(perception.detections)?perception.detections:[];
  $('perception-status').textContent=perception.error?`Perception fault: ${perception.error}`:tracks.length?tracks.some(t=>t.source==='worker-fiducial')?`${tracks.length} worker badge tracked`:`${tracks.length} visually tracked workers`:detections.length?`${detections.length} camera detections`:state.mode==='follow'?'Searching camera for a worker':'Worker detection starts in follow mode';
  const latency=perception.lastInferenceMs??perception.inferenceMs??perception.latencyMs;
  $('camera-inference').textContent=Number.isFinite(latency)?`${Math.round(latency)} ms`:'--';
  const cameraUrl=safeImageUrl(sensor.cameraUrl);
  if(!cameraUrl){placeholder.textContent='Waiting for the onboard camera';return;}
  if(cameraRequestPending||sensor.frameTime===lastCameraFrame||!connected)return;
  lastCameraFrame=sensor.frameTime;cameraRequestPending=true;
  image.onload=()=>{cameraRequestPending=false;image.hidden=false;placeholder.hidden=true;};
  image.onerror=()=>{cameraRequestPending=false;image.hidden=true;placeholder.hidden=false;placeholder.textContent='The onboard camera frame could not load.';};
  const url=new URL(cameraUrl);url.searchParams.set('frame',String(sensor.frameTime??Date.now()));image.src=url.href;
}

function updateNavigation() {
  if(!plannedLine)return;
  const path=state.navigation?.path;
  const points=Array.isArray(path)?path.filter((p)=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)):[];
  const signature=JSON.stringify(points);
  if(signature===plannedPathSignature)return;
  plannedPathSignature=signature;plannedLine.geometry.dispose();plannedLine.geometry=new THREE.BufferGeometry().setFromPoints(points.map((p)=>new THREE.Vector3(...p)));plannedLine.computeLineDistances();
}

function safeImageUrl(value) {
  if(typeof value!=='string')return null;
  try{const url=new URL(value,location.origin);return url.origin===location.origin?url.href:null;}catch{return null;}
}

function evidenceFrame(url,labelText) {
  const frame=document.createElement('div');frame.className='evidence-frame';
  const label=document.createElement('span');label.className='image-label';label.textContent=labelText;
  const safe=safeImageUrl(url);
  if(safe){const image=document.createElement('img');image.src=safe;image.alt=`${labelText.toLowerCase()} inspection camera evidence`;image.loading='lazy';image.addEventListener('error',()=>{image.replaceWith(imagePlaceholder());},{once:true});frame.append(image);}else frame.append(imagePlaceholder());
  frame.append(label);return frame;
}

function imagePlaceholder(){const missing=document.createElement('div');missing.className='image-missing';missing.textContent='Image unavailable';return missing;}

function renderIncidents(incidents) {
  ui['incident-list'].replaceChildren();
  for(const incident of incidents){
    const resolved=incident.status==='resolved';
    const card=document.createElement('article');card.className=`incident-card ${incident.type==='oil'?'oil':'fire'}${resolved?' resolved':''}`;
    const head=document.createElement('div');head.className='incident-card-head';
    const icon=document.createElement('div');icon.className='incident-icon';icon.textContent=resolved?'✓':incident.type==='fire'?'♨':incident.type==='oil'?'◒':'◎';
    const name=document.createElement('div');name.className='incident-name';
    const heading=document.createElement('h3');heading.textContent=resolved?(incident.type==='fire'?'Fire no longer visible':'Oil leak no longer visible'):incident.type==='fire'?'Possible fire':incident.type==='oil'?'Possible oil spill':incident.type==='person'?'Person in changed area':'Visual change detected';
    const detail=document.createElement('p');detail.textContent=`ZONE ${incident.zone||'?'} · ${sceneDescription?.zones?.[incident.zone]?.name||'Inspection area'}${Number.isFinite(incident.confidence)?` · Model score ${incident.confidence.toFixed(2)}`:''}`;
    detail.title='Model scores are uncalibrated estimates within this simulator.';
    name.append(heading,detail);
    const severity=document.createElement('span');severity.className=`severity ${String(incident.severity||'high').toLowerCase()}`;severity.textContent=resolved?'Resolved · info':incident.severity||'Detected';
    head.append(icon,name,severity);
    const pair=document.createElement('div');pair.className='evidence-pair';pair.append(evidenceFrame(incident.baselineUrl,'PREVIOUS FLIGHT'),evidenceFrame(incident.currentUrl,'CURRENT FLIGHT'));
    const foot=document.createElement('div');foot.className='incident-card-foot';
    const delta=document.createElement('span');const amount=document.createElement('strong');
    amount.textContent=Number.isFinite(incident.changePercent)?`${incident.changePercent.toFixed(1)}% visual change`:Number.isFinite(incident.changedPixels)?`${incident.changedPixels.toLocaleString()} changed pixels`:'Change detected';
    delta.append(amount);
    const link=document.createElement('button');link.textContent=`Inspect zone ${incident.zone||''}`;link.className='view-button';link.disabled=!connected;link.addEventListener('click',()=>sendAction('goto',incident.zone));
    foot.append(delta,link);card.append(head,pair,foot);ui['incident-list'].append(card);
  }
}

function toast(message,error=false) {
  clearTimeout(toastTimer);ui.toast.textContent=message;ui.toast.className=`toast visible${error?' error':''}`;
  toastTimer=setTimeout(()=>ui.toast.classList.remove('visible'),4500);
}

async function sendCommand(text) {
  const command=String(text).trim();if(!command)return;
  if(!connected){toast(physicsFault?`Physics fault: ${faultMessage}`:'The simulator is disconnected. Wait for it to reconnect.',true);return;}
  try{
    ui['voice-title'].textContent='Understanding your request…';
    ui['voice-help'].textContent=command;
    const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:command})});
    const result=await response.json();
    if(!response.ok||result.ok===false)throw new Error(result.message||result.error||'The flight command was not accepted.');
    if(result.state)receiveState(result.state);
    const message=result.message||'Command accepted.';toast(message);
    ui['voice-title'].textContent='Request accepted';ui['voice-help'].textContent=message;
    if(voiceReplies&&'speechSynthesis'in window){speechSynthesis.cancel();const utterance=new SpeechSynthesisUtterance(message);utterance.rate=1;utterance.volume=.8;speechSynthesis.speak(utterance);}
  }catch(error){ui['voice-title'].textContent='Request could not run';ui['voice-help'].textContent=error.message||'The command could not reach the simulator.';toast(ui['voice-help'].textContent,true);}
}
async function sendAction(action,zone) {
  if(!connected){toast(physicsFault?`Physics fault: ${faultMessage}`:'The simulator is disconnected.',true);return;}
  try{const response=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,...(zone?{zone}:{})})});const result=await response.json();if(!response.ok||result.ok===false)throw new Error(result.message||result.error||'The flight action could not run.');if(result.state)receiveState(result.state);toast(result.message||'Flight action accepted.');}catch(error){toast(error.message,true);}
}
const buttonActions={'baseline':'baseline','takeoff':'takeoff','hover':'hover','land':'land','follow me':'follow','start mission':'survey','clear incidents':'clear'};
document.querySelectorAll('[data-command]').forEach((button)=>button.addEventListener('click',()=>{const command=button.dataset.command;if(command==='return home')sendAction('goto','home');else if(command.startsWith('go to zone '))sendAction('goto',command.at(-1));else sendAction(buttonActions[command]);}));
$('command-form').addEventListener('submit',(event)=>{event.preventDefault();const command=ui['command-input'].value.trim();if(command){ui['command-input'].value='';sendCommand(command);}});

function selectIncident(type) {
  selectedIncident=selectedIncident===type?null:type;
  document.querySelector('.stage').classList.toggle('placing',Boolean(selectedIncident));
  document.querySelectorAll('[data-incident]').forEach((button)=>button.classList.toggle('placing',button.dataset.incident===selectedIncident));
  $('placement-help').textContent=selectedIncident?`Click the yard to place ${selectedIncident==='fire'?'a fire':'an oil leak'}. Click this button again to cancel.`:'Choose an incident, then click its location in the yard.';
  if(controls)controls.enabled=!selectedIncident;
}
document.querySelectorAll('[data-incident]').forEach((button)=>button.addEventListener('click',()=>selectIncident(button.dataset.incident)));
async function placeSelectedIncident(event) {
  if(!connected||!selectedIncident||!camera)return;
  const rectangle=renderer.domElement.getBoundingClientRect(),mouse=new THREE.Vector2((event.clientX-rectangle.left)/rectangle.width*2-1,-(event.clientY-rectangle.top)/rectangle.height*2+1),ray=new THREE.Raycaster(),point=new THREE.Vector3();
  ray.setFromCamera(mouse,camera);
  if(!ray.ray.intersectPlane(new THREE.Plane(localUp,0),point))return;
  const bounds=sceneDescription.bounds;
  if(point.x<bounds.x[0]||point.x>bounds.x[1]||point.y<bounds.y[0]||point.y>bounds.y[1]){toast('Choose a location inside the construction yard.',true);return;}
  try{const response=await fetch('/api/incident',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:selectedIncident,position:[point.x,point.y]})});const result=await response.json();if(!response.ok||result.ok===false)throw new Error(result.message||result.error||'The incident could not be placed.');if(result.state)receiveState(result.state);toast(result.message||'Incident placed in the physical environment.');selectIncident(selectedIncident);}catch(error){toast(error.message,true);}
}

async function moveWorker(dx,dy,stop=false) {
  if(!connected)return;
  try{const response=await fetch('/api/worker',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(stop?{stop:true}:{dx,dy})});const result=await response.json();if(!response.ok)throw new Error(result.message||result.error||'Worker movement is unavailable.');if(result.state)receiveState(result.state);}catch(error){toast(error.message,true);}
}
const heldWorkerDirections=new Map();
const arrowDirections={ArrowUp:[0,1],ArrowDown:[0,-1],ArrowLeft:[-1,0],ArrowRight:[1,0]};
let workerRequestPending=false;
function releaseWorkerDirection(key){heldWorkerDirections.delete(key);document.querySelectorAll('[data-worker]').forEach((button)=>button.classList.remove('held'));if(!heldWorkerDirections.size)moveWorker(0,0,true);}
document.querySelectorAll('[data-worker]').forEach((button)=>{
  button.addEventListener('pointerdown',(event)=>{if(!connected)return;event.preventDefault();button.setPointerCapture(event.pointerId);heldWorkerDirections.set(`pointer-${event.pointerId}`,button.dataset.worker.split(',').map(Number));button.classList.add('held');});
  ['pointerup','pointercancel','lostpointercapture'].forEach((name)=>button.addEventListener(name,(event)=>releaseWorkerDirection(`pointer-${event.pointerId}`)));
});
window.addEventListener('keydown',(event)=>{if(/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)||!connected)return;const delta=arrowDirections[event.key];if(delta){event.preventDefault();heldWorkerDirections.set(event.key,delta);}});
window.addEventListener('keyup',(event)=>{if(arrowDirections[event.key]&&heldWorkerDirections.has(event.key)){event.preventDefault();releaseWorkerDirection(event.key);}});
window.addEventListener('blur',()=>{if(heldWorkerDirections.size){heldWorkerDirections.clear();document.querySelectorAll('[data-worker]').forEach((button)=>button.classList.remove('held'));moveWorker(0,0,true);}cancelRecording();});
setInterval(async()=>{if(!connected||workerRequestPending||!heldWorkerDirections.size)return;const delta=[...heldWorkerDirections.values()].reduce((sum,p)=>[sum[0]+p[0],sum[1]+p[1]],[0,0]),length=Math.hypot(...delta);if(!length)return;workerRequestPending=true;try{await moveWorker(delta[0]/length*.14,delta[1]/length*.14);}finally{workerRequestPending=false;}},120);

function releaseRecordingResources(session) {
  clearTimeout(session.limitTimer);clearInterval(session.levelTimer);
  session.stream?.getTracks().forEach((track)=>track.stop());
  session.audioContext?.close().catch(()=>{});
  ui['mic-button'].classList.remove('listening');
  ui['mic-button'].setAttribute('aria-label','Start voice recording');
  if(recording===session)recording=null;
}

function cancelRecording() {
  if(!recording)return;
  const session=recording;session.cancelled=true;
  if(session.recorder?.state==='recording')session.recorder.stop();
  releaseRecordingResources(session);
}

async function submitVoiceAudio(audio) {
  if(!connected||!audio.size)return;
  if(audio.size>8*1024*1024){toast('The recording exceeds 8 MiB. Try a shorter request.',true);return;}
  voiceProcessing=true;ui['mic-button'].disabled=true;ui['mic-button'].classList.add('processing');
  ui['voice-title'].textContent='Transcribing inside the VM…';ui['voice-help'].textContent='Whisper is listening to the captured audio.';
  try{
    const response=await fetch('/api/voice',{method:'POST',headers:{'Content-Type':audio.type||'application/octet-stream'},body:audio});
    const result=await response.json();
    if(result.transcript)ui['command-input'].value=result.transcript;
    if(!response.ok||result.ok===false)throw new Error(result.message||result.error||'The voice request could not run.');
    if(result.state)receiveState(result.state);
    ui['voice-title'].textContent=result.transcript?`"${result.transcript}"`:'Voice request accepted';
    ui['voice-help'].textContent=result.message||'Request accepted.';toast(ui['voice-help'].textContent);
    if(voiceReplies&&window.speechSynthesis){const utterance=new SpeechSynthesisUtterance(ui['voice-help'].textContent);utterance.volume=.8;window.speechSynthesis.speak(utterance);}
  }catch(error){if(!physicsFault){ui['voice-title'].textContent='Voice request could not run';ui['voice-help'].textContent=error.message;ui['voice-help'].parentElement.classList.add('error');toast(error.message,true);}}
  finally{voiceProcessing=false;ui['mic-button'].classList.remove('processing');ui['mic-button'].disabled=!connected||!microphoneSupported;}
}

async function startVoiceRecording() {
  if(!connected||voiceProcessing)return;
  if(recording){if(recording.recorder?.state==='recording')recording.recorder.stop();else cancelRecording();return;}
  const session={cancelled:false,chunks:[],startedAt:performance.now(),lastSpeechAt:performance.now(),heardSpeech:false};
  recording=session;
  ui['voice-title'].textContent='Requesting microphone access…';
  ui['voice-help'].textContent='Allow your browser to record this voice request.';
  try{
    window.speechSynthesis?.cancel();
    session.stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
    if(session.cancelled||!connected){releaseRecordingResources(session);return;}
    const mime=['audio/webm;codecs=opus','audio/ogg;codecs=opus','audio/mp4'].find((type)=>MediaRecorder.isTypeSupported(type));
    session.recorder=new MediaRecorder(session.stream,mime?{mimeType:mime}:undefined);
    session.recorder.ondataavailable=(event)=>{if(event.data.size)session.chunks.push(event.data);};
    session.recorder.onstop=()=>{const audio=new Blob(session.chunks,{type:session.recorder.mimeType||mime||'audio/webm'});releaseRecordingResources(session);if(!session.cancelled)submitVoiceAudio(audio);};
    session.recorder.onerror=()=>{session.cancelled=true;releaseRecordingResources(session);toast('The microphone recording failed. Type your request instead.',true);};
    const AudioContext=window.AudioContext||window.webkitAudioContext;
    if(AudioContext){session.audioContext=new AudioContext();await session.audioContext.resume();if(session.cancelled||!connected){releaseRecordingResources(session);return;}const source=session.audioContext.createMediaStreamSource(session.stream),analyser=session.audioContext.createAnalyser();analyser.fftSize=2048;source.connect(analyser);const samples=new Float32Array(analyser.fftSize);session.levelTimer=setInterval(()=>{analyser.getFloatTimeDomainData(samples);const rms=Math.sqrt(samples.reduce((sum,value)=>sum+value*value,0)/samples.length),now=performance.now();if(rms>.012){session.heardSpeech=true;session.lastSpeechAt=now;}if(session.heardSpeech&&now-session.lastSpeechAt>1100&&now-session.startedAt>1000&&session.recorder.state==='recording')session.recorder.stop();},80);}
    session.startedAt=performance.now();session.lastSpeechAt=session.startedAt;
    session.recorder.start(200);
    session.limitTimer=setTimeout(()=>{if(session.recorder.state==='recording')session.recorder.stop();},8000);
    ui['mic-button'].classList.add('listening');ui['mic-button'].setAttribute('aria-label','Stop and send voice recording');
    ui['voice-title'].textContent='Recording your request…';ui['voice-help'].textContent='Speak naturally. Recording stops after silence or eight seconds.';ui['voice-help'].parentElement.classList.remove('error');
  }catch(error){session.cancelled=true;releaseRecordingResources(session);if(physicsFault)return;const message=error.name==='NotAllowedError'?'Microphone access was denied. Allow access in your browser or type your request.':error.name==='NotFoundError'?'No microphone is connected. Type your request instead.':error.name==='NotReadableError'?'The microphone is busy in another application. Type your request instead.':error.message||'The microphone could not record audio.';ui['voice-title'].textContent='Microphone unavailable';ui['voice-help'].textContent=message;ui['voice-help'].parentElement.classList.add('error');toast(message,true);}
}

function setupVoice() {
  if(!microphoneSupported){ui['mic-button'].disabled=true;ui['voice-title'].textContent='Type your flight request.';ui['voice-help'].textContent='This browser cannot record microphone audio. Use the text box or open Chrome on localhost.';return;}
  ui['mic-button'].addEventListener('click',startVoiceRecording);
}
$('sound-button').addEventListener('click',()=>{voiceReplies=!voiceReplies;$('sound-button').setAttribute('aria-pressed',String(voiceReplies));$('sound-button').textContent=`Voice replies ${voiceReplies?'on':'off'}`;});
setupVoice();

async function loadScene() {
  try{
    const response=await fetch('/api/scene');if(!response.ok)throw new Error('Scene is unavailable');
    sceneDescription=await response.json();
    try{buildScene(sceneDescription);}catch(error){console.error('Yard renderer failed:',error);ui['render-error'].hidden=false;ui['loading-state'].style.display='none';}
    if(state)updateUi();
  }catch{setTimeout(loadScene,2000);}
}
const stream=new EventSource('/api/stream');
stream.addEventListener('state',(event)=>{try{receiveState(JSON.parse(event.data));}catch(error){console.error('Invalid simulator state:',error);}});
stream.onmessage=(event)=>{try{receiveState(JSON.parse(event.data));}catch{}};
stream.onerror=()=>setConnection(false);
setInterval(()=>{if(transportConnected&&performance.now()-lastStateAt>3500)setConnection(false);},1000);
setConnection(false);
fetch('/api/state').then((response)=>response.ok?response.json():null).then((value)=>{if(value)receiveState(value);}).catch(()=>{});
loadScene();
