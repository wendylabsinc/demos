import * as THREE from './vendor/three.module.min.js';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['connection-dot','connection-label','mode-badge','run-status','speed','distance','heading','collisions','coordinates','coverage','coverage-progress','ring-progress','map-resolution','map-no-data','start-button','start-label','play-icon','stop-button','person-button','clear-button','reset-button','sound-button','sound-icon','sound-label','sass','sass-description','event-log','event-count','hardware-note','control-error','speech-bubble','speech-text','camera-button','lidar-button'].map(id => [id, $(id)]));
let state = null, soundEnabled = false, audioContext = null, lastEventId = null, firstPoll = true, pollBusy = false, bubbleTimer = null, topView = false, followView = false, lidarVisible = true, sceneSignature = '', currentPose = null, displayedPose = null, trail = [], soundVoices = [];
const sassDescriptions = ['A friendly beep. A firm request. Still has manners.', 'A sharp horn. A pointed remark. No apologies.', 'Sharper language. Personal space required.'];
const mapCanvas = $('map'), mapContext = mapCanvas.getContext('2d');
const mapScratch = document.createElement('canvas'), mapScratchContext = mapScratch.getContext('2d');
let lastCameraFrame=null;
function updateSensors(data){
  const cameraState=data.sensors?.camera||{},voice=data.sensors?.speech||{},explorer=data.sensors?.exploration||{};
  $('camera-status').textContent=(cameraState.state||'waiting').toUpperCase();
  $('vision-model').textContent=cameraState.model||'Camera inference';
  $('person-count').textContent=cameraState.ready?`${cameraState.detected_people||0} people detected · ${Number(cameraState.age_ms||0).toFixed(0)} ms old`:cameraState.error||'Waiting for inference';
  if(cameraState.captured_at&&cameraState.captured_at!==lastCameraFrame){lastCameraFrame=cameraState.captured_at;$('camera-image').src=`/api/camera?frame=${encodeURIComponent(lastCameraFrame)}`;$('camera-image').hidden=false;$('camera-placeholder').hidden=true;}
  $('explorer-status').textContent=(explorer.state||'waiting').toUpperCase();
  $('voice-status').textContent=(voice.state||'waiting').toUpperCase();
  $('explorer-status').classList.toggle('unavailable',explorer.state==='unavailable');
  $('voice-status').classList.toggle('unavailable',voice.state==='unavailable');
  $('decision-text').textContent=data.intention||'Observing the room.';
  $('decision-detail').textContent=explorer.error||'The rover chooses among open places seen in its current scan, then checks the route as it moves.';
  $('model-detail').textContent=voice.error||`${voice.model||explorer.model||'Local model'} · Generated remarks and exploration decisions`;
  const remark=voice.last_result?.text; $('last-remark').hidden=!remark;
  if(remark)$('last-remark').textContent=`Last remark: “${remark}”`;
}

let renderer, scene, camera, robot, robotWheels = [], frontSteers = [], obstacleGroup, peopleGroup, lidarGroup, trailLine, scanLine, scanDots, rangeHalo, personMeshes = new Map(), worldSize = {width:14,height:12};
const mat = (color, roughness=.65, metalness=0) => new THREE.MeshStandardMaterial({color, roughness, metalness});
const materials = {floor:mat('#bebeb0',.97), edge:mat('#9fa18f'), wall:mat('#d8d7c6'), shadow:mat('#3e443b'), chassis:mat('#303835',.35,.7), deck:mat('#bd8050',.45,.55), tire:mat('#202825',.8), steel:mat('#97a19a',.28,.8), circuit:mat('#1c6663',.5), gold:mat('#c5a26c',.4,.55), lidar:mat('#f0ede0',.4), black:mat('#242e29',.55)};
function mesh(geometry,material,parent,x=0,y=0,z=0){const m=new THREE.Mesh(geometry,material);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
function box(parent,w,h,d,material,x=0,y=0,z=0){return mesh(new THREE.BoxGeometry(w,h,d),material,parent,x,y,z);}
function cylinder(parent,r,h,material,x=0,y=0,z=0){return mesh(new THREE.CylinderGeometry(r,r,h,24),material,parent,x,y,z);}
function makeTextTexture(text,color='#566049',font='600 26px Arial',width=512,height=96){const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,width,height);ctx.font=font;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=color;ctx.fillText(text,width/2,height/2);return new THREE.CanvasTexture(canvas);}
function floorText(text,x,y,w=4){const texture=makeTextTexture(text);const plane=new THREE.Mesh(new THREE.PlaneGeometry(w,w*96/512),new THREE.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false,opacity:.6}));plane.rotation.x=-Math.PI/2;plane.position.set(x,.025,-y);scene.add(plane);return plane;}

function buildRobot(){
  const group=new THREE.Group();
  box(group,.48,.035,.245,materials.chassis,0,.135,0);
  box(group,.45,.015,.24,materials.deck,0,.163,0);
  box(group,.4,.025,.018,materials.steel,0,.111,.102);box(group,.4,.025,.018,materials.steel,0,.111,-.102);
  box(group,.19,.04,.14,materials.circuit,-.025,.192,0);
  box(group,.12,.045,.1,materials.black,-.115,.233,0);
  for(let i=0;i<4;i++)box(group,.015,.012,.025,materials.gold,-.035+i*.035,.22,-.075);
  for(const z of [-.095,.095]){cylinder(group,.006,.054,materials.steel,.13,.19,z);cylinder(group,.006,.054,materials.steel,-.18,.19,z);}
  box(group,.17,.012,.19,materials.chassis,.125,.216,0);
  cylinder(group,.065,.035,materials.lidar,.105,.244,0);
  cylinder(group,.054,.035,materials.black,.105,.272,0);
  cylinder(group,.053,.004,materials.deck,.105,.292,0);
  const sensor=box(group,.025,.009,.057,materials.black,.168,.274,0);
  const lens=new THREE.MeshBasicMaterial({color:'#e6b674'});box(group,.003,.006,.029,lens,.181,.274,0);
  const lightMat=new THREE.MeshStandardMaterial({color:'#fff0c7',emissive:'#d5ad68',emissiveIntensity:.55});
  box(group,.014,.022,.035,lightMat,.247,.15,-.078);box(group,.014,.022,.035,lightMat,.247,.15,.078);
  box(group,.03,.035,.13,materials.black,.24,.109,0);
  for(const x of [-.16,.16])for(const z of [-.148,.148]){
    const steer=new THREE.Group();steer.position.set(x,.069,z);group.add(steer);
    const wheel=new THREE.Group();steer.add(wheel);
    const tire=cylinder(wheel,.071,.045,materials.tire);tire.rotation.x=Math.PI/2;
    const rim=cylinder(wheel,.042,.049,materials.steel);rim.rotation.x=Math.PI/2;
    const hub=cylinder(wheel,.019,.052,materials.black);hub.rotation.x=Math.PI/2;
    for(let i=0;i<6;i++){const tread=box(wheel,.011,.015,.047,materials.tire,Math.cos(i*Math.PI/3)*.066,Math.sin(i*Math.PI/3)*.066,0);tread.rotation.z=i*Math.PI/3;}
    robotWheels.push(wheel);if(x>0)frontSteers.push(steer);
  }
  const wireCurve=new THREE.CatmullRomCurve3([new THREE.Vector3(-.12,.249,.046),new THREE.Vector3(-.05,.267,.066),new THREE.Vector3(.01,.232,.074),new THREE.Vector3(.1,.225,.05)]);
  mesh(new THREE.TubeGeometry(wireCurve,16,.005,5,false),mat('#cb925b'),group);
  const halo=new THREE.Mesh(new THREE.RingGeometry(.36,.375,64),new THREE.MeshBasicMaterial({color:'#765e32',transparent:true,opacity:.6,side:THREE.DoubleSide,depthWrite:false}));halo.rotation.x=-Math.PI/2;halo.position.y=.018;group.add(halo);
  const pointer=new THREE.Mesh(new THREE.ConeGeometry(.06,.13,3),new THREE.MeshBasicMaterial({color:'#9b733c'}));pointer.rotation.z=-Math.PI/2;pointer.position.set(.47,.019,0);pointer.scale.z=.05;group.add(pointer);
  return group;
}

function setupScene(){
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#c9c8ba');renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
  $('scene').prepend(renderer.domElement);scene=new THREE.Scene();camera=new THREE.OrthographicCamera(-8,8,6,-6,.1,100);camera.position.set(11,14,12);camera.lookAt(0,0,0);
  const ambient=new THREE.HemisphereLight('#fff8e7','#737e65',2.2);scene.add(ambient);const sun=new THREE.DirectionalLight('#fff4d6',3.5);sun.position.set(-5,12,5);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-12;sun.shadow.camera.right=12;sun.shadow.camera.top=12;sun.shadow.camera.bottom=-12;sun.shadow.normalBias=.025;sun.shadow.bias=-.0004;scene.add(sun);
  box(scene,14.5,.3,12.5,materials.edge,0,-.16,0);box(scene,14,.03,12,materials.floor,0,0,0);
  const grid=new THREE.GridHelper(14,28,'#9eaa90','#a9ae99');grid.position.y=.02;grid.material.transparent=true;grid.material.opacity=.22;grid.scale.z=12/14;scene.add(grid);
  // Cutaway perimeter walls keep the interior and the rover visible.
  box(scene,14,.6,.16,materials.wall,0,.3,-6.05);box(scene,.16,.6,12,materials.wall,-7.05,.3,0);
  box(scene,14,.08,.12,materials.shadow,0,.04,6.03);box(scene,.12,.08,12,materials.shadow,7.03,.04,0);
  box(scene,14.14,.03,.22,materials.edge,0,.615,-6.05);box(scene,.22,.03,12.14,materials.edge,-7.05,.615,0);
  floorText('W E N D Y O S   /   T E S T   A R E N A',1.4,-4.8,5.1);
  floorText('01',-5.9,4.6,.6);floorText('02',5.9,-4.9,.6);
  for(let x=-6;x<=6;x+=2){const mark=box(scene,.01,.005,.16,materials.shadow,x,.025,5.7);mark.castShadow=false;}
  for(let y=-4;y<=4;y+=2){const mark=box(scene,.16,.005,.01,materials.shadow,6.7,.025,-y);mark.castShadow=false;}
  obstacleGroup=new THREE.Group();scene.add(obstacleGroup);peopleGroup=new THREE.Group();scene.add(peopleGroup);robot=buildRobot();scene.add(robot);
  lidarGroup=new THREE.Group();scene.add(lidarGroup);
  const scanMaterial=new THREE.LineBasicMaterial({color:'#b58a47',transparent:true,opacity:.12,depthWrite:false});scanLine=new THREE.LineSegments(new THREE.BufferGeometry(),scanMaterial);lidarGroup.add(scanLine);
  scanDots=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:'#80602e',size:.038,sizeAttenuation:true,transparent:true,opacity:.8}));lidarGroup.add(scanDots);
  trailLine=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#667d53',transparent:true,opacity:.75}));scene.add(trailLine);
  rangeHalo=new THREE.Mesh(new THREE.RingGeometry(.75,.758,80),new THREE.MeshBasicMaterial({color:'#ad8750',transparent:true,opacity:.28,side:THREE.DoubleSide,depthWrite:false}));rangeHalo.rotation.x=-Math.PI/2;rangeHalo.position.y=.023;lidarGroup.add(rangeHalo);
  const resize=()=>{const rect=$('scene').getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);const ratio=rect.width/rect.height;const viewHeight=followView?(topView?4.2:3.4):(topView?12.7:10.7);camera.left=-viewHeight*ratio/2;camera.right=viewHeight*ratio/2;camera.top=viewHeight/2;camera.bottom=-viewHeight/2;camera.updateProjectionMatrix();};
  new ResizeObserver(resize).observe($('scene'));resize();let lastTime=performance.now();
  function animate(time){requestAnimationFrame(animate);const dt=Math.min((time-lastTime)/1000,.08);lastTime=time;
    if(currentPose){if(!displayedPose)displayedPose={...currentPose};const blend=1-Math.exp(-dt*18);displayedPose.x+=(currentPose.x-displayedPose.x)*blend;displayedPose.y+=(currentPose.y-displayedPose.y)*blend;let diff=Math.atan2(Math.sin(currentPose.yaw-displayedPose.yaw),Math.cos(currentPose.yaw-displayedPose.yaw));displayedPose.yaw+=diff*blend;robot.position.set(displayedPose.x,0,-displayedPose.y);robot.rotation.y=displayedPose.yaw;rangeHalo.position.set(displayedPose.x,.025,-displayedPose.y);for(const wheel of robotWheels)wheel.rotation.z-=Number(state?.speed||0)*dt/.071;for(const steer of frontSteers)steer.rotation.y=Number(state?.steering||0);}
    const center=followView&&displayedPose?new THREE.Vector3(displayedPose.x,0,-displayedPose.y):new THREE.Vector3();const target=topView?new THREE.Vector3(center.x+.001,19,center.z+.001):new THREE.Vector3(center.x+11,14,center.z+12);camera.position.lerp(target,1-Math.exp(-dt*8));camera.lookAt(center);renderer.render(scene,camera);
  }requestAnimationFrame(animate);
  $('follow-button').addEventListener('click',()=>{followView=!followView;$('follow-button').classList.toggle('active',followView);$('follow-button').setAttribute('aria-pressed',String(followView));$('follow-button').title=followView?'Show whole arena':'Follow rover';resize();});
  $('camera-button').addEventListener('click',()=>{topView=!topView;$('camera-button').classList.toggle('active',topView);$('camera-button').setAttribute('aria-label',topView?'Switch to isometric view':'Switch to overhead view');$('camera-button').title=topView?'Switch to isometric view':'Switch to overhead view';resize();});
}

function disposeGroup(group){while(group.children.length){const object=group.children[0];group.remove(object);object.traverse(child=>{child.geometry?.dispose();if(child.material&&!Object.values(materials).includes(child.material)){if(Array.isArray(child.material))child.material.forEach(m=>m.dispose());else child.material.dispose();}});}}
function updateObstacles(obstacles){if(!scene)return;const signature=JSON.stringify(obstacles);if(signature===sceneSignature)return;sceneSignature=signature;disposeGroup(obstacleGroup);for(const o of obstacles){const group=new THREE.Group();group.position.set(o.x,0,-o.y);const color=o.color||'#788573';const material=mat(color,.78);const h=Number(o.height||.65);if(o.kind==='circle'){cylinder(group,o.r||.4,h,material,0,h/2,0);cylinder(group,(o.r||.4)*.91,.025,mat('#9b9b80'),0,h+.012,0);if(h<1.1){const soil=cylinder(group,(o.r||.4)*.83,.025,mat('#555d43'),0,h+.027,0);for(let i=0;i<8;i++){const leaf=mesh(new THREE.SphereGeometry(.18,8,6),mat(i%2?'#657a50':'#829164'),group,Math.cos(i*2.4)*.17,h+.18+i*.016,Math.sin(i*2.4)*.17);leaf.scale.set(.8,1.6,.45);leaf.rotation.z=.5*Math.cos(i*2.4);}}}else{const w=o.w||1,d=o.h||1;box(group,w,h,d,material,0,h/2,0);const rimColor=mat('#aaa994',.85);box(group,w+.025,.03,d+.025,rimColor,0,h+.015,0);for(const x of [-w/2+.07,w/2-.07])for(const z of [-d/2+.07,d/2-.07]){if(h>.3)box(group,.075,.04,.075,materials.shadow,x,.025,z);}if(w>1.3&&d>.5&&h<.9){for(let i=0;i<3;i++)box(group,(w-.13)/3,.035,d-.13,mat('#8e9580'),-w/3+i*w/3,h+.05,0);box(group,w,.24,.12,material,0,h+.1,-d/2+.06);}}obstacleGroup.add(group);}}
function makePerson(){const g=new THREE.Group();const skin=mat('#bb9a76'),clothes=mat('#b27454'),pants=mat('#45534d');const leg1=box(g,.11,.37,.12,pants,-.08,.19,0),leg2=box(g,.11,.37,.12,pants,.08,.19,0);box(g,.32,.36,.21,clothes,0,.54,0);mesh(new THREE.SphereGeometry(.105,16,12),skin,g,0,.86,0);cylinder(g,.12,.035,materials.black,0,.955,0);cylinder(g,.09,.07,materials.black,0,.999,0);for(const x of [-.21,.21]){const arm=box(g,.075,.34,.09,clothes,x,.53,0);arm.rotation.z=x<0?-.1:.1;}box(g,.12,.07,.18,materials.black,-.08,.035,.02);box(g,.12,.07,.18,materials.black,.08,.035,.02);g.userData.legs=[leg1,leg2];const ring=new THREE.Mesh(new THREE.RingGeometry(.26,.28,40),new THREE.MeshBasicMaterial({color:'#c0835b',side:THREE.DoubleSide,transparent:true,opacity:.75}));ring.rotation.x=-Math.PI/2;ring.position.y=.02;g.add(ring);return g;}
function updatePeople(people){if(!scene)return;const alive=new Set();for(const p of people){alive.add(p.id);let person=personMeshes.get(p.id);if(!person){person=makePerson();peopleGroup.add(person);personMeshes.set(p.id,person);}person.position.set(p.x,0,-p.y);}for(const[id,person]of personMeshes)if(!alive.has(id)){peopleGroup.remove(person);person.traverse(child=>child.geometry?.dispose());personMeshes.delete(id);}}
function updateScan(data){if(!scene||!data.pose)return;const points=[],rays=[];for(const p of (data.scan||[])){if(!Number.isFinite(p.x)||!Number.isFinite(p.y))continue;points.push(p.x,.055,-p.y);rays.push(data.pose.x,.04,-data.pose.y,p.x,.04,-p.y);}scanLine.geometry.dispose();scanLine.geometry=new THREE.BufferGeometry();scanLine.geometry.setAttribute('position',new THREE.Float32BufferAttribute(rays,3));scanDots.geometry.dispose();scanDots.geometry=new THREE.BufferGeometry();scanDots.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));lidarGroup.visible=lidarVisible;
  if(data.pose&&data.running){const prev=trail.at(-1);if(!prev||Math.hypot(data.pose.x-prev.x,data.pose.y-prev.y)>.06){trail.push({x:data.pose.x,y:data.pose.y});if(trail.length>1800)trail.shift();trailLine.geometry.dispose();trailLine.geometry=new THREE.BufferGeometry().setFromPoints(trail.map(p=>new THREE.Vector3(p.x,.031,-p.y)));}}
}

function drawMap(data){const map=data.map;if(!map||!map.width||!map.height||!map.cells)return;mapScratch.width=map.width;mapScratch.height=map.height;const image=mapScratchContext.createImageData(map.width,map.height);for(let y=0;y<map.height;y++)for(let x=0;x<map.width;x++){const value=map.cells[y*map.width+x];const i=((map.height-1-y)*map.width+x)*4;const c=value<0?[39,46,32]:value>=50?[205,186,141]:[104,128,76];image.data[i]=c[0];image.data[i+1]=c[1];image.data[i+2]=c[2];image.data[i+3]=255;}mapScratchContext.putImageData(image,0,0);const bounds=mapCanvas.getBoundingClientRect();const scale=Math.min(devicePixelRatio,2);mapCanvas.width=Math.round(bounds.width*scale);mapCanvas.height=Math.round(bounds.height*scale);mapContext.imageSmoothingEnabled=false;mapContext.drawImage(mapScratch,0,0,mapCanvas.width,mapCanvas.height);
  const origin=map.origin||[-7,-6],resolution=map.resolution||.2;const toMap=p=>[(p.x-origin[0])/(map.width*resolution)*mapCanvas.width,(1-(p.y-origin[1])/(map.height*resolution))*mapCanvas.height];
  mapContext.strokeStyle='#c2cea377';mapContext.lineWidth=1;mapContext.beginPath();for(let i=0;i<trail.length;i++){const[x,y]=toMap(trail[i]);if(i===0)mapContext.moveTo(x,y);else mapContext.lineTo(x,y);}mapContext.stroke();
  for(const p of data.observed_people||[]){const[x,y]=toMap(p);mapContext.fillStyle='#ed9c73';mapContext.beginPath();mapContext.arc(x,y,3*scale,0,Math.PI*2);mapContext.fill();}
  if(data.pose){const[x,y]=toMap(data.pose);mapContext.save();mapContext.translate(x,y);mapContext.rotate(-data.pose.yaw);mapContext.fillStyle='#f6e5b9';mapContext.shadowColor='#f3d69a';mapContext.shadowBlur=7;mapContext.beginPath();mapContext.moveTo(6*scale,0);mapContext.lineTo(-4*scale,-3.8*scale);mapContext.lineTo(-2*scale,0);mapContext.lineTo(-4*scale,3.8*scale);mapContext.closePath();mapContext.fill();mapContext.restore();}
  ui['map-no-data'].hidden=Number(map.coverage)>0;ui['map-resolution'].textContent=`LIDAR · ${Number(resolution).toFixed(2)} M`;
}

function connected(ok){ui['connection-dot'].parentElement.classList.toggle('connected',ok);ui['connection-dot'].parentElement.classList.toggle('disconnected',!ok);ui['connection-label'].textContent=ok?'ROVER CONNECTED':'RECONNECTING';}
function fmt(n,d=1){return Number(n||0).toFixed(d);}
function updateState(data){state=data;connected(true);currentPose=data.pose;ui['mode-badge'].textContent=data.mode==='hardware'?'HARDWARE':'SIMULATION';ui['run-status'].classList.toggle('running',data.running);ui['run-status'].classList.toggle('fault',!!data.fault);ui['run-status'].innerHTML='<i></i>';ui['run-status'].append(document.createTextNode(data.fault?'FAULT · STOPPED':data.running?(data.status||'EXPLORING').toUpperCase():'STANDING BY'));ui['speed'].innerHTML=`${fmt(data.speed,2)} <small>m/s</small>`;ui['distance'].innerHTML=`${fmt(data.distance)} <small>m</small>`;ui['heading'].innerHTML=`${Math.round(((Number(data.pose?.yaw||0)*180/Math.PI)%360+360)%360)} <small>°</small>`;ui['collisions'].textContent=data.collisions||0;ui['coordinates'].textContent=`X ${fmt(data.pose?.x,2)} · Y ${fmt(data.pose?.y,2)}`;
  const cov=Number(data.map?.coverage||0);const coverage=Math.max(0,Math.min(100,cov));ui['coverage'].innerHTML=`${Math.round(coverage)}<small>%</small>`;ui['coverage-progress'].style.width=`${coverage}%`;ui['ring-progress'].style.strokeDashoffset=150.796*(1-coverage/100);ui['start-label'].textContent=data.running?'Pause exploration':data.mode==='hardware'?'Start rover':'Let it roam';ui['play-icon'].innerHTML=data.running?'<path d="M7 5h3v14H7V5Zm7 0h3v14h-3V5Z"/>':'<path d="m8 5 11 7-11 7V5Z"/>';
  if(document.activeElement!==ui.sass&&Number.isFinite(Number(data.sass)))ui.sass.value=data.sass;updateSassDescription();ui['hardware-note'].hidden=data.mode!=='hardware';ui['hardware-note'].textContent=data.fault?`Rover stopped. ${data.fault}`:'Hardware mode. Motion uses live ROS lidar and odometry. Keep the physical emergency stop within reach.';
  ui['person-button'].disabled=data.mode==='hardware';ui['clear-button'].disabled=data.mode==='hardware';ui['person-button'].title=data.mode==='hardware'?'People can only be added in simulation':'Place a person in the rover’s path';
  if(data.fault){ui['control-error'].textContent=String(data.fault);ui['control-error'].hidden=false;}updateObstacles(data.scene?.obstacles||[]);updatePeople(data.scene?.people||[]);updateScan(data);drawMap(data);handleEvents(data.events||[]);updateSensors(data);firstPoll=false;
}
function updateSassDescription(){ui['sass-description'].textContent=sassDescriptions[Number(ui.sass.value)-1]||sassDescriptions[1];const spans=document.querySelectorAll('.sass-labels span');spans.forEach((s,i)=>s.style.color=i===Number(ui.sass.value)-1?'var(--gold)':'');}
function formatTime(value){const date=typeof value==='number'?new Date(value<1e12?value*1000:value):new Date(value);return Number.isNaN(date.getTime())?'LIVE':date.toLocaleTimeString([], {hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});}
function handleEvents(events){const fresh=firstPoll?[]:events.filter(e=>lastEventId===null||Number(e.id)>Number(lastEventId));if(events.length)lastEventId=events.at(-1).id;ui['event-count'].textContent=`${String(events.filter(e=>e.kind==='speech').length).padStart(2,'0')} REMARKS`;if(events.length){ui['event-log'].replaceChildren();for(const e of [...events].reverse().slice(0,12)){const row=document.createElement('div');row.className=`event ${e.kind}`;const time=document.createElement('time');time.textContent=formatTime(e.time);const mark=document.createElement('span');mark.className='event-mark';mark.textContent=e.kind==='speech'?'“':e.kind==='horn'?'♪':'·';const text=document.createElement('p');text.textContent=e.text;row.append(time,mark,text);ui['event-log'].append(row);}}
  for(const e of fresh){if(e.kind==='speech'){ui['speech-text'].textContent=e.text;ui['speech-bubble'].hidden=false;clearTimeout(bubbleTimer);bubbleTimer=setTimeout(()=>ui['speech-bubble'].hidden=true,4500);if(soundEnabled)say(e.text);}if(e.kind==='horn'&&soundEnabled)honk();}
}
async function control(action,value){ui['control-error'].hidden=true;try{const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value===undefined?{action}:{action,value})});if(!response.ok){let detail;try{const body=await response.json();detail=body.error||body.detail;}catch{}throw new Error(detail||`The rover rejected the command (${response.status}).`);}const data=await response.json();if(action==='reset'){trail=[];displayedPose=null;if(trailLine){trailLine.geometry.dispose();trailLine.geometry=new THREE.BufferGeometry();}lastEventId=null;firstPoll=true;ui['event-log'].innerHTML='<p class="empty-log">Waiting for a generated remark.<span>ROSMASTER, STANDING BY</span></p>';ui['speech-bubble'].hidden=true;}updateState(data);}catch(error){ui['control-error'].textContent=error.message;ui['control-error'].hidden=false;}}
function honk(){if(!audioContext)return;const t=audioContext.currentTime;for(const frequency of [410,525]){const oscillator=audioContext.createOscillator(),gain=audioContext.createGain();oscillator.type='sawtooth';oscillator.frequency.value=frequency;gain.gain.setValueAtTime(0,t);gain.gain.linearRampToValueAtTime(.045,t+.025);gain.gain.setValueAtTime(.045,t+.2);gain.gain.linearRampToValueAtTime(0,t+.25);oscillator.connect(gain);gain.connect(audioContext.destination);oscillator.start(t);oscillator.stop(t+.26);}}
function say(text){if(!('speechSynthesis'in window))return;const speech=new SpeechSynthesisUtterance(text);speech.lang='en-GB';speech.rate=1.02;speech.pitch=.86;speech.volume=.9;const british=soundVoices.filter(v=>v.lang.startsWith('en-GB'));const voice=british.find(v=>/daniel|male|george|arthur|oliver/i.test(v.name))||british[0]||soundVoices.find(v=>v.lang.startsWith('en'));if(voice)speech.voice=voice;window.speechSynthesis.cancel();window.speechSynthesis.speak(speech);}
if('speechSynthesis'in window){soundVoices=window.speechSynthesis.getVoices();window.speechSynthesis.addEventListener('voiceschanged',()=>soundVoices=window.speechSynthesis.getVoices());}
ui['sound-button'].addEventListener('click',async()=>{soundEnabled=!soundEnabled;ui['sound-button'].setAttribute('aria-pressed',String(soundEnabled));ui['sound-label'].textContent=soundEnabled?'Sound on':'Enable sound';ui['sound-icon'].innerHTML=soundEnabled?'<path d="M3 9h4l5-4v14l-5-4H3V9Zm13-1a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>':'<path d="M3 9h4l5-4v14l-5-4H3V9Zm13 0 5 6m0-6-5 6"/>';if(soundEnabled){const AudioAPI=window.AudioContext||window.webkitAudioContext;if(AudioAPI){audioContext||=new AudioAPI();await audioContext.resume();honk();}}else window.speechSynthesis?.cancel();});
ui['start-button'].addEventListener('click',()=>control(state?.running?'stop':'start'));ui['stop-button'].addEventListener('click',()=>control('estop'));ui['person-button'].addEventListener('click',()=>control('person'));ui['clear-button'].addEventListener('click',()=>control('clear_people'));ui['reset-button'].addEventListener('click',()=>control('reset'));ui.sass.addEventListener('input',updateSassDescription);ui.sass.addEventListener('change',()=>control('sass',Number(ui.sass.value)));ui['lidar-button'].addEventListener('click',()=>{lidarVisible=!lidarVisible;ui['lidar-button'].classList.toggle('active',lidarVisible);ui['lidar-button'].setAttribute('aria-pressed',String(lidarVisible));if(lidarGroup)lidarGroup.visible=lidarVisible;});
try{setupScene();}catch(error){$('scene-error').textContent='The 3D view needs WebGL. The live map and rover controls are still available.';$('scene-error').hidden=false;console.warn('3D view unavailable:',error);}
async function poll(){if(pollBusy)return;pollBusy=true;try{const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);updateState(await response.json());}catch(error){connected(false);}finally{pollBusy=false;}}
updateSassDescription();poll();setInterval(poll,125);window.addEventListener('resize',()=>state&&drawMap(state));
