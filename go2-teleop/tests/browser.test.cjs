// Deterministic DOM/socket/timer fakes; this is not an iOS/device test.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const script = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
function browser() {
  const elements = new Map(), listeners = new Map(), timers = new Map(), sockets = [];
  let sequence = 0, timerId = 0;
  const element = id => {
    if (!elements.has(id)) elements.set(id, {disabled:false, textContent:'', dataset:{}, setPointerCapture(){}});
    return elements.get(id);
  };
  const moves = ['forward','back','left','right'].map(element);
  moves[0].dataset = {f:'0.15',y:'0'};moves[1].dataset = {f:'-0.15',y:'0'};
  moves[2].dataset = {f:'0',y:'0.3'};moves[3].dataset = {f:'0',y:'-0.3'};
  class Socket {
    static OPEN = 1;
    constructor(){this.readyState=1;this.frames=[];sockets.push(this)}
    send(raw){this.frames.push(JSON.parse(raw))}
    close(){this.readyState=3;this.closed=true}
    ack(){this.onmessage({data:JSON.stringify({challenge:`challenge-${++sequence}`,timeout_ms:350,mode:'fake'})})}
  }
  const document={hidden:false,hasFocus:()=>true,getElementById:element,querySelectorAll:()=>moves,addEventListener:(name,fn)=>listeners.set(name,fn)};
  vm.runInNewContext(script, {document,window:{addEventListener:(name,fn)=>listeners.set(name,fn)},location:{protocol:'http:',host:'localhost:8882'},WebSocket:Socket,
    setTimeout:(fn,ms)=>{const id=++timerId;timers.set(id,{fn,ms});return id},clearTimeout:id=>timers.delete(id)});
  const tick = ms => {const item=[...timers].find(([,timer])=>timer.ms===ms);assert.ok(item,`no ${ms}ms timer`);timers.delete(item[0]);item[1].fn()};
  const down = () => moves[0].onpointerdown({preventDefault(){},pointerId:7});
  const up = () => moves[0].onpointerup({pointerId:7});
  const enable = () => {element('enable').onclick();return sockets.at(-1)};
  return {elements,element,listeners,timers,sockets,moves,document,tick,down,up,enable};
}
test('Enable does not send movement before initial acknowledgement',()=>{
  const b=browser(),s=b.enable();b.down();assert.equal(s.frames.length,0);s.ack();b.tick(100);assert.equal(s.frames[0].forward,0);
});
test('release while request is in flight drops held movement, not queues it',()=>{
  const b=browser(),s=b.enable();s.ack();b.down();assert.equal(s.frames[0].forward,.15);b.up();assert.equal(s.frames.length,1);s.ack();b.tick(100);assert.equal(s.frames[1].forward,0);assert.equal(s.frames[1].yaw,0);
});
test('release with available challenge sends zero immediately',()=>{
  const b=browser(),s=b.enable();s.ack();b.down();s.ack();b.up();assert.equal(s.frames[1].forward,0);
});
for(const event of ['blur','pagehide','visibilitychange','pointercancel','stop','watchdog','disconnect']) test(`${event} disables, ignores late replies and never reconnects`,()=>{
  const b=browser(),s=b.enable();s.ack();b.down();
  if(event==='visibilitychange'){b.document.hidden=true;b.listeners.get(event)()}
  else if(event==='pointercancel')b.moves[0].onpointercancel();
  else if(event==='stop')b.element('stop').onclick();
  else if(event==='watchdog')b.tick(300);
  else if(event==='disconnect')s.onclose();
  else b.listeners.get(event)();
  assert.ok(s.closed);s.ack();b.down();assert.equal(s.frames.length,1);assert.equal(b.sockets.length,1);assert.ok(b.moves.every(m=>m.disabled));
});
test('fresh explicit Enable resets all held movement',()=>{
  const b=browser(),s=b.enable();s.ack();b.down();b.element('stop').onclick();const next=b.enable();next.ack();b.tick(100);assert.equal(next.frames[0].forward,0);
});
