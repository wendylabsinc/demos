"""Single-owner RealSense capture module.

Interface:
  GET /health    capture health and freshness
  GET /frame.bgr latest uncompressed BGR8 frame for the detector
  GET /frame.jpg latest JPEG, encoded lazily for human/debug clients
  POST /lease/*  short-lived exclusive reader ownership
  GET /debug     bounded diagnostic state
"""

from __future__ import annotations

import io
import json
import logging
import math
import threading
import time
import os
import uuid
from urllib.request import Request as Request_, urlopen
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .lease import CameraLeaseBusy, CameraLeaseDenied, CameraLeaseRegistry
from .settings import CameraSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coke-detector.camera")

SEGMENTATION_URL = os.environ.get("SEGMENTATION_URL", "http://127.0.0.1:8003").rstrip("/")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8117").rstrip("/")
WRAPPER_URL = os.environ.get("PHYSICAL_WRAPPER_URL", "http://127.0.0.1:8121").rstrip("/")
FSM_BRIDGE_URL = os.environ.get("G1_FSM_BRIDGE_URL", "http://127.0.0.1:8100").rstrip("/")

#: What the policy consumes. The segmentation service scales detector pixels and
#: camera intrinsics from the source size into this one, and hardcodes the
#: source, so these four belong together and are asserted, not assumed.
SOURCE_WIDTH, SOURCE_HEIGHT = 640, 480
POLICY_WIDTH, POLICY_HEIGHT = 320, 240




#: From docs/measurements/2026-09-17-camera-field-of-view.md: between the can
#: entering view and the grasp it sweeps +/-8.9 deg vertically, so this much
#: vertical field holds it. Colour's 43.4 deg clears it comfortably; the earlier
#: 53.5 figure counted the opening waist turn, during which the can is behind
#: the camera and nothing is looking at it.
REQUIRED_VERTICAL_DEG = 17.8

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>G1 cameras</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#0f1115;--fg:#e6e8ec;--dim:#9aa3b2;--ok:#3ddc84;--bad:#ff6b6b;--card:#171a21;--line:#262b36}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--fg:#11151c;--dim:#5b6472;--card:#fff;--line:#e2e6ec}}
*{box-sizing:border-box}
html,body{max-width:100%;overflow-x:hidden}
body{margin:0 auto;background:var(--bg);color:var(--fg);
font:13px/1.45 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
padding:14px 16px;max-width:1600px}
h1{font-size:17px;margin:0 0 2px}.sub{color:var(--dim);margin:0 0 14px}
/* auto-fit would stretch three cards across a 4K display and leave every table
   swimming in whitespace; auto-fill with a ceiling keeps a readable column
   width and simply uses more rows. */
.grid{display:grid;gap:14px;
grid-template-columns:repeat(auto-fill,minmax(300px,380px));
justify-content:start;align-items:start}
@media(min-width:2000px){.grid{grid-template-columns:repeat(auto-fill,minmax(320px,400px))}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 9px}
img{width:100%;max-width:100%;border-radius:6px;background:#000;display:block;
aspect-ratio:4/3;object-fit:contain}
/* a long reason string must wrap rather than widen its column */
td,th{overflow-wrap:anywhere}
.card{min-width:0}
table{width:100%;border-collapse:collapse;table-layout:fixed}
td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);font-weight:500}td.n{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok)}.bad{color:var(--bad)}.k{color:var(--dim)}
.note{margin-top:12px;padding:10px 12px;border-left:3px solid var(--bad);
background:rgba(255,107,107,.08);border-radius:0 6px 6px 0;font-size:12px}
.note code{overflow-wrap:anywhere}
.bar{height:7px;border-radius:4px;background:var(--line);overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;border-radius:4px;background:var(--ok);transition:width .3s}
.bar.warn>i{background:#f0b429}.bar.over>i{background:var(--bad)}
.off{opacity:.45}
</style></head><body>
<h1>G1 cameras</h1><p class="sub" id="sub">loading…</p>
<div class="grid">
  <div class="card"><h2>Colour &mdash; what the policy is given</h2>
    <div style="position:relative">
      <img id="rgb" alt="colour">
      <div id="aim" style="position:absolute;inset:0;pointer-events:none">
        <div style="position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(255,255,255,.45)"></div>
        <div style="position:absolute;top:0;bottom:0;left:50%;width:1px;background:rgba(255,255,255,.45)"></div>
        <!-- Where the can should sit through the whole reach, measured over
             frames 94-400 of the sealed reference at 43.36 deg: v 111-143 of
             240 once the camera is tilted ~18 deg further down. -->
        <div style="position:absolute;left:38%;right:22%;top:46.2%;bottom:40.4%;
                    border:2px solid var(--ok);border-radius:4px"></div>
        <div style="position:absolute;left:38%;top:39%;color:var(--ok);font-size:11px;
                    text-shadow:0 1px 3px #000">put the can here</div>
      </div>
    </div>
    <label class="sub" style="display:block;margin:8px 0">
      <input type="checkbox" id="aimon" checked> aiming guide
    </label>
    <table><tr><th>detector sees the can</th><td class="n" id="dconf">—</td></tr>
    <tr><th>segmentation mask</th><td class="n" id="mvalid">—</td></tr>
    <tr><th>mask found, last 60 s</th><td class="n" id="srate">—</td></tr>
    <tr><th>longest unbroken miss</th><td class="n" id="sgap">—</td></tr>
    <tr><th>segmentation rate</th><td class="n" id="shz">—</td></tr></table>
    <p class="sub" style="margin:8px 0 0">Aim for the can inside the box. The
    grasp gate still wants 0.30; a mask no longer does.</p></div>
  <div class="card"><h2>Target &mdash; :8003 &mdash; the gate</h2>
    <div style="position:relative">
      <img id="det" alt="detector">
      <div id="detbox" style="position:absolute;border:2px solid var(--bad);border-radius:2px;display:none"></div>
      <div id="detlab" style="position:absolute;font-size:11px;padding:1px 4px;border-radius:3px;
           background:rgba(0,0,0,.65);color:#fff;display:none"></div>
    </div>
    <table style="margin-top:10px"><tr><th>confidence</th><td class="n" id="g_conf">—</td></tr>
      <tr><th>clear of edge</th><td class="n" id="g_edge">—</td></tr>
      <tr><th>plausible shape</th><td class="n" id="g_shape">—</td></tr>
      <tr><th><b>eligible to grasp</b></th><td class="n" id="g_all">—</td></tr>
      <tr><th><b>eligible for a mask</b></th><td class="n" id="g_mask">—</td></tr>
      <tr><th>segmentation says</th><td class="n" id="g_seg">—</td></tr></table>
    <div style="display:flex;gap:8px;align-items:center;margin-top:10px">
      <label class="sub" style="flex:1">confidence gate
        <input type="number" id="thr" step="0.01" min="0.01" max="0.99" style="width:70px;
          padding:5px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--fg)">
      </label>
      <button id="setthr" style="padding:7px 12px;border-radius:6px;border:1px solid var(--line);
        background:var(--card);color:var(--fg);font:inherit;cursor:pointer">Apply</button>
    </div>
    <p class="sub" id="thrnote" style="margin:8px 0 0">All three must pass to grasp;
    the lower two are all a mask needs. Boxes and mask now come from one model, so
    they cannot disagree about which frame they describe.</p></div>
  <div class="card"><h2>Segmentation mask &mdash; :8003</h2><img id="seg" alt="mask"></div>
  <div class="card"><h2>Depth, unaligned &mdash; the stereo module's own 63.6&deg;</h2>
    <img id="dep" alt="depth"><p class="sub" style="margin:8px 0 0">Everything outside
    the colour panel's framing is field the policy never receives: alignment crops it away.</p></div>
  <div class="card"><h2>Fields of view, measured</h2><table id="fov"></table>
    <div class="note" style="border-left-color:var(--dim);background:rgba(128,128,128,.08)">
    Vision runs from the can entering view to the grasp, and over that window it
    sweeps <b>17.8&deg;</b> vertical &mdash; colour's 43.4&deg; holds it. Depth is
    still warped into the colour frame, so the wider stereo field is captured and
    discarded; that costs context, not the can.</div></div>
  <div class="card"><h2>Where the robot thinks the can is</h2>
    <table id="rt"></table>
    <div class="note" id="rtnote" style="display:none"></div></div>

  <div class="card"><h2>Is the camera where the code thinks?</h2>
    <table id="cal"></table>
    <button id="recal" style="margin-top:10px;padding:8px 12px;border-radius:6px;
      border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit;
      cursor:pointer">Measure again</button>
    <div class="note" id="calnote" style="display:none"></div>
    <p class="sub" style="margin:10px 0 0">Measured by fitting the surface the camera
    is already looking at &mdash; no target, nothing moved. Tilt and height only; it
    says nothing about the sideways offset.</p></div>

  <div class="card"><h2>How close to the limits</h2>
    <div id="guards"><p class="sub">waiting for the wrapper…</p></div>
    <p class="sub" style="margin:10px 0 0">A guard you can watch approaching is a
    warning. A guard that has fired is a post-mortem.</p></div>

  <div class="card"><h2>Was it looking</h2><table id="vis"></table></div>

  <div class="card"><h2>Is the robot willing</h2><table id="rdy"></table>
    <p class="sub" style="margin:10px 0 0">The checks a run is refused on, before
    anything moves.</p></div>

  <div class="card"><h2>All modes this camera offers</h2><table id="modes"></table></div>
  <div class="card"><h2>Inspect any mode</h2>
    <select id="pick" style="width:100%;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);font:inherit"></select>
    <div style="display:flex;gap:8px;margin:10px 0">
      <button id="go" style="flex:1;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit;cursor:pointer">Switch</button>
      <button id="back" style="flex:1;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit;cursor:pointer">Back to policy mode</button>
    </div>
    <p class="sub" id="sel" style="margin:0 0 10px"></p>
    <img id="insp" alt="inspection">
    <div class="note">Inspecting takes the camera <b>out of policy service</b> —
    segmentation asserts a 640&times;480 source. It is refused while the camera
    lease is held, so it cannot happen under a running policy.</div></div>
</div>
<script>
const R=17.8;
const cls=v=>v>=R?'ok':'bad';
async function j(u){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw 0;return r.json()}
async function tick(){
 try{
  const s=await j('/streams'), h=await j('/health');
  document.getElementById('sub').textContent=
    `serial ${s.serial} · ${h.width}\u00d7${h.height} @${h.fps} fps · ${s.alignment} · frame ${h.frame_id}`;
  let t='<tr><th>stream</th><th>size</th><th>horiz</th><th>vert</th><th>lens centre</th></tr>';
  for(const [k,v] of Object.entries(s.active||{}))
    t+=`<tr><td>${k}</td><td class="n k">${v.width}\u00d7${v.height}</td>`+
       `<td class="n">${v.horizontal_deg.toFixed(1)}\u00b0</td>`+
       `<td class="n ${cls(v.vertical_deg)}">${v.vertical_deg.toFixed(1)}\u00b0</td>`+
       `<td class="n k">${v.principal_point_offset_px.y>=0?'+':''}${v.principal_point_offset_px.y.toFixed(1)} px</td></tr>`;
  document.getElementById('fov').innerHTML=t;
  let m='<tr><th>sensor</th><th>stream</th><th>size</th><th>horiz</th><th>vert</th></tr>',seen=new Set();
  for(const sn of (s.device?.sensors||[]))for(const d of sn.modes){
    const key=sn.sensor+d.stream+d.vertical_deg.toFixed(1);if(seen.has(key))continue;seen.add(key);
    m+=`<tr><td class="k">${sn.sensor}</td><td>${d.stream}</td>`+
       `<td class="n k">${d.width}\u00d7${d.height}</td><td class="n">${d.horizontal_deg.toFixed(1)}\u00b0</td>`+
       `<td class="n ${cls(d.vertical_deg)}">${d.vertical_deg.toFixed(1)}\u00b0</td></tr>`;}
  document.getElementById('modes').innerHTML=m;
 }catch(e){document.getElementById('sub').textContent='camera unreachable'}
}
function feeds(){const t=Date.now();
 document.getElementById('rgb').src='/frame.jpg?t='+t;
 document.getElementById('seg').src=location.protocol+'//'+location.hostname+':8003/preview.jpg?t='+t;
 document.getElementById('dep').src='/depth.jpg?t='+t;
 document.getElementById('det').src='/frame.jpg?t='+t;}
let MODES=[];
function fillPicker(s){
 if(MODES.length)return;
 const sel=document.getElementById('pick');
 for(const sn of (s.device?.sensors||[]))for(const d of sn.modes)for(const f of d.fps){
   MODES.push({stream:d.stream,width:d.width,height:d.height,fps:f,v:d.vertical_deg,h:d.horizontal_deg});}
 MODES.sort((a,b)=>a.stream.localeCompare(b.stream)||b.v-a.v||b.fps-a.fps);
 sel.innerHTML=MODES.map((m,i)=>`<option value="${i}">${m.stream} ${m.width}\u00d7${m.height} @${m.fps} \u2014 V ${m.v.toFixed(1)}\u00b0 H ${m.h.toFixed(1)}\u00b0</option>`).join('');
}
async function post(body){
 const r=await fetch('/streams/select',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(body)});
 const v=await r.json().catch(()=>({}));
 document.getElementById('sel').textContent=r.ok
   ?(v.policy_mode?'policy mode \u2014 serving the policy path':
     `inspecting ${v.requested.stream} ${v.requested.width}\u00d7${v.requested.height} @${v.requested.fps} \u2014 NOT serving the policy path`)
   :('refused: '+(v.detail||r.status));
}
document.getElementById('go').onclick=()=>{const m=MODES[document.getElementById('pick').value];
 if(m)post({stream:m.stream,width:m.width,height:m.height,fps:m.fps})};
document.getElementById('back').onclick=()=>post(null);
document.getElementById('setthr').onclick=async()=>{
  const v=Number(document.getElementById('thr').value);
  const n=document.getElementById('thrnote');
  try{const r=await fetch('/recognition/grasp-threshold',{method:'POST',
       headers:{'Content-Type':'application/json'},
       body:JSON.stringify({grasp_min_confidence:v})});
   const j=await r.json();
   n.textContent=r.ok?`gate now ${j.grasp_min_confidence} (not persistent — resets on restart)`
                     :`refused: ${j.detail||r.status}`;
  }catch(e){n.textContent='could not reach the detector'}};
document.getElementById('recal').onclick=calibration;
document.getElementById('aimon').onchange=e=>{
  document.getElementById('aim').style.display=e.target.checked?'block':'none'};
async function recognition(){
 try{const v=await j('/recognition');
  const c=Number(v.detector_confidence||0), t=Number(v.detector_threshold||0.3);
  const d=document.getElementById('dconf');
  d.textContent=c.toFixed(3); d.className='n '+(c>=t?'ok':'bad');
  const m=document.getElementById('mvalid');
  m.textContent=v.target_mask_valid?'valid':(v.segmentation_reason||'not valid');
  m.className='n '+(v.target_mask_valid?'ok':'bad');
  // the detector's own box, drawn over the frame it was found in
  const img=document.getElementById('det'), bx=document.getElementById('detbox'),
        lb=document.getElementById('detlab');
  const fw=(v.frame_size&&v.frame_size[0])||640, fh=(v.frame_size&&v.frame_size[1])||480;
  if(v.detector_box&&img.clientWidth){
    const sx=img.clientWidth/fw, sy=img.clientHeight/fh, b=v.detector_box;
    bx.style.display=lb.style.display='block';
    bx.style.left=(b.x1*sx)+'px'; bx.style.top=(b.y1*sy)+'px';
    bx.style.width=((b.x2-b.x1)*sx)+'px'; bx.style.height=((b.y2-b.y1)*sy)+'px';
    bx.style.borderColor=v.detector_eligible?'var(--ok)':'var(--bad)';
    lb.style.left=(b.x1*sx)+'px'; lb.style.top=Math.max(0,b.y1*sy-16)+'px';
    lb.textContent=Number(v.detector_confidence||0).toFixed(2)+(v.detector_eligible?' ✓':' ✗');
  } else { bx.style.display=lb.style.display='none'; }
  const g=v.gates||{};
  const mark=(id,ok,text)=>{const e=document.getElementById(id); if(!e)return;
    e.textContent=text; e.className='n '+(ok?'ok':'bad')};
  // When there is no box at all the server sends gates:null. Leaving the rows
  // alone then shows the last box's numbers as though they were current -- a
  // reading of 0.184 beside "detector sees the can 0.000", seconds after the
  // can was last seen. Say "no box" instead.
  if(g.confidence) mark('g_conf', g.confidence.pass,
     `${Number(g.confidence.value||0).toFixed(3)} vs ${g.confidence.bound}`);
  else mark('g_conf', false, 'no box');
  if(g.clear_of_edge) mark('g_edge', g.clear_of_edge.pass,
     `${Number(g.clear_of_edge.value||0).toFixed(0)} px`);
  else mark('g_edge', false, 'no box');
  if(g.plausible_shape) mark('g_shape', g.plausible_shape.pass,
     `${g.plausible_shape.value.w}×${g.plausible_shape.value.h}  ar ${Number(g.plausible_shape.value.aspect).toFixed(2)}`);
  else mark('g_shape', false, 'no box');
  mark('g_all', v.detector_eligible, v.detector_eligible?'yes':'no');
  mark('g_mask', v.mask_eligible, v.mask_eligible?'yes':'no');
  mark('g_seg', !!v.target_mask_valid,
     v.segmentation_confidence!=null?Number(v.segmentation_confidence).toFixed(3):(v.segmentation_reason||'—'));
  const thrInput=document.getElementById('thr');
  if(thrInput&&document.activeElement!==thrInput&&v.detector_threshold!=null)
    thrInput.value=v.detector_threshold;
  const s=v.stats||{}; const r=document.getElementById('srate');
  if(r&&s.frames){const f=Number(s.valid_fraction||0);
   r.textContent=`${(100*f).toFixed(0)}% of ${s.frames} frames`; r.className='n '+(f>=0.9?'ok':'bad');
   const g=document.getElementById('sgap');
   if(g)g.textContent=`${s.longest_invalid_run||0} frames`;
   const h=document.getElementById('shz');
   if(h)h.textContent=`${s.effective_hz||0} Hz`;}
 }catch(e){document.getElementById('dconf').textContent='—'}
}
async function inspection(){
 try{const v=await j('/streams/selection');
  const t=Date.now(), k=v.requested?.stream;
  document.getElementById('insp').src = k==='infrared'?'/infrared.jpg?t='+t
    : k==='depth'?'/depth.jpg?t='+t : '/frame.jpg?t='+t;
  if(!document.getElementById('sel').textContent)
    document.getElementById('sel').textContent=v.policy_mode?'policy mode \u2014 serving the policy path':'inspecting';
 }catch(e){}
}
const _tick=tick; tick=async()=>{await _tick();try{fillPicker(await j('/streams'))}catch(e){}};
const fmt=(v,d=2,u='')=>v==null?'—':(typeof v==='number'?v.toFixed(d)+u:String(v));
const row=(k,v,cls='')=>`<tr><th>${k}</th><td class="n ${cls}">${v}</td></tr>`;
const yn=(v,good=true)=>v==null?['—','']:[v?'yes':'no',(v===good)?'ok':'bad'];

function guardBar(name,g){
 if(!g||g.used==null)return `<div class="off"><div class="sub">${name} — no reading</div></div>`;
 const pct=Math.min(100,Math.round(g.used*100));
 const cls=g.used>=1?'over':g.used>=.8?'warn':'';
 return `<div style="margin-bottom:12px"><div class="sub">${name}
   <span style="float:right">${pct}% of ${fmt(g.limit,3)}</span></div>
   <div class="bar ${cls}"><i style="width:${pct}%"></i></div></div>`;
}

async function calibration(){
 const t=document.getElementById('cal'); if(!t)return;
 t.innerHTML='<tr><th>measuring…</th><td class="n">—</td></tr>';
 let v; try{ v=await j('/camera-check'); }catch(e){
   t.innerHTML='<tr><th>unavailable</th><td class="n bad">—</td></tr>'; return; }
 const m=v.measured||{}, b=v.believed||{}, d=v.difference||{};
 let h='';
 h+=row('measured tilt', m.optical_axis_to_plane_deg!=null?fmt(m.optical_axis_to_plane_deg,2,'°'):(m.unavailable||'—'));
 h+=row('code believes', b.looking_below_horizontal_deg!=null?fmt(b.looking_below_horizontal_deg,2,'°'):(b.unavailable||'—'));
 const off=Math.abs(Number(d.degrees||0));
 h+=row('difference', d.degrees!=null?((d.degrees>0?'+':'')+fmt(d.degrees,2,'°')):'—', off>=3?'bad':(off>=1?'':'ok'));
 h+=row('reach error at 56 cm', d.reach_error_mm_at_560mm!=null?fmt(d.reach_error_mm_at_560mm,0,' mm'):'—',
        (d.reach_error_mm_at_560mm||0)>=30?'bad':'ok');
 h+=row('camera above surface', m.height_above_plane_m!=null?fmt(m.height_above_plane_m,3,' m'):'—');
 h+=row('implied surface height', d.implied_surface_height_m!=null?fmt(d.implied_surface_height_m,3,' m'):'—');
 h+=row('plane fit', m.inlier_fraction!=null?`${(100*m.inlier_fraction).toFixed(0)}% of ${m.points} points`:'—',
        (m.inlier_fraction||0)>=0.2?'':'bad');
 t.innerHTML=h;
 const c=v.calibration||{};
 h+=row('measured calibration', c.qualified===true?'qualified':(c.qualified===false?'not qualified':'—'),
        c.qualified===true?'ok':'bad');
 if(c.residual_rms_m!=null) h+=row('its residual', fmt(1000*c.residual_rms_m,1,' mm'),
        c.residual_rms_m<=0.01?'ok':'bad');
 if(c.sample_count!=null) h+=row('its samples', `${c.sample_count}, spread ${fmt(c.spread_m,2,' m')}`);
 t.innerHTML=h;
 const n=document.getElementById('calnote'); n.style.display='block';
 n.innerHTML = c.qualified===true
   ? `<b>A measured calibration is active.</b> ${c.source||''} &mdash; it corrects the
      detected point downstream, not the transform compared above, so the
      difference can stay while the reach error is already fixed. Trust the
      residual (${fmt(1000*(c.residual_rms_m||0),1)} mm), not the degrees.`
   : `<b>Nothing here is measured yet.</b> The comparison is against the CAD
      extrinsic, so read it as "these two disagree", not "the camera is wrong by
      this much". ${c.unqualified_reason||''} To fix rather than observe:
      <code>python3 -m runtime.calibrate record --x .. --y .. --z ..</code>`;
}
async function diagnostics(){
 let d; try{ d=await j('/diagnostics'); }catch(e){ return; }
 const r=d.retarget||{};
 let t='';
 t+=row('can seen', ...(r.can_visible==null?['—','']:[r.can_visible?'yes':'no', r.can_visible?'ok':'bad']));
 t+=row('offset in use', ...(r.stepping?[r.available?'yes':'no', r.available?'ok':'bad']:['no run','']));
 t+=row('reason', r.stepping?(r.reason||'—'):'no policy run in progress');
 t+=row('how far off', r.distance_to_optimal_cm!=null?fmt(r.distance_to_optimal_cm,1,' cm'):'—',
        (r.distance_to_optimal_cm||0)>4?'bad':'');
 t+=row('offset frozen', ...(r.latched==null?['—','']:[r.latched?`yes, frame ${r.reference_frame??'?'}`:'no','']));
 t+=row('measurement age', r.detection_age_ms!=null?fmt(r.detection_age_ms,0,' ms'):'—',
        (r.detection_age_ms||0)>350?'bad':'');
 t+=row('camera calibrated', ...(r.calibration_qualified==null?['—','']:yn(r.calibration_qualified,true)));
 document.getElementById('rt').innerHTML=t;
 const n=document.getElementById('rtnote');
 if(r.calibration_qualified===false){n.style.display='block';
   n.innerHTML=`<b>Extrinsics are not measured.</b> ${r.calibration_source||''} — every
   distance above inherits that. See docs/measurements/2026-09-17-camera-field-of-view.md.`;}

 const g=d.guards||{};
 document.getElementById('guards').innerHTML=
   guardBar('state freshness', g.state_skew_s)+
   guardBar('distance from start pose', g.distance_from_start_pose_rad)+
   guardBar('entry speed', g.entry_speed_rad_s)+
   `<div class="sub">joint step limit ${fmt(g.maximum_policy_step_rad,3)} rad/tick</div>`;

 const v=d.vision||{}, run=d.run||{};
 let w='';
 w+=row('policy steps', run.steps_completed!=null?`${run.steps_completed} of ${run.steps_total??'?'}`:'—');
 w+=row('frame admitted', ...(v.admitted==null?['—','']:yn(v.admitted,true)));
 w+=row('overlay only', ...(v.overlay_only==null?['—','']:yn(v.overlay_only,false)));
 w+=row('proposals', v.proposals??'—');
 w+=row('running at', v.effective_policy_hz!=null?`${fmt(v.effective_policy_hz,1)} Hz of ${fmt(v.selected_execution_hz,0)}`:'—');
 w+=row('sealed rate', v.policy_contract_hz!=null?fmt(v.policy_contract_hz,0,' Hz'):'—');
 w+=row('vision error', v.last_error||'none', v.last_error?'bad':'ok');
 document.getElementById('vis').innerHTML=w;

 const k=d.readiness||{};
 let y='';
 y+=row('control mode (FSM)', k.fsm_id??'—', k.fsm_id===500?'ok':'bad');
 y+=row('mode machine', k.mode_machine??'—');
 y+=row('remote neutral', ...(k.remote_neutral==null?['—','']:yn(k.remote_neutral,true)));
 y+=row('motor errors', (k.motor_error_groups||[]).length?k.motor_error_groups.join(', '):'none',
        (k.motor_error_groups||[]).length?'bad':'ok');
 y+=row('publishers armed', ...(k.publishers_armed==null?['—','']:[k.publishers_armed?'ARMED':'no', k.publishers_armed?'bad':'ok']));
 y+=row('DDS writes', k.dds_writes_accepted??'—');
 y+=row('io fault', k.io_fault||'none', k.io_fault?'bad':'ok');
 y+=row('run state', run.running?`running ${run.command_id||''}`:(run.holding_after_fault?'FAULT HOLD':'idle'),
        run.holding_after_fault?'bad':'');
 document.getElementById('rdy').innerHTML=y;
}
tick();feeds();inspection();diagnostics();calibration();setInterval(diagnostics,1000);setInterval(feeds,200);setInterval(inspection,250);setInterval(tick,2000);recognition();setInterval(recognition,600);
</script></body></html>"""


def _field_of_view(intrinsics) -> dict[str, float]:
    """Horizontal and vertical FOV in degrees, from focal length and size."""
    return {
        "horizontal_deg": 2 * math.degrees(math.atan(intrinsics.width / (2 * intrinsics.fx))),
        "vertical_deg": 2 * math.degrees(math.atan(intrinsics.height / (2 * intrinsics.fy))),
    }


def _intrinsics_dict(intrinsics) -> dict[str, Any]:
    fov = _field_of_view(intrinsics)
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
        **fov,
        # The lens is not necessarily centred. MuJoCo's camera has no
        # principal-point parameter, so this offset is the part of a sim/real
        # camera match that field of view alone cannot carry.
        "principal_point_offset_px": {
            "x": intrinsics.ppx - (intrinsics.width - 1) / 2,
            "y": intrinsics.ppy - (intrinsics.height - 1) / 2,
        },
    }


def _raw_frame_metadata(frame) -> dict[str, Any]:
    """Sensor-reported timing/geometry, before alignment or unit conversion."""
    intr = frame.profile.as_video_stream_profile().intrinsics
    metadata = {}
    for name, field in getattr(getattr(rs, "frame_metadata_value", None), "__members__", {}).items():
        if frame.supports_frame_metadata(field):
            metadata[name] = frame.get_frame_metadata(field)
    return {
        "sensor_timestamp_ms": frame.get_timestamp(),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
        "sensor_frame_number": frame.get_frame_number() if hasattr(frame, "get_frame_number") else None,
        "intrinsics": _intrinsics_dict(intr),
        "sensor_metadata": metadata,
    }


def _describe_device(device) -> dict[str, Any]:
    """Every video profile this camera offers, with its true field of view.

    Enumerated once at pipeline start, from the device this process already
    owns -- the RealSense is exclusively held, so nothing else can ask it.
    """
    sensors: list[dict[str, Any]] = []
    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name)
        except Exception:  # noqa: BLE001 - a sensor without a name still has profiles
            name = "unknown"
        modes: dict[tuple, dict[str, Any]] = {}
        for profile in sensor.get_stream_profiles():
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            try:
                intrinsics = video.get_intrinsics()
            except Exception:  # noqa: BLE001 - some profiles carry none
                continue
            stream = str(profile.stream_type()).rsplit(".", 1)[-1]
            key = (stream, video.width(), video.height())
            entry = modes.setdefault(
                key,
                {
                    "stream": stream,
                    "width": video.width(),
                    "height": video.height(),
                    "formats": set(),
                    "fps": set(),
                    **_field_of_view(intrinsics),
                },
            )
            entry["formats"].add(str(profile.format()).rsplit(".", 1)[-1])
            entry["fps"].add(profile.fps())
        for entry in modes.values():
            entry["formats"] = sorted(entry["formats"])
            entry["fps"] = sorted(entry["fps"])
        sensors.append(
            {
                "sensor": name,
                "modes": sorted(
                    modes.values(),
                    key=lambda m: (m["stream"], -m["width"] * m["height"]),
                ),
            }
        )
    return {"sensors": sensors}


class CameraRuntime:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_bgr: bytes | None = None
        self._frame_id = 0
        self._captured_at_unix_ns = 0
        self._pipeline_started = False
        self._serial: str | None = None
        self._last_error: str | None = None
        self._restart_count = 0
        self._hardware_reset_count = 0
        self._last_hardware_reset_at = 0.0
        self._rgbd_frames: OrderedDict[int, dict] = OrderedDict()
        self._device_streams: dict[str, Any] = {}
        self._raw_depth: np.ndarray | None = None
        self._depth_scale: float | None = None
        self._restart = threading.Event()
        #: None means the policy profile from settings. Anything else is an
        #: operator inspecting the sensor, and takes the camera out of policy
        #: service until it is cleared -- see select_profile().
        self._requested: dict[str, Any] | None = None
        self._latest_infrared: bytes | None = None
        self._active_streams: dict[str, Any] = {}
        self._stream_id: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="realsense", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    @staticmethod
    def _devices() -> list[dict]:
        devices: list[dict] = []
        for device in rs.context().query_devices():
            try:
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
            except RuntimeError:
                continue
            if serial:
                devices.append({"serial": serial, "name": name, "device": device})
        return sorted(devices, key=lambda item: item["serial"])

    def _select_device(self) -> dict:
        devices = self._devices()
        if self.settings.serial:
            for item in devices:
                if item["serial"] == self.settings.serial:
                    return item
            visible = ", ".join(item["serial"] for item in devices) or "none"
            raise RuntimeError(
                f"configured RealSense {self.settings.serial} is absent; "
                f"visible serials: {visible}"
            )
        if len(devices) == 1:
            return devices[0]
        if not devices:
            raise RuntimeError("no RealSense camera detected")
        raise RuntimeError(
            "multiple RealSense cameras detected; set G1_REALSENSE_SERIAL to: "
            + ", ".join(item["serial"] for item in devices)
        )

    def _maybe_hardware_reset(self, serial: str, failures: int) -> bool:
        threshold = self.settings.hardware_reset_after_failures
        if threshold == 0 or failures < threshold:
            return False
        now = time.monotonic()
        if now - self._last_hardware_reset_at < 30.0:
            return False
        for item in self._devices():
            if item["serial"] != serial:
                continue
            logger.warning(
                "Explicitly enabled hardware reset for RealSense %s after %d failures",
                serial,
                failures,
            )
            item["device"].hardware_reset()
            self._last_hardware_reset_at = now
            with self._lock:
                self._hardware_reset_count += 1
            return True
        return False

    def _capture_loop(self) -> None:
        start_failures = 0
        while not self._stop.is_set():
            pipeline: rs.pipeline | None = None
            serial = self.settings.serial
            try:
                selected = self._select_device()
                serial = str(selected["serial"])
                with self._lock:
                    requested = dict(self._requested) if self._requested else None
                configuration = rs.config()
                configuration.enable_device(serial)
                pipeline = rs.pipeline()
                if requested is None:
                    policy_mode = True
                    width, height, fps = (
                        self.settings.width_px, self.settings.height_px, self.settings.fps,
                    )
                    configuration.enable_stream(
                        rs.stream.color, width, height, rs.format.bgr8, fps
                    )
                    if self.settings.rgbd_enabled:
                        configuration.enable_stream(
                            rs.stream.depth, width, height, rs.format.z16, fps
                        )
                else:
                    # Operator inspection. One stream, exactly as asked for.
                    policy_mode = False
                    width, height, fps = requested["width"], requested["height"], requested["fps"]
                    stream, fmt = {
                        "color": (rs.stream.color, rs.format.bgr8),
                        "depth": (rs.stream.depth, rs.format.z16),
                        "infrared": (rs.stream.infrared, rs.format.y8),
                    }[requested["stream"]]
                    configuration.enable_stream(stream, width, height, fmt, fps)
                if policy_mode and self.settings.rgbd_enabled:
                    for sensor in selected["device"].query_sensors():
                        if sensor.supports(rs.option.global_time_enabled):
                            sensor.set_option(rs.option.global_time_enabled, 1)
                profile = pipeline.start(configuration)
                try:
                    device_streams = _describe_device(profile.get_device())
                    active = {}
                    for stream in profile.get_streams():
                        if not stream.is_video_stream_profile():
                            continue
                        video = stream.as_video_stream_profile()
                        active[str(stream.stream_type()).rsplit(".", 1)[-1]] = {
                            "fps": stream.fps(),
                            "format": str(stream.format()).rsplit(".", 1)[-1],
                            **_intrinsics_dict(video.get_intrinsics()),
                        }
                    with self._lock:
                        self._device_streams = device_streams
                        self._active_streams = active
                except Exception as error:  # noqa: BLE001 - reporting must not stop capture
                    logger.warning("stream description unavailable: %s", error)
                align = (
                    rs.align(rs.stream.color)
                    if policy_mode and self.settings.rgbd_enabled
                    else None
                )
                depth_scale = (
                    profile.get_device().first_depth_sensor().get_depth_scale()
                    if align
                    else None
                )
                with self._lock:
                    self._depth_scale = depth_scale
                rays = None
                start_failures = 0
                consecutive_timeouts = 0
                with self._lock:
                    self._pipeline_started = True
                    self._serial = serial
                    self._last_error = None
                    self._stream_id = str(uuid.uuid4())
                    self._rgbd_frames.clear()
                logger.info(
                    "RealSense %s (%s) started at %dx%d @ %d FPS",
                    serial,
                    selected["name"],
                    self.settings.width_px,
                    self.settings.height_px,
                    self.settings.fps,
                )

                while not self._stop.is_set():
                    if self._restart.is_set():
                        self._restart.clear()
                        break
                    try:
                        frames = pipeline.wait_for_frames(timeout_ms=1000)
                    except RuntimeError as error:
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= self.settings.frame_timeout_limit:
                            raise RuntimeError(
                                f"frame watchdog reached {consecutive_timeouts} timeouts: {error}"
                            ) from error
                        continue
                    if not policy_mode:
                        # Inspection: keep one preview and nothing else. No
                        # policy consumer is served in this mode by design.
                        self._store_inspection_frame(frames, requested)
                        consecutive_timeouts = 0
                        continue
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    rgbd = None
                    if align is not None:
                        # Keep the depth frame BEFORE alignment. Aligning warps
                        # it into the colour frame and discards everything
                        # outside colour's 43.4 deg; this is the stereo module's
                        # own 63.6 deg view, which is the wider field the task
                        # needs and the only place it still exists.
                        unaligned = frames.get_depth_frame()
                        raw_color = np.asanyarray(color.get_data()).copy()
                        raw_color_meta = _raw_frame_metadata(color)
                        raw_received_unix_ns = time.time_ns()
                        raw_received_monotonic_ns = time.monotonic_ns()
                        raw_depth = None
                        raw_depth_meta = None
                        if unaligned:
                            raw_depth = np.asanyarray(unaligned.get_data()).copy()
                            raw_depth_meta = _raw_frame_metadata(unaligned)
                            if hasattr(unaligned.profile, "get_extrinsics_to"):
                                transform = unaligned.profile.get_extrinsics_to(color.profile)
                                raw_depth_meta["depth_to_color"] = {
                                    "rotation_column_major": list(transform.rotation),
                                    "translation_m": list(transform.translation),
                                }
                            with self._lock:
                                self._raw_depth = raw_depth
                        aligned = align.process(frames)
                        depth_frame = aligned.get_depth_frame()
                        color = aligned.get_color_frame()
                        if not depth_frame or not color:
                            continue
                        if rays is None:
                            intr = (
                                depth_frame.profile.as_video_stream_profile().intrinsics
                            )
                            rays = np.asarray(
                                [
                                    rs.rs2_deproject_pixel_to_point(intr, [x, y], 1.0)
                                    for y in range(intr.height)
                                    for x in range(intr.width)
                                ],
                                dtype=np.float32,
                            ).reshape(intr.height, intr.width, 3)
                        received_mono = time.monotonic()
                        wall = time.time()
                        color_domain = color.get_frame_timestamp_domain()
                        depth_domain = depth_frame.get_frame_timestamp_domain()
                        valid_domains = {
                            rs.timestamp_domain.global_time,
                            rs.timestamp_domain.system_time,
                        }
                        color_ms, depth_ms = (
                            color.get_timestamp(),
                            depth_frame.get_timestamp(),
                        )
                        timestamp_valid = (
                            color_domain in valid_domains
                            and depth_domain in valid_domains
                        )
                        rgbd = {
                            "raw_color_bgr": raw_color,
                            "raw_depth_z16": raw_depth,
                            "aligned_depth_z16": np.asanyarray(depth_frame.get_data()).copy(),
                            "raw_sources": {"color": raw_color_meta, "depth": raw_depth_meta},
                            "color_bgr": np.asanyarray(color.get_data()).copy(),
                            "depth_m": np.asanyarray(depth_frame.get_data()).astype(
                                np.float32
                            )
                            * depth_scale,
                            "rays": rays,
                            "meta": {
                                "schema": "wendy.g1.rgbd.v1",
                                "raw_received_unix_ns": raw_received_unix_ns,
                                "raw_received_monotonic_ns": raw_received_monotonic_ns,
                                "aligned_to": "color",
                                "camera_serial": serial,
                                "clock": "host_monotonic",
                                "timestamp_valid": timestamp_valid,
                                "color_time": received_mono + color_ms / 1000 - wall
                                if timestamp_valid
                                else None,
                                "depth_time": received_mono + depth_ms / 1000 - wall
                                if timestamp_valid
                                else None,
                                "received_monotonic": received_mono,
                                "color_sensor_ms": color_ms,
                                "depth_sensor_ms": depth_ms,
                                "color_timestamp_domain": str(color_domain),
                                "depth_timestamp_domain": str(depth_domain),
                                "depth_scale_m": depth_scale,
                                "intrinsics": {
                                    "width": intr.width,
                                    "height": intr.height,
                                    "fx": intr.fx,
                                    "fy": intr.fy,
                                    "ppx": intr.ppx,
                                    "ppy": intr.ppy,
                                    "model": str(intr.model),
                                    "coeffs": list(intr.coeffs),
                                },
                            },
                        }
                    consecutive_timeouts = 0
                    frame = np.asanyarray(color.get_data())
                    with self._lock:
                        # bytes() makes one immutable snapshot whose lifetime is
                        # independent of librealsense's next frame buffer.
                        self._latest_bgr = frame.tobytes()
                        self._frame_id += 1
                        self._captured_at_unix_ns = time.time_ns()
                        self._last_error = None
                        if rgbd is not None:
                            rgbd["meta"].update(
                                frame_id=self._frame_id,
                                captured_at_unix_ns=self._captured_at_unix_ns,
                                stream_id=self._stream_id,
                            )
                            self._rgbd_frames[self._frame_id] = rgbd
                            while (
                                len(self._rgbd_frames)
                                > self.settings.rgbd_buffer_frames
                            ):
                                self._rgbd_frames.popitem(last=False)
            except Exception as error:
                start_failures += 1
                with self._lock:
                    self._pipeline_started = False
                    self._last_error = f"{type(error).__name__}: {error}"
                    self._restart_count += 1
                logger.error("RealSense capture failed: %s", error)
                reset = False
                if serial:
                    try:
                        reset = self._maybe_hardware_reset(serial, start_failures)
                    except Exception as reset_error:
                        logger.warning(
                            "RealSense hardware reset failed: %s", reset_error
                        )
                self._stop.wait(5.0 if reset else self.settings.retry_delay_s)
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception as error:
                        logger.warning("RealSense pipeline stop failed: %s", error)

    def snapshot(self) -> tuple[bytes, int, int]:
        with self._lock:
            if self._latest_bgr is None:
                raise RuntimeError("no camera frame available")
            return self._latest_bgr, self._frame_id, self._captured_at_unix_ns

    def status(self) -> dict:
        with self._lock:
            age_s = (
                max(0.0, (time.time_ns() - self._captured_at_unix_ns) / 1e9)
                if self._captured_at_unix_ns
                else None
            )
            healthy = bool(
                self._pipeline_started
                and age_s is not None
                and age_s <= self.settings.freshness_limit_s
            )
            return {
                "healthy": healthy,
                "pipeline_started": self._pipeline_started,
                "serial": self._serial,
                "frame_id": self._frame_id,
                "frame_age_s": round(age_s, 4) if age_s is not None else None,
                "last_error": self._last_error,
                "restart_count": self._restart_count,
                "hardware_reset_count": self._hardware_reset_count,
                "hardware_reset_enabled": self.settings.hardware_reset_after_failures
                > 0,
                "width": self.settings.width_px,
                "height": self.settings.height_px,
                "fps": self.settings.fps,
                "rgbd_enabled": self.settings.rgbd_enabled,
                "rgbd_buffered_frames": len(self._rgbd_frames),
                "rgbd_stream_id": self._stream_id,
                "rgbd_frame_ids": list(self._rgbd_frames),
                "raw_recording_schema": "wendy.realsense.raw-rgbd.v1",
            }

    def rgbd_snapshot(self, frame_id: int) -> bytes:
        with self._lock:
            if not self.settings.rgbd_enabled:
                raise RuntimeError("RGBD_DISABLED: enable G1_REALSENSE_RGBD_ENABLED=1")
            if frame_id not in self._rgbd_frames:
                raise KeyError("EXACT_RGBD_FRAME_UNAVAILABLE")
            snapshot = self._rgbd_frames[frame_id]
        # Encoding outside the capture lock; snapshots own immutable arrays.
        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            color_bgr=snapshot["color_bgr"],
            depth_m=snapshot["depth_m"],
            rays=snapshot["rays"],
            metadata_json=json.dumps(snapshot["meta"], allow_nan=False),
        )
        return stream.getvalue()

    def _store_inspection_frame(self, frames, requested: dict[str, Any]) -> None:
        """Encode one preview for whichever single stream the operator picked."""
        kind = requested["stream"]
        if kind == "color":
            frame = frames.get_color_frame()
            if not frame:
                return
            image = np.asanyarray(frame.get_data())
        elif kind == "infrared":
            frame = frames.get_infrared_frame()
            if not frame:
                return
            image = cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_GRAY2BGR)
        else:
            frame = frames.get_depth_frame()
            if not frame:
                return
            with self._lock:
                self._raw_depth = np.asanyarray(frame.get_data()).copy()
            return
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._frame_id += 1
            self._captured_at_unix_ns = time.time_ns()
            self._last_error = None
            if kind == "infrared":
                self._latest_infrared = encoded.tobytes()
            else:
                self._latest_bgr = np.asanyarray(frame.get_data()).tobytes()

    def infrared_preview(self) -> bytes:
        with self._lock:
            payload = self._latest_infrared
        if payload is None:
            raise RuntimeError("no infrared frame; select an infrared profile first")
        return payload

    def select_profile(self, request: dict[str, Any] | None) -> dict[str, Any]:
        """Reconfigure the sensor, or return it to policy service with None.

        Refused while the camera lease is held: the lease is how a policy
        consumer says it is reading frames, and reconfiguring underneath it
        would change resolution and field of view mid-run. Even unleased, any
        profile other than the policy one stops serving the policy path --
        segmentation asserts a 640x480 source -- which is why the answer says
        so rather than leaving it to be discovered.
        """
        held = leases.status()
        if held.get("exclusive"):
            raise RuntimeError(
                f"camera lease held by {held.get('owner')}; release it first"
            )
        if request is not None:
            offered = {
                (m["stream"], m["width"], m["height"], fps)
                for sensor in (self._device_streams.get("sensors") or [])
                for m in sensor["modes"]
                for fps in m["fps"]
            }
            key = (request["stream"], request["width"], request["height"], request["fps"])
            if key not in offered:
                raise ValueError(f"this camera does not offer {key}")
        with self._lock:
            self._requested = dict(request) if request else None
            self._latest_infrared = None
        self._restart.set()
        return self.selection()

    def selection(self) -> dict[str, Any]:
        with self._lock:
            requested = dict(self._requested) if self._requested else None
        return {
            "requested": requested,
            "policy_mode": requested is None,
            "serves_policy_path": requested is None,
        }

    def raw_depth_preview(self, near_m: float = 0.25, far_m: float = 2.0) -> bytes:
        """Colourised depth as the stereo module sees it -- unaligned, 63.6 deg.

        Rendered on request rather than per frame: nothing on the policy path
        consumes it, and it exists so an operator can see how much field
        alignment is throwing away.
        """
        with self._lock:
            depth = None if self._raw_depth is None else self._raw_depth.copy()
            scale = self._depth_scale
        if depth is None:
            raise RuntimeError("no depth frame yet")
        metres = depth.astype(np.float32) * float(scale or 0.001)
        valid = metres > 0
        span = np.clip((metres - near_m) / max(far_m - near_m, 1e-6), 0, 1)
        image = cv2.applyColorMap((255 * (1 - span)).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[~valid] = 0
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("depth preview encoding failed")
        return encoded.tobytes()

    def streams(self) -> dict[str, Any]:
        """What this camera can see: every offered mode, and the live ones.

        `active` carries the intrinsics actually in force, so field of view is
        measured from the running stream rather than quoted from a datasheet.
        `alignment` matters as much as the numbers: depth is warped into the
        colour frame, so the delivered depth is cropped to the COLOUR field of
        view no matter how much wider the depth sensor is.
        """
        with self._lock:
            device_streams = json.loads(json.dumps(self._device_streams))
            active = json.loads(json.dumps(self._active_streams))
        return {
            "schema": "wendy.g1.camera-streams.v1",
            "serial": self.settings.serial,
            "alignment": "depth aligned into color" if self.settings.rgbd_enabled else "color only",
            "delivered_field_of_view": "color" if self.settings.rgbd_enabled else None,
            "active": active,
            "device": device_streams,
        }

    def table_plane(self, tolerance_m: float = 0.01) -> dict[str, Any]:
        """Measure the camera's tilt and height from the surface it can see.

        A table is flat, and this camera already produces metric depth, so the
        plane it is looking at gives two of the six numbers a hand-eye
        calibration would -- the angle between the optical axis and that plane,
        and the perpendicular height above it -- with no target, no tape measure
        and nothing moved.

        It does NOT give the sideways offset or the full orientation, and it
        assumes the dominant plane is the work surface and that the surface is
        level. Both are visible in the output (`inlier_fraction`, and the height
        against a plausible table) rather than assumed silently.
        """
        with self._lock:
            if not self.settings.rgbd_enabled:
                raise RuntimeError("RGBD_DISABLED: enable G1_REALSENSE_RGBD_ENABLED=1")
            if not self._rgbd_frames:
                raise RuntimeError("no depth frame yet")
            frame_id, snapshot = next(reversed(self._rgbd_frames.items()))
            depth = snapshot["depth_m"].astype(np.float64)
            source = dict((snapshot["meta"] or {}).get("intrinsics") or {})
        height, width = depth.shape
        if not source:
            raise RuntimeError("depth intrinsics unavailable")
        fx = source["fx"] * width / source["width"]
        fy = source["fy"] * height / source["height"]
        cx = source["ppx"] * width / source["width"]
        cy = source["ppy"] * height / source["height"]

        rows, columns = np.indices(depth.shape)
        usable = np.isfinite(depth) & (depth > 0.2) & (depth < 2.5)
        if usable.sum() < 2000:
            raise RuntimeError(f"only {int(usable.sum())} usable depth points")
        points = np.stack(
            [
                (columns[usable] - cx) / fx * depth[usable],
                (rows[usable] - cy) / fy * depth[usable],
                depth[usable],
            ],
            axis=1,
        )

        generator = np.random.default_rng(0)      # deterministic: same frame, same answer
        # RANSAC scores against a subsample: 600 iterations over a full 640x480
        # frame is 180M dot products for an answer a tenth of the points gives
        # just as well. The winning plane is then refined on every inlier.
        scoring = points if len(points) <= 30_000 else points[
            generator.choice(len(points), 30_000, replace=False)
        ]
        best = None
        for _ in range(600):
            sample = points[generator.choice(len(points), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-9:
                continue
            normal = normal / length
            offset = -(normal @ sample[0])
            score = int((np.abs(scoring @ normal + offset) < tolerance_m).sum())
            if best is None or score > best[0]:
                best = (score, normal, offset)
        if best is None:
            raise RuntimeError("no plane could be fitted")
        _, normal, offset = best
        inliers = np.abs(points @ normal + offset) < tolerance_m
        count = int(inliers.sum())
        accepted = points[inliers]
        centre = accepted.mean(axis=0)
        # The plane normal is the least-variance direction of the inliers, which
        # is the smallest eigenvector of their 3x3 scatter matrix. NOT an SVD of
        # the point matrix: numpy defaults to full_matrices=True, and asking for
        # U on 115k points requests a 115k x 115k array -- 98 GiB, which is
        # exactly how this first failed on a 640x480 frame.
        centred = accepted - centre
        normal = np.linalg.eigh(centred.T @ centred)[1][:, 0]
        normal = normal / np.linalg.norm(normal)
        if normal[1] > 0:                          # optical +Y is down; point it up
            normal = -normal
        offset = -(normal @ centre)
        axis = np.asarray([0.0, 0.0, 1.0])         # optical +Z is the viewing direction
        tilt = 90 - math.degrees(math.acos(abs(float(np.clip(axis @ normal, -1, 1)))))
        return {
            "schema": "wendy.g1.camera-plane-measurement.v1",
            "frame_id": frame_id,
            "points": int(len(points)),
            "inliers": count,
            "inlier_fraction": round(count / len(points), 4),
            "plane_normal_optical": [round(float(v), 5) for v in normal],
            "height_above_plane_m": round(abs(float(offset)), 4),
            "optical_axis_to_plane_deg": round(tilt, 3),
            "assumes": "the dominant plane is a level work surface",
        }

    def policy_rgbd_snapshot(self, frame_id: int) -> bytes:
        """One exact frame at the policy's 320x240, as color_bgr + depth_m + meta.

        The two arrays are resized by DIFFERENT rules, and the difference is the
        point.  Colour is averaged (INTER_AREA): discarding three pixels in four
        without averaging aliases the can's edge.  Depth is NOT averaged
        (INTER_NEAREST): averaging a 0.40 m can pixel with a 0.80 m table pixel
        yields 0.60 m, a surface that is not there, all the way round the rim --
        and that number reaches the retargeter, which turns it into a position.
        A nearest sample is always a distance something actually had.

There is no training-time resize to match, and that is not an oversight:
        the policy was trained in simulation on `robot_rgbd` renders produced
        natively at 320x240 (sealed scene contract, `intrinsics_320x240`). No
        640x480 sensor frame was ever reduced. The sealed sensor contract
        delegates this step -- "the camera driver must align depth into RGB
        pixels, resize/calibrate both to this grid" -- and names no method.
        So these rules are ours to choose, and choosing them is not deferred
        work.

        What IS unreconciled is larger than interpolation. Measured on this
        robot against the sealed contract:

            trained   fy 156.39 @ 320x240  ->  75.0 deg vertical FOV
            real      fy 603.70 @ 640x480  ->  43.4 deg vertical FOV
                      (fy 301.85 once scaled to 320x240)

        The real camera's focal length at policy resolution is 1.93x what the
        policy was trained on: it sees a little over half the angle, and
        everything in frame is about twice the size the policy expects. No
        choice of interpolation reaches across that, because a narrow camera
        cannot be resampled into a wide one. Fixing it means adapting the
        simulator to 43.4 deg or changing the lens -- not touching this method.
        """
        with self._lock:
            if not self.settings.rgbd_enabled:
                raise RuntimeError("RGBD_DISABLED: enable G1_REALSENSE_RGBD_ENABLED=1")
            if frame_id not in self._rgbd_frames:
                raise KeyError("EXACT_RGBD_FRAME_UNAVAILABLE")
            snapshot = self._rgbd_frames[frame_id]
            colour = snapshot["color_bgr"]
            depth = snapshot["depth_m"]
            meta = dict(snapshot["meta"])
        height, width = depth.shape[:2]
        # The consumer hardcodes 640x480 when it scales detector pixels and
        # intrinsics into policy space, so a different source is a silent
        # mismatch there rather than an error. Refuse instead.
        if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
            raise RuntimeError(
                f"policy RGB-D expects a {SOURCE_WIDTH}x{SOURCE_HEIGHT} source, "
                f"got {width}x{height}; the consumer assumes the former"
            )
        stream = io.BytesIO()
        np.savez(
            stream,
            color_bgr=cv2.resize(
                colour, (POLICY_WIDTH, POLICY_HEIGHT), interpolation=cv2.INTER_AREA
            ),
            depth_m=cv2.resize(
                depth, (POLICY_WIDTH, POLICY_HEIGHT), interpolation=cv2.INTER_NEAREST
            ),
            metadata_json=json.dumps(meta, allow_nan=False),
        )
        return stream.getvalue()

    def recorder_rgbd_snapshot(self, frame_id: int) -> tuple[bytes, dict]:
        """Encode one exact frame without unused rays or nested compression."""
        with self._lock:
            if not self.settings.rgbd_enabled:
                raise RuntimeError("RGBD_DISABLED: enable G1_REALSENSE_RGBD_ENABLED=1")
            if frame_id not in self._rgbd_frames:
                raise KeyError("EXACT_RGBD_FRAME_UNAVAILABLE")
            snapshot = self._rgbd_frames[frame_id]
        meta = snapshot["meta"]
        scale = float(meta["depth_scale_m"])
        depth = snapshot["depth_m"]
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(depth).all():
            raise RuntimeError("invalid metric depth")
        raw = np.rint(depth.astype(np.float64) / scale)
        if (raw < 0).any() or (raw > 65535).any():
            raise RuntimeError("depth outside Z16 range")
        ok_depth, png = cv2.imencode(
            ".png", raw.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )
        ok_color, jpeg = cv2.imencode(
            ".jpg",
            snapshot["color_bgr"],
            [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality],
        )
        if not ok_depth or not ok_color:
            raise RuntimeError("RGB-D image encoding failed")
        height, width = depth.shape
        metadata = {
            "schema": "wendy.realsense.rgbd-frame.v1",
            "source_frame_id": frame_id,
            "captured_at_unix_ns": meta["captured_at_unix_ns"],
            "calibration": {
                "schema": "wendy.realsense.aligned-depth-calibration.v1",
                "serial": meta["camera_serial"],
                "aligned_to": meta["aligned_to"],
                "depth_encoding": "z16_png",
                "depth_scale_m": scale,
                "width": width,
                "height": height,
                "intrinsics": meta["intrinsics"],
            },
        }
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("color.jpg", jpeg.tobytes())
            archive.writestr("depth.png", png.tobytes())
            archive.writestr("metadata.json", json.dumps(metadata, allow_nan=False))
        return stream.getvalue(), metadata

    def raw_rgbd_snapshot(self, frame_id: int, stream_id: str) -> bytes:
        """Lossless source arrays and derived previews from ONE identified frame.

        No corrected depth replaces a sensor sample. Keep native Z16 values,
        including zeros, beside the scale and the separately aligned arrays.
        Frames own their buffers; encoding runs outside the capture lock.
        """
        with self._lock:
            if not self.settings.rgbd_enabled:
                raise RuntimeError("RGBD_DISABLED")
            snapshot = self._rgbd_frames.get(frame_id)
            if snapshot is None or snapshot["meta"].get("stream_id") != stream_id:
                raise KeyError("EXACT_RGBD_FRAME_UNAVAILABLE")
        arrays = {"color_bgr.npy": snapshot.get("raw_color_bgr"),
                  "depth_z16.npy": snapshot.get("raw_depth_z16"),
                  "aligned_depth_z16.npy": snapshot.get("aligned_depth_z16"),
                  "aligned_depth_m.npy": snapshot["depth_m"]}
        if any(value is None for value in arrays.values()):
            raise RuntimeError("RAW_SOURCE_UNAVAILABLE")
        meta = snapshot["meta"]
        ok_color, jpeg = cv2.imencode(".jpg", snapshot["color_bgr"],
                                    [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
        ok_depth, png = cv2.imencode(".png", snapshot["aligned_depth_z16"],
                                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok_color or not ok_depth:
            raise RuntimeError("RGB-D image encoding failed")
        height, width = snapshot["depth_m"].shape
        metadata = {
            "schema": "wendy.realsense.rgbd-frame.v1",
            "raw_schema": "wendy.realsense.raw-rgbd.v1",
            "source_frame_id": frame_id, "stream_id": stream_id,
            "captured_at_unix_ns": meta["captured_at_unix_ns"],
            "source_metadata": meta, "raw_sources": snapshot["raw_sources"],
            "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                       for name, array in arrays.items()},
            "values": "as_received_no_correction",
            "calibration": {
                "schema": "wendy.realsense.aligned-depth-calibration.v1",
                "serial": meta["camera_serial"], "aligned_to": "color",
                "depth_encoding": "z16_png", "depth_scale_m": meta["depth_scale_m"],
                "width": width, "height": height, "intrinsics": meta["intrinsics"],
            },
        }
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("color.jpg", jpeg.tobytes())
            archive.writestr("depth.png", png.tobytes())
            archive.writestr("metadata.json", json.dumps(metadata, allow_nan=False))
            for name, array in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                archive.writestr(name, buffer.getvalue())
        return stream.getvalue()


settings = CameraSettings.from_env()
runtime = CameraRuntime(settings)
leases = CameraLeaseRegistry(settings.lease_timeout_s)


class LeaseRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=64)


def _require_local(request: Request) -> None:
    host = request.client.host if request.client is not None else ""
    if host not in {"127.0.0.1", "::1"}:
        raise HTTPException(
            status_code=403, detail="camera leases are available only on loopback"
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    status = runtime.status()
    return JSONResponse(status, status_code=200 if status["healthy"] else 503)


@app.get("/debug")
async def debug():
    return {**runtime.status(), "lease": leases.status()}


@app.post("/lease/acquire", status_code=201)
async def acquire_lease(update: LeaseRequest, request: Request):
    _require_local(request)
    try:
        grant = leases.acquire(update.owner)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except CameraLeaseBusy as error:
        raise HTTPException(status_code=423, detail=str(error)) from error
    return {
        "token": grant.token,
        "owner": grant.owner,
        "expires_in_s": grant.expires_in_s,
    }


@app.post("/lease/renew")
async def renew_lease(
    request: Request, x_camera_lease: str | None = Header(default=None)
):
    _require_local(request)
    try:
        grant = leases.renew(x_camera_lease or "")
    except CameraLeaseDenied as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    return {"owner": grant.owner, "expires_in_s": grant.expires_in_s}


@app.post("/lease/release")
async def release_lease(
    request: Request, x_camera_lease: str | None = Header(default=None)
):
    _require_local(request)
    try:
        released = leases.release(x_camera_lease or "")
    except CameraLeaseDenied as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    return {"released": released}


def _authorize_frame(token: str | None) -> None:
    try:
        leases.authorize_frame(token)
    except CameraLeaseBusy as error:
        raise HTTPException(status_code=423, detail=str(error)) from error


@app.get("/frame.jpg")
async def frame(x_camera_lease: str | None = Header(default=None)):
    _authorize_frame(x_camera_lease)
    try:
        bgr, frame_id, captured_at = runtime.snapshot()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    image = np.frombuffer(bgr, dtype=np.uint8).reshape(
        settings.height_px, settings.width_px, 3
    )
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
    )
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encoding failed")
    return Response(
        encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Frame-Id": str(frame_id),
            "X-Captured-At-Unix-Ns": str(captured_at),
            "Cache-Control": "no-store",
        },
    )


@app.get("/frame.bgr")
async def raw_frame(x_camera_lease: str | None = Header(default=None)):
    _authorize_frame(x_camera_lease)
    try:
        bgr, frame_id, captured_at = runtime.snapshot()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        bgr,
        media_type="application/x-bgr8",
        headers={
            "X-Frame-Id": str(frame_id),
            "X-Captured-At-Unix-Ns": str(captured_at),
            "X-Frame-Width": str(settings.width_px),
            "X-Frame-Height": str(settings.height_px),
            "X-Frame-Channels": "3",
            "X-Frame-Stride": str(settings.width_px * 3),
            "Cache-Control": "no-store",
        },
    )


@app.get("/frame.rgbd")
def rgbd_frame(frame_id: int, x_camera_lease: str | None = Header(default=None)):
    """Exact-frame GET only. Never pair a detection with a newer depth frame."""
    _authorize_frame(x_camera_lease)
    try:
        payload = runtime.rgbd_snapshot(frame_id)
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        payload, media_type="application/x-npz", headers={"Cache-Control": "no-store"}
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """Operator view: both feeds, the measured fields of view, every mode."""
    return HTMLResponse(_PAGE)


@app.get("/infrared.jpg")
def infrared_preview():
    try:
        payload = runtime.infrared_preview()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/streams/selection")
def stream_selection():
    return runtime.selection()


@app.post("/streams/select")
async def select_stream(request: Request):
    """Reconfigure the sensor for inspection, or pass null to return to policy."""
    try:
        body = await request.json()
    except Exception as error:  # noqa: BLE001 - a bad body is a client error
        raise HTTPException(status_code=400, detail="invalid JSON") from error
    profile = body if body else None
    if profile is not None:
        try:
            profile = {
                "stream": str(profile["stream"]),
                "width": int(profile["width"]),
                "height": int(profile["height"]),
                "fps": int(profile["fps"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400, detail="stream, width, height and fps are required"
            ) from error
    try:
        return runtime.select_profile(profile)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/depth.jpg")
def raw_depth_preview():
    """The stereo module's own field, before it is cropped into colour's."""
    try:
        payload = runtime.raw_depth_preview()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


#: One page refresh must not be able to occupy a worker for longer than this,
#: however many neighbours are down. Each call was individually bounded at 1.5 s
#: and nothing bounded the SET, so /diagnostics could spend 4.5 s and
#: /recognition 3 s -- on endpoints a dashboard polls on a timer.
NEIGHBOUR_BUDGET_S = 2.5
NEIGHBOUR_TIMEOUT_S = 1.5


def _neighbour(url: str, deadline: float | None = None) -> dict[str, Any] | None:
    remaining = (
        NEIGHBOUR_TIMEOUT_S if deadline is None
        else min(NEIGHBOUR_TIMEOUT_S, deadline - time.monotonic())
    )
    if remaining <= 0:
        return None
    try:
        with urlopen(url, timeout=remaining) as response:
            return json.loads(response.read(600_000))
    except Exception:  # noqa: BLE001 - a neighbour being down is data, not an error
        return None


def _headroom(value, limit):
    """Fraction of a limit in use, so a guard can be watched approaching."""
    if not isinstance(value, (int, float)) or not isinstance(limit, (int, float)) or limit <= 0:
        return None
    return {"value": value, "limit": limit, "used": round(abs(value) / limit, 4)}


@app.get("/camera-check")
def camera_check():
    """Measured tilt against the tilt the policy path believes.

    The measurement is ours -- we own the depth. The belief belongs to
    inference, which owns the transform, so it is asked rather than
    reimplemented here. The waist angles come from the wrapper, because the
    transform depends on them and only it subscribes to robot state.

    A difference here is not cosmetic: the retargeter turns every can detection
    into a world position through that transform, so degrees of error become
    millimetres of reach error, and this says how many of each.
    """
    out: dict[str, Any] = {"schema": "wendy.g1.camera-check.v1"}
    try:
        out["measured"] = runtime.table_plane()
    except RuntimeError as error:
        out["measured"] = {"unavailable": str(error)}

    live = _neighbour(f"{WRAPPER_URL}/health") or {}
    joints = (live.get("live") or {}).get("q_43")
    believed = None
    if isinstance(joints, list) and len(joints) == 43:
        yaw, roll, pitch = joints[12], joints[13], joints[14]
        believed = _neighbour(
            f"{INFERENCE_URL}/camera-pose?yaw={yaw}&roll={roll}&pitch={pitch}"
        )
    out["believed"] = believed or {"unavailable": "wrapper or inference unreachable"}

    # Since #52 a measured optical-frame correction can sit on top of the CAD
    # extrinsic. It corrects the detected POINT, not this transform, so
    # /camera-pose still reports the uncorrected tilt -- and a panel that did
    # not say so would keep showing a discrepancy that had since been
    # corrected downstream.
    contract = _neighbour(f"{INFERENCE_URL}/contract") or {}
    out["calibration"] = contract.get("camera_calibration") or {
        "unavailable": "inference did not report a calibration"
    }

    measured = out["measured"].get("optical_axis_to_plane_deg")
    claimed = (believed or {}).get("looking_below_horizontal_deg")
    if isinstance(measured, (int, float)) and isinstance(claimed, (int, float)):
        difference = measured - claimed
        reach_m = 0.56                      # the reference trajectory's range to the can
        out["difference"] = {
            "degrees": round(difference, 3),
            "reach_error_mm_at_560mm": round(
                1000 * reach_m * math.tan(math.radians(abs(difference))), 1
            ),
            "compares_against": "the uncorrected CAD extrinsic",
            "note": (
                "tilt only, and against the uncorrected transform: a measured "
                "calibration corrects the detected point downstream, so this "
                "difference can persist after the reach error has been fixed"
            ),
        }
        height = out["measured"].get("height_above_plane_m")
        world_z = (believed or {}).get("world_z_m")
        if isinstance(height, (int, float)) and isinstance(world_z, (int, float)):
            out["difference"]["implied_surface_height_m"] = round(world_z - height, 4)
    return out


@app.get("/table-plane")
def table_plane():
    """Measured tilt and height, from the surface the camera is already seeing."""
    try:
        return runtime.table_plane()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - a diagnostic that 500s tells nobody anything
        logger.exception("table-plane failed")
        raise HTTPException(
            status_code=500, detail=f"{type(error).__name__}: {error}"
        ) from error


@app.get("/diagnostics")
def diagnostics():
    """Everything the edge already knows about why a policy might fail.

    Aggregated here because this is the page an operator has open, and because
    every one of these numbers exists on a service that cannot be reached from
    a browser without CORS headers it does not send. Read-only, fail-soft: a
    service that is down becomes null rather than an error, since half the
    answer is still worth showing.

    Nothing here is on the policy path. It is a window, not a dependency.
    """
    deadline = time.monotonic() + NEIGHBOUR_BUDGET_S
    inference = _neighbour(f"{INFERENCE_URL}/health", deadline) or {}
    wrapper = _neighbour(f"{WRAPPER_URL}/health", deadline) or {}
    fsm = _neighbour(f"{FSM_BRIDGE_URL}/state", deadline) or {}

    retarget = inference.get("object_retarget") or {}
    vision = (inference.get("camera") or {}).get("latest") or {}
    runtime_state = wrapper.get("policy_runtime") or {}
    live = wrapper.get("live") or {}
    io = wrapper.get("io") or {}

    errors = [
        key for key in ("body_errors", "left_hand_errors", "right_hand_errors",
                        "left_hand_system_errors", "right_hand_system_errors")
        if any(int(v) != 0 for v in (live.get(key) or []))
    ]
    return {
        "schema": "wendy.g1.edge-diagnostics.v1",
        # --- where the robot thinks the can is -------------------------------
        "retarget": {
            "available": retarget.get("available"),
            "reason": retarget.get("reason"),
            # Live, from observe(). `available` is written only while a policy
            # is stepping, so on an idle robot it says nothing about the can.
            "can_visible": retarget.get("can_visible"),
            "stepping": retarget.get("stepping"),
            "latched": retarget.get("latched"),
            "reference_frame": retarget.get("reference_frame"),
            "distance_to_optimal_cm": retarget.get("distance_to_optimal_cm"),
            "filtered_distance_to_optimal_m": retarget.get("filtered_distance_to_optimal_m"),
            "raw_offset_world_xy_m": retarget.get("raw_offset_world_xy_m"),
            "detection_age_ms": retarget.get("detection_age_ms"),
            "detection_valid": retarget.get("detection_valid"),
            "calibration_qualified": retarget.get("physical_camera_calibration_qualified"),
            "calibration_source": retarget.get("calibration_source"),
        },
        # --- how close to the limits -----------------------------------------
        "guards": {
            "state_skew_s": _headroom(live.get("state_skew_s"), wrapper.get("maximum_state_skew_s")),
            # `alignment` is the live commanded state against reference frame 0 --
            # how far the arm is from the START pose, not in-run tracking error.
            # Before entry it says whether a run can begin; it is not the guard
            # that aborts mid-run, and calling it that would mislead.
            "distance_from_start_pose_rad": _headroom(
                (wrapper.get("alignment") or {}).get("maximum_position_error_rad"),
                runtime_state.get("maximum_policy_tracking_error_rad")
                or wrapper.get("maximum_policy_tracking_error_rad"),
            ),
            "entry_speed_rad_s": _headroom(
                (wrapper.get("alignment") or {}).get("maximum_speed_rad_s"),
                (wrapper.get("alignment") or {}).get("speed_limit_rad_s"),
            ),
            "maximum_policy_step_rad": runtime_state.get("maximum_policy_step_rad"),
            "within_entry_limits": (wrapper.get("alignment") or {}).get("within_entry_limits"),
        },
        # --- was it looking ---------------------------------------------------
        "vision": {
            "frame_id": vision.get("frame_id"),
            "admitted": vision.get("admitted"),
            "overlay_only": vision.get("overlay_only"),
            "last_error": (inference.get("camera") or {}).get("last_error"),
            "proposals": inference.get("proposals"),
            "effective_policy_hz": runtime_state.get("effective_policy_hz"),
            "selected_execution_hz": runtime_state.get("selected_execution_hz"),
            "policy_contract_hz": runtime_state.get("policy_contract_hz"),
        },
        # --- is the robot willing ---------------------------------------------
        "readiness": {
            "fsm_id": fsm.get("fsm_id"),
            "fsm_healthy": fsm.get("healthy"),
            "mode_machine": live.get("mode_machine"),
            "remote_neutral": (live.get("remote") or {}).get("neutral"),
            "motor_error_groups": errors,
            "publishers_armed": io.get("publishers_armed"),
            "owner": io.get("owner"),
            "dds_writes_accepted": io.get("dds_writes_accepted"),
            "io_fault": io.get("fault"),
            "motion_enabled": io.get("motion_enabled"),
        },
        # --- run state ---------------------------------------------------------
        "run": {
            "running": runtime_state.get("running"),
            "command_id": runtime_state.get("command_id"),
            "steps_completed": runtime_state.get("policy_steps_completed"),
            "steps_total": runtime_state.get("policy_steps_total"),
            "elapsed_s": runtime_state.get("policy_elapsed_s"),
            "holding_after_fault": runtime_state.get("holding_after_fault"),
            "fault_hold": runtime_state.get("fault_hold"),
            "last_error": runtime_state.get("last_error") or wrapper.get("last_error"),
            "timing": runtime_state.get("timing"),
        },
        "reachable": {
            "inference": bool(inference), "wrapper": bool(wrapper), "fsm_bridge": bool(fsm),
        },
    }


@app.get("/recognition")
def recognition():
    """Detector and segmentation health, fetched server-side.

    The page cannot read :8003 itself: a cross-origin `fetch` needs
    CORS headers neither service sends, so every reading came back blank. An
    `<img>` has no such restriction, which is why the mask pictures worked while
    the numbers beside them did not. Proxying makes it same-origin.
    """
    out: dict[str, Any] = {"segmentation": None}
    deadline = time.monotonic() + NEIGHBOUR_BUDGET_S
    for key, url in (
        ("segmentation", f"{SEGMENTATION_URL}/state"),
        ("segmentation_stats", f"{SEGMENTATION_URL}/stats?seconds=60"),
    ):
        remaining = min(NEIGHBOUR_TIMEOUT_S, deadline - time.monotonic())
        if remaining <= 0:
            out[key] = {"unreachable": "NeighbourBudgetExhausted"}
            continue
        try:
            with urlopen(url, timeout=remaining) as response:
                out[key] = json.loads(response.read(400_000))
        except Exception as error:  # noqa: BLE001 - a neighbour being down is data
            out[key] = {"unreachable": f"{type(error).__name__}"}
    # One source now. Boxes, tracking and the grasp gate all come from the
    # model that produces the mask, so a box and the mask beside it cannot
    # disagree about which frame they describe.
    segmentation = out.get("segmentation") or {}
    detector = segmentation if isinstance(segmentation, dict) else {}
    latest = (detector.get("latest") or {}) if isinstance(detector, dict) else {}
    boxes = latest.get("boxes") or []
    target = next((b for b in boxes if b.get("is_target")), boxes[0] if boxes else None)
    threshold = latest.get("grasp_min_confidence")

    gates = None
    if target:
        geometry = target.get("geometry") or {}
        width, height = (geometry.get("size_px") or [0, 0])[:2]
        aspect = geometry.get("aspect_ratio") or 0.0
        # The three conditions of policy_eligible_2d, evaluated here so the page
        # can say WHICH one refuses rather than only that something did.
        #
        # Two of the three are also the whole of segmentation_eligible_2d: a
        # mask needs a box it can be associated with, not a box worth reaching
        # for. Confidence is the one that separates them, so a can can be
        # segmentable and not yet graspable -- which is the ordinary state of
        # this robot, not an edge case.
        gates = {
            "confidence": {
                "value": target.get("conf"),
                "bound": threshold,
                "pass": bool(threshold is not None and target.get("conf", 0) >= threshold),
            },
            "clear_of_edge": {
                "value": geometry.get("edge_margin_px"),
                "bound": 2.0,
                "pass": not geometry.get("border_clipped"),
            },
            "plausible_shape": {
                "value": {"w": width, "h": height, "aspect": aspect},
                "bound": "w>=20, h>=40, 0.25<=aspect<=1.25",
                "pass": bool(width >= 20 and height >= 40 and 0.25 <= aspect <= 1.25),
            },
        }

    return {
        "schema": "wendy.g1.recognition-summary.v1",
        "detector_confidence": target.get("conf") if target else 0.0,
        "detector_threshold": threshold,
        "detector_backend": ((latest.get("model") or {}).get("backend")),
        "detector_eligible": bool(
            ((target or {}).get("geometry") or {}).get("policy_eligible_2d")
        ),
        # Falls back to the grasp gate for a detector predating the split, so
        # the page reports what that detector actually enforces.
        "mask_eligible": bool(
            ((target or {}).get("geometry") or {}).get(
                "segmentation_eligible_2d",
                ((target or {}).get("geometry") or {}).get("policy_eligible_2d"),
            )
        ),
        "detector_box": (
            {k: target[k] for k in ("x1", "y1", "x2", "y2")} if target else None
        ),
        "frame_size": [latest.get("frame_w"), latest.get("frame_h")],
        "gates": gates,
        "target_mask_valid": segmentation.get("target_mask_valid"),
        "segmentation_reason": (
            ((segmentation.get("latest") or {}).get("selection") or {}).get("reason")
        ),
        "segmentation_confidence": (
            (((segmentation.get("latest") or {}).get("selection") or {}).get("candidates")
             or [{}])[0].get("confidence")
        ),
        "stats": out.get("segmentation_stats"),
    }


@app.post("/recognition/grasp-threshold")
async def set_grasp_threshold(request: Request):
    """Proxy the detector's eligibility threshold, same-origin.

    It is the gate the whole policy path waits on, and it was never calibrated
    against what this detector actually scores. Making it adjustable from the
    page is how that gets measured rather than argued about.
    """
    try:
        body = await request.json()
        value = float(body["grasp_min_confidence"])
    except Exception as error:  # noqa: BLE001 - a bad body is a client error
        raise HTTPException(status_code=400, detail="grasp_min_confidence required") from error
    payload = json.dumps({"grasp_min_confidence": value}).encode()
    call = Request_(
        f"{SEGMENTATION_URL}/config/grasp-confidence", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(call, timeout=2.0) as response:
            return json.loads(response.read(10_000))
    except Exception as error:  # noqa: BLE001 - surface the detector's refusal
        raise HTTPException(status_code=502, detail=f"{type(error).__name__}") from error


@app.get("/streams")
def streams():
    return runtime.streams()


@app.get("/frame.policy-rgbd")
def policy_rgbd_frame(frame_id: int, x_camera_lease: str | None = Header(default=None)):
    """The policy vision hop: segmentation reads this, adds the mask, serves on."""
    _authorize_frame(x_camera_lease)
    try:
        payload = runtime.policy_rgbd_snapshot(frame_id)
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        payload, media_type="application/x-npz", headers={"Cache-Control": "no-store"}
    )


@app.get("/frame.recorder-rgbd")
def recorder_rgbd_frame(
    frame_id: int, x_camera_lease: str | None = Header(default=None)
):
    _authorize_frame(x_camera_lease)
    try:
        payload, metadata = runtime.recorder_rgbd_snapshot(frame_id)
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        payload,
        media_type="application/zip",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Id": str(frame_id),
            "X-Captured-At-Unix-Ns": str(metadata["captured_at_unix_ns"]),
        },
    )


@app.get("/frame.raw-rgbd")
def raw_rgbd_frame(frame_id: int, stream_id: str,
                   x_camera_lease: str | None = Header(default=None)):
    _authorize_frame(x_camera_lease)
    try:
        payload = runtime.raw_rgbd_snapshot(frame_id, stream_id)
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(payload, media_type="application/zip", headers={"Cache-Control": "no-store"})


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
