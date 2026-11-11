const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes=new Map();
const node=id=>{if(!nodes.has(id))nodes.set(id,{hidden:true,textContent:'',classList:{add(){},remove(){},toggle(){}},addEventListener(){}});return nodes.get(id)};
const preview=node('sensor-preview');
let width=640,height=480,pixels='previous',resizes=0,draws=0;
Object.defineProperties(preview,{
 width:{get:()=>width,set:value=>{width=value;pixels=null;resizes++;}},
 height:{get:()=>height,set:value=>{height=value;pixels=null;resizes++;}}
});
preview.getContext=()=>({drawImage:image=>{pixels=image.src;draws++;}});
const decodes=[],revoked=[];let nextURL=0,sequence=10;
class DecodedImage{
 constructor(){this.naturalWidth=640;this.naturalHeight=480;}
 decode(){return new Promise((resolve,reject)=>decodes.push({image:this,resolve,reject}));}
}
const context={document:{getElementById:node,querySelectorAll:()=>[]},window:{addEventListener(){}},setInterval(){},Date,Map,Set,console,Image:DecodedImage,
 URL:{createObjectURL:()=>`blob:frame-${++nextURL}`,revokeObjectURL:url=>revoked.push(url)},
 fetch:async()=>({status:200,ok:true,headers:{get:()=>String(sequence++)},blob:async()=>({size:55000})})};
vm.createContext(context);
const source=fs.readFileSync(require('path').join(__dirname,'../index.html'),'utf8').split('<script>')[1].split('</script>')[0].replace('});refresh();setInterval(refresh,600);','});setInterval(refresh,600);');
vm.runInContext(source,context);
async function begin(){
 const promise=vm.runInContext('pollSensor()',context);
 await new Promise(resolve=>setImmediate(resolve));
 return {promise,decode:decodes.shift()};
}
(async()=>{
 vm.runInContext('sensorActive=true;sensorFreshAt=Date.now();sensorSequence=9;sensorGeneration=1;connected=true;snapshot={vision_model:"vision"}',context);
 let frame=await begin();
 assert.equal(pixels,'previous','pending decode must keep the last visible frame');
 assert.equal(resizes,0,'pending decode must not clear the canvas');
 assert.equal(vm.runInContext('sensorSequence',context),9,'capture must still refer to the displayed frame');
 frame.decode.resolve();await frame.promise;
 assert.equal(pixels,'blob:frame-1');assert.equal(draws,1);assert.equal(resizes,0,'same-size frames must not reset the canvas');
 assert.equal(vm.runInContext('sensorSequence',context),10);assert.deepEqual(revoked,['blob:frame-1']);
 frame=await begin();frame.decode.reject(Error('JPEG decode failed'));await frame.promise;
 assert.equal(pixels,'blob:frame-1','bad frames must not erase the last visible image');assert.equal(draws,1);
 assert.equal(vm.runInContext('cameraReady()',context),false,'failed frames must not remain eligible for inference');
 frame=await begin();vm.runInContext('sensorGeneration++',context);frame.decode.resolve();await frame.promise;
 assert.equal(draws,1,'a frame from a stopped or changed source must not be painted');
 frame=await begin();frame.decode.image.naturalWidth=1600;frame.decode.image.naturalHeight=1200;
 assert.equal(pixels,'blob:frame-1');frame.decode.resolve();await frame.promise;
 assert.equal(width,1600);assert.equal(height,1200);assert.equal(pixels,'blob:frame-4');assert.equal(draws,2);
 assert.deepEqual(revoked,['blob:frame-1','blob:frame-2','blob:frame-3','blob:frame-4'],'release every decoded, failed, or superseded JPEG');
 console.log('SensorLink keeps visible pixels during decoding, rejects stale draws, and releases frame URLs');
})().catch(error=>{console.error(error);process.exitCode=1});
