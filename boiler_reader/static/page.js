let refreshMs=BOILER.refreshMs;
const frame=document.getElementById('frame');
const stage=document.getElementById('stage');
const roiBox=document.getElementById('roi-box');
let imageSize=null;
let currentRoi=null;
let drag=null;
let saving=false;
let phase='idle';
let llmImageVersion=null;
let flashValuePending=false;
let llmFlashPending=false;
let llmAwaiting=false;
const llmImage=document.getElementById('llm-image');
// Orange pulse on the LLM image from the end of the green flash until the reading is decided.
function applyLlmWait(){
    llmImage.classList.toggle('waiting',phase==='analyzing'&&llmAwaiting&&!llmImage.classList.contains('flash'));
}
llmImage.onload=()=>{
    llmImage.hidden=false;
    if(llmFlashPending){
        llmFlashPending=false;
        llmAwaiting=true;
        llmImage.classList.remove('flash');void llmImage.offsetWidth;llmImage.classList.add('flash');
        applyLlmWait();
    }
};
llmImage.addEventListener('animationend',()=>{llmImage.classList.remove('flash');applyLlmWait();});
llmImage.onerror=()=>{llmImage.hidden=true;llmFlashPending=false;llmAwaiting=false;applyLlmWait();};
let roiResult=null;
function applyRoiState(){
    roiBox.classList.toggle('ok',roiResult==='ok');
    roiBox.classList.toggle('bad',roiResult==='bad');
    document.getElementById('grab-dot').classList.toggle('on',phase==='grabbing');
    applyLlmWait();
}
async function pollActivity(){
    try{
        const response=await fetch('/activity.json?t='+Date.now(),{cache:'no-store'});
        const activity=await response.json();
        const previous=phase;
        phase=activity.phase;
        if(phase!=='analyzing'||previous!=='analyzing')llmAwaiting=false;
        if(phase==='analyzing'&&previous!=='analyzing'&&!drag&&!saving)frame.src='/last.jpg?t='+Date.now();
        if(phase==='analyzing'&&activity.roi&&!drag&&!saving){
            currentRoi=activity.roi;imageSize=activity.image_size;drawRoi();
        }
        if(activity.llm_image!==llmImageVersion){
            const first=llmImageVersion===null;
            llmImageVersion=activity.llm_image;
            // The first poll after opening the page only loads the existing image.
            if(!first){llmFlashPending=true;clearReplies();}
            if(!first||activity.llm_image>0)llmImage.src='/llm-image.jpg?t='+Date.now();
        }
        applyRoiState();
        if(previous==='analyzing'&&phase!=='analyzing'){flashValuePending=true;refresh();}
    }catch(error){}
}
function drawRoi(){
    if(!currentRoi||!imageSize||!frame.naturalWidth)return;
    const imageRect=frame.getBoundingClientRect();
    const stageRect=stage.getBoundingClientRect();
    const scaleX=imageRect.width/imageSize.width;
    const scaleY=imageRect.height/imageSize.height;
    roiBox.style.left=(imageRect.left-stageRect.left+currentRoi.x*scaleX)+'px';
    roiBox.style.top=(imageRect.top-stageRect.top+currentRoi.y*scaleY)+'px';
    roiBox.style.width=(currentRoi.width*scaleX)+'px';
    roiBox.style.height=(currentRoi.height*scaleY)+'px';
}
function showReading(reading){
    imageSize=reading.image_size||imageSize;
    currentRoi=reading.roi||currentRoi;
    const valid=reading.valid;
    roiResult=(valid&&!reading.issue)?'ok':'bad';
    applyRoiState();
    const unit=reading.unit||'bar';
    document.getElementById('value').textContent=valid?Number(reading.value).toFixed(1):'--';
    if(flashValuePending){
        flashValuePending=false;
        const valueEl=document.getElementById('value');
        valueEl.classList.remove('flash');void valueEl.offsetWidth;valueEl.classList.add('flash');
    }
    document.getElementById('unit').textContent=unit;
    document.getElementById('unit').hidden=!valid;
    document.getElementById('display').textContent=reading.display_text||'--';
    document.getElementById('error-code').textContent=reading.error_code||'--';
    showReplies(reading);
    const delay=reading.delay;
    document.getElementById('issue-delay').textContent=delay&&delay.issue
        ?(delay.issue==='low_pressure'?'Low pressure: ':'Error: ')+(delay.eligible?'ready':delay.remaining_seconds+'s remaining')
        :'No active issue';
    const status=document.getElementById('status');
    status.textContent=valid?(unit==='°C'?'VALID TEMPERATURE':'VALID PRESSURE'):'INVALID / ERROR';
    status.classList.toggle('invalid',!valid);
    document.getElementById('updated').textContent=BOILER.buildLabel;
    if(reading.camera)showLastGrabbed(reading.camera);
    drawRoi();
}
function clearReplies(){document.getElementById('llm-replies').replaceChildren();}
// The replies belong to the final reading, so they stay hidden while the LLM is still working.
function showReplies(reading){
    if(phase==='analyzing')return;
    const list=document.getElementById('llm-replies');
    list.replaceChildren();
    (reading.llm_replies||[]).forEach(reply=>{
        const item=document.createElement('li');
        item.textContent=reply;
        if(reply===reading.raw_pattern)item.className='win';
        list.appendChild(item);
    });
}
function showLastGrabbed(camera){
    const lastGrabbed=document.getElementById('last-grabbed');
    if(!camera||!camera.checked_at){lastGrabbed.textContent='Last frame: unknown';return;}
    const timestamp=new Date(camera.checked_at).toLocaleString();
    lastGrabbed.textContent=(camera.available?'Last frame: ':'Last grab failed: ')+timestamp;
}
let hasReading=false;
async function refresh(){
        if(drag||saving)return;
        frame.src='/last.jpg?t='+Date.now();
    const status=document.getElementById('status');
    try{
        const response=await fetch('/reading.json?t='+Date.now(),{cache:'no-store'});
        const reading=await response.json();
        if(!response.ok)throw new Error(reading.error||'Reading unavailable');
        if(reading.busy){
            if(!hasReading){
                status.textContent='INITIALISING';
                status.classList.remove('invalid');
                document.getElementById('updated').textContent='Initialising: waiting for the first frame...';
            }
            return;
        }
        hasReading=true;
        showReading(reading);
    }catch(error){
        document.getElementById('value').textContent='--';
        document.getElementById('unit').hidden=true;
        document.getElementById('display').textContent=error.message;
        document.getElementById('error-code').textContent='--';
        document.getElementById('issue-delay').textContent=error.message;
        status.textContent='NO IMAGE';
        status.classList.add('invalid');
        document.getElementById('updated').textContent='Waiting for latest frame';
        fetch('/camera-status.json?t='+Date.now(),{cache:'no-store'})
            .then(response=>response.json()).then(showLastGrabbed).catch(()=>{});
    }
}
frame.addEventListener('load',()=>{frame.hidden=false;drawRoi();});
frame.addEventListener('error',()=>{frame.hidden=true;});
let grabbing=false;
frame.title='Tap to grab a new frame';
frame.addEventListener('click',async()=>{
    if(grabbing)return;
    grabbing=true;
    document.getElementById('updated').textContent='Grabbing frame...';
    try{
        const response=await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Camera grab failed');
        showLastGrabbed(result.camera);
        await refresh();
        document.getElementById('updated').textContent='Frame grabbed and decoded';
    }catch(error){
        document.getElementById('status').textContent='CAMERA GRAB FAILED';
        document.getElementById('status').classList.add('invalid');
        document.getElementById('updated').textContent=error.message;
    }finally{grabbing=false;}
});
document.getElementById('send-test-webhook').addEventListener('click',async event=>{
    if(!confirm('Send a simulated ECS error to Home Assistant? This may trigger a mobile notification.'))return;
    const button=event.currentTarget;
    button.disabled=true;button.textContent='Sending test...';
    let outcome='Send test HA message';
    try{
        const response=await fetch('/test-webhook',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Webhook delivery failed');
        document.getElementById('updated').textContent='Test HA message sent';
        outcome='Sent ✓';
    }catch(error){
        document.getElementById('updated').textContent=error.message;
        outcome='Failed: '+error.message;
    }finally{
        button.disabled=false;button.textContent=outcome;
        if(outcome!=='Send test HA message')setTimeout(()=>{button.textContent='Send test HA message';},4000);
    }
});
roiBox.addEventListener('pointerdown',event=>{
    if(!imageSize||!currentRoi)return;
    event.preventDefault();
    const resizing=event.target.id==='resize-handle';
    drag={pointerId:event.pointerId,resizing,startX:event.clientX,startY:event.clientY,roi:{...currentRoi}};
    roiBox.setPointerCapture(event.pointerId);
});
roiBox.addEventListener('pointermove',event=>{
    if(!drag||drag.pointerId!==event.pointerId)return;
    const rect=frame.getBoundingClientRect();
    const dx=Math.round((event.clientX-drag.startX)*imageSize.width/rect.width);
    const dy=Math.round((event.clientY-drag.startY)*imageSize.height/rect.height);
    if(drag.resizing){
        currentRoi.width=Math.max(25,Math.min(imageSize.width-drag.roi.x,drag.roi.width+dx));
        currentRoi.height=Math.max(25,Math.min(imageSize.height-drag.roi.y,drag.roi.height+dy));
    }else{
        currentRoi.x=Math.max(0,Math.min(imageSize.width-drag.roi.width,drag.roi.x+dx));
        currentRoi.y=Math.max(0,Math.min(imageSize.height-drag.roi.height,drag.roi.y+dy));
    }
    drawRoi();
});
async function commitRoi(){
    if(!drag)return;
    drag=null;saving=true;
    const status=document.getElementById('status');
    status.textContent='SCANNING';
    try{
        const response=await fetch('/roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentRoi)});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Could not update ROI');
        showReading(result);
    }catch(error){
        status.textContent='ROI UPDATE FAILED';
        status.classList.add('invalid');
        document.getElementById('updated').textContent=error.message;
        refresh();
    }finally{saving=false;}
}
roiBox.addEventListener('pointerup',commitRoi);
roiBox.addEventListener('pointercancel',commitRoi);
window.addEventListener('resize',drawRoi);
let refreshTimer=setInterval(refresh,refreshMs);
setInterval(pollActivity,400);
document.getElementById('admin-btn').addEventListener('click',async()=>{
    const adminOverlay=document.getElementById('admin-overlay');
    adminOverlay.hidden=false;
    adminOverlay.style.display='flex';
    try{
        const r=await fetch('/admin-settings');
        const s=await r.json();
        document.getElementById('a-rtsp-url').value=s.rtsp_url||'';
        document.getElementById('a-rtsp-user').value=s.rtsp_user||'';
        document.getElementById('a-rtsp-pass').value=s.rtsp_pass||'';
        document.getElementById('a-poll-seconds').value=s.poll_seconds||30;
        document.getElementById('a-ha-webhook-url').value=s.ha_webhook_url||'';
        document.getElementById('a-ha-webhook-timeout').value=s.ha_webhook_timeout||5;
        document.getElementById('a-low-threshold').value=s.low_threshold||1.0;
        document.getElementById('a-high-threshold').value=s.high_threshold||1.5;
        document.getElementById('a-alert-delay-minutes').value=s.alert_delay_minutes||5;
        document.getElementById('a-brightness-threshold').value=s.brightness_threshold||30;
        document.getElementById('a-expected-format').value=s.expected_format||'auto';
        document.getElementById('a-ollama-url').value=s.ollama_url||'';
        document.getElementById('a-ollama-model').value=s.ollama_model||'lfm3-vl:latest';
    }catch(e){
        document.getElementById('updated').textContent='Could not load admin settings: '+e.message;
    }
});
document.querySelectorAll('details.grp').forEach(d=>d.addEventListener('toggle',()=>{
    if(d.open)document.querySelectorAll('details.grp').forEach(o=>{if(o!==d)o.open=false;});
}));
let logLast=0;
let logTimer=null;
let logEntries=[];
function logLine(entry){
    let text=entry.message;
    let bad=entry.level==='ERROR'||entry.level==='WARNING';
    try{
        const data=JSON.parse(entry.message);
        const {event,...rest}=data;
        text=event+Object.entries(rest).map(([k,v])=>' '+k+'='+(typeof v==='string'?v:JSON.stringify(v))).join('');
        if(/fail|error|unavailable/.test(event))bad=true;
    }catch(e){}
    return {time:entry.time.slice(11,19),text,bad};
}
function renderLog(){
    const box=document.getElementById('log-lines');
    const errorsOnly=document.getElementById('log-errors-only').checked;
    const stick=box.scrollTop+box.clientHeight>=box.scrollHeight-20;
    box.replaceChildren();
    logEntries.filter(e=>!errorsOnly||e.bad).forEach(e=>{
        const row=document.createElement('div');
        if(e.bad)row.className='bad';
        const time=document.createElement('span');
        time.className='t';time.textContent=e.time+' ';
        row.append(time,e.text);
        box.appendChild(row);
    });
    if(stick)box.scrollTop=box.scrollHeight;
}
async function pollLog(){
    try{
        const response=await fetch('/log.json?since='+logLast,{cache:'no-store'});
        const data=await response.json();
        if(data.last<logLast){logLast=0;logEntries=[];}
        if(data.entries.length){
            logEntries=logEntries.concat(data.entries.map(logLine)).slice(-500);
            renderLog();
        }
        logLast=data.last;
    }catch(e){}
}
function showTab(name){
    document.querySelectorAll('button.tab').forEach(t=>t.classList.toggle('on',t.dataset.tab===name));
    document.getElementById('settings-pane').hidden=name!=='settings';
    document.getElementById('settings-pane').style.display=name==='settings'?'flex':'none';
    document.getElementById('log-pane').hidden=name!=='log';
    clearInterval(logTimer);logTimer=null;
    if(name==='log'){
        pollLog();
        logTimer=setInterval(pollLog,2000);
        const box=document.getElementById('log-lines');
        box.scrollTop=box.scrollHeight;
    }
}
document.querySelectorAll('button.tab').forEach(t=>t.addEventListener('click',()=>showTab(t.dataset.tab)));
document.getElementById('log-errors-only').addEventListener('change',renderLog);
document.getElementById('log-copy').addEventListener('click',async event=>{
    const button=event.currentTarget;
    const text=Array.from(document.getElementById('log-lines').children).map(r=>r.textContent).join('\n');
    try{await navigator.clipboard.writeText(text);button.textContent='Copied';}
    catch(e){button.textContent='Copy failed';}
    setTimeout(()=>{button.textContent='Copy';},2000);
});
function closeAdmin(){const o=document.getElementById('admin-overlay');o.hidden=true;o.style.display='none';showTab('settings');}
document.addEventListener('click',e=>{const o=document.getElementById('admin-overlay');if(o&&e.target===o)closeAdmin();});
document.addEventListener('keydown',e=>{const o=document.getElementById('admin-overlay');if(e.key==='Escape'&&o&&!o.hidden)closeAdmin();});
document.addEventListener('click',e=>{if(e.target&&e.target.id==='admin-close')closeAdmin();});
document.getElementById('admin-form').addEventListener('submit',async event=>{
    event.preventDefault();
    const button=document.getElementById('admin-save');
    button.disabled=true;button.textContent='Saving...';
    try{
        const payload={
            rtsp_url:document.getElementById('a-rtsp-url').value.trim(),
            rtsp_user:document.getElementById('a-rtsp-user').value.trim(),
            rtsp_pass:document.getElementById('a-rtsp-pass').value,
            poll_seconds:Number(document.getElementById('a-poll-seconds').value),
            ha_webhook_url:document.getElementById('a-ha-webhook-url').value.trim(),
            ha_webhook_timeout:Number(document.getElementById('a-ha-webhook-timeout').value),
            low_threshold:Number(document.getElementById('a-low-threshold').value),
            high_threshold:Number(document.getElementById('a-high-threshold').value),
            alert_delay_minutes:Number(document.getElementById('a-alert-delay-minutes').value),
            brightness_threshold:Number(document.getElementById('a-brightness-threshold').value),
            expected_format:document.getElementById('a-expected-format').value,
            ollama_url:document.getElementById('a-ollama-url').value.trim(),
            ollama_model:document.getElementById('a-ollama-model').value.trim(),
        };
        const response=await fetch('/admin-settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Could not save admin settings');
        closeAdmin();
        document.getElementById('updated').textContent='Admin settings saved';
    }catch(error){
        document.getElementById('updated').textContent=error.message;
    }finally{button.disabled=false;button.textContent='Save';}
});
function openCaptureModal(){const o=document.getElementById('capture-overlay');o.hidden=false;o.style.display='flex';document.getElementById('capture-value').focus();}
function closeCaptureModal(){const o=document.getElementById('capture-overlay');o.hidden=true;o.style.display='none';document.getElementById('capture-form').reset();}
llmImage.addEventListener('click',openCaptureModal);
llmImage.title='Click to capture a misread';
document.getElementById('capture-close').addEventListener('click',closeCaptureModal);
document.getElementById('capture-cancel').addEventListener('click',closeCaptureModal);
document.addEventListener('click',e=>{const o=document.getElementById('capture-overlay');if(o&&e.target===o)closeCaptureModal();});
document.addEventListener('keydown',e=>{const o=document.getElementById('capture-overlay');if(e.key==='Escape'&&o&&!o.hidden)closeCaptureModal();});
document.getElementById('capture-form').addEventListener('submit',async event=>{
    event.preventDefault();
    const btn=document.getElementById('capture-save');
    const value=document.getElementById('capture-value').value.trim();
    const unit=document.querySelector('input[name="capture-unit"]:checked').value;
    if(!value){document.getElementById('updated').textContent='Enter a reading value';return;}
    btn.disabled=true;btn.textContent='Saving...';
    try{
        const response=await fetch('/save-mismatch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value,unit})});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Save failed');
        closeCaptureModal();
        document.getElementById('updated').textContent='Saved: '+result.saved;
    }catch(error){
        document.getElementById('updated').textContent=error.message;
    }finally{btn.disabled=false;btn.textContent='Save';}
});
refresh();
