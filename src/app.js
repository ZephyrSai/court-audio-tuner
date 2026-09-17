/* Court Audio Tuner — DSP engine + UI. Mirrors the Python prototypes (dsplib2.py) and the Android port. */
(() => {
'use strict';
const $ = id => document.getElementById(id);
const clamp = (v,a,b) => v<a?a:(v>b?b:v);
const db2lin = d => Math.pow(10, d/20);
const lin2db = l => 20*Math.log10(l+1e-12);

/* ---------------- FFT (radix-2, in place) ---------------- */
const N = 1024, HOP = 512, BINS = N/2+1;
const cosT = new Float32Array(N/2), sinT = new Float32Array(N/2), rev = new Uint16Array(N);
for (let i=0;i<N/2;i++){ const a=-2*Math.PI*i/N; cosT[i]=Math.cos(a); sinT[i]=Math.sin(a); }
{ const bits=10; for (let i=0;i<N;i++){ let r=0; for(let b=0;b<bits;b++) r=(r<<1)|((i>>b)&1); rev[i]=r; } }
function fft(re, im){
  for (let a=0;a<N;a++){ const b=rev[a]; if(b>a){ let t=re[a];re[a]=re[b];re[b]=t; t=im[a];im[a]=im[b];im[b]=t; } }
  for (let len=2;len<=N;len<<=1){ const half=len>>1, step=N/len;
    for (let s=0;s<N;s+=len){ for (let j=0,t=0;j<half;j++,t+=step){ const wr=cosT[t],wi=sinT[t],p=s+j,q=p+half;
      const xr=re[q]*wr-im[q]*wi, xi=re[q]*wi+im[q]*wr; re[q]=re[p]-xr; im[q]=im[p]-xi; re[p]+=xr; im[p]+=xi; } } }
}
const win = new Float32Array(N); for (let n=0;n<N;n++) win[n]=Math.sqrt(0.5-0.5*Math.cos(2*Math.PI*n/N));

/* ---------------- Biquad HPF (4th-order Butterworth) ---------------- */
function hpfCoeffs(sr, fc){ const Q=[0.54119610,1.30656296], w0=2*Math.PI*fc/sr, c=Math.cos(w0), s=Math.sin(w0);
  return Q.map(q=>{ const al=s/(2*q), a0=1+al; return {b0:((1+c)/2)/a0,b1:(-(1+c))/a0,b2:((1+c)/2)/a0,a1:(-2*c)/a0,a2:(1-al)/a0}; }); }
function applyHPF(x, sr, fc){ if(!(fc>0)) return x; const y=Float32Array.from(x);
  for (const k of hpfCoeffs(sr,fc)){ let z1=0,z2=0; for(let i=0;i<y.length;i++){ const inp=y[i], out=k.b0*inp+z1; z1=k.b1*inp-k.a1*out+z2; z2=k.b2*inp-k.a2*out; y[i]=out; } }
  return y; }

/* ---------------- Bands ---------------- */
const USER_EDGES = [0,150,300,600,1200,2500,4000,6000,8000,12000,1e9];
const USER_NAMES = ['<150','150–300','300–600','600–1.2k','1.2–2.5k','2.5–4k','4–6k','6–8k','8–12k','>12k'];
function makeBands(sr, nb, fLo){ // triangular bands, columns sum to 1 => interpolation weights
  const freqs = new Float32Array(BINS); for (let k=0;k<BINS;k++) freqs[k]=k*sr/N;
  const edges=[0]; for (let b=0;b<nb;b++) edges.push(fLo*Math.pow(sr/2/fLo, b/(nb-1)));
  const centers=[]; for (let b=0;b<nb;b++) centers.push((edges[b]+edges[b+1])/2); centers[0]=0; centers[nb-1]=sr/2;
  const W = Array.from({length:nb},()=>new Float32Array(BINS));
  for (let b=0;b<nb;b++){ const lo=b>0?centers[b-1]:0, c=centers[b], hi=b<nb-1?centers[b+1]:sr/2;
    for (let k=0;k<BINS;k++){ const f=freqs[k]; if(f>=lo&&f<=c) W[b][k]=(f-lo)/Math.max(c-lo,1e-9); else if(f>c&&f<=hi) W[b][k]=(hi-f)/Math.max(hi-c,1e-9); } }
  for (let k=0;k<BINS;k++){ if(freqs[k]<=centers[0]) W[0][k]=1; if(freqs[k]>=centers[nb-1]) W[nb-1][k]=1; }
  for (let k=0;k<BINS;k++){ let s=0; for(let b=0;b<nb;b++) s+=W[b][k]; if(s>1e-9) for(let b=0;b<nb;b++) W[b][k]/=s; }
  const userIdx = centers.map(c=>{ for(let u=0;u<10;u++) if(c>=USER_EDGES[u]&&c<USER_EDGES[u+1]) return u; return 9; });
  const nbins = W.map(w=>w.reduce((a,v)=>a+(v>0?1:0),0));
  return {W, centers, userIdx, nbins, freqs};
}

/* ---------------- Engine ---------------- */
/* Returns {out: Float32Array, gain: Float32Array[frames*BINS] (linear), frames} */
function process(x, sr, cfg){
  const algo=cfg.algo; const y = algo==='off' ? x : applyHPF(x, sr, cfg.hpfHz);
  const padded = new Float32Array(y.length + 2*N); padded.set(y, HOP); // HOP pre-pad so frame 0 is centred
  const out = new Float32Array(padded.length);
  const nFrames = Math.floor((padded.length - N)/HOP);
  const gainField = new Float32Array(nFrames*BINS);
  const re=new Float32Array(N), im=new Float32Array(N), P=new Float32Array(BINS);
  const {W, userIdx, nbins} = makeBands(sr, cfg.bands, 60);
  const nb=cfg.bands;
  // per-band floors / eq from user strips
  const floorDbBand = new Float32Array(nb), eqLinBand = new Float32Array(nb);
  for (let b=0;b<nb;b++){ const u=userIdx[b]; floorDbBand[b]=clamp(cfg.floorDb + cfg.floorTrim[u], 0, 40); eqLinBand[b]=db2lin(cfg.eq[u]); }
  const eqBin = new Float32Array(BINS); for(let k=0;k<BINS;k++){ let s=0; for(let b=0;b<nb;b++) s+=W[b][k]*eqLinBand[b]; eqBin[k]=algo==='off'?1:s; }
  const userOfBin = new Uint8Array(BINS); for(let k=0;k<BINS;k++){ const f=k*sr/N; let u=9; for(let i=0;i<10;i++) if(f>=USER_EDGES[i]&&f<USER_EDGES[i+1]){ u=i; break; } userOfBin[k]=u; }
  const floorLinUser = new Float32Array(10); for(let u=0;u<10;u++) floorLinUser[u]=db2lin(-clamp(cfg.floorDb+cfg.floorTrim[u],0,40));
  // state
  const E=new Float32Array(nb), Es=new Float32Array(nb), env=new Float32Array(nb), noise=new Float32Array(nb), prevClean=new Float32Array(nb);
  const G=new Float32Array(nb), Gprev=new Float32Array(nb).fill(1), GdbPrev=new Float32Array(nb);
  let Gexp=null; // expander gain state (starts at floor)
  const perbinNoise=new Float32Array(BINS), perbinSmooth=new Float32Array(BINS), perbinPrev=new Float32Array(BINS); let perbinInit=false;
  const fr=HOP/sr, warmFrames=Math.max(1,Math.round(cfg.warmupS/fr));
  const aPsd = new Float32Array(nb); for(let b=0;b<nb;b++) aPsd[b]=clamp(1-nbins[b]/cfg.targetAvg,0.5,0.95);
  const aEa = cfg.envAttackMs>0?Math.exp(-fr/(cfg.envAttackMs/1000)):0, aEr=Math.exp(-fr/(cfg.envReleaseMs/1000));
  const aGa = cfg.gainAttackMs>0?Math.exp(-fr/(cfg.gainAttackMs/1000)):0, aGr=Math.exp(-fr/(cfg.gainReleaseMs/1000));
  const aG40 = Math.exp(-fr/(cfg.gainTcMs/1000));
  const gbin = new Float32Array(BINS);
  let init=false;
  for (let f=0; f<nFrames; f++){
    const off=f*HOP;
    for (let n=0;n<N;n++){ re[n]=padded[off+n]*win[n]; im[n]=0; }
    fft(re,im);
    for (let k=0;k<BINS;k++) P[k]=re[k]*re[k]+im[k]*im[k];
    if (algo==='off'){ gbin.fill(1); }
    else if (algo==='eq'){ for(let k=0;k<BINS;k++) gbin[k]=1; }
    else if (algo==='perbin'){
      for (let k=0;k<BINS;k++){ const p=P[k]; let ps;
        if(!perbinInit){ ps=p; perbinSmooth[k]=p; perbinNoise[k]=p; }
        else { ps=0.7*perbinSmooth[k]+0.3*p; perbinSmooth[k]=ps; let nz=perbinNoise[k];
          if(f<warmFrames) nz+=0.2*(ps-nz); else if(ps>nz) nz+=cfg.noiseUp*(ps-nz); else nz+=cfg.noiseDown*(ps-nz); perbinNoise[k]=nz; }
        const nzb=cfg.noiseBias*perbinNoise[k]+1e-12, gamma=p/nzb, gm1=gamma>1?gamma-1:0;
        const xi=perbinInit? cfg.alpha*(perbinPrev[k]/nzb)+(1-cfg.alpha)*gm1 : gm1;
        let g=xi/(1+xi); if(cfg.transientRelease && gamma>8){ const gt=1-1/gamma; if(gt>g) g=gt; }
        const gminB=floorLinUser[userOfBin[k]]; if(g<gminB) g=gminB;
        gbin[k]=g; perbinPrev[k]=p*g*g; }
      perbinInit=true;
    } else {
      // band energies
      for (let b=0;b<nb;b++){ let s=0; const w=W[b]; for(let k=0;k<BINS;k++) s+=w[k]*P[k]; E[b]=s; }
      if(!init){ for(let b=0;b<nb;b++){ Es[b]=E[b]; env[b]=E[b]; noise[b]=E[b]; } }
      else for(let b=0;b<nb;b++){
        const a = algo==='bandwiener'? aPsd[b] : cfg.noiseSmooth;
        Es[b]=a*Es[b]+(1-a)*E[b];
        env[b]= E[b]>env[b] ? aEa*env[b]+(1-aEa)*E[b] : aEr*env[b]+(1-aEr)*E[b];
        let nz=noise[b]; if(f<warmFrames) nz+=0.2*(Es[b]-nz); else if(Es[b]>nz) nz+=cfg.noiseUp*(Es[b]-nz); else nz+=cfg.noiseDown*(Es[b]-nz); noise[b]=nz; }
      if (algo==='bandwiener'){
        for (let b=0;b<nb;b++){ const Nn=cfg.noiseBias*noise[b]+1e-12, gS=Es[b]/Nn, gR=E[b]/Nn, gm1=gS>1?gS-1:0;
          const xi = init? cfg.alpha*(prevClean[b]/Nn)+(1-cfg.alpha)*gm1 : gm1;
          let g=xi/(1+xi); const trans = cfg.transientRelease && gR>cfg.transientGamma; if(trans){ const gt=1-1/gR; if(gt>g) g=gt; }
          let gdb=Math.max(lin2db(g), -floorDbBand[b]);
          if (init){ const prev=GdbPrev[b];
            if (gdb>prev) gdb = trans? gdb : Math.min(gdb, prev+cfg.riseDb); else gdb = Math.max(gdb, aG40*prev+(1-aG40)*gdb);
            gdb = Math.max(gdb, prev-cfg.fallDb); }
          GdbPrev[b]=gdb; G[b]=db2lin(gdb); }
      } else { // expander
        if(!Gexp){ Gexp=new Float32Array(nb); for(let b=0;b<nb;b++) Gexp[b]=db2lin(-floorDbBand[b]); }
        for (let b=0;b<nb;b++){ const gmin=db2lin(-floorDbBand[b]); const snr=10*Math.log10(env[b]/(cfg.noiseBias*noise[b]+1e-12)+1e-9);
          let o=clamp((snr-cfg.thrLo)/(cfg.thrHi-cfg.thrLo),0,1); o=o*o*(3-2*o);
          const gt=gmin+(1-gmin)*o; Gexp[b] = gt>Gexp[b] ? aGa*Gexp[b]+(1-aGa)*gt : aGr*Gexp[b]+(1-aGr)*gt; G[b]=Gexp[b]; }
      }
      for (let k=0;k<BINS;k++){ let s=0; for(let b=0;b<nb;b++) s+=W[b][k]*G[b]; gbin[k]=s; }
      init=true;
    }
    for (let k=0;k<BINS;k++){ const g=gbin[k]*eqBin[k]; gainField[f*BINS+k]=g; re[k]*=g; im[k]*=g; }
    if (algo==='bandwiener') for (let b=0;b<nb;b++){ let s=0; const w=W[b]; for(let k=0;k<BINS;k++) s+=w[k]*(re[k]*re[k]+im[k]*im[k]); prevClean[b]=s; }
    for (let k=1;k<N/2;k++){ re[N-k]=re[k]; im[N-k]=-im[k]; }
    for (let n=0;n<N;n++) im[n]=-im[n];
    fft(re,im);
    const inv=1/N; for (let n=0;n<N;n++) out[off+n]+=re[n]*inv*win[n];
  }
  const res = out.subarray(HOP, HOP+y.length);
  const outGain = db2lin(cfg.outputGainDb); const final=new Float32Array(y.length);
  for (let i=0;i<y.length;i++) final[i]=clamp(res[i]*outGain,-1,1);
  return {out: final, gain: gainField, frames: nFrames};
}

/* ---------------- Metrics ---------------- */
function frameRmsDb(x, sr, winS){ const h=Math.floor(winS*sr), r=[]; for(let i=0;i+h<=x.length;i+=h){ let s=0; for(let j=i;j<i+h;j++) s+=x[j]*x[j]; r.push(10*Math.log10(s/h+1e-12)); } return r; }
function pct(arr,p){ const a=Array.from(arr).sort((x,y)=>x-y); return a[Math.floor(p*(a.length-1))]; }
function stftBandDb(x, sr, lo, hi){ // per-frame energy (dB) in [lo,hi]
  const re=new Float32Array(N), im=new Float32Array(N), nF=Math.floor((x.length-N)/HOP), r=new Float32Array(nF);
  const k0=Math.floor(lo/sr*N), k1=Math.min(BINS-1,Math.ceil(hi/sr*N));
  for (let f=0;f<nF;f++){ for(let n=0;n<N;n++){ re[n]=x[f*HOP+n]*win[n]; im[n]=0; } fft(re,im); let s=0; for(let k=k0;k<=k1;k++) s+=re[k]*re[k]+im[k]*im[k]; r[f]=10*Math.log10(s/(k1-k0+1)+1e-16); }
  return r; }
function measure(orig, proc, gain, frames, sr){
  const fo=frameRmsDb(orig,sr,0.1), fp=frameRmsDb(proc,sr,0.1);
  const floorO=pct(fo,0.1), floorP=pct(fp,0.1);
  const hfO=stftBandDb(orig,sr,2000,8000), hfP=stftBandDb(proc,sr,2000,8000);
  const med=pct(hfO,0.5); let hits=[]; for(let i=0;i<hfO.length;i++) if(hfO[i]>med+10) hits.push(i);
  const hitO=hits.length?hits.reduce((a,i)=>a+hfO[i],0)/hits.length:NaN, hitP=hits.length?hits.reduce((a,i)=>a+hfP[i],0)/hits.length:NaN;
  const lfO=pct(stftBandDb(orig,sr,60,400),0.5), lfP=pct(stftBandDb(proc,sr,60,400),0.5);
  // blips from the gain field: 2-12 kHz
  const k0=Math.floor(2000/sr*N), k1=Math.min(BINS-1,Math.floor(12000/sr*N)); let cells=0, blips=0;
  for (let k=k0;k<=k1;k++){ for (let f=1;f<frames-2;f++){ const g0=lin2db(gain[(f-1)*BINS+k]), g1=lin2db(gain[f*BINS+k]), g2=lin2db(gain[(f+1)*BINS+k]), g3=lin2db(gain[(f+2)*BINS+k]);
    cells++; if((g1-g0>=6&&g2-g1<=-6)||(g1-g0>=6&&g3-g1<=-6)) blips++; } }
  return {floorO,floorP,hitO,hitP,lfO,lfP,blips: cells?blips/cells*1000:0, hits:hits.length};
}

/* ---------------- Config / UI ---------------- */
const DEFAULTS = {
  algo:'expander', hpfHz:120, floorDb:15, bands:32, outputGainDb:0,
  alpha:0.97, noiseBias:1.5, noiseUp:0.004, noiseDown:0.25, warmupS:0.75, targetAvg:24, noiseSmooth:0.8,
  riseDb:1.5, fallDb:4, gainTcMs:40, transientRelease:true, transientGamma:6,
  thrLo:4, thrHi:14, envAttackMs:5, envReleaseMs:40, gainAttackMs:0, gainReleaseMs:150,
  floorTrim:[0,0,0,0,0,0,0,0,0,0], eq:[0,0,0,0,0,0,0,0,0,0]
};
const KNOBS = [ // id, label, min, max, step, unit, algos
  ['hpfHz','Low-cut (HPF)',0,300,10,'Hz',['expander','bandwiener','eq','perbin']],
  ['floorDb','Floor / max attenuation',0,30,1,'dB',['expander','bandwiener','perbin']],
  ['bands','Band count',8,64,4,'',['expander','bandwiener']],
  ['outputGainDb','Output gain',-12,12,0.5,'dB',['expander','bandwiener','eq','perbin','off']],
  ['thrLo','Open threshold (band SNR)',0,15,0.5,'dB',['expander']],
  ['thrHi','Fully open at',4,30,0.5,'dB',['expander']],
  ['envAttackMs','Envelope attack',0,50,1,'ms',['expander']],
  ['envReleaseMs','Envelope release',5,300,5,'ms',['expander']],
  ['gainAttackMs','Gain open time',0,50,1,'ms',['expander']],
  ['gainReleaseMs','Gain close time',20,800,10,'ms',['expander']],
  ['alpha','Decision-directed α',0.8,0.995,0.005,'',['bandwiener','perbin']],
  ['riseDb','Gain rise limit / frame',0.25,12,0.25,'dB',['bandwiener']],
  ['fallDb','Gain fall limit / frame',0.5,24,0.5,'dB',['bandwiener']],
  ['gainTcMs','Gain smoothing τ',5,300,5,'ms',['bandwiener']],
  ['transientGamma','Transient release above',3,20,0.5,'×',['bandwiener']],
  ['noiseBias','Noise bias',1,3,0.05,'×',['expander','bandwiener','perbin']],
  ['noiseUp','Noise track up',0.0005,0.05,0.0005,'',['expander','bandwiener','perbin']],
  ['noiseDown','Noise track down',0.02,0.6,0.01,'',['expander','bandwiener','perbin']],
  ['warmupS','Noise learn warm-up',0.2,3,0.05,'s',['expander','bandwiener','perbin']],
];
const NOTES = {
  expander:'Each band sits at a constant floor while only noise is present and opens when its SNR exceeds the threshold. No flutter by construction; tune thresholds and close time for naturalness.',
  bandwiener:'Wiener gain per band, decision-directed, with rise/fall limits and τ smoothing. More reduction than the expander, small residual modulation.',
  eq:'No adaptive processing: just the low-cut and the static per-band EQ row. Cannot produce artifacts. Use it as the baseline.',
  perbin:'The first implementation you heard (per-FFT-bin gains). Kept so you can hear the crackle for comparison.',
  off:'Bypass — original audio (output gain still applies).'
};
let cfg = JSON.parse(JSON.stringify(DEFAULTS));
const BUILTIN = {
  'Expander · gentle': {algo:'expander',hpfHz:100,floorDb:12},
  'Expander · balanced': {algo:'expander',hpfHz:120,floorDb:15},
  'Expander · hum-focused': {algo:'expander',hpfHz:150,floorDb:12,floorTrim:[8,8,6,3,0,0,0,0,0,0]},
  'Band-Wiener · 15 dB': {algo:'bandwiener',hpfHz:120,floorDb:15},
  'Static EQ · low-cut + shelf': {algo:'eq',hpfHz:120,eq:[-12,-10,-8,-4,-1,0,0,0,0,0]},
  'OLD per-bin (crackly)': {algo:'perbin',hpfHz:100,floorDb:12},
};

function buildKnobs(){ const host=$('globalKnobs'); host.innerHTML='';
  for (const [id,label,min,max,step,unit,algos] of KNOBS){
    const wrap=document.createElement('div'); wrap.className='knob'; wrap.dataset.algos=algos.join(' ');
    wrap.innerHTML=`<div><div class="lab"><span>${label}</span><span>${unit}</span></div><input type="range" id="k_${id}" min="${min}" max="${max}" step="${step}" aria-label="${label}"></div><input type="number" id="n_${id}" min="${min}" max="${max}" step="${step}" aria-label="${label} value">`;
    host.appendChild(wrap);
    const r=$('k_'+id), n=$('n_'+id);
    r.addEventListener('input',()=>{ cfg[id]=parseFloat(r.value); n.value=r.value; schedule(); });
    n.addEventListener('change',()=>{ cfg[id]=clamp(parseFloat(n.value)||0,min,max); r.value=cfg[id]; n.value=cfg[id]; schedule(); });
  }
  const tr=document.createElement('div'); tr.className='row'; tr.dataset.algos='bandwiener perbin';
  tr.innerHTML='<label for="k_transientRelease">Transient release (let hits through instantly)</label><input type="checkbox" id="k_transientRelease">';
  host.appendChild(tr); $('k_transientRelease').addEventListener('change',e=>{ cfg.transientRelease=e.target.checked; schedule(); });
}
function buildBands(){
  const mk=(hostId,key,min,max,step)=>{ const host=$(hostId); host.innerHTML='';
    for (let u=0;u<10;u++){ const d=document.createElement('div'); d.className='band';
      d.innerHTML=`<span class="v mono" id="${key}_v${u}">0</span><input type="range" class="vslider" id="${key}_${u}" min="${min}" max="${max}" step="${step}" orient="vertical" aria-label="${USER_NAMES[u]} ${key}"><span class="f mono">${USER_NAMES[u]}</span>`;
      host.appendChild(d); const s=$(`${key}_${u}`); s.addEventListener('input',()=>{ cfg[key][u]=parseFloat(s.value); $(`${key}_v${u}`).textContent=fmt(cfg[key][u]); schedule(); }); } };
  mk('floorBands','floorTrim',-15,15,1); mk('eqBands','eq',-18,6,1);
}
const fmt = v => (Math.round(v*1000)/1000).toString();
function syncUI(){
  $('algo').value=cfg.algo; $('algoNote').textContent=NOTES[cfg.algo];
  for (const [id] of KNOBS){ $('k_'+id).value=cfg[id]; $('n_'+id).value=cfg[id]; }
  $('k_transientRelease').checked=!!cfg.transientRelease;
  document.querySelectorAll('#globalKnobs [data-algos]').forEach(el=>{ el.classList.toggle('hidden', !el.dataset.algos.split(' ').includes(cfg.algo)); });
  for (let u=0;u<10;u++){ $('floorTrim_'+u).value=cfg.floorTrim[u]; $('floorTrim_v'+u).textContent=fmt(cfg.floorTrim[u]); $('eq_'+u).value=cfg.eq[u]; $('eq_v'+u).textContent=fmt(cfg.eq[u]); }
  $('cfgOut').value='{\n'+Object.entries(cfg).map(([k,v])=>`  "${k}": ${JSON.stringify(v)}`).join(',\n')+'\n}';
}
function applyPreset(p){ cfg=Object.assign(JSON.parse(JSON.stringify(DEFAULTS)), JSON.parse(JSON.stringify(p))); syncUI(); schedule(0); }
function buildPresets(){ const row=$('presetRow'); row.innerHTML='';
  const saved=(()=>{ try{ return JSON.parse(localStorage.getItem('cat_presets')||'{}'); }catch(e){ return {}; } })();
  for (const [name,p] of [...Object.entries(BUILTIN),...Object.entries(saved)]){ const b=document.createElement('button'); b.textContent=name; b.addEventListener('click',()=>applyPreset(p)); row.appendChild(b); }
}

/* ---------------- Audio ---------------- */
let ctx=null, orig=null, sr=44100, procBuf=null, origBuf=null, srcO=null, srcP=null, gO=null, gP=null, anO=null, anP=null;
let playing=false, hearing='proc', t0=0, startOffset=0, lastResult=null, timer=null;
function ensureCtx(){ if(!ctx){ ctx=new (window.AudioContext||window.webkitAudioContext)(); gO=ctx.createGain(); gP=ctx.createGain(); anO=ctx.createAnalyser(); anP=ctx.createAnalyser(); anO.fftSize=4096; anP.fftSize=4096; anO.smoothingTimeConstant=0.85; anP.smoothingTimeConstant=0.85;
  gO.connect(anO).connect(ctx.destination); gP.connect(anP).connect(ctx.destination); setHearing(hearing); } return ctx; }
function setHearing(h){ hearing=h; const b=$('abBtn'); b.dataset.state=h; b.textContent = h==='proc'?'Hearing: PROCESSED':'Hearing: ORIGINAL';
  if(gO&&gP){ const t=ctx.currentTime; gO.gain.setTargetAtTime(h==='orig'?1:0,t,0.01); gP.gain.setTargetAtTime(h==='proc'?1:0,t,0.01); } }
function makeBuffer(data){ const b=ctx.createBuffer(1,data.length,sr); b.getChannelData(0).set(data); return b; }
function loopRange(){ const dur=orig.length/sr; let a=clamp(parseFloat($('loopStart').value)||0,0,dur-0.1), b=clamp(parseFloat($('loopEnd').value)||dur,a+0.1,dur); return [a,b]; }
function stopSources(){ for (const s of [srcO,srcP]) if(s){ try{ s.stop(); }catch(e){} } srcO=srcP=null; }
function startSources(offset){ stopSources(); const [a,b]=loopRange(); offset = (offset>=a&&offset<b)? offset : a;
  srcO=ctx.createBufferSource(); srcO.buffer=origBuf; srcO.loop=true; srcO.loopStart=a; srcO.loopEnd=b; srcO.connect(gO);
  srcP=ctx.createBufferSource(); srcP.buffer=procBuf; srcP.loop=true; srcP.loopStart=a; srcP.loopEnd=b; srcP.connect(gP);
  const t=ctx.currentTime+0.02; srcO.start(t,offset); srcP.start(t,offset); t0=t; startOffset=offset; }
function currentPos(){ if(!playing) return startOffset; const [a,b]=loopRange(); let p=startOffset+(ctx.currentTime-t0); if(p>=b) p=a+((p-a)%(b-a)); return p; }
function play(){ ensureCtx(); if(ctx.state==='suspended') ctx.resume(); if(!procBuf) return; if(playing){ startOffset=currentPos(); stopSources(); playing=false; $('playBtn').textContent='▶ Play loop'; return; }
  startSources(startOffset); playing=true; $('playBtn').textContent='■ Stop'; }

/* ---------------- Render loop ---------------- */
let pending=null;
function schedule(delay=120){ clearTimeout(pending); pending=setTimeout(run, delay); }
function run(){ if(!orig) return; syncUI(); const s=performance.now(); const r=process(orig, sr, cfg); const ms=performance.now()-s; lastResult=r;
  ensureCtx(); procBuf=makeBuffer(r.out); if(playing) startSources(currentPos());
  const m=measure(orig, r.out, r.gain, r.frames, sr);
  const dF=m.floorP-m.floorO, dH=m.hitP-m.hitO, dL=m.lfP-m.lfO;
  const cls=(v,goodBelow,badAbove)=> v<=goodBelow?'good':(v>=badAbove?'bad':'');
  $('metrics').innerHTML=[
    ['Noise floor (10th pct)',`${m.floorO.toFixed(1)} → ${m.floorP.toFixed(1)} dBFS`,''],
    ['Floor change',`${dF>0?'+':''}${dF.toFixed(1)} dB`,cls(dF,-8,0)],
    ['Hum band 60–400 Hz',`${dL>0?'+':''}${dL.toFixed(1)} dB`,cls(dL,-8,0)],
    [`Ball hits (${m.hits} events, 2–8 kHz)`,`${isNaN(dH)?'—':(dH>0?'+':'')+dH.toFixed(1)+' dB'}`, isNaN(dH)?'':(dH>=-2?'good':(dH<=-5?'bad':''))],
    ['Blips (musical noise)',`${m.blips.toFixed(2)} ‰`, m.blips<1?'good':(m.blips>4?'bad':'')],
    ['Processing',`${ms.toFixed(0)} ms for ${(orig.length/sr).toFixed(1)} s`,'']
  ].map(([k,v,c])=>`<span class="k">${k}</span><span class="v mono ${c}">${v}</span>`).join('');
  drawGainMap(r); $('status').textContent=`rendered · ${cfg.algo}`; }

function drawGainMap(r){ const c=$('gainmap'); const W=c.clientWidth||800; c.width=W; const H=c.height; const g=c.getContext('2d'); const img=g.createImageData(W,H);
  const kMax=Math.min(BINS-1,Math.floor(12000/sr*N));
  for (let x=0;x<W;x++){ const f=Math.floor(x/W*r.frames); for (let y=0;y<H;y++){ const k=Math.floor((1-y/H)*kMax); const v=clamp((lin2db(r.gain[f*BINS+k])+30)/30,0,1); const i=(y*W+x)*4;
    img.data[i]=Math.round(20+235*v*v); img.data[i+1]=Math.round(20+170*v); img.data[i+2]=Math.round(30+40*v); img.data[i+3]=255; } }
  g.putImageData(img,0,0); }

function drawSpectrum(){ requestAnimationFrame(drawSpectrum); const c=$('spec'); if(!anO) return; const W=c.clientWidth||800; if(c.width!==W) c.width=W; const H=c.height; const g=c.getContext('2d');
  const css=getComputedStyle(document.documentElement); g.fillStyle=css.getPropertyValue('--panel2'); g.fillRect(0,0,W,H);
  const fo=new Float32Array(anO.frequencyBinCount), fp=new Float32Array(anP.frequencyBinCount); anO.getFloatFrequencyData(fo); anP.getFloatFrequencyData(fp);
  const nyq=ctx.sampleRate/2, fmin=20, fmax=16000, xOf=f=>W*Math.log(f/fmin)/Math.log(fmax/fmin), yOf=d=>H-(clamp(d,-110,-10)+110)/100*H;
  g.strokeStyle=css.getPropertyValue('--line'); g.lineWidth=1; g.font='10px IBM Plex Mono, monospace'; g.fillStyle=css.getPropertyValue('--muted');
  for (const f of [50,100,200,500,1000,2000,5000,10000]){ const x=xOf(f); g.beginPath(); g.moveTo(x,0); g.lineTo(x,H); g.stroke(); g.fillText(f>=1000?(f/1000)+'k':f, x+3, H-4); }
  for (const d of [-20,-40,-60,-80,-100]){ const y=yOf(d); g.beginPath(); g.moveTo(0,y); g.lineTo(W,y); g.stroke(); g.fillText(d+' dB',3,y-2); }
  const draw=(arr,color)=>{ g.strokeStyle=color; g.lineWidth=1.6; g.beginPath(); let first=true;
    for (let x=0;x<W;x+=2){ const f=fmin*Math.pow(fmax/fmin,x/W); const b=f/nyq*arr.length; const b0=Math.max(0,Math.floor(b/1.12)), b1=Math.min(arr.length-1,Math.ceil(b*1.12)); let s=0,n=0; for(let i=b0;i<=b1;i++){ s+=Math.pow(10,arr[i]/10); n++; } const d=10*Math.log10(s/n+1e-12); const y=yOf(d); if(first){ g.moveTo(x,y); first=false; } else g.lineTo(x,y); } g.stroke(); };
  draw(fo, css.getPropertyValue('--orig')); draw(fp, css.getPropertyValue('--proc'));
  if(playing){ const p=currentPos(); $('posSlider').value=p; $('posLabel').textContent=p.toFixed(2)+' s'; } }

/* ---------------- Export (WAV inside a stored ZIP) ---------------- */
function encodeWav(data, sr){ const n=data.length, buf=new ArrayBuffer(44+n*2), v=new DataView(buf); const str=(o,s)=>{ for(let i=0;i<s.length;i++) v.setUint8(o+i,s.charCodeAt(i)); };
  str(0,'RIFF'); v.setUint32(4,36+n*2,true); str(8,'WAVE'); str(12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true); v.setUint32(24,sr,true); v.setUint32(28,sr*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true); str(36,'data'); v.setUint32(40,n*2,true);
  for (let i=0;i<n;i++){ const x=clamp(data[i],-1,1); v.setInt16(44+i*2, x<0?x*32768:x*32767, true); } return new Uint8Array(buf); }
const CRC_T=(()=>{ const t=new Uint32Array(256); for(let i=0;i<256;i++){ let c=i; for(let k=0;k<8;k++) c=c&1?(0xEDB88320^(c>>>1)):(c>>>1); t[i]=c>>>0; } return t; })();
function crc32(u){ let c=0xFFFFFFFF; for(let i=0;i<u.length;i++) c=CRC_T[(c^u[i])&0xFF]^(c>>>8); return (c^0xFFFFFFFF)>>>0; }
function zipStore(files){ // files: [{name, data:Uint8Array}] -> Blob (method 0, no compression)
  const enc=new TextEncoder(), parts=[], central=[]; let off=0;
  const u16=n=>[n&255,(n>>8)&255], u32=n=>[n&255,(n>>8)&255,(n>>16)&255,(n>>>24)&255];
  for (const f of files){ const name=enc.encode(f.name), crc=crc32(f.data), sz=f.data.length;
    const lh=new Uint8Array([0x50,0x4b,3,4,...u16(20),...u16(0x0800),...u16(0),...u16(0),...u16(0),...u32(crc),...u32(sz),...u32(sz),...u16(name.length),...u16(0)]);
    parts.push(lh,name,f.data);
    central.push(new Uint8Array([0x50,0x4b,1,2,...u16(20),...u16(20),...u16(0x0800),...u16(0),...u16(0),...u16(0),...u32(crc),...u32(sz),...u32(sz),...u16(name.length),...u16(0),...u16(0),...u16(0),...u16(0),...u32(0),...u32(off)]),name);
    off+=lh.length+name.length+sz; }
  const cdStart=off, cdSize=central.reduce((a,c)=>a+c.length,0);
  const end=new Uint8Array([0x50,0x4b,5,6,...u16(0),...u16(0),...u16(files.length),...u16(files.length),...u32(cdSize),...u32(cdStart),...u16(0)]);
  return new Blob([...parts,...central,end],{type:'application/zip'}); }
async function saveProcessed(){ if(!lastResult){ $('copyStatus').textContent='Nothing rendered yet.'; return; }
  let data=lastResult.out; let tag='full';
  if ($('saveLoopOnly').checked){ const [a,b]=loopRange(); data=data.subarray(Math.floor(a*sr),Math.floor(b*sr)); tag=`${a.toFixed(1)}-${b.toFixed(1)}s`; }
  const base=`processed_${cfg.algo}_${cfg.floorDb}dB_hpf${cfg.hpfHz}_${tag}`;
  const zip=zipStore([{name:base+'.wav',data:encodeWav(data,sr)},{name:base+'.config.json',data:new TextEncoder().encode(JSON.stringify(cfg,null,2))}]);
  $('copyStatus').textContent='Preparing '+(zip.size/1048576).toFixed(1)+' MB…';
  let dl=null; try{ if(window.claude&&claude.use) dl=await claude.use('downloads'); }catch(e){ dl=null; }
  if (dl){ try{ await dl.save({filename:base+'.zip',data:zip}); $('copyStatus').textContent='Saved '+base+'.zip'; }
    catch(err){ $('copyStatus').textContent = err&&err.code==='declined'?'Save cancelled.':('Save failed: '+(err&&err.message||err)); } return; }
  const url=URL.createObjectURL(zip); const a=document.createElement('a'); a.href=url; a.download=base+'.zip'; document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),5000);
  $('copyStatus').textContent='Downloaded '+base+'.zip'; }

/* ---------------- Loading ---------------- */
function b64ToBuf(b64){ const bin=atob(b64); const u=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i); return u.buffer; }
async function loadArrayBuffer(ab, name){ ensureCtx(); const ab2=await ctx.decodeAudioData(ab.slice(0)); sr=ab2.sampleRate; const ch=ab2.numberOfChannels; const d=new Float32Array(ab2.length);
  for (let c=0;c<ch;c++){ const x=ab2.getChannelData(c); for(let i=0;i<d.length;i++) d[i]+=x[i]/ch; }
  orig=d; origBuf=makeBuffer(d); const dur=d.length/sr; $('posSlider').max=dur; $('loopEnd').value=dur.toFixed(1); $('loopEnd').max=dur; $('loopStart').max=dur; $('durLabel').textContent=dur.toFixed(2)+' s';
  $('sampleInfo').textContent=`${name} · ${(sr/1000).toFixed(1)} kHz · ${ch===1?'mono':ch+' ch (mixed to mono)'} · ${dur.toFixed(1)} s`; startOffset=0; if(playing){ stopSources(); playing=false; $('playBtn').textContent='▶ Play loop'; }
  run(); }

function init(){
  buildKnobs(); buildBands(); buildPresets(); syncUI();
  $('algo').addEventListener('change',e=>{ cfg.algo=e.target.value; syncUI(); schedule(0); });
  $('playBtn').addEventListener('click',play);
  $('abBtn').addEventListener('click',()=>{ ensureCtx(); setHearing(hearing==='proc'?'orig':'proc'); });
  $('posSlider').addEventListener('change',e=>{ startOffset=parseFloat(e.target.value); $('posLabel').textContent=startOffset.toFixed(2)+' s'; if(playing) startSources(startOffset); });
  for (const id of ['loopStart','loopEnd']) $(id).addEventListener('change',()=>{ const [a]=loopRange(); startOffset=a; if(playing) startSources(a); });
  $('zeroBands').addEventListener('click',()=>{ cfg.floorTrim=Array(10).fill(0); cfg.eq=Array(10).fill(0); syncUI(); schedule(0); });
  $('resetBtn').addEventListener('click',()=>applyPreset({}));
  $('savePreset').addEventListener('click',()=>{ const name=prompt('Preset name'); if(!name) return; try{ const s=JSON.parse(localStorage.getItem('cat_presets')||'{}'); s[name]=cfg; localStorage.setItem('cat_presets',JSON.stringify(s)); }catch(e){} buildPresets(); });
  $('copyCfg').addEventListener('click',async()=>{ const txt=JSON.stringify(cfg); try{ await navigator.clipboard.writeText(txt); $('copyStatus').textContent='Copied — paste it in chat.'; }catch(e){ $('cfgOut').select(); $('copyStatus').textContent='Select the text above and copy it.'; } });
  $('loadCfg').addEventListener('click',async()=>{ try{ const t=await navigator.clipboard.readText(); applyPreset(JSON.parse(t)); $('copyStatus').textContent='Loaded config from clipboard.'; }catch(e){ $('copyStatus').textContent='Clipboard had no valid config JSON.'; } });
  $('saveWav').addEventListener('click',saveProcessed);
  $('fileInput').addEventListener('change',async e=>{ const f=e.target.files[0]; if(!f) return; $('status').textContent='decoding '+f.name+'…'; try{ await loadArrayBuffer(await f.arrayBuffer(), f.name); }catch(err){ $('status').textContent='could not decode '+f.name; } });
  document.addEventListener('keydown',e=>{ if(e.code==='Space'&&e.target.tagName!=='INPUT'&&e.target.tagName!=='TEXTAREA'){ e.preventDefault(); play(); } if(e.key==='a'||e.key==='A'){ if(e.target.tagName!=='INPUT'&&e.target.tagName!=='TEXTAREA'){ ensureCtx(); setHearing(hearing==='proc'?'orig':'proc'); } } });
  drawSpectrum();
  const b64=$('wavdata').textContent.trim();
  loadArrayBuffer(b64ToBuf(b64),'1-original.wav').then(()=>{ $('status').textContent='ready · press Play loop'; }).catch(e=>{ $('status').textContent='decode failed: '+e.message; });
}
if (document.readyState==='loading') document.addEventListener('DOMContentLoaded',init); else init();
})();
