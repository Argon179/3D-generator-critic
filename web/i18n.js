/* Interface dictionaries are presentation-only; scientific decisions remain unchanged. */
(()=>{
 const stored=localStorage.getItem('gai.language');let language=stored==='en'?'en':'ru',dictionary={},runPairs={},reversePairs={};
 const originals=new WeakMap(),attributeSources=new WeakMap();let scheduled=false;
 function replace(text,map){
  if(map[text]!==undefined)return map[text];
  const keys=Object.keys(map).filter(k=>k&&text.includes(k)).sort((a,b)=>b.length-a.length);
  if(!keys.length)return text;
  const escape=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  return text.replace(new RegExp(keys.map(escape).join('|'),'gu'),m=>map[m]);
 }
 function translate(value){
  const source=String(value??'');
  if(language==='ru')return replace(source,reversePairs);
  const result=replace(source,{...runPairs,...dictionary});
  return Object.keys(dictionary).length&&/[А-Яа-яЁё]/.test(result)&&source.length>100?'English translation pending…':result;
 }
 function apply(){
  scheduled=false;document.documentElement.lang=language;
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  while(walker.nextNode()){
   const node=walker.currentNode;if(node.parentElement?.closest('script,style,pre,textarea,[data-no-translate]'))continue;
   if(!originals.has(node))originals.set(node,node.nodeValue);
   const next=translate(originals.get(node));if(next!==node.nodeValue)node.nodeValue=next;
  }
  for(const element of document.querySelectorAll('[placeholder],[title],[aria-label]')){
   if(!attributeSources.has(element))attributeSources.set(element,{});
   const attrs=attributeSources.get(element);
   for(const key of ['placeholder','title','aria-label'])if(element.hasAttribute(key)){
    attrs[key]??=element.getAttribute(key);const next=translate(attrs[key]);if(next!==element.getAttribute(key))element.setAttribute(key,next);
   }
  }
  const button=document.getElementById('languageToggle');if(button){const label=language==='en'?'🌐 RU':'🌐 EN';if(button.textContent!==label)button.textContent=label;button.title=language==='en'?'Switch to Russian':'Переключить на английский';button.setAttribute('aria-label',button.title)}
 }
 function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(apply)}}
 function pairs(value){
  if(Array.isArray(value)){value.forEach(pairs);return}
  if(!value||typeof value!=='object')return;
  for(let i=0;i<(value.quality_warnings?.length??0);i++){const ru=value.quality_warnings[i],en=value.quality_warnings_en?.[i];if(typeof ru==='string'&&typeof en==='string'){runPairs[ru]=en;reversePairs[en]=ru}}
  for(const [key,text]of Object.entries(value)){
   if(key.endsWith('_ru')&&typeof text==='string'&&typeof value[key.slice(0,-3)+'_en']==='string'){runPairs[text]=value[key.slice(0,-3)+'_en'];reversePairs[value[key.slice(0,-3)+'_en']]=text;}
   pairs(text);
  }
 }
 let runId=null,fetching=false;
 async function loadRun(id,data){
  if(id!==runId){runId=id;runPairs={};reversePairs={};fetching=false}
  pairs(data);const notice=document.getElementById('translationNotice');if(notice){notice.hidden=language!=='en'||!data?.iterations?.some(i=>!i.critique?.summary_en);notice.textContent=notice.hidden?'':'Legacy critiques are translated from Russian. New critiques use English originals.'}schedule();const incomplete=data?.iterations?.some(i=>i.critique?.translation_status==='incomplete'||i.verification?.translation_status==='incomplete');if((language!=='en'&&!incomplete)||fetching||!id)return;
  fetching=true;
  try{
   let response=await fetch('/api/runs/'+id+'/translations');let result=await response.json();
   if(result.status==='missing'&&data?.status==='completed'){
    response=await fetch('/api/runs/'+id+'/translations',{method:'POST'});result=await response.json();
   }
   if(id===runId&&result.pairs){Object.assign(runPairs,result.pairs);for(const [ru,en]of Object.entries(result.pairs))reversePairs[en]=ru;schedule()}
   if(['queued','running'].includes(result.status))setTimeout(()=>{fetching=false;loadRun(id,data)},2500);
   else if(result.status!=='completed')fetching=false;
  }catch{fetching=false}
 }
 window.gaiI18n={get language(){return language},apply,schedule,loadRun,translate};
 document.getElementById('languageToggle').onclick=()=>{
  language=language==='ru'?'en':'ru';localStorage.setItem('gai.language',language);fetching=false;apply();window.dispatchEvent(new Event('gai-language-change'));
 };
 new MutationObserver(schedule).observe(document.body,{childList:true,subtree:true});
 fetch('/translations.json').then(r=>r.json()).then(d=>{dictionary=d;apply()});apply();
})();
