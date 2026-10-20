(()=>{
 const theme=document.getElementById('theme');
 try{const saved=localStorage.getItem('ml-theme');if(saved)document.body.dataset.theme=saved;}catch{}
 theme.addEventListener('click',()=>{const current=document.body.dataset.theme??(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');const value=current==='dark'?'light':'dark';document.body.dataset.theme=value;try{localStorage.setItem('ml-theme',value);}catch{}});
 const field=document.getElementById('search'),results=document.getElementById('results');
 const prefix=location.pathname.includes('/chapters/')?'../':'';
 field.addEventListener('input',()=>{const q=field.value.trim().toLowerCase();results.replaceChildren();if(!q)return;const matches=(window.ML_SEARCH??[]).filter(x=>(x.title+x.text).toLowerCase().includes(q)).slice(0,12);const status=document.createElement('p');status.textContent=matches.length?`找到 ${matches.length} 个相关页面`:'未找到相关页面';results.append(status);for(const item of matches){const a=document.createElement('a');a.href=prefix+item.url;a.textContent=item.title;const snippet=document.createElement('div');snippet.className='search-snippet';const i=item.text.toLowerCase().indexOf(q);snippet.textContent=item.text.slice(Math.max(0,i-22),Math.max(0,i-22)+100);results.append(a,snippet);}});
})();
