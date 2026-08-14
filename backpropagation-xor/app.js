const sigmoid = t => 1 / (1 + Math.exp(-t));
const r6 = n => Number(n.toFixed(6));
const initial = {w11:.15,w12:.20,b1:.10,w21:-.10,w22:.05,b2:-.05,v1:.30,v2:-.25,bo:.05};
const after1 = {w11:.15,w12:.204350,b1:.104350,w21:-.10,w22:.046292,b2:-.053708,v1:.334076,v2:-.220340,bo:.109320};
const after2 = {w11:.144587,w12:.198938,b1:.098938,w21:-.096250,w22:.050043,b2:-.049957,v1:.292242,v2:-.252647,bo:.041043};
function forward(p,x1,x2,y){const z1=p.w11*x1+p.w12*x2+p.b1,h1=sigmoid(z1),z2=p.w21*x1+p.w22*x2+p.b2,h2=sigmoid(z2),zo=p.v1*h1+p.v2*h2+p.bo,yh=sigmoid(zo);return{z1,h1,z2,h2,zo,yh,loss:.5*(y-yh)**2};}
function rows(p,x1,x2,y){const f=forward(p,x1,x2,y),do_=((f.yh-y)*f.yh*(1-f.yh)),d1=do_*p.v1*f.h1*(1-f.h1),d2=do_*p.v2*f.h2*(1-f.h2),n=n=>r6(n).toFixed(6);return[
["1","z₁ = w₁₁x₁ + w₁₂x₂ + b₁","w₁₁="+n(p.w11)+", w₁₂="+n(p.w12)+", b₁="+n(p.b1)+", x₁="+x1+", x₂="+x2,p.w11.toFixed(2)+"("+x1+") + "+n(p.w12)+"("+x2+") + "+n(p.b1),r6(f.z1),"Linear score entering hidden unit 1"],
["2","h₁ = sigmoid(z₁)","z₁="+n(f.z1),"sigmoid("+n(f.z1)+")",r6(f.h1),"Output of hidden unit 1"],
["3","z₂ = w₂₁x₁ + w₂₂x₂ + b₂","w₂₁="+n(p.w21)+", w₂₂="+n(p.w22)+", b₂="+n(p.b2)+", x₁="+x1+", x₂="+x2,p.w21.toFixed(2)+"("+x1+") + "+n(p.w22)+"("+x2+") + "+n(p.b2),r6(f.z2),"Linear score entering hidden unit 2"],
["4","h₂ = sigmoid(z₂)","z₂="+n(f.z2),"sigmoid("+n(f.z2)+")",r6(f.h2),"Output of hidden unit 2"],
["5","zₒ = v₁h₁ + v₂h₂ + bₒ","v₁="+n(p.v1)+", v₂="+n(p.v2)+", bₒ="+n(p.bo)+", h₁="+n(f.h1)+", h₂="+n(f.h2),n(p.v1)+"("+n(f.h1)+") + "+n(p.v2)+"("+n(f.h2)+") + "+n(p.bo),r6(f.zo),"Output linear score / logit"],
["6","ŷ = sigmoid(zₒ)","zₒ="+n(f.zo),"sigmoid("+n(f.zo)+")",r6(f.yh),"Predicted y; actual y="+y],
["7","E = ½(y − ŷ)²","y="+y+", ŷ="+n(f.yh),"0.5("+y+" − "+n(f.yh)+")²",r6(f.loss),"Sample loss"],
["8","δₒ = (ŷ − y)ŷ(1 − ŷ)","ŷ="+n(f.yh)+", y="+y,"("+n(f.yh)+" − "+y+")("+n(f.yh)+")(1 − "+n(f.yh)+")",r6(do_),"Output error signal"],
["9","δ₁ = δₒv₁h₁(1 − h₁)","δₒ="+n(do_)+", v₁="+n(p.v1)+", h₁="+n(f.h1),n(do_)+" × "+n(p.v1)+" × "+n(f.h1)+" × (1 − "+n(f.h1)+")",r6(d1),"Hidden error signal 1"],
["10","δ₂ = δₒv₂h₂(1 − h₂)","δₒ="+n(do_)+", v₂="+n(p.v2)+", h₂="+n(f.h2),n(do_)+" × "+n(p.v2)+" × "+n(f.h2)+" × (1 − "+n(f.h2)+")",r6(d2),"Hidden error signal 2"]];}
const round1={rows:rows(initial,0,1,1),round:1},round2={rows:rows(after1,1,1,0),round:2};let revealed=0,active=1;
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
function render(id,data,count){$(id).innerHTML=data.map((r,i)=>"<tr class='"+(i<count?"revealed ":"")+(i===count-1?"active-row":"")+"'><td>"+r[0]+"</td><td>"+r[1]+"</td><td>"+r[2]+"</td><td>"+r[3]+"</td><td>"+Number(r[4]).toFixed(6)+"</td><td>"+r[5]+"</td></tr>").join("");}
function parameters(){const keys=["w11","w12","b1","w21","w22","b2","v1","v2","bo"];$("#parameter-table").innerHTML=keys.map(k=>"<tr><td>"+k+"</td><td>"+initial[k].toFixed(6)+"</td><td>"+after1[k].toFixed(6)+"</td><td>"+after2[k].toFixed(6)+"</td></tr>").join("");}
function variables(count,calc,p){const f=forward(p,active===1?0:1,1,active===1?1:0),n=v=>v===null?"—":Number(v).toFixed(6),items=[["x₁",active===1?0:1,"input feature",0],["x₂",1,"input feature",0],["y",active===1?1:0,"actual target",0],["z₁",count>=1?f.z1:null,"hidden pre-activation",1],["h₁",count>=2?f.h1:null,"hidden activation",2],["z₂",count>=3?f.z2:null,"hidden pre-activation",3],["h₂",count>=4?f.h2:null,"hidden activation",4],["zₒ",count>=5?f.zo:null,"output logit",5],["ŷ",count>=6?f.yh:null,"predicted value",6],["E",count>=7?f.loss:null,"sample loss",7],["δₒ",count>=8?calc.rows[7][4]:null,"output error signal",8],["δ₁",count>=9?calc.rows[8][4]:null,"hidden error signal",9],["δ₂",count>=10?calc.rows[9][4]:null,"hidden error signal",10]];$("#variable-table").innerHTML=items.map(v=>"<tr class='"+(v[3]===count?"current-variable ":"")+(v[1]===null?"pending-variable":"")+"'><td>"+v[0]+"</td><td>"+n(v[1])+"</td><td>"+v[2]+"</td></tr>").join("");}
function text(id,value){$(id).textContent=value;}
function updateGraph(count,calc,p){const f=forward(p,active===1?0:1,1,active===1?1:0),n=v=>v===null?"—":Number(v).toFixed(6);text("#w11-value",p.w11.toFixed(3));text("#w12-value",p.w12.toFixed(3));text("#w21-value",p.w21.toFixed(3));text("#w22-value",p.w22.toFixed(3));text("#v1-value",p.v1.toFixed(3));text("#v2-value",p.v2.toFixed(3));text("#z1-value","z₁ = "+(count>=1?n(f.z1):"—"));text("#z2-value","z₂ = "+(count>=3?n(f.z2):"—"));text("#h1-value",count>=2?n(f.h1):"—");text("#h2-value",count>=4?n(f.h2):"—");text("#zo-value","zₒ = "+(count>=5?n(f.zo):"—"));text("#yhat-value",count>=6?n(f.yh):"—");text("#loss-value","E = "+(count>=7?n(f.loss):"—"));text("#deltao-value","δₒ = "+(count>=8?n(calc.rows[7][4]):"—"));text("#delta1-value","δ₁ = "+(count>=9?n(calc.rows[8][4]):"—"));text("#delta2-value","δ₂ = "+(count>=10?n(calc.rows[9][4]):"—"));text("#graph-state-tag","state "+count+" / 10");}
function highlightGraph(count){
  $$(".graph-current,.graph-edge-current,.graph-value-current").forEach(el=>el.classList.remove("graph-current","graph-edge-current","graph-value-current"));
  const edge=id=>$(id)?.classList.add("graph-edge-current"),node=id=>$(id)?.classList.add("graph-current"),value=id=>$(id)?.classList.add("graph-value-current");
  const targets={
    1:()=>{edge("#edge-x1-h1");edge("#edge-x2-h1");value("#z1-value")},
    2:()=>{node("#node-h1");value("#h1-value")},
    3:()=>{edge("#edge-x1-h2");edge("#edge-x2-h2");value("#z2-value")},
    4:()=>{node("#node-h2");value("#h2-value")},
    5:()=>{edge("#edge-h1-y");edge("#edge-h2-y");value("#zo-value")},
    6:()=>{node("#node-output");value("#yhat-value")},
    7:()=>{node("#node-output");value("#loss-value")},
    8:()=>{edge("#edge-y-h1-back");edge("#edge-y-h2-back");value("#deltao-value")},
    9:()=>{node("#node-h1");edge("#edge-y-h1-back");value("#delta1-value")},
    10:()=>{node("#node-h2");edge("#edge-y-h2-back");value("#delta2-value")}
  };
  if(targets[count])targets[count]();
}
function updateState(count,calc){const sample=active===1?"(x₁, x₂) = (0, 1)":"(x₁, x₂) = (1, 1)",target=active===1?"actual y = 1":"actual y = 0";text("#sample-x",sample);text("#sample-y",target);if(!count){text("#state-step","Step 0 · Press “Next step” to begin");text("#state-equation","—");["#state-values","#state-calculation","#state-result","#state-meaning"].forEach(id=>text(id,"—"));return;}const r=calc.rows[count-1];text("#state-step","Step "+r[0]+" / 10 · "+(count<=7?"forward / loss":"backward signal"));text("#state-equation",r[1]);text("#state-values",r[2]);text("#state-calculation",r[3]);text("#state-result",Number(r[4]).toFixed(6));text("#state-meaning",r[5]);}
function update(){const a=Math.min(revealed,10),b=Math.max(0,Math.min(revealed-10,10)),count=active===1?a:b,calc=active===1?round1:round2,p=active===1?(a===10?after1:initial):after1;render("#trace-1",round1.rows,a);render("#trace-2",round2.rows,b);parameters();updateGraph(count,calc,p);highlightGraph(count);updateState(count,calc);$("#round-1-state").textContent=a===10?"complete":a+" / 10";$("#round-2-state").textContent=b===10?"complete":b+" / 10";const pct=Math.round(revealed*5);$("#progress-label").textContent=revealed<10?"Round 1 · "+a+" / 10 steps":"Round 2 · "+b+" / 10 steps";$("#progress-percent").textContent=pct+"%";$("#progress-bar").style.width=pct+"%";$("#next-btn").disabled=revealed>=20;$("#phase-tag").textContent=revealed<7?"Forward pass":revealed<20?"Backward + update":"Two rounds complete";$("#coach-note").textContent=revealed===0?"Start by calculating the hidden pre-activation z₁.":revealed<7?"Keep the equation and the values used visible together.":revealed<20?"Now follow the error signal backward before changing any parameter.":"Compare the parameter snapshots: every update came from a gradient.";}
function show(n){active=n;$("#round-1").classList.toggle("hidden",n!==1);$("#round-2").classList.toggle("hidden",n!==2);$$(".round-tab").forEach(b=>{const on=Number(b.dataset.round)===n;b.classList.toggle("active",on);b.setAttribute("aria-selected",on);});}
$("#next-btn").addEventListener("click",()=>{revealed=Math.min(20,revealed+1);if(revealed>10)active=2;update();});$("#round-btn").addEventListener("click",()=>{revealed=active===1?10:20;update();});$("#show-all-btn").addEventListener("click",()=>{revealed=20;show(1);update();});$("#reset-btn").addEventListener("click",()=>{revealed=0;show(1);update();});$$(".round-tab").forEach(b=>b.addEventListener("click",()=>show(Number(b.dataset.round))));update();
