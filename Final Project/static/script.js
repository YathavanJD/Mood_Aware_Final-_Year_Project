window.onload = () => {
    createAmbientParticles();
};

/* ============ SOFT PARTICLES (NOT DISTRACTING) ============ */

function createAmbientParticles(){

    setInterval(()=>{

        const p=document.createElement("div");

        p.style.position="fixed";
        p.style.width="3px";
        p.style.height="3px";
        p.style.borderRadius="50%";
        p.style.background="rgba(255,255,255,.4)";
        p.style.left=Math.random()*100+"vw";
        p.style.top="100vh";
        p.style.zIndex="0";
        p.style.transition="transform 10s linear, opacity 10s";

        document.body.appendChild(p);

        setTimeout(()=>{
            p.style.transform="translateY(-120vh)";
            p.style.opacity="0";
        },50);

        setTimeout(()=>p.remove(),10000);

    },800);
}
