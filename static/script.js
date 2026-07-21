/* =====================================================================
   MOOD AWARE AI — shared front-end utilities
   Toasts, scroll-reveal, and small progressive-enhancement helpers
   used across every page.
   ===================================================================== */

/* ---------- Toasts (replaces alert()) ---------- */
(function () {
  function ensureStack() {
    let stack = document.getElementById("toast-stack");
    if (!stack) {
      stack = document.createElement("div");
      stack.id = "toast-stack";
      document.body.appendChild(stack);
    }
    return stack;
  }

  window.showToast = function (message, type = "info", duration = 3800) {
    const stack = ensureStack();
    const item = document.createElement("div");
    item.className = `toast-item ${type}`;
    item.textContent = message;
    stack.appendChild(item);
    setTimeout(() => {
      item.style.transition = "opacity .3s ease, transform .3s ease";
      item.style.opacity = "0";
      item.style.transform = "translateY(8px)";
      setTimeout(() => item.remove(), 300);
    }, duration);
  };
})();

/* ---------- Scroll reveal ---------- */
(function () {
  const items = document.querySelectorAll(".reveal");
  if (!items.length) return;

  if (!("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("in-view"));
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  items.forEach((el) => io.observe(el));
})();

/* ---------- Auto-dismiss flash alerts ---------- */
(function () {
  document.querySelectorAll(".alert").forEach((el) => {
    setTimeout(() => {
      el.style.transition = "opacity .4s ease";
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 400);
    }, 5000);
  });
})();
