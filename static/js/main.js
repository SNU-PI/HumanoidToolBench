(() => {
  "use strict";

  document.documentElement.classList.add("js");

  /* ---------- Top bar ---------- */
  const topbar = document.querySelector(".topbar");
  const onScroll = () => topbar.classList.toggle("is-stuck", window.scrollY > 8);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  // Only in-page links take part in the scroll highlight.
  const navLinks = [...document.querySelectorAll('.topnav a[href^="#"]')];
  const targets = navLinks
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);
  if ("IntersectionObserver" in window && targets.length) {
    const seen = new Map();
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => seen.set(e.target.id, e.isIntersecting));
        const current = targets.find((t) => seen.get(t.id));
        navLinks.forEach((a) =>
          a.classList.toggle("is-active", !!current && a.getAttribute("href") === "#" + current.id)
        );
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    targets.forEach((t) => io.observe(t));
  }

  /* ---------- Clips: play while on screen ---------- */
  const clips = [...document.querySelectorAll("video[data-clip]")];
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  clips.forEach((v) => {
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    if (calm) v.controls = true;
  });
  if (!calm && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          const v = e.target;
          if (e.isIntersecting) {
            const p = v.play();
            if (p && p.catch) p.catch(() => (v.controls = true));
          } else {
            v.pause();
          }
        });
      },
      { threshold: 0.25 }
    );
    clips.forEach((v) => io.observe(v));
  } else {
    clips.forEach((v) => (v.controls = true));
  }

  /* ---------- Standard / decoy switch ---------- */
  document.querySelectorAll("[data-tabs]").forEach((group) => {
    const card = group.closest(".card");
    const buttons = [...group.querySelectorAll("button")];
    const show = (mode) => {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
      card.querySelectorAll("[data-pane]").forEach((p) => (p.hidden = p.dataset.pane !== mode));
    };
    buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset.mode)));
    group.hidden = false;
    show("S");
  });

  /* ---------- Heat-table tooltip ---------- */
  const tip = document.createElement("div");
  tip.className = "tip";
  // A visual echo of the cell under the pointer; the table itself carries the values.
  tip.setAttribute("aria-hidden", "true");
  const tipValue = document.createElement("b");
  const tipLabel = document.createElement("span");
  tip.append(tipValue, tipLabel);
  document.body.append(tip);

  const place = (x, y) => {
    const r = tip.getBoundingClientRect();
    let left = x + 14;
    let top = y + 16;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 12;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  };

  document.addEventListener("pointermove", (e) => {
    const cell = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (!cell) {
      tip.classList.remove("on");
      return;
    }
    tipValue.textContent = cell.dataset.value || cell.textContent.trim();
    tipLabel.textContent = cell.dataset.tip;
    tip.classList.add("on");
    place(e.clientX, e.clientY);
  });
  window.addEventListener("scroll", () => tip.classList.remove("on"), { passive: true });

  /* ---------- Live counts: GitHub stars, Hugging Face likes ---------- */
  const SOURCES = {
    github: {
      urls: ["https://api.github.com/repos/SNU-PI/HumanoidToolBench"],
      field: "stargazers_count",
    },
    "hf-dataset": {
      // expand[]=likes keeps the reply to the one field; the full record lists every file.
      urls: ["https://huggingface.co/api/datasets/snupilab/humanoidtoolbench-teleop?expand[]=likes"],
      field: "likes",
    },
  };
  const CACHE_MS = 10 * 60 * 1000;
  const compact = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n));

  const readCache = (key) => {
    try {
      const hit = JSON.parse(sessionStorage.getItem("htb-count-" + key));
      return hit && Date.now() - hit.t < CACHE_MS ? hit.v : null;
    } catch (err) {
      return null;
    }
  };
  const writeCache = (key, v) => {
    try {
      sessionStorage.setItem("htb-count-" + key, JSON.stringify({ t: Date.now(), v }));
    } catch (err) {
      /* storage can be unavailable; the count is simply fetched again next time */
    }
  };

  const loadCount = async (key) => {
    const cached = readCache(key);
    if (cached !== null) return cached;
    const { urls, field } = SOURCES[key];
    const values = await Promise.all(
      urls.map(async (url) => {
        const reply = await fetch(url, { headers: { Accept: "application/json" } });
        if (!reply.ok) throw new Error(String(reply.status));
        const value = (await reply.json())[field];
        if (!Number.isFinite(value)) throw new Error("no " + field);
        return value;
      })
    );
    const total = values.reduce((a, b) => a + b, 0);
    writeCache(key, total);
    return total;
  };

  Object.keys(SOURCES).forEach((key) => {
    const slots = document.querySelectorAll(`[data-count="${key}"]`);
    if (!slots.length) return;
    loadCount(key)
      .then((n) =>
        slots.forEach((slot) => {
          slot.querySelector("span").textContent = compact(n);
          slot.setAttribute("aria-label", `${slot.title}: ${n}`);
          slot.classList.remove("is-pending");
        })
      )
      .catch(() => {
        /* rate limit or offline: the button stays as it is, without a count */
        slots.forEach((slot) => (slot.hidden = true));
      });
  });

  /* ---------- Copy button ---------- */
  document.querySelectorAll(".copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.parentElement.querySelector("pre").dataset.copy;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
      } catch (err) {
        btn.textContent = "Press Ctrl+C";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  });
})();
