(function () {
  var PALETTES = {
    slate: {
      bg: "12,12,11",
      accent: [159, 152, 240],
      teal: [93, 202, 165],
      ringBase: 0.055,
      ringStep: 0.009,
      crosshair: "rgba(255,255,255,0.02)",
      nodeAlphaMin: 0.15,
      nodeAlphaRange: 0.35,
      centerAlpha: 0.7,
      vignetteEnd: 0.88,
    },
    default: {
      bg: "250,249,247",
      accent: [79, 70, 229],
      teal: [16, 163, 127],
      ringBase: 0.09,
      ringStep: 0.014,
      crosshair: "rgba(0,0,0,0.04)",
      nodeAlphaMin: 0.25,
      nodeAlphaRange: 0.4,
      centerAlpha: 0.8,
      vignetteEnd: 0.85,
    },
  };

  function getScheme() {
    var s = document.body && document.body.getAttribute("data-md-color-scheme");
    return s === "default" ? "default" : "slate";
  }

  function initHeroCanvas() {
    var canvas = document.getElementById("ao-hero-canvas");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var W, H, nodes, animId;
    var scheme = getScheme();
    var P = PALETTES[scheme];

    function resize() {
      var hero = canvas.parentElement;
      W = canvas.width = hero.offsetWidth;
      H = canvas.height = hero.offsetHeight;
      seedNodes();
    }

    function seedNodes() {
      var cx = W / 2,
        cy = H / 2;
      var count = Math.min(28, Math.floor((W * H) / 22000));

      nodes = [
        {
          x: cx, y: cy, r: 3.5,
          color: P.accent, alpha: P.centerAlpha,
          fixed: true, baseX: cx, baseY: cy, phase: 0, speed: 0,
        },
      ];

      for (var i = 1; i < count; i++) {
        var angle = Math.random() * Math.PI * 2;
        var dist = 80 + Math.random() * Math.min(W, H) * 0.42;
        var color = Math.random() > 0.3 ? P.accent : P.teal;
        nodes.push({
          x: cx + Math.cos(angle) * dist,
          y: cy + Math.sin(angle) * dist,
          r: 1.2 + Math.random() * 1.8,
          color: color,
          alpha: P.nodeAlphaMin + Math.random() * P.nodeAlphaRange,
          fixed: false,
          baseX: cx + Math.cos(angle) * dist,
          baseY: cy + Math.sin(angle) * dist,
          phase: Math.random() * Math.PI * 2,
          speed: 0.003 + Math.random() * 0.005,
        });
      }
    }

    function draw(t) {
      ctx.clearRect(0, 0, W, H);
      var cx = W / 2,
        cy = H / 2;

      // Concentric rings
      [55, 120, 210, 320, 450].forEach(function (r, i) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle =
          "rgba(" + P.accent[0] + "," + P.accent[1] + "," + P.accent[2] + "," +
          (P.ringBase - i * P.ringStep) + ")";
        ctx.lineWidth = 0.5;
        ctx.stroke();
      });

      // Crosshair
      ctx.strokeStyle = P.crosshair;
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(cx - 500, cy);
      ctx.lineTo(cx + 500, cy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx, cy - 400);
      ctx.lineTo(cx, cy + 400);
      ctx.stroke();

      // Animate nodes
      nodes.forEach(function (n) {
        if (n.fixed) return;
        n.x = n.baseX + Math.sin(t * n.speed + n.phase) * 12;
        n.y = n.baseY + Math.cos(t * n.speed * 0.7 + n.phase) * 8;
      });

      // Edges
      var maxDist = Math.min(W, H) * 0.3;
      for (var i = 0; i < nodes.length; i++) {
        for (var j = i + 1; j < nodes.length; j++) {
          var a = nodes[i], b = nodes[j];
          var dx = a.x - b.x, dy = a.y - b.y;
          var dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > maxDist) continue;
          var opacity = (1 - dist / maxDist) * 0.18 * Math.min(a.alpha, b.alpha) * 2;
          var c = a.color;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + opacity + ")";
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }

      // Nodes
      nodes.forEach(function (n) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(" + n.color[0] + "," + n.color[1] + "," + n.color[2] + "," + n.alpha + ")";
        ctx.fill();
      });

      // Radial vignette
      var grad = ctx.createRadialGradient(cx, cy, H * 0.2, cx, cy, H * 0.85);
      grad.addColorStop(0, "rgba(" + P.bg + ",0)");
      grad.addColorStop(1, "rgba(" + P.bg + "," + P.vignetteEnd + ")");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);

      // Top fade
      var topFade = ctx.createLinearGradient(0, 0, 0, 100);
      topFade.addColorStop(0, "rgba(" + P.bg + ",1)");
      topFade.addColorStop(1, "rgba(" + P.bg + ",0)");
      ctx.fillStyle = topFade;
      ctx.fillRect(0, 0, W, 100);

      animId = requestAnimationFrame(function (t2) {
        draw(t2 * 0.001);
      });
    }

    window.addEventListener("resize", function () {
      cancelAnimationFrame(animId);
      resize();
      draw(0);
    });
    resize();
    draw(0);

    // Watch for light/dark toggle
    var observer = new MutationObserver(function () {
      var newScheme = getScheme();
      if (newScheme !== scheme) {
        scheme = newScheme;
        P = PALETTES[scheme];
        cancelAnimationFrame(animId);
        resize();
        draw(0);
      }
    });
    observer.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initHeroCanvas);
  } else {
    initHeroCanvas();
  }
  document.addEventListener("DOMContentLoaded", initHeroCanvas);

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      initHeroCanvas();
    });
  }
})();
